# Phase 1A: SFT Loss Correction & Verification — Investigation Report

**Document:** Forensic Investigation of Existing SFT Loss Calculation  
**Target File:** `modern_gpt/train_sft_4k.py`  
**Status:** COMPLETE — PHASE 0 HYPOTHESIS CONFIRMED WITH MATHEMATICAL PROOF

---

## 1. Executive Summary

This investigation conducted a line-by-line mathematical trace of the loss calculation, gradient accumulation, and validation evaluation in `modern_gpt/train_sft_4k.py`.

### Core Findings:
1. **Hypothesis Confirmed**: The Phase 0 forensic hypothesis is **100% confirmed**. The training loop computed cross-entropy using PyTorch's default `reduction="mean"` independently for each microbatch, divided each microbatch mean by `window_size`, and accumulated gradients sequentially.
2. **Objective Discrepancy**: Training optimized the unweighted mean of microbatch means ($L_{\text{incorrect}} = \frac{1}{K}\sum_{i=1}^K \frac{S_i}{N_i}$), whereas the mathematically correct objective is the global token-weighted mean ($L_{\text{correct}} = \frac{\sum_{i=1}^K S_i}{\sum_{i=1}^K N_i}$).
3. **Severe Gradient Skew**: A token in a 20-token Commonsense QA response exerted **80 times more gradient influence** on model parameters than a token in a 1,600-token retrieval or code response.
4. **Train vs. Validation Objective Mismatch**: In `train_sft_4k.py`, `evaluate()` was *already* computing $L_{\text{val}} = \frac{\sum S_i}{\sum N_i}$ using `reduction="sum"`. Thus, validation loss was measuring a fundamentally different objective than the one being optimized during training.
5. **Zero-Token Vulnerability**: When all tokens in a microbatch are masked (`-100`), PyTorch's `F.cross_entropy(..., reduction="mean")` returns `NaN`, crashing the trainer at line 404 (`assert torch.isfinite(loss)`).

---

## 2. Detailed Code Trace of Previous Implementation

In `modern_gpt/train_sft_4k.py` (lines 388–434):

```python
# Exact partial-window normalization:
# Determine size of current accumulation window (8 for full windows, 4 for final partial window)
window_start = (i // self.grad_accum_steps) * self.grad_accum_steps
window_size = min(self.grad_accum_steps, len(train_loader) - window_start)

with torch.amp.autocast(device_type="cuda", dtype=self.torch_dtype):
    logits = self.model(input_ids)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, self.model_config.vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    loss = loss / window_size

assert torch.isfinite(loss), f"Encountered non-finite loss at micro_step {self.micro_step}"
accum_loss += loss.item()

if self.scaler:
    self.scaler.scale(loss).backward()
else:
    loss.backward()

# Step optimizer when accumulation boundary reached or end of epoch
is_accum_boundary = ((i + 1) % self.grad_accum_steps == 0)
is_epoch_end = (i == len(train_loader) - 1)

if is_accum_boundary or is_epoch_end:
    ...
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    self.global_step += 1
```

---

## 3. Mathematical Analysis of the Bug

Let the gradient accumulation window consist of $K$ microbatches ($i = 1, \dots, K$).  
For microbatch $i$:
- Let $N_i$ be the number of valid supervised tokens (`shift_labels != -100`).
- Let $\ell_{i,t} = -\log P(y_{i,t} \mid x_{i,<t})$ be the negative log-likelihood of token $t$.
- Let $S_i = \sum_{t=1}^{N_i} \ell_{i,t}$ be the sum of cross-entropy losses in microbatch $i$.

### What PyTorch Computes under `reduction="mean"`:
By default, `F.cross_entropy(..., ignore_index=-100)` internally averages only over non-ignored tokens:
$$\mathcal{L}_i = \frac{1}{N_i} \sum_{t=1}^{N_i} \ell_{i,t} = \frac{S_i}{N_i}$$

Line 402 then scales by `window_size` ($K$):
$$\mathcal{L}_{\text{scaled}, i} = \frac{\mathcal{L}_i}{K} = \frac{S_i}{K \cdot N_i}$$

When `loss.backward()` accumulates gradients across the $K$ microbatches, the resulting parameter gradient is:
$$\nabla_{\theta} \mathcal{L}_{\text{accum}} = \sum_{i=1}^K \nabla_{\theta} \mathcal{L}_{\text{scaled}, i} = \frac{1}{K} \sum_{i=1}^K \frac{\nabla_{\theta} S_i}{N_i}$$

And the logged training loss is:
$$\text{accum\_loss} = \sum_{i=1}^K \mathcal{L}_{\text{scaled}, i} = \frac{1}{K} \sum_{i=1}^K \frac{S_i}{N_i}$$

### The True Target Objective:
In supervised fine-tuning, the effective batch consists of the union of all tokens across the $K$ microbatches. The true objective is the average loss per supervised token across the effective batch:
$$\mathcal{L}_{\text{true}} = \frac{\sum_{i=1}^K S_i}{\sum_{i=1}^K N_i} = \frac{\sum_{i=1}^K \sum_{t=1}^{N_i} \ell_{i,t}}{\sum_{i=1}^K N_i}$$

The true gradient must therefore be:
$$\nabla_{\theta} \mathcal{L}_{\text{true}} = \frac{\sum_{i=1}^K \nabla_{\theta} S_i}{\sum_{i=1}^K N_i}$$

### Equivalence Condition:
$\nabla_{\theta} \mathcal{L}_{\text{accum}} \equiv \nabla_{\theta} \mathcal{L}_{\text{true}}$ **if and only if** $N_1 = N_2 = \dots = N_K$.  
If all microbatches have identical numbers of supervised tokens, then:
$$\frac{1}{K} \sum_{i=1}^K \frac{\nabla S_i}{N} = \frac{\sum_{i=1}^K \nabla S_i}{K \cdot N} = \frac{\sum_{i=1}^K \nabla S_i}{\sum_{i=1}^K N_i}$$

However, in conversational SFT, sequences vary widely:
- Commonsense QA samples have $N \approx 20$ supervised tokens.
- Mathematics (GSM8K) samples have $N \approx 90$ supervised tokens.
- Python coding samples have $N \approx 250$ supervised tokens.
- Retrieval/document samples have $N \approx 1,600$ supervised tokens.

### Gradient Weight Ratio Demonstration:
Consider an accumulation window with two microbatches:
- Microbatch $A$: $N_A = 20$ supervised tokens (Commonsense QA).
- Microbatch $B$: $N_B = 1,600$ supervised tokens (Python code / retrieval).

Under the previous implementation:
$$\nabla_{\theta} \mathcal{L} = \frac{1}{2} \left[ \frac{\nabla S_A}{20} + \frac{\nabla S_B}{1600} \right] = \frac{\nabla S_A}{40} + \frac{\nabla S_B}{3200}$$

Individual token gradient weights:
$$\text{Weight of a token in } A = \frac{1}{40} = 0.025$$
$$\text{Weight of a token in } B = \frac{1}{3200} = 0.0003125$$
$$\text{Ratio} = \frac{0.025}{0.0003125} = 80.0\times$$

Under the correct global token-mean objective:
$$\nabla_{\theta} \mathcal{L}_{\text{true}} = \frac{\nabla S_A + \nabla S_B}{20 + 1600} = \frac{\nabla S_A + \nabla S_B}{1620}$$
$$\text{Weight of any valid token in } A \text{ or } B = \frac{1}{1620} \approx 0.0006173 \quad (\text{Equal weight } 1:1)$$

---

## 4. Audit of Component Mechanics

| Component | Code Inspection | Status | Identified Flaw |
| :--- | :--- | :--- | :--- |
| **Cross-Entropy Reduction** | Line 397: `F.cross_entropy(..., ignore_index=-100)` | Verified | Defaults to `reduction="mean"`. Normalizes per microbatch. |
| **Ignored Label Masking** | Labels set to `-100` in `collate_sft_batch` | Verified | Masking is respected within microbatch, but ignored across microbatch boundaries. |
| **Gradient Accumulation** | Sequential `backward()` across `grad_accum_steps = 8` | Verified | Gradients accumulate additively, but were pre-divided by $K \cdot N_i$ instead of $N_{\text{total}}$. |
| **Partial Window Handling** | `window_size = min(grad_accum_steps, len - window_start)` | Verified | Correctly counts microbatches in partial window, but still averages microbatch means. |
| **Validation Loss** | Lines 230–236: `F.cross_entropy(..., reduction="sum")` | Verified | **Mismatch**: Validation was already token-weighted ($\frac{\sum S}{\sum N}$), while training was not! |
| **Distributed Training** | Not present | Verified | Single-device script (`cuda` or `cpu`). No DDP logic to conflict. |
| **Zero Supervised Tokens** | No guard on `N_i == 0` | **Vulnerable** | PyTorch returns `NaN` when all targets are `-100` under `reduction="mean"`, crashing on `assert torch.isfinite`. |

---

## 5. Conclusion & Action Plan

The Phase 0 forensic hypothesis is completely validated:
1. The SFT objective in `train_sft_4k.py` was flawed. It placed up to 80x more weight on short multiple-choice tokens than on long reasoning or retrieval tokens.
2. Training loss and validation loss were computing different mathematical functions.
3. The training loop must be updated to accumulate unnormalized (summed) token gradients across the window, or normalize microbatch losses by the exact total number of supervised tokens in the effective accumulation window.

