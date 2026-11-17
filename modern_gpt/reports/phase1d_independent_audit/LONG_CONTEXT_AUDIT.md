# Phase 1D Long-Context Audit

**Claim under review:**
> *"Positional Loss: First 512 tokens = 3.0662, Last 512 tokens = 2.8642 (lower loss at sequence end demonstrates effective RoPE attention context utilization)"*

**Source:** `validation_loss_report.md`, `resource_usage_report.md`

---

## 1. What Was Measured

```python
loss_per_token = F.cross_entropy(logits[0], y[0], reduction="none")
early_losses.append(loss_per_token[:512].mean().item())
late_losses.append(loss_per_token[-512:].mean().item())
```

For each 2,048-token sequence, the per-token cross-entropy loss is computed:
- **Early loss:** Mean over tokens 0–511 (first 512 positions)
- **Late loss:** Mean over tokens 1536–2047 (last 512 positions)

These are averaged across all 175 sequences (25 sequences × 7 domains).

**Reported values:**
- Early loss (mean, tokens 0–511): **3.0662**
- Late loss (mean, tokens 1536–2047): **2.8642**
- Difference: **−0.0020** (late is lower)

---

## 2. Why This Does Not Prove Long-Context Utilization

### 2.1 The Autoregressive Ramp Effect
In standard autoregressive language modeling, early positions in a sequence always exhibit higher loss than later positions. This is a universal property of LMs, not a property specific to long-context or RoPE models:

- **Position 0:** The model has zero context. It predicts from prior distribution alone. Loss is close to `log(vocab_size)` = `log(32768)` ≈ 10.4 nats in the worst case.
- **Position 511:** The model has 511 tokens of context. Loss is typically lower.
- **Position 2047:** The model has 2,047 tokens of context. Loss is typically even lower.

This gradient exists in models trained with 128-token context windows too — the last token in a 128-token sequence still benefits from 127 tokens of context and shows lower loss than position 0.

### 2.2 The 2,048-Context Claim Requires Ablation
To determine whether positions 1536–2047 specifically benefit from attending to tokens 0–1535 (rather than just 1536–2047 immediately adjacent tokens), you must run a **context truncation ablation**:

| Condition | Prefix Provided | Target Positions | Expected Effect |
|---|---|---|---|
| No context | 0 tokens | 1536–2047 | Loss ≈ prior distribution |
| Short context | 128 tokens of prefix | 1664–2047 | Slight decrease |
| Medium context | 512 tokens | 1536–2047 | Moderate decrease |
| Full context | 1,536 tokens | 1536–2047 | Measured decrease (baseline) |

**If the 1,536-token prefix contributes meaningfully to attention patterns at positions 1536–2047, then the loss in the "Full context" condition should be measurably lower than in the "Short context" condition.**

This ablation was **not performed** in Phase 1D.

### 2.3 RoPE Does Not Guarantee Context Utilization
RoPE (Rotary Position Embedding) encodes relative positional relationships in attention. At 501.68M pretraining tokens, the model has seen sequences up to 2,048 tokens, but the amount of training data containing meaningful long-range dependencies (as opposed to padding or unrelated concatenated documents) is unknown. RoPE enables long-context attention; it does not guarantee the model has learned to exploit it.

---

## 3. Controlled Ablation Design (Requires GPU Authorization)

If GPU inference is authorized for a short session, the following ablation would take approximately 15–25 minutes on the RTX 5060:

```python
# For each domain, evaluate 5 sequences at positions 1500-2000
# using different prefix lengths: 0, 128, 256, 512, 1024, 1536
context_lengths = [0, 128, 256, 512, 1024, 1536]
target_start = 1536
target_end = 2048

for ctx_len in context_lengths:
    # pad prefix with eos tokens or truncate
    prefix = seq[target_start - ctx_len : target_start]
    target = seq[target_start : target_end]
    # measure CE loss on target_end - target_start = 512 positions
```

If loss at target positions decreases as `ctx_len` increases from 0 → 1536, that is evidence of long-range context utilization. If loss plateaus after `ctx_len=128` or `ctx_len=256`, it suggests the model only effectively attends to short local context.

---

## 4. Verdict

| Claim | Verdict | Evidence |
|---|---|---|
| "Lower loss in final 512 positions" | **VERIFIED** — 2.8642 vs 3.0662 | Recalculated from code |
| "Demonstrates effective 2,048-token RoPE utilization" | **UNSUPPORTED** | No ablation performed |
| "RoPE attention context utilization" | **UNSUPPORTED** | Autoregressive ramp is a confound |

**Defensible replacement statement:**
> *"The model exhibits the standard autoregressive positional loss gradient: loss in the final 512 positions (2.8642) is lower than loss in the first 512 positions (3.0662). Whether this difference is attributable to effective long-range attention versus local context utilization requires a context-truncation ablation. No such ablation was conducted in Phase 1D."*

