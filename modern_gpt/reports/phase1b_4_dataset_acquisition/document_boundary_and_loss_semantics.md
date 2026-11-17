# ModernGPT-S 500M — Phase 1B.4: Document Boundary Semantics & Next-Token Slicing Report

## Executive Summary

This report formally resolves the semantic and notation audit requirements from Phase 1B.3 regarding document boundary handling in packed pretraining sequences and next-token autoregressive tensor slicing in the streaming PyTorch DataLoader.

---

## 1. Document Boundary Semantics: Continuous-Stream Causal LM

### 1.1 Architectural Implementation
In `modern_gpt/data_500m/packing.py`, tokenized documents from multiple sources are packed contiguously into fixed 2,048-token sequences. The delimiter token `<|endoftext|>` (Token ID 9) is appended immediately after each document:
$$\dots, t_{k-1}^{(D_1)}, t_k^{(D_1)}, \mathbf{\langle|endoftext|\rangle}, t_1^{(D_2)}, t_2^{(D_2)}, \dots$$

### 1.2 Training Objective Clarification
- **Objective Type**: **Continuous-stream causal language modelling** (the standard architectural choice of GPT-2, GPT-3, LLaMA, and Mistral).
- **Attention Visibility**: Standard lower-triangular causal attention masking is applied. Later tokens in document $D_2$ **can and do attend causally** to earlier tokens from document $D_1$ when co-located in the same 2,048-token context window.
- **Boundary Delimitation**: The presence of `<|endoftext|>` provides a statistical boundary signal. The model learns that context across `<|endoftext|>` transitions between disparate topics and styles.
- **Explicit Negative Confirmation**: **The `<|endoftext|>` token alone does NOT prevent cross-document attention.** We do not claim strict document isolation. Strict document isolation (preventing cross-document attention via 2D block-diagonal masking) is reserved for conversational Supervised Fine-Tuning (SFT) and is not active during pretraining.

---

## 2. Next-Token Slicing & Resolution of Slice Notation

### 2.1 Code Implementation in `dataloader.py`
In `modern_gpt/data_500m/dataloader.py` (`BinaryShardDataset.__iter__`), each packed 2,048-token sequence is converted to PyTorch tensor slices:
```python
# seq_tensor has shape [2048]
x = seq_tensor[:-1]  # Exact length 2,047
y = seq_tensor[1:]   # Exact length 2,047
```

### 2.2 Mathematical Definition vs Python Half-Open Slice Syntax
In the Phase 1B.3 preliminary notes, a textual notation ambiguity existed between mathematical closed intervals and Python's zero-indexed half-open slice notation `[start:stop)`. This report formally establishes the exact correspondence:

| Representation | Input Slice $x$ | Target Slice $y$ | Length |
|---|---|---|:---:|
| **Python Code Syntax** | `seq[:-1]` or `seq[0:2047]` | `seq[1:]` or `seq[1:2048]` | **2,047** |
| **0-Indexed Closed Interval** | Positions $[0, 2046]$ | Positions $[1, 2047]$ | **2,047** |
| **Mathematical Sequence Indices** | $\{s_0, s_1, \dots, s_{2046}\}$ | $\{s_1, s_2, \dots, s_{2047}\}$ | **2,047** |

- **Resolution of Discrepancy**: A description referring to Python syntax `[0:2046]` would evaluate to 2,046 elements ($2046 - 0 = 2046$). The actual codebase uses `[:-1]`, which is Python's half-open slice equivalent to `[0:2047]`, selecting index positions 0 through 2046 inclusive (exactly 2,047 tokens).
- **Autoregressive Invariant**: For all $t \in [0, 2045]$, $x[t+1] \equiv y[t] = s_{t+1}$. Therefore:
  $$(x[:, 1:] == y[:, :-1]).\text{all}() \equiv \mathbf{True}$$
  This was verified bitwise across 100 consecutive streaming batches in Phase 1B.3 and reconfirmed here with zero off-by-one errors.
- **Padding and Loss**: When training on fully packed sequences, loss is computed over all 2,047 positions. If terminal padding (Token ID 0) occurs, `CrossEntropyLoss(ignore_index=0)` masks padding tokens from loss and gradient accumulation.

---

## 3. Fixed Validation Partition Architecture

### 3.1 Domain-Balanced Validation Suite
To ensure reliable evaluation without domain drift, the final validation corpus is structured as an isolated, fixed partition:
- **Total Validation Allocation**: 20,000,000 tokens (2.0% of total pretraining scale).
- **Proportional Domain Balancing**:
  - Quality Web: 8,000,000 tokens (40%)
  - Educational: 3,000,000 tokens (15%)
  - Source Code: 3,000,000 tokens (15%)
  - Wikipedia: 1,500,000 tokens (7.5%)
  - Mathematics: 1,500,000 tokens (7.5%)
  - Books: 2,000,000 tokens (10%)
  - Dialogue: 1,000,000 tokens (5%)
- **Isolation Mechanism**: Validation documents are quarantined into SHA-256 Bloom/hash sets prior to training set ingestion. Any training document exhibiting text hash collision or Jaccard similarity $\ge 0.80$ is strictly purged.

