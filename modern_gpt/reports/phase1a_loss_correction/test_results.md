# ModernGPT-S — Phase 1A: SFT Loss Correction Test Results Report

**Status:** ALL TESTS PASSED  
**Date:** 2026-11-09  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Test Suite Overview

A dedicated automated test suite was constructed at `modern_gpt/tests/test_sft_token_weighted_loss.py`. It explicitly tests the mathematical correctness, numerical stability, and partition invariance of the corrected token-weighted SFT loss function across 9 distinct scenarios (Tests A through J).

### Summary Table

| Test Identifier | Test Description | Scope / Scenario | Max Absolute Error | Max Relative Error | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test A** | Equal supervised token counts | 2 microbatches, 50 tokens each | $0.00$ | $0.00$ | **PASS** |
| **Test B** | Unequal token counts (20 vs 150) | 2 microbatches, severe ratio | Diverges from old as expected | N/A (proves old bug) | **PASS** |
| **Test C** | Extreme imbalance (1 vs 200) | 2 microbatches, 200× ratio | Diverges from old as expected | N/A (proves old bug) | **PASS** |
| **Test D** | Variable padding & ignored labels | Random padding & masking | Bitwise identical | Bitwise identical | **PASS** |
| **Test E** | Partial accumulation window | 5 microbatches at epoch tail | Exact normalization by $N_{\text{partial}}$ | N/A | **PASS** |
| **Test F** | Zero supervised tokens safety | Microbatch with 0 valid tokens | Zero gradient flow, no NaN | N/A | **PASS** |
| **Test G** | Full-model gradient equivalence | Full 6-layer model (K=4) | $2.98 \times 10^{-8}$ | $7.25 \times 10^{-4}$ | **PASS** |
| **Test H** | Full optimizer update equivalence | AdamW step comparison | $3.73 \times 10^{-9}$ | $3.73 \times 10^{-9}$ | **PASS** |
| **Test J** | Validation partition invariance | Partition into 1, 2, 4 batches | $< 1.0 \times 10^{-7}$ | $< 1.0 \times 10^{-7}$ | **PASS** |

---

## 2. Detailed Test Analysis

### 2.1 Test A: Equal Supervised Token Counts
- **Setup**: Two microbatches with exactly 50 supervised tokens each ($N_1 = 50, N_2 = 50, N_{\text{total}} = 100$).
- **Result**: Because $N_1 = N_2 = N_{\text{total}} / 2$, the old unweighted microbatch average $\frac{1}{2}(\frac{S_1}{50} + \frac{S_2}{50})$ equals $\frac{S_1 + S_2}{100}$. Both loss and parameter gradients are bitwise identical.

### 2.2 Test B & C: Token Imbalance Demonstration
- **Setup**: Microbatch 1 has 20 tokens (Test B) or 1 token (Test C); Microbatch 2 has 150 tokens (Test B) or 200 tokens (Test C).
- **Result**:
  - The accumulated gradient from the corrected token-weighted method matches the reference full-batch gradient to machine precision ($< 10^{-7}$).
  - The unweighted method produces significant gradient deviation, proving the existence of the distortion and confirming that the fix properly eliminates it.

### 2.3 Test D: Variable Padding & Masking
- **Setup**: Microbatches with arbitrary prefix and suffix `-100` masking simulating prompt masking and padding.
- **Result**: Ignored labels are strictly excluded from both the numerator ($S_i$) and the denominator ($N_i$), with no leakage into gradient computations.

### 2.4 Test E: Partial Accumulation Windows
- **Setup**: 5-microbatch residual window at epoch boundary.
- **Result**: Gradients are normalized by $N_{\text{partial}} = \sum_{i=1}^5 N_i$ rather than a hardcoded 8-microbatch denominator or unweighted average, ensuring exact epoch-tail weighting.

### 2.5 Test F: Zero Supervised Tokens Safety
- **Setup**: A microbatch containing purely prompt tokens (`label == -100` everywhere).
- **Result**: `compute_token_weighted_cross_entropy` gracefully detects `valid_tokens == 0`, produces zero gradient, and avoids `0 / 0` division or NaN errors.

### 2.6 Test G: Full-Model Gradient Equivalence
- **Setup**: A full 6-layer `ModernGPT` model with 16k vocabulary and multi-head attention. We compare:
  1. *Reference*: Concatenated forward/backward pass with full-batch loss.
  2. *Accumulated*: 4 microbatches with variable supervised token counts (50, 100, 150, 200) accumulated across steps.
- **Results**:
  - **Max Absolute Gradient Difference**: $2.9802 \times 10^{-8}$
  - **Max Relative Gradient Difference**: $7.2524 \times 10^{-4}$ (isolated to parameters with near-zero gradients $< 10^{-6}$)
  - Confirms exact mathematical equivalence within float32 floating-point rounding limits.

### 2.7 Test H: Optimizer Parameter Update Equivalence
- **Setup**: Apply an AdamW optimizer step to parameters updated via accumulated token-weighted gradients vs reference full-batch gradients.
- **Results**:
  - **Max Absolute Weight Difference**: $3.7253 \times 10^{-9}$
  - The model weights after the optimizer step are identical to 9 decimal places.

### 2.8 Test J: Validation Loss Partition Invariance
- **Setup**: Evaluate a dataset of 4 batches under three different partitioning schemes:
  1. 1 large batch (all 4 items)
  2. 2 batches (2 items each)
  3. 4 microbatches (1 item each)
- **Results**:
  - Partition 1 Loss: $3.791522$
  - Partition 2 Loss: $3.791522$
  - Partition 4 Loss: $3.791522$
  - Difference: $< 1.0 \times 10^{-7}$
  - Validation loss is completely invariant to batching, microbatching, or sequence lengths.

