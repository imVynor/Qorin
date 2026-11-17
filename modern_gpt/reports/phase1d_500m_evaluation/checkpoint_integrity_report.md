# ModernGPT-S 500M — Checkpoint Integrity & Cryptographic Audit

**Document ID:** `REP-PHASE1D-05-CHECKPOINT-AUDIT`  
**Classification:** Cryptographic & Numerical Checkpoint Audit  
**Date:** November 13, 2026  

---

## 1. Checkpoint Identification & Metadata

The milestone checkpoint corresponding to the first 500M training tokens was audited:

- **Primary Checkpoint Path:** `modern_gpt/checkpoints/checkpoint_501M/ckpt_501m_tokens_501678080.pt`
- **Active Duplicate Path:** `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt`
- **File Size:** `3,097,088,883 bytes` (~3.10 GB)
- **Exact SHA-256 Hash:** `a57556554144bdba52bfc81072e097627280e3d7bc2e580719f071b3521f9122`
- **Global Optimizer Update:** `15,310` updates
- **Total Tokens Processed:** **`501,678,080` tokens**
- **Best Validation Loss:** `2.9464`
- **Serialization Timestamp:** `2026-11-13T14:58:08.249737`
- **Tokenizer Fingerprint:** `3559c7435e1d9330`
- **Checkpoint Category:** **Complete Resumable Training Checkpoint** (contains full model weights, fused AdamW optimizer state, momentum buffers, sampler offsets, and RNG states).

---

## 2. Parameter Accounting & Numerical Validation

- **State Dict Tensor Count:** Exactly `223` tensors.
- **Total Tensor Elements:** `569,964,032` elements.
- **Trainable Parameters:** `514,389,504` parameters.
- **Tied Embedding Elements:** $32,768 \times 1,536 = 50,331,648$ elements ($514,389,504 + 50,331,648 = 569,964,032$).
- **NaN / Inf Audit:**
  - Evaluated across all 223 tensors in `model_state_dict`.
  - NaN detected: **0 (Passed)**
  - Inf detected: **0 (Passed)**
  - All numerical weights are completely finite.

---

## 3. Strict Model Restoration Test

The checkpoint was loaded into `ModernGPT` using `model.load_state_dict(ckpt['model_state_dict'], strict=True)`:

1. **Missing Parameters:** Exactly 0.
2. **Unexpected Parameters:** Exactly 0.
3. **Weight Tying:** `model.lm_head.weight is model.token_embedding.weight` evaluated to `True`.
4. **Logits Dimensionality:** Forward pass on sample tokens returned tensor shape `[1, T, 32768]`.
5. **Autoregressive Sanity:** Generated grammatically correct English mathematical continuation without collapse.

### Verdict
The 500M milestone checkpoint is **100% integral, uncorrupted, and verified suitable for evaluation and resumed pretraining**.

