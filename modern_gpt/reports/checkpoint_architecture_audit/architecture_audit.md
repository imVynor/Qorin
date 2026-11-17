# QORIN Checkpoint Architecture & Forensic Integrity Audit

**Audit Date:** November 17, 2026  
**Audited Target:** `modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt`  
**Expected Cryptographic Hash:** `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`  
**Audit Status:** **VERIFIED & CERTIFIED (Bit-Exact)**  
**Verification Method:** Read-Only Forensic CPU Inspection (PyTorch 2.6, SHA-256 streaming hash)

---

## 1. Executive Summary & Verdict

A comprehensive, read-only forensic audit was performed on the frozen QORIN Experiment C checkpoint (`ckpt_experiment_c_final.pt`). The primary objective was to definitively resolve the architectural discrepancy between the historical Experiment B configuration (20 layers, 1536 dim, 24 heads, 2048 context) and the recent sprint report narrative (12 layers, 1536 dim, 12 heads, 4096 context, 514,480,128 parameters).

### Definitive Verdict:
1. **The physical checkpoint is unequivocally 20 layers, 1536 hidden dimension, 24 query heads, 4 KV heads (6:1 GQA), and 2048 context length.**
2. **The "12 layers, 4096 context, 514,480,128 parameters" description in the recent sprint report was an unverified narrative draft.** It originated from a theoretical preset authored in `modern_gpt/reports/final_research_audit/generate_experiment_registry.py` and does not describe any physical checkpoint in the 500M lineage.
3. **The unique parameter count is exactly 514,389,504** when accounting for weight-tied embeddings (`lm_head.weight` $\equiv$ `token_embedding.weight`).
4. **The checkpoint's SHA-256 hash matches the expected hash bit-for-bit.**

---

## 2. Checkpoint Verification Summary

| Metric | Target / Expected | Observed / Verified | Status |
| :--- | :--- | :--- | :--- |
| **File Path** | `modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt` | Present (3,124,032,607 bytes / 2.91 GB) | **MATCH** |
| **SHA-256 Hash** | `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b` | `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b` | **MATCH (Bit-Exact)** |
| **Number of Transformer Layers** | 20 (Hist. Exp B) vs 12 (Sprint Report) | **20** (`blocks.0` through `blocks.19`) | **20 LAYERS (Exp B confirmed)** |
| **Hidden Dimension ($d_{\text{model}}$)** | 1536 | **1536** | **MATCH** |
| **Query Heads ($n_{\text{heads}}$)** | 24 (Head dim 64) | **24** (`query.weight` shape `[1536, 1536]`) | **MATCH** |
| **Key/Value Heads ($n_{\text{kv}}$)** | 4 (GQA 6:1 ratio) | **4** (`key.weight`, `value.weight` shape `[256, 1536]`) | **MATCH** |
| **Head Dimension ($d_{\text{head}}$)** | 64 | **64** ($1536 / 24 = 64$) | **MATCH** |
| **Context Length ($T$)** | 2048 vs 4096 | **2048** (RoPE buffers `[1, 1, 2048, 64]`, stored config: 2048) | **2048 TOKENS** |
| **Vocabulary Size ($V$)** | 32,768 (BPE-32k) | **32,768** (`token_embedding.weight` `[32768, 1536]`) | **MATCH** |
| **FFN Hidden Dimension** | 3840 (SwiGLU 2.5x) | **3840** (`gate_proj`, `up_proj` `[3840, 1536]`) | **MATCH** |
| **Linear Biases** | Disabled (`False`) | `qkv_bias=False`, `out_proj_bias=False` | **MATCH** |
| **RoPE Parameters** | Default base ($\theta=10000.0, s=1.0$) | $\theta=10000.0$, scale=1.0, type="default" | **MATCH** |
| **Normalization** | RMSNorm ($\epsilon=1\times 10^{-5}$) | `final_norm.weight` `[1536]`, block norms `[1536]` | **MATCH** |
| **Activation Function** | SwiGLU | SwiGLU (`gate_proj`, `up_proj`, `down_proj`) | **MATCH** |
| **Weight Tying** | Tied embedding & LM Head | Bit-identical tensors (Max abs diff: 0.0) | **MATCH** |

---

## 3. Discrepancy Resolution: 20 Layers vs 12 Layers

### 3.1 The Two Conflicting Narratives
Prior to this audit, two conflicting architectural specifications existed across project documentation:
1. **The Historical Specification:**  
   *Layers:* 20, *Hidden Dim:* 1536, *Query Heads:* 24, *KV Heads:* 4, *Context:* 2048, *Parameters:* 514,389,504.
2. **The Recent Sprint Report Narrative:**  
   *Layers:* 12, *Hidden Dim:* 1536, *Query Heads:* 12, *KV Heads:* 4, *Context:* 4096, *Parameters:* 514,480,128.

### 3.2 Root Cause Analysis
Forensic code archaeology identified the exact origin of the 12-layer narrative:
- In `modern_gpt/reports/final_research_audit/generate_experiment_registry.py` (authored during the preparation for the final research audit), a hardcoded dictionary entry was manually written:
  ```python
  "architecture_baseline": "ModernGPT-S (514M parameters, 12 layers, 1536 dim, GQA 12/4 heads, RoPE, RMSNorm, SwiGLU, 4096 context)",
  "parameters": 514480128,
  "layers": 12,
  "hidden_dim": 1536,
  "heads": 12,
  "kv_heads": 4,
  "context_length": 4096,
  ```
- This dictionary was propagated into `modern_gpt/reports/final_research_audit/website_data/checkpoint_lineage.json` and `final_evaluation_audit.md`.
- In reality, this metadata block was created based on an unverified theoretical 12-layer calculation, rather than reading the physical checkpoint metadata.
- When `ckpt_experiment_c_final.pt` is inspected with PyTorch, the stored metadata dictionary `state['config']` explicitly contains:
  ```python
  {
    'vocab_size': 32768,
    'context_length': 2048,
    'emb_dim': 1536,
    'n_heads': 24,
    'n_layers': 20,
    'ffn_hidden_dim': 3840,
    'dropout': 0.0,
    'qkv_bias': False,
    'rope_theta': 10000.0,
    'num_kv_heads': 4,
    'gradient_checkpointing': True,
    'rope_scale': 1.0,
    'rope_type': 'default',
    'out_proj_bias': False
  }
  ```
- Furthermore, the state dictionary contains weights for `blocks.0` through `blocks.19`—exactly 20 Transformer blocks. If a 12-layer model is loaded against this checkpoint, 88 tensors from layers 12–19 are left unloaded, proving that the model is undeniably 20 layers.

---

## 4. Parameter Count Reconciliation

Three different parameter numbers have appeared in various reports. Here is the exact mathematical and physical derivation of each:

### 4.1 Component-by-Component Parameter Accounting (20-Layer Architecture)

1. **Token Embeddings:**
   $$\text{vocab\_size} \times \text{emb\_dim} = 32,768 \times 1,536 = 50,331,648$$

2. **Per-Block Parameters (20 Blocks):**
   - **Attention:**
     - Query Projection: $1536 \times 1536 = 2,359,296$
     - Key Projection: $1536 \times 256 = 393,216$ ($4 \text{ KV heads} \times 64 = 256$)
     - Value Projection: $1536 \times 256 = 393,216$
     - Output Projection: $1536 \times 1536 = 2,359,296$
     - Attention Layer Norm (`attn_norm.weight`): $1,536$
     - *Total Attention per block:* $5,506,560$ parameters.
   - **SwiGLU MLP:**
     - Gate Projection: $1536 \times 3840 = 5,898,240$
     - Up Projection: $1536 \times 3840 = 5,898,240$
     - Down Projection: $3840 \times 1536 = 5,898,240$
     - MLP Layer Norm (`mlp_norm.weight`): $1,536$
     - *Total MLP per block:* $17,696,256$ parameters.
   - **Total Learnable Parameters per block:**
     $$5,506,560 + 17,696,256 = 23,202,816$$
   - **Total for 20 Blocks:**
     $$20 \times 23,202,816 = 464,056,320$$

3. **Final Layer Normalization:**
   - `final_norm.weight`: $1,536$

4. **Language Model Head:**
   - `lm_head.weight`: $32,768 \times 1,536 = 50,331,648$ (Tied to `token_embedding.weight`).

5. **RoPE Rotary Buffers (Non-trainable persistent state):**
   - Per block: `rope.cos` ($1 \times 1 \times 2048 \times 64 = 131,072$) + `rope.sin` ($131,072$) = $262,144$ elements.
   - For 20 blocks: $20 \times 262,144 = 5,242,880$ elements.

### 4.2 Parameter Count Reconciliations:

| Metric | Calculation | Parameter Count |
| :--- | :--- | :--- |
| **Unique Trainable Parameters (Tied LM Head)** | $\text{Embedding} + 20 \times \text{Block} + \text{Final Norm}$ | **514,389,504** |
| **Unique Parameters (Untied LM Head)** | $\text{Embedding} + 20 \times \text{Block} + \text{Final Norm} + \text{LM Head}$ | **564,721,152** |
| **Total Tensor Elements in State Dict** | Untied Parameters + 40 RoPE Buffers ($5,242,880$) | **569,964,032** |
| **Theoretical 12-layer estimate (Sprint Narrative)** | Manual script estimate ($32768 \times 1536 \times 2 + 12 \times \dots$) | **514,480,128** (Erroneous artifact) |

---

## 5. Concrete Tensor Evidence

Inspecting `ckpt_experiment_c_final.pt` yields exactly 223 tensors:
- $1 \times$ `token_embedding.weight`: `[32768, 1536]`
- $20 \times$ `blocks.<L>.attn_norm.weight`: `[1536]`
- $20 \times$ `blocks.<L>.attn.query.weight`: `[1536, 1536]`
- $20 \times$ `blocks.<L>.attn.key.weight`: `[256, 1536]`
- $20 \times$ `blocks.<L>.attn.value.weight`: `[256, 1536]`
- $20 \times$ `blocks.<L>.attn.output.weight`: `[1536, 1536]`
- $20 \times$ `blocks.<L>.attn.rope.cos`: `[1, 1, 2048, 64]`
- $20 \times$ `blocks.<L>.attn.rope.sin`: `[1, 1, 2048, 64]`
- $20 \times$ `blocks.<L>.mlp_norm.weight`: `[1536]`
- $20 \times$ `blocks.<L>.mlp.gate_proj.weight`: `[3840, 1536]`
- $20 \times$ `blocks.<L>.mlp.up_proj.weight`: `[3840, 1536]`
- $20 \times$ `blocks.<L>.mlp.down_proj.weight`: `[1536, 3840]`
- $1 \times$ `final_norm.weight`: `[1536]`
- $1 \times$ `lm_head.weight`: `[32768, 1536]`

Total tensor count: $1 + (20 \times 11) + 1 + 1 = 223$ tensors.

**Tied LM Head Verification:**
A tensor equality check between `token_embedding.weight` and `lm_head.weight` in `ckpt_experiment_c_final.pt`:
- `torch.equal(token_embedding.weight, lm_head.weight) == True`
- `max(|token_embedding.weight - lm_head.weight|) == 0.0`
This proves that the embedding and language model head are bit-identical, confirming tied embeddings.

---

## 6. Audit Conclusion

The QORIN Experiment C checkpoint `ckpt_experiment_c_final.pt` is a 20-layer, 1536-hidden-dim, 24-query-head, 4-KV-head (GQA), 2048-context Transformer containing 514,389,504 unique parameters. The historical Experiment B architecture is 100% correct and accurate. The 12-layer narrative is completely refuted by physical evidence.
