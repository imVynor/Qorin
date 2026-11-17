# Model Checkpoint & Tokenizer Inventory

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/MODEL_CHECKPOINT_INVENTORY.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Authoritative Deployment Candidate Checkpoints

The table below catalogs the 12 model checkpoints registered in the backend model provider (`backend/models/qorin_provider.py`) and verified against the filesystem and `canonical_model_manifest.json`:

| Checkpoint Path | File Size | SHA-256 Hash | Architecture | Param Count | Tokenizer Dependency | Loading Implementation | Current Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt` | 1,982.32 MB | `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 514,389,504 | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | **Production Candidate (Autonomous Multi-Turn Tool Chaining)** |
| `modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_d_combined/checkpoints/ckpt_branch_d_best.pt` | 2,979.33 MB | `7322b9aee559945c32f02f59835e7143534c3982178b5aacf4b31470934f9dcc` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 514,389,504 | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | Validated Checkpoint (Decision Calibration & Grounding) |
| `modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt` | 1,982.33 MB | `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 514,389,504 | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | Validated Checkpoint (Balanced Web + Python) |
| `modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt` | 2,979.33 MB | `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 514,389,504 | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | Validated Checkpoint (Targeted Autonomous Two-Tool SFT) |
| `modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt` | 1,988.23 MB | `acbecc699da76387de9ef6d6a40a64ad73b324b6eee2bd1842a6fd9ab1972c4f` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 514,389,504 | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | Intermediate Checkpoint (35M General + Schema Demo) |
| `modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt` | 1,988.22 MB | `249fb467a9cf76e732890cc0501ec39a443756cb2f1c86c66402960f4abbadc4` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 514,389,504 | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | Baseline SFT Checkpoint (25M General Instruction Pilot) |
| `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt` | 2,953.61 MB | `28ac47b07e9616967d6a2ea0458e68432682085a2e05fba6ac2e55b82b6d639d` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, RMSNorm, ctx 2048 | 569,964,032 (raw) | `modern_gpt/artifacts/tokenizer_32k` (BPE-32k) | `ModernGPT.load_state_dict(strict=False)` | Pretraining Foundation (252M Tokens Processed / Unconverged) |
| `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt` | 588.69 MB | `d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510` | 14L-12Q-4KV (GQA 3:1), SwiGLU 2048, RoPE s=8.0, ctx 4096 | 120,029,440 | `modern_gpt/artifacts/tokenizer_16k_v2` (BPE-16k) | `ModernGPT.load_state_dict(strict=False)` | **Production Candidate (4K Extended Context & ChatML)** |
| `modern_gpt/checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt` | 588.70 MB | `3b23f37b82279c63d5913933fef22ea487ed28f28e08878cfa4d531064900817` | 14L-12Q-4KV (GQA 3:1), SwiGLU 2048, RoPE s=8.0, ctx 4096 | 120,029,440 | `modern_gpt/artifacts/tokenizer_16k_v2` (BPE-16k) | `ModernGPT.load_state_dict(strict=False)` | Evaluation Pilot (100M 4K RoPE Linear Extension) |
| `modern_gpt/checkpoints/modern_gpts_100m/best_val.pt` | 1,152.70 MB | `bded509c44eabc2932dc1e293967bb0524acc31182fe9df0295b9ad2a4671e31` | 14L-12Q-4KV (GQA 3:1), SwiGLU 2048, RoPE base, ctx 512 | 113,606,912 | `modern_gpt/artifacts/tokenizer_16k_v2` (BPE-16k) | `ModernGPT.load_state_dict(strict=False)` | Completed Pretraining Baseline (1.5B Tokens at 512 Ctx) |
| `modern_gpt/checkpoints/sft_50m_targeted_v2/best_val.pt` | 565.12 MB | `32c831873d5407f013456340fc331484b0e0170adabeae9e103f9901f21eb971` | 12L-8H, SwiGLU 1365, RMSNorm, RoPE base, ctx 512 | 58,083,840 | `modern_gpt/artifacts/tokenizer_16k_v2` (BPE-16k) | `ModernGPT.load_state_dict(strict=False)` | Completed Fine-Tuning Checkpoint (50M Multi-Domain) |
| `modern_gpt/checkpoints/exp_100m/best_val.pt` | 193.49 MB | `3bf4c51485f727b6a3acf7861299eec408cc7e59b195a0a4a09ee7927057dd16` | 6L-6H MHA, GELU, LayerNorm, Learned RoPE, ctx 512 | 23,305,344 | `modern_gpt/artifacts/tokenizer_16k_v2` (BPE-16k) | `ModernGPT.load_state_dict(strict=False)` | Lightweight Proof-of-Concept Baseline (16M Architecture) |

---

## 2. Tokenizer Dependencies & Special Tokens

The repository uses two custom BPE tokenizers trained with Hugging Face `tokenizers`:

1. **BPE-32k (`modern_gpt/artifacts/tokenizer_32k`):**
   - **Target Families:** QORIN-500M series.
   - **Vocabulary Size:** 32,768.
   - **Files:** `tokenizer.json` (SHA-256: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`), `vocab.json`, `merges.txt`.
   - **Special Tokens:**
     - `<|im_start|>`: Begins ChatML turn (`system`, `user`, `assistant`, `tool`).
     - `<|im_end|>`: Ends ChatML turn.
     - `<|tool_call|>`: Delimits beginning of tool execution JSON.
     - `<|tool_result|>`: Delimits tool observation payload.
     - `<|tool_error|>`: Delimits tool execution error.

2. **BPE-16k v2 (`modern_gpt/artifacts/tokenizer_16k_v2`):**
   - **Target Families:** QORIN-16M, QORIN-50M, QORIN-100M series.
   - **Vocabulary Size:** 16,000.
   - **Files:** `tokenizer.json` (SHA-256: `6c155b6fb30153aac2f206fc14c30b78e3c3b45b16748c87faf7bf9b04ccbe69`), `vocab.json`, `merges.txt`.
   - **Special Tokens:** `<|im_start|>`, `<|im_end|>`.

---

## 3. Checkpoint-Loading Compatibility & External Storage Audit

### How the Model is Loaded:
In `backend/models/qorin_provider.py`:
```python
ckpt_path = REPO_ROOT / meta["checkpoint_path"]
tok_path = REPO_ROOT / meta["tokenizer_path"]
state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
```

### Can weights be stored outside the Git repository?
- **Current Behavior:** `ckpt_path = REPO_ROOT / meta["checkpoint_path"]` joins the relative path with the local repository root directory.
- **Limitation:** If `meta["checkpoint_path"]` is an absolute path or points to an external mount (e.g. `/mnt/qorin_models/best_val.pt` or `D:\models\qorin`), `Path(REPO_ROOT / absolute_path)` will collapse to the absolute path on POSIX, but on Windows or when containerizing, hardcoded relative paths inside `KNOWN_QORIN_MODELS` assume the checkpoints reside within `modern_gpt/`.
- **Recommended Deployment Modification:**
  Introduce an optional environment variable `QORIN_CHECKPOINT_DIR`:
  ```python
  base_dir = Path(os.environ.get("QORIN_CHECKPOINT_DIR", REPO_ROOT))
  ckpt_path = base_dir / meta["checkpoint_path"]
  ```
  This will allow mounting weights from high-speed persistent volumes, S3 buckets, or external SSDs without modifying Git tracking.

---

## 4. Lineage and Training Relationships

```
[QORIN-500M Pretraining Foundation]
  ckpt_latest.pt (Step 7,694 / 252M Tokens)
       │
       ▼
[Experiment A: General SFT Pilot]
  ckpt_pilot_final.pt (25M Tokens)
       │
       ▼
[Experiment B: General + Schema Demonstration]
  ckpt_experiment_b_final.pt (35M Tokens)
       │
       ├──► [Experiment C: Autonomous Two-Tool SFT] (ckpt_experiment_c_final.pt)
       ├──► [Experiment D: Balanced Web + Python] (ckpt_experiment_d_best.pt)
       └──► [Experiment E: Branch D Calibration] (ckpt_branch_d_best.pt)
                 │
                 ▼
       [Experiment F: Autonomous Multi-Turn Chaining] (ckpt_pilot_best.pt)
       ★ PRIMARY 500M DEPLOYMENT CANDIDATE ★
```

```
[QORIN-100M Baseline Pretraining]
  best_val.pt (1.5B Tokens at 512 Context)
       │
       ▼
[QORIN-100M 4K Linear RoPE Extension Pilot]
  checkpoint_step_2000.pt (Step 2,000 / RoPE scale=8.0)
       │
       ▼
[QORIN-100M SFT 4K v3]
  best_val.pt (ChatML Alignment + 4K Context)
  ★ PRIMARY 100M DEPLOYMENT CANDIDATE ★
```
