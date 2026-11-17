# Hugging Face Model Repository Architecture & Packaging Plan

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/huggingface_cpu_deployment/MODEL_REPOSITORY_PLAN.md`  
**Target Repository:** `https://huggingface.co/vynor/qorin-models`  
**Audit Date:** November 17, 2026  
**Status:** Repository Created; Zero Model Weights Uploaded (Pre-Upload Verification Complete)  

---

## 1. Selected Checkpoints & Cryptographic Verification

Two complementary checkpoints have been designated for cloud hosting in the separate Hugging Face model repository:

### Checkpoint A: Flagship Autonomous Tool Agent (`QORIN-500M`)
- **Model Identifier:** `qorin-500m-sft-exp-f`
- **Source Path:** `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt`
- **File Size:** 1,982.32 MB (1.94 GB)
- **SHA-256 Hash:** `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4`
- **Architecture Configuration:**
  - `n_layers`: 20
  - `emb_dim`: 1536
  - `n_heads`: 24
  - `num_kv_heads`: 4 (Grouped-Query Attention 6:1)
  - `head_dim`: 64
  - `ffn_hidden_dim`: 3840 (SwiGLU)
  - `norm_type`: `RMSNorm` (eps = 1e-5)
  - `context_length`: 2048
  - `vocab_size`: 32768
  - `tied_embeddings`: True
- **Tokenizer Dependency:**
  - Directory: `modern_gpt/artifacts/tokenizer_32k/`
  - Key File: `tokenizer.json` (SHA-256: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`)
  - Auxiliary Files: `vocab.json`, `merges.txt`
  - Special Tokens: `<|im_start|>`, `<|im_end|>`, `<|tool_call|>`, `<|tool_result|>`, `<|tool_error|>`

### Checkpoint B: Extended Context CPU Baseline (`QORIN-100M`)
- **Model Identifier:** `qorin-100m-sft-v3`
- **Source Path:** `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt`
- **File Size:** 588.69 MB (0.57 GB)
- **SHA-256 Hash:** `d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510`
- **Architecture Configuration:**
  - `n_layers`: 14
  - `emb_dim`: 768
  - `n_heads`: 12
  - `num_kv_heads`: 4 (Grouped-Query Attention 3:1)
  - `head_dim`: 64
  - `ffn_hidden_dim`: 2048 (SwiGLU)
  - `norm_type`: `RMSNorm` (eps = 1e-5)
  - `context_length`: 4096 (RoPE Linear Scale = 8.0)
  - `vocab_size`: 16000
- **Tokenizer Dependency:**
  - Directory: `modern_gpt/artifacts/tokenizer_16k_v2/`
  - Key File: `tokenizer.json` (SHA-256: `6c155b6fb30153aac2f206fc14c30b78e3c3b45b16748c87faf7bf9b04ccbe69`)
  - Auxiliary Files: `vocab.json`, `merges.txt`
  - Special Tokens: `<|im_start|>`, `<|im_end|>`

---

## 2. Target Directory Hierarchy on Hugging Face

The repository `vynor/qorin-models` will be structured hierarchically to allow granular downloading:

```
vynor/qorin-models/
├── README.md                               # Comprehensive model card & rubric mapping
├── .gitattributes                          # Git LFS tracking for *.pt and *.bin
├── qorin-500m/
│   ├── config.json                         # ModelConfig serialization
│   ├── pytorch_model.pt                    # 1.98 GB (ckpt_pilot_best.pt)
│   └── tokenizer/
│       ├── tokenizer.json                  # BPE-32k Rust tokenizer
│       ├── vocab.json                      # Vocabulary mappings
│       ├── merges.txt                      # BPE merge rules
│       └── special_tokens_map.json         # ChatML & tool token definitions
└── qorin-100m/
    ├── config.json                         # ModelConfig serialization
    ├── pytorch_model.pt                    # 588.7 MB (best_val.pt)
    └── tokenizer/
        ├── tokenizer.json                  # BPE-16k v2 Rust tokenizer
        ├── vocab.json
        ├── merges.txt
        └── special_tokens_map.json
```

---

## 3. Storage & Upload Protocol

- **Total Binary Payload:**
  $$\text{Total Size} = 1,982.32\text{ MB} + 588.69\text{ MB} \approx \mathbf{2,571.01\text{ MB}~(2.51\text{ GB})}$$
- **LFS Requirements:**
  - Files over 50 MB on Hugging Face must be tracked via Git LFS.
  - Both `pytorch_model.pt` files exceed this threshold and will be uploaded via the Hugging Face chunked blob upload API:
    ```python
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo="qorin-500m/pytorch_model.pt",
        repo_id="vynor/qorin-models",
        repo_type="model"
    )
    ```
- **Strict Execution Boundary:**
  - In compliance with the user directive, **zero model weight binaries have been uploaded during this stage**.
  - Only the repository scaffold and metadata documentation are prepared.
