# ModernGPT-S 500M — Repository Onboarding & Architecture Audit

**Document ID:** `REP-PHASE1D-02-ONBOARDING`  
**Classification:** Onboarding & Architectural Trace  
**Auditor:** Autonomous Systems Agent  
**Date:** November 13, 2026  

---

## 1. Codebase Structure & Core Modules

The ModernGPT-S codebase is an end-to-end small language model research environment structured cleanly across model definitions, binary data pipelines, training orchestrators, evaluation benchmarks, and an interactive inference backend.

```
ML-course/
├── modern_gpt/                      # Core modeling and research package
│   ├── model/                       # PyTorch neural network modules
│   │   ├── config.py                # ModelConfig dataclass with parameter validation
│   │   ├── gpt.py                   # ModernGPT causal decoder-only Transformer
│   │   ├── attention.py             # GroupedQueryAttention (GQA) & KV-caching
│   │   ├── block.py                 # TransformerBlock with pre-norm RMSNorm
│   │   ├── norm.py                  # RMSNorm module
│   │   ├── mlp.py                   # SwiGLU gated activation MLP
│   │   ├── rope.py                  # Rotary Position Embedding implementation
│   │   └── loss.py                  # Global token-weighted cross-entropy loss
│   ├── data_500m/                   # 5B pretraining dataset & sampler
│   │   ├── domain_sampler.py        # Interleaved zero-copy memory-mapped 7-domain sampler
│   │   └── final_5b/                # 101 train shards + 7 val shards (uint16 binary)
│   ├── artifacts/                   # Vocabularies, tokenizers, and configs
│   │   ├── tokenizer_32k/           # Production BPE-32K (SHA-256: d10624a9...)
│   │   └── tokenizer_16k_v2/        # Preserved 16K BPE vocabulary for 100M/50M
│   ├── checkpoints/                 # Model checkpoints
│   │   ├── checkpoint_501M/         # Milestone 501.68M checkpoint
│   │   └── 500m_pretrain/           # Live pretraining checkpoints (ckpt_latest.pt)
│   ├── archive/                     # Frozen historical baselines
│   │   └── 100m_baseline/           # Cryptographically sealed 100M artifacts
│   ├── scripts/                     # Pretraining, evaluation, and diagnostic scripts
│   │   ├── pretrain_500m.py         # 500M pretraining orchestrator with duty cycle
│   │   ├── monitor_gpu.py           # Real-time GPU thermals, power & clock watchdog
│   │   └── run_phase1d_eval...py    # Phase 1D comprehensive evaluation runner
│   └── reports/                     # Formal engineering audit & milestone reports
├── backend/                         # FastAPI inference harness & tool sandboxes
└── web/                             # React 19 + TypeScript + Vite research showcase
```

---

## 2. Complete End-to-End Execution Path

The full lifecycle from raw data to evaluation follows a strict, deterministic sequence:

```
[1. Raw Corpora]
   │ FineWeb-Edu, SmolLM, CodeParrot, Wikipedia, OpenWebMath, Gutenberg, UltraChat
   ▼
[2. Filtering & Deduplication]
   │ MinHash LSH deduplication, heuristic quality filtering, regex normalization
   ▼
[3. BPE-32K Tokenization]
   │ Production BPE tokenizer (SHA-256: d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59)
   ▼
[4. Binary Shard Packing]
   │ uint16 little-endian token packing + 80-byte cryptographic headers
   │ 101 training shards (50M tokens each) + 7 validation shards (20M tokens total)
   ▼
[5. Interleaved Domain Sampler]
   │ Zero-copy np.memmap(mode='r', offset=80); continuous 7-domain mixture preservation
   ▼
[6. PyTorch Microbatching & DataLoader]
   │ Microbatch size = 1 (2,048 tokens), autoregressive alignment x[:, 1:] == y[:, :-1]
   ▼
[7. ModernGPT-S 500M Transformer]
   │ 20 layers, 1536 dim, 24Q-4KV GQA, SwiGLU, RMSNorm, RoPE, tied embeddings
   ▼
[8. Pretraining Loop & Duty Cycle]
   │ Fused AdamW, 16 accum steps (32,768 tok/update), 105m active / 15m cooldown schedule
   ▼
[9. Checkpointing & Recovery]
   │ Periodic atomic serialization of model, optimizer, sampler offsets, and RNG states
   ▼
[10. Phase 1D Evaluation Suite]
   │ Validation loss, behavioral QA, coding pass@1, JSON schema, math, tools, retrieval
```

---

## 3. Dependency & Environment Audit

The execution environment is fully isolated in `.\.venv`:
- **Python Version:** `3.12.7` (64-bit Windows)
- **PyTorch Version:** `2.16.0.dev20261022+cu134` (CUDA 13.4 runtime)
- **Hardware Target:** NVIDIA GeForce RTX 5060 Laptop GPU (7.93 GiB physical VRAM, SM 12.0)
- **Primary Dependencies:** `torch`, `numpy`, `tokenizers`, `fastapi`, `uvicorn`, `sse-starlette`, `beautifulsoup4`
- **Integrity Status:** Zero missing dependencies for evaluation; GPU hardware fully operational with native `bfloat16` acceleration.

