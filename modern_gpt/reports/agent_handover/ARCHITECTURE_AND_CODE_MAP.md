# ModernGPT-S / QORIN — Architecture & Codebase Map

**Scope:** Source Code Inventory, Module Responsibilities & Execution Paths  
**Target:** New Agent & Developer Onboarding  

---

## 1. Top-Level Repository Organization

```
ML-course/
├── modern_gpt/                   # Core language model research package
│   ├── model/                    # Model architecture & neural network modules
│   │   ├── config.py             # ModelConfig dataclass & architecture factory methods
│   │   ├── gpt.py                # ModernGPT causal decoder-only Transformer
│   │   ├── attention.py          # Grouped-Query Attention (GQA) & KV cache
│   │   ├── block.py              # Pre-norm TransformerBlock
│   │   ├── norm.py               # RMSNorm implementation
│   │   ├── mlp.py                # SwiGLU activation MLP
│   │   ├── rope.py               # Rotary Position Embedding (RoPE) engine
│   │   └── loss.py               # Global token-weighted cross-entropy loss
│   ├── data_500m/                # Production 5B dataset & sampling infrastructure
│   │   ├── domain_sampler.py     # Interleaved zero-copy memory-mapped 7-domain sampler
│   │   └── final_5b/             # Final 101 train shards + 7 val shards (uint16 binary)
│   ├── artifacts/                # Production tokenizers, vocabularies & configs
│   │   ├── tokenizer_32k/        # Production BPE-32K (SHA-256: d10624a9..., fp: 3559c7435e1d9330)
│   │   ├── tokenizer_16k_v2/     # Historical 16k vocabulary for 16M/50M/100M
│   │   └── corpus/               # Intermediate document manifests
│   ├── checkpoints/              # Checkpoint storage
│   │   ├── checkpoint_501M/      # 501.68M milestone checkpoint (ckpt_501m_tokens_501678080.pt)
│   │   └── 500m_pretrain/        # Active 500M training runs (ckpt_latest.pt, ckpt_best_val.pt)
│   ├── archive/                  # Frozen historical baselines
│   │   └── 100m_baseline/        # Cryptographically sealed 100M model artifacts
│   ├── scripts/                  # Pretraining, fine-tuning & evaluation scripts
│   │   ├── pretrain_500m.py      # Production 500M training orchestrator with duty cycle
│   │   ├── monitor_gpu.py        # Real-time GPU thermals, power & clock watchdog
│   │   ├── evaluate_passkey.py   # Needle-in-a-haystack context retrieval test
│   │   └── run_phase1d_eval...py # Phase 1D comprehensive evaluation runner
│   └── reports/                  # Detailed engineering & audit reports
│       ├── phase1b_7_final_sharding/  # Corpus verification reports
│       ├── phase1c_pretraining/       # Pretraining feasibility & telemetry reports
│       ├── phase1d_500m_evaluation/   # Phase 1D evaluation suite reports
│       └── agent_handover/            # Master agent onboarding documentation
├── backend/                      # Production inference API & experimentation harness
│   ├── app.py                    # FastAPI server exposing SSE streaming & REST API
│   ├── models/                   # Model provider abstraction (QORIN native + Ollama)
│   ├── retrieval/                # TF-IDF knowledge base indexer & router
│   ├── tools/                    # Sandboxed Python execution & live Web researcher
│   └── training/                 # Observability telemetry decimation & parsing
└── web/                          # React 19 + TypeScript + Vite research showcase & UI
    └── src/
        ├── pages/Harness.tsx     # Experimentation testbed (Chat, A/B Lab, Dashboard, Traces)
        └── components/harness/   # Interactive chat, inspection drawer & SVG chart components
```

---

## 2. Core Model Architecture Specifications (`modern_gpt/model/`)

### A. Architectural Hyperparameters (ModernGPT-S 500M)
- **Trainable Parameters:** `514,389,504` (instantiated parameter count)
- **Checkpoint Tensors:** `569,964,032` (includes tied LM head weights in state dict)
- **Layers ($N_{\text{layers}}$):** `20`
- **Embedding Dimension ($d_{\text{model}}$):** `1536`
- **Attention Heads ($N_{\text{heads}}$):** `24` (query heads)
- **Key-Value Heads ($N_{\text{kv\_heads}}$):** `4` (Grouped-Query Attention ratio: **6:1**)
- **Head Dimension ($d_{\text{head}}$):** `64` ($1536 / 24 = 64$)
- **Feed-Forward Activation:** `SwiGLU` with intermediate dimension `3840` ($2.5 \times d_{\text{model}}$)
- **Normalization:** `RMSNorm` applied in pre-norm configuration (epsilon $10^{-5}$)
- **Positional Encoding:** Rotary Position Embeddings (`RoPE`) with base frequency $\theta = 10,000.0$
- **Embedding Tie:** Output linear projection (`lm_head.weight`) is tied directly to `token_embedding.weight`
- **Biases:** Out-projection, QKV projections, and FFN linear biases are disabled (`bias=False`)

### B. Module Responsibilities
1. `config.py`:
   - Enforces strict constraints in `__post_init__`: `emb_dim % n_heads == 0`, `n_heads % num_kv_heads == 0`, head dim is even for RoPE.
   - Provides standardized factories: `get_16m_config()`, `get_50m_config()`, `get_100m_gqa4_config()`, `get_500m_config()`.
2. `gpt.py`:
   - Instantiates `token_embedding`, `blocks` (`TransformerBlock`), `final_norm`, and `lm_head`.
   - Forward pass supports `use_cache` for fast autoregressive generation and `gradient_checkpointing` for training memory conservation.
3. `attention.py`:
   - Implements `GroupedQueryAttention`.
   - Computes causal scaled dot-product attention via `F.scaled_dot_product_attention` (FlashAttention-2 / SDPA kernel).
   - Repeats KV heads $6\times$ to align with 24 query heads.
4. `mlp.py`:
   - Computes $\text{SwiGLU}(x) = \text{down\_proj}(\text{silu}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x))$.

---

## 3. End-to-End Data Pipeline (`modern_gpt/data_500m/`)

```
Raw Multi-Domain Web & Code Corpora
  │
  ▼
Domain Filtering, Deduplication & Quality Heuristics
  │
  ▼
BPE-32K Tokenization (SHA-256: d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59)
  │
  ▼
Binary uint16 Packing with 80-byte Cryptographic Header
  │
  ├── 101 Training Shards (50M tokens each) ──> modern_gpt/data_500m/final_5b/train/
  └── 7 Validation Shards (domain-isolated) ──> modern_gpt/data_500m/final_5b/val/
        │
        ▼
InterleavedDomainSampler (zero-copy np.memmap, continuous 7-domain mixture)
        │
        ▼
PyTorch DataLoader (Batch size 1, Context length 2048, Autoregressive alignment x[:, 1:] == y[:, :-1])
```

---

## 4. Training Engine & Safety Orchestrator (`modern_gpt/scripts/pretrain_500m.py`)

- **Optimizer:** Fused `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, weight decay $0.1$).
- **Learning Rate Schedule:** Linear warmup across 2,000 updates to peak $3.0 \times 10^{-4}$, followed by Cosine Annealing decay down to minimum $3.0 \times 10^{-5}$.
- **Batching:** Microbatch size 1 ($2,048$ tokens) $\times 16$ gradient accumulation steps = **32,768 tokens per optimizer update**.
- **Hardware Watchdog:**
  - Enforces **1:45 active GPU work / 15 minutes cooling** duty cycle.
  - Monitors GPU core temperature via `nvidia-smi`; triggers emergency pause if temperature reaches $\ge 92^\circ\text{C}$.
  - Saves atomic checkpoints every 50M tokens to `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt`.

