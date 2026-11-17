# ModernGPT-S 500M — Repository and Baseline Inspection Report

**Phase:** Phase 1B.1 Tokenizer Finalization & Pretraining Corpus Audit  
**Date:** November 11, 2026  
**Status:** **INSPECTION COMPLETE & PRESERVATION VERIFIED**  
**Auditor:** Antigravity Autonomous Agent  

---

## 1. Executive Summary

In accordance with Phase 1B.1 Part A requirements, a comprehensive inspection of the `modern_gpt` repository was conducted to assess baseline preservation, review historical reports, inspect candidate tokenizers, audit data preprocessing and loading pipelines, and verify that all 100M baseline artifacts remain frozen and unmodified.

**Key Findings:**
1. **100M Baseline Preservation:** The archive at `modern_gpt/archive/100m_baseline/` is 100% intact, immutable, and protected with master SHA-256 manifests (`archive_manifest.json`, checksum `60040fe90df04e765b49af0cc98418fcc269d500e4777a0bece1302062163a7a`).
2. **Phase 1A Deliverables:** All 9 Phase 1A feasibility reports and the `model_500m.yaml` configuration are fully present and verified under `modern_gpt/reports/phase1a_500m_feasibility/` and `modern_gpt/configs/modern_gpts_500m/`.
3. **32K Tokenizer Candidate Located:** The candidate tokenizer files reside in `modern_gpt/reports/phase1a_500m_feasibility/candidate_32k/` with fingerprint `a1274e6fd57b5b63`.
4. **Data Pipelines & Artifacts:** Existing tokenization (`bpe_tokenizer.py`), sharding (`sharding.py`), and dataset packing (`dataset.py`) are fully functional. Pretraining validation documents (`val_documents.jsonl`, 9,348 records) are verified as held-out from training.
5. **SFT v4 Data Preserved:** Raw and curated pools (`curated_candidate_pool.jsonl`, 16,422 records; `greetings_casual_v1.jsonl`, 1,200 records; `identity_retention_v1.jsonl`, 600 records) are isolated under `modern_gpt/data/sft_4k_v4/` and strictly separated from pretraining data.

---

## 2. Repository Structure & Artifact Layout

```
ML-course/modern_gpt/
├── archive/
│   └── 100m_baseline/                  # FROZEN 100M BASELINE (Read-Only)
│       ├── checkpoints/                # 100M pretrain & SFT checkpoints
│       ├── configs/                    # 100M configuration YAMLs
│       ├── environment/                # Python environment specifications
│       ├── evaluations/                # Forensic evaluation logs
│       ├── manifests/                  # Master archive SHA-256 manifest
│       ├── reports/                    # Phase 0 forensic audit & Phase 1A loss reports
│       ├── tokenizer/                  # bpe_16k_v2 frozen tokenizer
│       ├── ARCHIVE_README.md
│       ├── EXPERIMENT_RESULTS_INDEX.md
│       └── REPRODUCIBILITY_AUDIT.md
├── artifacts/
│   ├── corpus_100m/                    # 100M pretraining dataset (train: 84k, val: 9.3k docs)
│   ├── tokenizer_16k_v2/               # Active 16K tokenizer (fingerprint: be6c3d60c5baf217)
│   └── experiments/                    # Historical experiment checkpoints & logs
├── configs/
│   ├── modern_gpts_100m/               # 100M architecture & training YAMLs
│   └── modern_gpts_500m/               # 500M architecture YAML (model_500m.yaml)
├── data/
│   ├── bpe_tokenizer.py                # Byte-level BPE tokenizer trainer & wrapper
│   ├── dataset.py                      # Packed token dataset for PyTorch DataLoader
│   ├── sharding.py                     # Binary shard chunking and prefetching
│   ├── sft_4k_v3/                      # Historical SFT v3 data
│   └── sft_4k_v4/                      # Curated SFT v4 candidate pool & supplements
├── model/
│   ├── attention.py                    # GQA & SDPA attention implementation
│   ├── block.py                        # Transformer block definition
│   ├── config.py                       # ModelConfig dataclass
│   ├── gpt.py                          # ModernGPT container with gradient checkpointing
│   ├── mlp.py                          # SwiGLU feed-forward network
│   ├── norm.py                         # RMSNorm
│   └── rope.py                         # Rotary Position Embedding
├── reports/
│   ├── phase0_forensic_audit/          # Initial 100M audit
│   ├── phase1a_loss_correction/        # SFT loss correction proof
│   ├── phase1a_500m_feasibility/       # 500M architecture & feasibility reports
│   └── phase1b_1_tokenizer_corpus_audit/# Current Phase 1B.1 reports
├── train.py                            # Pretraining runner
└── train_sft_4k.py                     # SFT runner with Phase 1A loss weighting
```

---

## 3. Phase 0 & Phase 1A Documentation Review

### 3.1 Phase 0 Baseline Audit Review
- The frozen 100M baseline established that ModernGPT-S 100M achieved reasonable syntactic fluency but suffered from severe multi-turn conversational degradation, zero greeting examples, context amnesia, and microbatch token-weighting asymmetry.
- All 100M artifacts were locked in `archive/100m_baseline/` with explicit instructions never to overwrite or alter them.

### 3.2 Phase 1A Feasibility Review
- **Architecture Consensus:** ModernGPT-S 500M ($L=20, d=1536, H_q=24, H_{kv}=4, d_{ffn}=3840$, context 2,048, tied embeddings, zero linear biases).
- **Parameter Count Verification:** Exactly $514,389,504$ parameters with 32K vocabulary; $464,057,856$ non-embedding core parameters.
- **Hardware Simulation:** Confirmed feasible on NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM) at $5.10\text{ GB}$ peak VRAM allocated ($2.68\text{ GB}$ headroom).
- **Candidate Tokenizer:** Preliminary 32K candidate BPE trained on 25k docs, demonstrating initial sequence length compression of $17.62\%$ overall and $25.18\%$ on code.

---

## 4. Candidate 32K Tokenizer Inspection

The candidate tokenizer was inspected in `modern_gpt/reports/phase1a_500m_feasibility/candidate_32k/`:
- **Files Present:** `tokenizer.json` (2.32 MB), `tokenizer_config.json` (289 B), `training_metrics.json` (283 B).
- **Vocabulary Size:** Exactly $32,768$ tokens.
- **Fingerprint:** `a1274e6fd57b5b63` (SHA-256 truncated).
- **Initial Special Tokens:** Trained with default `special_tokens=["<|endoftext|>"]` (ID: 0).
- **Critical Architectural Observation:** The preliminary candidate tokenizer does NOT yet define the reserved special tokens required by Phase 1A's `tool_token_specification.md` (`<|pad|>`, `<|unk|>`, `<|bos|>`, `<|eos|>`, `<|im_start|>`, `<|im_end|>`, `<|tool_call|>`, `<|tool_result|>`, `<|tool_error|>`). This will be thoroughly evaluated in Part B.

---

## 5. Data Processing & SFT v4 Status

1. **Pretraining Corpus (`artifacts/corpus_100m/`):**
   - `train_documents.jsonl`: 84,144 documents ($394.4\text{ MB}$, $102.3\text{M tokens}$).
   - `val_documents.jsonl`: 9,348 documents ($43.1\text{ MB}$, $11.2\text{M tokens}$).
   - Provenance: FineWeb-Edu ($70\%$), Cosmopedia-v2 ($20\%$), CodeParrot-clean ($10\%$).
   - `val_documents.jsonl` provides a pristine, genuine held-out pretraining corpus for tokenizer validation.
2. **SFT v4 Candidate Data (`data/sft_4k_v4/`):**
   - Curated candidate pool: 16,422 high-quality records across 7 balanced categories.
   - Greetings pool: 1,200 verified casual and professional conversational records.
   - Identity retention pool: 600 verified multi-turn continuity records.
   - **Boundary Enforcement:** SFT v4 data is strictly quarantined from pretraining corpora and will not be contaminated into pretraining data.

---

## 6. Preservation Compliance Checklist

- [x] `modern_gpt/archive/100m_baseline/` unmodified and read-only.
- [x] `modern_gpt/artifacts/tokenizer_16k_v2/` unmodified and active for 100M baseline.
- [x] Historical 100M checkpoints and configs untouched.
- [x] Phase 1A feasibility reports preserved intact in `modern_gpt/reports/phase1a_500m_feasibility/`.
- [x] All new Phase 1B.1 work isolated under `modern_gpt/reports/phase1b_1_tokenizer_corpus_audit/`.
- [x] SFT v4 artifacts preserved in `modern_gpt/data/sft_4k_v4/`.

