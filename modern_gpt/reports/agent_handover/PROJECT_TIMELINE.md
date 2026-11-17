# ModernGPT-S / QORIN — Master Chronological Project Timeline

**Status:** Verified & Cryptographically Grounded  
**Scope:** Inception through Phase 1D 500M Milestone  
**Target Repository:** ModernGPT-S / QORIN Research Suite  

---

## 1. Timeline Overview

```
OCTOBER 2026                 NOVEMBER 2-6            NOVEMBER 6-8            NOVEMBER 9-11           NOVEMBER 12-13
[5M / 25M PoC] ----------> [16M & 50M Ablations] -> [100M Baseline 1.5B] -> [SFT Forensics &] -> [500M Pretrain]
Loss: 6.03 -> 5.08          16M: Syntax vs Logic    100M Val: 2.71          Phase 1A Loss Corr     501.7M Tokens Done
Degenerative loops          50M GQA-4: 50% KV save  4K Context Pilot        5B Shards Packed       Phase 1D Evaluation
```

---

## 2. Detailed Milestone Chronology

### October 2026: Foundations & Exploratory Proof-of-Concept Pretraining
- **PoC 5M Run (`checkpoints/exp_5m/best_val.pt`):**
  - **Tokens:** 5,000,000 tokens (`bpe_16k`).
  - **Terminal Validation Loss:** `6.0346` (Perplexity: `417.63`).
  - **Key Observation:** Validated causal autoregressive training loop and rotary embeddings, but model collapsed into repetitive degenerative loops (`"the first of the first..."`).
- **PoC 25M Run (`checkpoints/exp_25m/best_val.pt`):**
  - **Tokens:** 25,000,000 tokens.
  - **Terminal Validation Loss:** `5.0837` (Perplexity: `161.38`).
  - **Key Observation:** Perplexity declined by 61.4%; model learned basic English syntax, token grouping, and sentence boundaries.

---

### November 2, 2026: ModernGPT-16M & The 500-Example Capability Diagnostic
- **Architecture:** 6 layers, 6 attention heads, 384 embedding dimension, SwiGLU FFN (1024 intermediate), tied embeddings, 16,768,128 parameters (16,252,800 active non-embedding weights).
- **Pretraining Run (`EXP-PRETRAIN-16M-100M`):**
  - Trained on 102.3M tokens (`bpe_16k_v2`, fingerprint `be6c3d60c5baf217`).
  - Terminal Validation Loss: `3.8615` (Perplexity: `47.54`).
- **500-Example Capability Diagnostic (`EXP-SFT-16M-MIDSCALE-500`):**
  - Rigorous evaluation across 68 task families (400 training / 100 isolated validation examples) spanning atomic Python, JSON generation, direct QA, and conceptual explanations.
  - **The Semantic / Generalization Bottleneck Discovered:**
    - Python AST syntax validity reached **83.1% train / 77.5% validation**.
    - JSON structural validity reached **74.4% train / 67.5% validation**.
    - **Held-out functional algorithmic execution was 0.0%**: The 16M parameter model quickly mastered structural surface templates, but utterly lacked capacity to synthesize novel execution logic.

---

### November 2–6, 2026: 50M Architectural Exploration & Iso-Data Ablations
- **Corpus:** 250M-token standardized ablation corpus.
- **Architectural Variants Evaluated:**
  1. `modern_gpts_v2_gqa4_250m` (12 layers, 8 query heads, 4 KV heads [GQA 2:1], 512 dim, SwiGLU 1376, 48,934,912 params):
     - Best Validation Loss: `3.1637` (Perplexity: `22.67`).
     - Cut KV cache memory consumption by **50.0%** while preserving **99.4%** of MHA perplexity.
  2. `modern_gpts_v2_deep_thin_250m` (16 layers, 7 heads, 7 KV heads, 448 dim, SwiGLU 1200, 49,120,000 params):
     - Best Validation Loss: `3.1687` (Perplexity: `22.77`).
- **Decision:** Grouped-Query Attention established as the permanent architectural standard for all subsequent models.

---

### November 6–8, 2026: ModernGPT-S 100M Baseline & 4K Context Scaling
- **Architecture:** 14 layers, 768 embedding dimension, 12 query heads, 4 KV heads (GQA 3:1), SwiGLU FFN (2048 dim), tied embeddings, **100,401,408 trainable parameters** (113,606,912 checkpoint tensor elements).
- **100M Baseline Pretraining Run:**
  - Trained on 1,499,987,968 tokens (1.5B) across 45,770 steps.
  - Final Validation Loss: `2.7137` (Perplexity: `15.08`).
  - Saved to `modern_gpt/archive/100m_baseline/checkpoints/base_100m_best_val.pt`.
- **4K Context Extension Pilot (Step 4):**
  - Scaled context window from 512 to 4,096 tokens using linear RoPE frequency scaling ($s = 8.0$).
  - Pretrained for 2,000 steps on 4K sequences; validation loss remained stable at `2.583`.
  - **Forensic Discovery:** Despite stable loss, passkey needle-in-a-haystack retrieval collapsed to **4.0%** across 4,096 tokens, proving that low loss during continued pretraining does not guarantee effective deep-context retrieval.

---

### November 8–9, 2026: Supervised Fine-Tuning Progression & Forensic Diagnosis
- **SFT v1 (Step 6B):** 9,997 records, best val loss `1.442`. Established ChatML token delimiters (`<|im_start|>`, `<|im_end|>`).
- **SFT v2 (Step 7B):** 26,734 records, best val loss `1.241`. Improved JSON formatting.
- **SFT v3 (Step 8D):** 31,949 records, best val loss `0.765`.
- **Phase 0 Forensic Audit Findings:**
  1. *Conversational Starvation:* Exactly **9 out of 34,700 training samples (0.026%)** were conversational greetings or repair turns. Casual greetings caused catastrophic template collapse into formal business resignation letters.
  2. *Template Domination:* Two corporate memo templates accounted for **56.5% of all supervised training tokens**.
  3. *Microbatch Weighting Asymmetry:* Microbatches were averaged without weighting by token count, allowing 20-token samples to exert up to $80\times$ more gradient impact per token than full sequences.
- **Phase 1A Loss Correction:**
  - Re-implemented mathematically correct token-weighted cross-entropy loss with global token normalization.
  - Verified across 10 deterministic synthetic suites (Tests A–J).

---

### November 11, 2026: 100M Baseline Archival & 500M Architecture Finalization
- **100M Baseline Sealed:**
  - Entire 100M experimental lifecycle frozen and cryptographically sealed under `modern_gpt/archive/100m_baseline/` (Manifest SHA-256: `60040fe90df04e765b49af0cc98418fcc269d500e4777a0bece1302062163a7a`).
- **ModernGPT-S 500M Specification Approved:**
  - 20 layers, 1536 hidden dimension, 24 query heads, 4 KV heads (6:1 GQA), head dimension 64, SwiGLU FFN (3840 intermediate dimension, 2.5x multiplier), tied embeddings, linear biases completely disabled.
  - Exactly **514,389,504 trainable parameters** (569,964,032 checkpoint tensor elements).
- **Production BPE-32K Tokenizer Sealed:**
  - `modern_gpt/artifacts/tokenizer_32k/tokenizer.json` (32,768 vocabulary, SHA-256 `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`, fingerprint `3559c7435e1d9330`).
  - Special tokens: `<|pad|>` (0), `<|unk|>` (1), `<|bos|>` (2), `<|im_start|>` (4), `<|im_end|>` (5), `<|tool_call|>` (6), `<|tool_result|>` (7), `<|endoftext|>` (9).

---

### November 11–12, 2026: 5B Pretraining Corpus Finalization & Binary Packing
- **Corpus Finalized:** 5,000,000,000 training tokens and 20,000,000 validation tokens across 7 balanced domains:
  1. Quality Web (`fineweb-edu`): 2,000,000,000 tokens (40.0%) — 40 shards
  2. Educational (`smollm-corpus`): 750,000,000 tokens (15.0%) — 15 shards
  3. Source Code (`codeparrot-clean`): 750,000,000 tokens (15.0%) — 15 shards
  4. Wikipedia (`wikipedia:en`): 375,000,000 tokens (7.5%) — 8 shards
  5. Mathematics (`open-web-math`): 375,000,000 tokens (7.5%) — 8 shards
  6. Books / Prose (`gutenberg`): 500,000,000 tokens (10.0%) — 10 shards
  7. Curated Dialogue (`ultrachat_200k`): 250,000,000 tokens (5.0%) — 5 shards
- **Binary Packing:** Sealed into 101 training shards and 7 validation shards under little-endian uint16 layout with 80-byte cryptographic headers. Zero validation contamination verified against 25,137 quarantine document hashes.

---

### November 12, 2026: Phase 1C 500M Pretraining Launch
- **Hardware Profile:** Dedicated NVIDIA GeForce RTX 5060 Laptop GPU (8GB physical VRAM, Blackwell SM 12.0).
- **VRAM Budget:** Microbatch size 1 (2,048 tokens) with gradient checkpointing and fused AdamW consumes **4.87 GiB VRAM**, leaving **+2.68 GiB (34%) safe headroom** against OOM.
- **Operating Schedule (1:45 / 15 Rule):** Strict cycles of 105 minutes active training followed by 15 minutes mandatory cooling ($< 65^\circ\text{C}$). Emergency thermal override at $\ge 92^\circ\text{C}$.
- **Interleaved Sampler:** Zero-copy memory-mapped sampling maintaining the exact 7-domain mixture continuously (partial-training protection).

---

### November 13, 2026: 500M Milestone & Phase 1D Evaluation
- Pretraining crosses the 500M-token milestone:
  - **Global Step:** `15,310` updates.
  - **Tokens Processed:** `501,678,080` tokens.
  - **Best Validation Loss:** `2.9464`.
  - **Checkpoint Preserved:** `modern_gpt/checkpoints/checkpoint_501M/ckpt_501m_tokens_501678080.pt` (SHA-256: `a57556554144bdba52bfc81072e097627280e3d7bc2e580719f071b3521f9122`).
- **Phase 1D Evaluation Executed:** Full audit of validation loss, behavioral compliance, coding, JSON, mathematics, tool calling, decoding diagnostics, and 4-way retrieval augmentation.

