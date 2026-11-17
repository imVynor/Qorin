# ModernGPT-S / QORIN — Master Project Context & Onboarding Guide

**Document ID:** `DOC-HANDOVER-MASTER-01`  
**Classification:** Authoritative Technical Specification & Research Context  
**Primary Audience:** Autonomous Coding Agents, ML Researchers & Project Maintainers  
**Last Updated:** November 13, 2026  

---

## 1. Executive Research Mission

ModernGPT-S (also developed under the research moniker **QORIN**) is an independent small language model project investigating the boundaries of small decoder-only Transformers:

> **Central Research Question:**  
> *"How does increasing the capacity of a small decoder-only Transformer affect conversational language-model performance, and can external retrieval compensate for a smaller model?"*

### Two Orthogonal Investigation Tracks
1. **Capacity Scaling Track:**
   Tracing the behavioral and capability transitions across a tightly controlled, modern architectural lineage:
   $$\text{16M (6L, 384D)} \longrightarrow \text{50M (12L, 512D)} \longrightarrow \text{100M (14L, 768D)} \longrightarrow \text{500M (20L, 1536D)}$$
2. **Retrieval & System Augmentation Track:**
   Testing whether structured non-parametric retrieval (sparse TF-IDF and dense vector search) over local knowledge bases and conversational memory can bridge the factual hallucination and parametric capacity limits of sub-billion parameter models.

### Critical Conceptual Invariant
- **Intrinsic Model Capability:** What the standalone neural network computes purely through forward-pass matrix multiplications over its feed-forward and attention weights.
- **System-Level Capability:** What the coordinated agentic harness achieves when combining the model with external retrieval, sandboxed tools, and structured goal loops.
*Rule: Never attribute retrieval-augmented or tool-assisted outputs to intrinsic model capability.*

---

## 2. Why the Project Transitioned from 100M to 500M

The 100M baseline pretraining campaign (1.5B tokens, validation loss `2.7137`, perplexity `15.08`) established a solid syntactic model, but systematic diagnostics revealed several critical plateaus:

1. **The Semantic / Reasoning Bottleneck:**
   While the 100M model reliably learned surface templates for code and structured data, its capacity to synthesize multi-step reasoning, perform arithmetic without hallucination, and retain complex multi-turn context remained severely bounded.
2. **Context Extension vs Attention Degradation:**
   Extending the 100M context window from 512 to 4,096 tokens via RoPE linear frequency scaling ($s=8.0$) maintained low cross-entropy validation loss (`2.583`), but passkey needle-in-a-haystack retrieval collapsed to **4.0%** across 4K tokens. The model lacked the parametric attention capacity to attend across 4,096 tokens without dedicated long-context training.
3. **Conversational SFT Failure (Forensic Finding):**
   Supervised fine-tuning v3 achieved low loss (`0.7648`), but forensic analysis uncovered catastrophic template collapse driven by greeting starvation (only 0.026% greeting records) and severe template domination (56.5% of tokens in two corporate templates).
4. **Capacity Scaling Hypothesis:**
   Scaling to 500M parameters ($5\times$ increase in weights, $2\times$ increase in embedding width, $1.43\times$ increase in depth) combined with a production 5B-token balanced 7-domain mixture provides the minimum required capacity for stable algorithmic reasoning and rich multi-turn conversational retention.

---

## 3. Approved ModernGPT-S 500M Architecture

Instantiated via `ModelConfig.get_500m_config()` in `modern_gpt/model/config.py`:

```
                            Input Tokens (B, T <= 2048)
                                       │
                                       ▼
                       Token Embedding (32,768 x 1,536)
                                       │
                                       ▼  (Dropout p = 0.0)
                        ┌──────────────────────────────┐
                        │   Transformer Block (x20)    │
                        │                              │
                        │  x ──► RMSNorm ──────────┐   │
                        │  │                       ▼   │
                        │  │   24Q-4KV GQA (6:1)   │   │
                        │  │   RoPE (theta=10000)  │   │
                        │  │                       │   │
                        │  └───► (+) ◄─────────────┘   │
                        │        │                     │
                        │  x ──► RMSNorm ──────────┐   │
                        │  │                       ▼   │
                        │  │   SwiGLU FFN (3840)   │   │
                        │  │   (2.5x multiplier)   │   │
                        │  │                       │   │
                        │  └───► (+) ◄─────────────┘   │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                              Final RMSNorm (1,536)
                                       │
                                       ▼
                          Tied LM Head (1,536 x 32,768)
                                       │
                                       ▼
                          Logits (B, T, 32,768)
```

### Exact Parameter Accounting
- **Active / Trainable Parameters:** **`514,389,504`**
- **State Dict Tensor Elements:** **`569,964,032`**
- **Tied Embedding Elements:** $32,768 \times 1,536 = \mathbf{50,331,648}$ ($514,389,504 + 50,331,648 = 569,964,032$).
- **Linear Biases:** All projection biases (`qkv_bias`, `out_proj_bias`, FFN biases) are disabled (`bias=False`).

---

## 4. Production Tokenizer Specification

- **File Path:** `modern_gpt/artifacts/tokenizer_32k/tokenizer.json`
- **Vocabulary Size:** `32,768`
- **SHA-256 Hash:** `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`
- **Tokenizer Fingerprint:** `3559c7435e1d9330`
- **Key Special Tokens:**
  - `<|pad|>`: ID `0`
  - `<|unk|>`: ID `1`
  - `<|bos|>`: ID `2`
  - `<|im_start|>`: ID `4` (ChatML message opener)
  - `<|im_end|>`: ID `5` (ChatML message closer)
  - `<|tool_call|>`: ID `6`
  - `<|tool_result|>`: ID `7`
  - `<|endoftext|>`: ID `9` (Canonical End-Of-Text / EOS delimiter)

---

## 5. Production 5B Dataset & Shard Semantics

- **Storage Directory:** `modern_gpt/data_500m/final_5b/`
- **Training Tokens:** `5,000,000,000` (101 binary shards, ~50M tokens each)
- **Validation Tokens:** `20,000,000` (7 domain-isolated shards)
- **Binary Format:** Little-endian `uint16` tokens with 80-byte cryptographic headers.
- **7-Domain Balanced Mixture:**
  1. `web` (FineWeb-Edu): 40.0% (2.0B tokens)
  2. `edu` (SmolLM-Corpus): 15.0% (750M tokens)
  3. `code` (CodeParrot-Clean): 15.0% (750M tokens)
  4. `wiki` (Wikipedia EN): 7.5% (375M tokens)
  5. `math` (Open-Web-Math): 7.5% (375M tokens)
  6. `books` (Gutenberg): 10.0% (500M tokens)
  7. `dialogue` (UltraChat-200K): 5.0% (250M tokens)
- **Sampling Engine:** `InterleavedDomainSampler` (`modern_gpt/data_500m/domain_sampler.py`) provides zero-copy memory-mapped loading that enforces the target mixture across every rolling window.

---

## 6. Training Configuration & Thermal Protocol

- **Hardware Profile:** Dedicated NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, Blackwell SM 12.0).
- **VRAM Allocation:** Microbatch size 1 ($2,048$ tokens) $\times$ gradient checkpointing = **4.87 GiB allocated** (5.25 GiB peak reserved), leaving **34% safe headroom**.
- **Optimizer:** Fused `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{-8}$, weight decay 0.1).
- **Batching:** 16 gradient accumulation steps $\times 2,048 = \mathbf{32,768}$ tokens per optimizer update.
- **Mandatory 1:45 / 15 Operating Schedule:**
  - 105 minutes active gradient descent $\to$ atomic checkpoint $\to$ 15 minutes mandatory cooling ($< 65^\circ\text{C}$).
  - Hardware thermal safety override triggers cooldown if temperature reaches $\ge 92^\circ\text{C}$.

---

## 7. Current Project Status & Milestone Checkpoints

- **500M Milestone Checkpoint:**
  - **Filename:** `modern_gpt/checkpoints/checkpoint_501M/ckpt_501m_tokens_501678080.pt`
  - **SHA-256:** `a57556554144bdba52bfc81072e097627280e3d7bc2e580719f071b3521f9122`
  - **Global Update:** `15,310`
  - **Tokens Processed:** `501,678,080` (crossing 10.0% of the 5B campaign)
  - **Best Validation Loss:** `2.9464`
- **Active Training Checkpoints:**
  - `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt` (identically updated)
  - `modern_gpt/checkpoints/500m_pretrain/ckpt_best_val.pt` (Step 14,756, loss 2.9464)
- **Archived Baseline Reference:**
  - `modern_gpt/archive/100m_baseline/checkpoints/base_100m_best_val.pt` (SHA-256 `bded509c...`)

