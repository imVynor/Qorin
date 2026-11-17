# ModernGPT-S 500M — Technical Decisions, Risk Register & Mitigations

**Phase:** Phase 1A Architecture & Feasibility  
**Date:** November 11, 2026  
**Status:** **RISK AUDIT COMPLETE**  
**Target Model:** ModernGPT-S 500M  

---

## 1. Architectural & Engineering Decision Record

| Decision ID | Area | Selected Choice | Rejected Alternatives | Primary Justification |
| :---: | :--- | :--- | :--- | :--- |
| **D-01** | **Depth / Width** | $L=20, d=1536$ | $L=24, d=1280$ or $L=16, d=1792$ | Optimal balance between representational depth and tensor parallelism efficiency. Matches standard scaling laws. |
| **D-02** | **FFN Scaling** | $d_{ffn}=3840$ ($2.50\times$) | $d_{ffn}=4096$ ($2.67\times$) | Aligns with Llama 3 SwiGLU design, preserving capacity while keeping non-embedding parameter count at exactly $464,057,856$. |
| **D-03** | **Attention Type** | GQA ($24$ query, $4$ KV heads) | MHA ($24$ query, $24$ KV) or MQA ($24$ query, $1$ KV) | Reduces KV cache footprint by $6\times$ during auto-regressive generation without measurable quality loss compared to full MHA. |
| **D-04** | **Vocabulary Size** | $32,768$ ($32\text{K}$) | $16,000$ ($16\text{K}$) | Yields $17.62\%$ global sequence reduction ($25.18\%$ on code) for only $+5.14\%$ parameter cost. Fits comfortably within 8GB VRAM. |
| **D-05** | **Linear Biases** | Strictly False (`bias=False`) | Default `bias=True` on output proj | Eliminates 30,720 unnecessary parameters; aligns with modern zero-bias architectures; matches analytical budget. |
| **D-06** | **Weight Tying** | Tied (`True`) | Untied (`False`) | Avoids wasting $+50.3\text{M}$ parameters on a redundant output projection head; channels parameters into transformer reasoning depth. |
| **D-07** | **Tool Syntax** | ChatML + `<|tool_call|>` | Raw text / function tag regex | Dedicated special tokens guarantee zero syntax collision and enable reliable JSON validation. |
| **D-08** | **Activation Memory**| Gradient Checkpointing = `True` | Non-checkpointed full activation cache | Reduces peak activation VRAM by $>2\text{ GB}$, ensuring the 500M model fits in 8GB VRAM with $2.68\text{ GB}$ headroom. |

---

## 2. Technical Risk Register & Mitigation Strategy

### Risk 1: Laptop GPU Thermal Throttling & Hardware Longevity
- **Risk Level:** **HIGH** (Severity: High, Likelihood: Medium)
- **Description:** Pretraining a 500M model on consumer laptop hardware (NVIDIA RTX 5060 Laptop GPU) involves sustained 100% GPU utilization over multiple days. Laptop chassis have constrained thermal dissipation, which may trigger core thermal throttling, clock degradation, or fan fatigue.
- **Mitigation Strategy:**
  1. **Power Target Clamping:** Set NVIDIA GPU power target to 80W–85W (using `nvidia-smi -pl 85`) to keep GPU core temperatures $<75^\circ\text{C}$.
  2. **Thermal Cooldown Sleep:** Inject a configurable 2-second sleep every 100 optimizer steps in the training loop during overnight runs.
  3. **Frequent State Checkpointing:** Save model weights and optimizer states every 500 steps ($\sim 40\text{ minutes}$) to ensure zero lost work if system sleep or Windows updates occur.

---

### Risk 2: Tokenizer Retraining & Dataset Re-Encoding Overhead
- **Risk Level:** **MEDIUM** (Severity: Medium, Likelihood: High)
- **Description:** Moving from 16K to 32K vocabulary means existing pre-tokenized binary files (`.bin`) from 100M cannot be directly ingested without re-tokenization.
- **Mitigation Strategy:**
  1. The raw text corpora (`corpus_train.jsonl` and SFT source files) are preserved intact.
  2. The custom Byte-level BPE tokenizer processes $>800,000$ tokens/second on CPU. Re-tokenizing a 2B token dataset takes $<45\text{ minutes}$ using multi-process chunking.
  3. The candidate tokenizer vocabulary and merges are deterministic and fingerprinted (`a1274e6fd57b5b63`).

---

### Risk 3: Compute Budget for 10B Pretraining on a Single GPU
- **Risk Level:** **MEDIUM** (Severity: High, Likelihood: Medium)
- **Description:** A full 10-Billion-token Chinchilla pretraining run requires $\sim 200$ GPU hours ($\sim 8.4$ continuous days) on a single RTX 5060 Laptop GPU.
- **Mitigation Strategy:**
  1. **Staged Pretraining Strategy:**
     - **Stage 1 (Pilot):** 1 Billion tokens ($\sim 20$ GPU hours, $2.5$ calendar days). Validate loss curves, perplexity, and gradient norms.
     - **Stage 2 (Expansion):** 5 Billion tokens ($\sim 100$ GPU hours).
  2. **High-Throughput SFT Track:** For conversational and tool-use research, SFT fine-tuning requires only 50M–100M tokens, which trains in **$1.0\text{ to }2.0\text{ hours}$** on this GPU.

---

### Risk 4: Long Context ($S > 2048$) VRAM Spill
- **Risk Level:** **LOW** (Severity: Medium, Likelihood: Low)
- **Description:** If context length is extended to 4,096 tokens during fine-tuning, activation memory and KV cache will expand.
- **Mitigation Strategy:**
  1. With GQA ($4$ KV heads) and PyTorch FlashAttention / SDPA, memory scales linearly rather than quadratically.
  2. Empirical testing showed $2.68\text{ GB}$ of free headroom at $S=2048$. At $S=4096, B=1$ with gradient checkpointing, peak VRAM is estimated at $\sim 6.2\text{ GB}$, comfortably below the $8.0\text{ GB}$ ceiling.
  3. Linear RoPE scaling (proven in Phase 0 / 100M baseline) allows smooth extension from 2048 to 4096 without retraining from scratch.

---

### Risk 5: Tool Call Hallucination in Small Models
- **Risk Level:** **MEDIUM** (Severity: Medium, Likelihood: Medium)
- **Description:** A 500M model might generate malformed JSON or invoke non-existent tool names if trained on noisy tool data.
- **Mitigation Strategy:**
  1. Assistant-only SFT loss masking forces the model to learn exact JSON schema delimiters.
  2. Runtime validation catches malformed JSON and injects `<|tool_error|>` back into the context, enabling one-step conversational self-correction.
  3. Phase 1B dataset curation will incorporate structured tool examples with varied schemas.

