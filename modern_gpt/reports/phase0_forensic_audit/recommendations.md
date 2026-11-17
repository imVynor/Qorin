# Phase 0: Forensic Diagnosis & Baseline Audit — Prioritized Action Plan & Recommendations

This document outlines the strategic engineering roadmap to resolve conversational failures in ModernGPT-S. It provides an immediate remediation plan for the **100M model** and architectural specifications for the upcoming **250M–300M model**.

---

## 1. Dual-Track Strategy Overview

| Track | Objective | Primary Target | Expected Outcome |
| :--- | :--- | :--- | :--- |
| **Track 1: 100M Refinement** | Maximize the efficiency and conversational usability of the existing 100M architecture | Fast, ultra-light edge conversational assistant (4K context, ~200MB memory footprint) | Fluid casual chat, accurate identity retention, clean turn endings, elimination of boilerplate |
| **Track 2: 250M–300M Scaling** | Build a high-capacity foundation model with robust multi-step reasoning, arithmetic, and code generation | High-performance small LLM (competitive with SmolLM-360M / Qwen2.5-0.5B) | Strong zero-shot generalization, emergent arithmetic reasoning, reliable JSON/tool calling |

---

## 2. Track 1: Action Plan for ModernGPT-S 100M

### Phase 1: Inference & Runtime Remediations (Zero-Training Required)
1. **Fix Delimiter Repetition Penalty Leak**:
   - In `chat.py`, explicitly exclude the token IDs representing `<|im_start|>` and `<|im_end|>` from the repetition penalty calculation.
2. **System-Aware Context Truncation**:
   - Replace raw left-token slicing (`input_ids[-(context_length - max_tokens):]`) with a structured eviction queue that pins the system prompt at index 0 and evicts the oldest user/assistant turns in pairs.
3. **Calibrated Sampling Defaults**:
   - For factual/QA tasks, set `temperature=0.2, repetition_penalty=1.05`.
   - For conversational chat, set `temperature=0.6, repetition_penalty=1.10, top_p=0.90`. Avoid aggressive repetition penalties ($>1.15$) that destabilize delimiter generation.

### Phase 2: SFT v4 Dataset Surgery & Re-balancing
The current SFT v3 dataset (31,949 samples) is over-concentrated on rigid templates. SFT v4 should be condensed into a **clean, balanced 16,000-sample curriculum**:

| Category | SFT v3 Count | SFT v4 Recommended Target | Rationale |
| :--- | :--- | :--- | :--- |
| **Long Context Retrieval** | 9,781 (30.6%) | **1,500 (9.4%)** | 1,500 samples are sufficient to preserve 4K needle-in-a-haystack retrieval without crowding out conversational priors. |
| **Commonsense QA** | 8,269 (25.9%) | **1,500 (9.4%)** | Drastically prune redundant `"The correct answer is [A-E]"` lines. Diversify phrasing. |
| **Casual & Greeting Dialogues** | **0 (0.0%)** | **1,000 (6.3%)** | Inject clean greetings (`"Hello"`, `"Hi"`, `"How are you?"`), identity questions (`"What is your name?"` $\rightarrow$ `"I am ModernGPT-S..."`), and short farewells. |
| **Multi-Turn Dialogue** | 1,343 (4.2%) | **4,000 (25.0%)** | Restore multi-turn depth (2–6 turns) from curated UltraChat and SmolTalk conversational subsets. |
| **Python Coding & Tools** | 2,990 (9.4%) | **3,000 (18.8%)** | Maintain high coding quality; ensure separation between code scripts and raw JSON formats. |
| **Structured JSON** | 1,895 (5.9%) | **2,000 (12.5%)** | Purge python function wrappers; enforce raw RFC 8259 JSON output targets. |
| **Mathematics & Logic** | 2,907 (9.1%) | **1,500 (9.4%)** | Retain basic arithmetic; add explicit multi-step explanations. |
| **Safety & Crisis Refusal** | **0 (0.0%)** | **1,500 (9.4%)** | Add standard refusals for dangerous tasks and crisis referrals (988 lifeline). |
| **TOTAL** | **31,949** | **16,000** | **High-quality, balanced, attractor-free distribution** |

### Phase 3: Training Pipeline Correction
1. **Token-Normalized Loss**:
   - Update `train_sft_4k.py` to divide accumulated loss by total supervised tokens in the gradient accumulation window, eliminating the 80x per-token gradient bias toward short QA lines.
2. **Tokenizer Special Tokens**:
   - Register `<|im_start|>` and `<|im_end|>` as atomic special tokens in the tokenizer vocab, ensuring they are assigned single, indivisible token IDs.

---

## 3. Track 2: Architectural Blueprint for ModernGPT-S 250M–300M

To achieve emergent multi-step reasoning, robust world knowledge, and reliable instruction following, scale the architecture to the **250M–300M parameter regime**:

### Recommended Architecture Specification:
- **Parameter Count**: ~270M parameters.
- **Layers ($n_{\text{layer}}$)**: 24 (provides deep hierarchical feature extraction).
- **Hidden Dimension ($d_{\text{model}}$)**: 1,024.
- **Query Heads ($n_{\text{head}}$)**: 16 (head dimension: 64).
- **Key-Value Heads ($n_{\text{kv\_head}}$)**: 4 (Grouped Query Attention, 4:1 ratio for memory efficiency).
- **FFN Hidden Dimension**: 2,816 (SwiGLU activation).
- **Vocabulary Size**: 32,768 (BPE with dedicated atomic control tokens).
- **Context Length**: 4,096 tokens native (RoPE base frequency $\theta = 500,000$).
- **Precision**: BFloat16 native with FlashAttention-2 / SDPA.

### Pre-Training & Post-Training Curriculum:
1. **Pre-training Scale**: Train on a curated 15B–25B token deduplicated corpus (Fineweb-Edu, OpenWebMath, GitHub code).
2. **Post-training**: Apply SFT v4 curriculum followed by Direct Preference Optimization (DPO) on pair-wise conversational preferences to strictly suppress repetitive attractors.

---

## 4. Immediate Decision Gate

- **Status on Further Training**: **PAUSED**. Do not launch Step 8E or retrain 100M until the SFT v4 dataset re-balancing (Phase 2) and token-level loss normalization (Phase 3) are completed and validated.
- **Immediate Next Step**: Approve the SFT v4 dataset curation plan and tokenizer special-token registration.

