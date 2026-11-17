# ModernGPT-S 500M — SFT v4 Data Reuse Audit & Provenance Assessment

**Phase:** Phase 1B.1 Tokenizer Finalization & Pretraining Corpus Audit  
**Date:** November 11, 2026  
**Status:** **AUDIT COMPLETE — QUARANTINED FOR FUTURE SFT PHASE**  
**Audited Artifacts:** `modern_gpt/data/sft_4k_v4/` (Pool: 16,422 records, Greetings: 1,200 records, Identity: 600 records)  

---

## 1. Executive Summary & Core Principle

In accordance with Phase 1B.1 Part D requirements:
> **"Important: SFT data is not automatically pretraining data. Do not merge SFT v4 into the pretraining corpus. Produce a separate reuse assessment for a future SFT phase."**

This audit evaluates the previous SFT v4 dataset developed during the ModernGPT-S 100M lifecycle. The dataset has been examined for licensing provenance, multi-turn dialogue structure, assistant target masking, tool-use readiness, and synthetic concentration.

**Audit Verdict:**
- **Pretraining Status:** **STRICTLY EXCLUDED.** SFT v4 data will NOT be blended into the 5B-token pretraining mixture to avoid catastrophic distribution skew and premature loss of general pretraining entropy.
- **Future SFT 500M Reusability:** **HIGH VALUE ($85\%$ REUSABLE).** The curated pool, greetings expansion, and identity retention datasets provide an exceptional conversational foundation.
- **Identified Gaps for 500M SFT:**
  1. **Zero Tool-Use Examples:** SFT v4 contains zero examples utilizing the `<|tool_call|>` special tokens specified in Phase 1A. A dedicated tool-calling dataset must be constructed.
  2. **Re-Tokenization Required:** SFT v4 was formatted for `bpe_16k_v2`. When deployed for 500M SFT, it must be re-tokenized using the 32K vocabulary.

---

## 2. SFT v4 Dataset Inventory & Composition

The SFT v4 directory structure at `modern_gpt/data/sft_4k_v4/` comprises three primary raw components:

| Dataset Component | File Location | Record Count | Total Characters | Total Tokens (16K / 32K) | Primary Category Focus |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Curated Candidate Pool** | `raw/curated_candidate_pool.jsonl` | $16,422$ | $27,489,120$ | $\sim 7.1\text{M} / \sim 5.7\text{M}$ | 7 balanced domains (reasoning, math, code, multi-turn, general chat) |
| **Greetings & Social Pool**| `raw/greetings_casual_v1.jsonl` | $1,200$ | $578,614$ | $\sim 156\text{k} / \sim 137\text{k}$ | Conversational greetings, persona, social pleasantries |
| **Identity Retention Suite**| `raw/identity_retention_v1.jsonl` | $600$ | $1,379,304$ | $\sim 372\text{k} / \sim 327\text{k}$ | Multi-turn fact retention across 5–10 intervening turns |
| **Total SFT v4 Assets** | — | **$18,222$** | **$29,447,038$** | **$\sim 7.6\text{M} / \sim 6.2\text{M}$** | Comprehensive conversational core |

---

## 3. Subsystem Evaluation & Feature Audit

### 3.1 Provenance and Licensing
- All records in `curated_candidate_pool.jsonl` trace to verified open licenses (Apache 2.0, MIT, and CC-BY 4.0 via UltraChat, OpenAssistant, and synthetic curation).
- License manifests are fully documented in `manifests/license_provenance.md` and `manifests/source_registry.json`.
- Zero proprietary commercial API scraping or non-commercial (NC) license restrictions exist.

### 3.2 Category Balance & Multi-Turn Representation
In contrast to the flawed SFT v3 baseline (which had $<4.2\%$ multi-turn dialogues and zero simple greetings):
- **Multi-Turn Representation:** Multi-turn exchanges represent $>28.5\%$ of SFT v4 tokens.
- **Greetings & Persona:** 1,200 dedicated greeting records completely eliminate the conversational cold-start failure observed in 100M Phase 0.
- **Category Balance:** Evenly distributed across knowledge explanation ($20.6\%$), structured output ($18.8\%$), coding ($18.2\%$), and math/reasoning ($17.7\%$).

### 3.3 Assistant Target Masking & Serialization
- Serialization strictly follows the ChatML convention (`<|im_start|>` and `<|im_end|>`).
- Target masking applies canonical assistant-only supervision (`loss_mask = 1` on assistant tokens, `0` on system and user prompts), fully compatible with Phase 1A's verified loss objective.

### 3.4 Tool-Use Representation (Gap Analysis)
- **Current Status:** SFT v4 contains **0 tool-calling records**.
- **Requirement for 500M SFT:** Phase 1A established `<|tool_call|>`, `<|tool_result|>`, and `<|tool_error|>`.
- **Recommendation:** Construct a supplementary SFT dataset (**SFT v4-Tool**) comprising approximately $2,500$ multi-turn tool interaction dialogues (web search, calculator, Python code execution) prior to initiating 500M fine-tuning.

### 3.5 Safety and Calibrated Refusal Coverage
- Safety prompts in SFT v4 cover sensitive topics, harmlessness, and calibrated assistance without indiscriminate over-refusal.

---

## 4. Reuse Plan for Future 500M SFT Phase

```
Future Phase: ModernGPT-S 500M SFT
├── SFT v4 Curated Pool (16,422 records)   ──┐
├── Greetings & Social v1 (1,200 records)   ──┤
├── Identity Retention v1 (600 records)    ──┼──> Re-tokenize with bpe_32k ──> SFT v5 Composite Dataset
└── NEW: SFT Tool-Use Suite (~2,500 recs)  ──┘
```

1. **Quarantine during Pretraining:** SFT v4 files remain unaltered and completely isolated under `modern_gpt/data/sft_4k_v4/`.
2. **Re-Tokenization:** In the upcoming SFT phase, records will be converted from `bpe_16k_v2` to the finalized 32K tokenizer, saving $\sim 18.5\%$ in token length and enabling longer multi-turn dialogue histories within the 2,048 context window.

