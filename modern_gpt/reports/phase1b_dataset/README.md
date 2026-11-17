# ModernGPT-S — Phase 1B.1: Dataset Inspection and Analysis

**Status:** COMPLETE & VERIFIED  
**Phase:** 1B.1 (Dataset Inspection & Forensic Analysis)  
**Date:** 2026-11-09  
**Directory:** `modern_gpt/reports/phase1b_dataset/`  
**Dataset Analyzed:** `modern_gpt/data/sft_4k_v3/processed/` (`train.jsonl`, `val.jsonl`)  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

Phase 1B.1 was conducted as a rigorous, read-only inspection and forensic analysis of the existing Supervised Fine-Tuning v3 dataset (`sft_4k_v3`).

Its purpose is to uncover the exact data-level root causes behind ModernGPT-S 100M's conversational failures (formulaic greeting responses, identity amnesia, corporate email speech, and multi-turn context collapse) and provide an empirically grounded plan for engineering **SFT v4**.

### Key Findings
1. **Structural Monoculture**: Two rigid synthetic formats account for **56.50% of all training records** (`synthetic_long_retrieval_v3` at 30.61% and `tau/commonsense_qa` at 25.88%). Over 18,000 records begin with one of four opening template phrases.
2. **Single-Turn Dominance**: **95.80% of all records** (30,606 of 31,949) are strictly single-turn queries. True multi-turn dialogues (≥4 messages) account for only **4.20% of records** (1,343 records).
3. **Conversational Greeting Starvation**: The training dataset contains **only 9 records (216 supervised tokens)** from the diverse conversational greeting pool, because an unstratified split dumped the first items into the validation split.
4. **Zero Identity-Retention Examples**: There are zero records where a user states their identity, converses across intervening turns, and prompts the assistant to recall or maintain that context.
5. **Zero Safety / Responsible Alignment Data**: SFT v3 contains exactly 0 records and 0 tokens dedicated to safety, responsible refusal, or crisis helpline resources.

---

## 2. Phase 1B.1 Documentation Directory Map

The following reports constitute the complete Phase 1B.1 deliverable in `modern_gpt/reports/phase1b_dataset/`:

| Report File | Title & Scope | Key Empirical Findings / Deliverables |
| :--- | :--- | :--- |
| [`dataset_inventory.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/dataset_inventory.md) | Volumetric & Structural Accounting | 31,949 train records (85.72 MB); 3,277 val records (8.67 MB); 3,362,374 train supervised tokens; mean sup tokens = 105.24. |
| [`category_distribution.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/category_distribution.md) | Content Distribution Audit | Record share vs supervised token share divergence; 57.61% of context input tokens consumed by retrieval. |
| [`conversation_analysis.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/conversation_analysis.md) | Dialogue Depth & Structure | 95.80% single-turn tasks; only 4.20% true multi-turn; 54.73% of responses <50 tokens; zero identity recall chains. |
| [`repetition_analysis.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/repetition_analysis.md) | Template & Attractor Forensic | 0 byte-level duplicate rows, but 4 template families dominate 56.5% of records; `"the correct answer"` opens 8,271 records. |
| [`source_inventory.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/source_inventory.md) | Provenance & License Tracking | Master table of all 20 upstream sources; 72.7% MIT, 19.4% Apache-2.0, 6.6% CC-BY-4.0, 6.9% CC-BY-SA. All permissive. |
| [`tokenizer_compatibility.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/tokenizer_compatibility.md) | Tokenizer & ChatML Verification | Tokenizer fingerprint `be6c3d60c5baf217` verified; 100% records adhere to ChatML; assistant-only loss masking verified. |
| [`recommendations.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/recommendations.md) | Technical Roadmap for SFT v4 | Target composition by supervised tokens (25% conversation, 20% multi-turn, 15% knowledge, 12% instruction, 10% math, 8% code, 5% retrieval, 5% safety). |
| [`README.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_dataset/README.md) | Master Summary & Index | Executive overview, summary tables, compliance confirmation, and next steps. |

---

## 3. High-Level Empirical Metrics Summary

| Metric | Train Split | Validation Split | Full SFT v3 Dataset |
| :--- | :---: | :---: | :---: |
| **Total Records** | 31,949 | 3,277 | 35,226 |
| **Total Messages** | 77,750 | 7,998 | 85,748 |
| **Total Supervised Tokens** | 3,362,374 | 362,248 | 3,724,622 |
| **Total Input Sequence Tokens** | 24,081,703 | 2,443,646 | 26,525,349 |
| **Single-Turn Records (2–3 msgs)** | 30,606 (95.80%) | 3,115 (95.06%) | 33,721 (95.73%) |
| **Multi-Turn Records (≥4 msgs)** | 1,343 (4.20%) | 162 (4.94%) | 1,505 (4.27%) |
| **Mean Supervised Tokens / Record** | 105.24 | 110.54 | 105.73 |
| **Median Supervised Tokens / Record** | 45.00 | 47.00 | 45.00 |
| **Top 2 Dominant Sources Share** | 18,050 records (56.50%) | 1,797 records (54.84%) | 19,847 records (56.34%) |
| **Simple Greetings in Train** | 9 records (216 tokens) | 11 records (362 tokens) | 20 records (578 tokens) |
| **Safety / Responsible Records** | 0 records (0.0%) | 0 records (0.0%) | 0 records (0.0%) |
| **Exact Byte-Level Duplicate Rows** | 0 | 0 | 0 |

---

## 4. Compliance & Safety Statement

In strict adherence to Phase 1B.1 non-negotiable boundaries:
- **No training was started, resumed, or initialized.**
- **No model checkpoints were modified or overwritten.**
- **No tokenizer vocabulary or configuration files were changed.**
- **The SFT v3 dataset files remain 100% intact and byte-for-byte identical to their recorded SHA-256 hashes.**
- **No new training data was generated or serialized into active training directories.**

