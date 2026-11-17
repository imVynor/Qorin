# ModernGPT-S — Phase 1B.2: Source Curation & Dataset Preparation Summary

**Status:** COMPLETE & AUDITED  
**Phase:** 1B.2 (Source Curation & Candidate Pool Assembly)  
**Date:** 2026-11-09  
**Candidate Pool Path:** `modern_gpt/data/sft_4k_v4/raw/curated_candidate_pool.jsonl`  
**Candidate Pool SHA-256:** `f6d5cccf21d4df476c42fe78fbf0520de49c43387a58f9532df28b75606a192e`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

Phase 1B.2 successfully established the curated candidate foundation for **SFT v4** by curating existing source records, pruning massive template repetition, reclassifying misclassified tasks, and establishing full token accounting.

In strict compliance with instructions:
- **Zero synthetic conversations were generated.**
- **Zero training was started.**
- **SFT v3 and historical checkpoints were preserved byte-for-byte.**
- **The tokenizer (`be6c3d60c5baf217`) and model architecture were unchanged.**

The resulting curated candidate pool contains **16,422 high-quality records** and **2,528,376 supervised target tokens**, providing 70.2% of the eventual 3.6M token training target from existing, vetted data.

---

## 2. Key Achievements & Curation Metrics

1. **Elimination of Template Monoculture (-15,350 records)**:
   - `tau/commonsense_qa`: Pruned from 8,269 records down to **1,500 diverse records** (eliminating 6,769 repetitive `"The correct answer is [Choice]"` templates).
   - `synthetic_long_retrieval_v3`: Pruned from 9,781 records down to **1,200 balanced records** across 512, 1024, 2048, and 3072 context lengths (eliminating 8,581 repetitive document templates).
2. **Category Reclassifications (12,774 records reclassified)**:
   - `smoltalk:smol-rewrite` (460 records) and `local_sft_v2:dolly_instruction` (941 records) moved from `general_conversation` to `instruction_following`.
   - `local_sft_v2:squad` (946 records) and `smoltalk:explore-instruct` (916 records) moved from `general_conversation` to `knowledge_and_explanations`.
   - Commonsense QA (1,500 records) moved to `knowledge_and_explanations`.
   - Structured JSON and retrieval consolidated under `retrieval_and_structured_output`.
   - `general_conversation` purified down to **1,224 genuine conversational dialogue records** (268,359 supervised tokens).
3. **Exact Deduplication (-493 records)**:
   - Canonicalized SHA-256 content hashing identified and purged 493 cross-source duplicate conversations. Zero duplicate hashes remain in the pool.
4. **Programmatic Quality Gates (100% Pass Rate)**:
   - Python AST Syntax: 2,990 coding records parsed through `ast.parse()` with zero syntax errors.
   - JSON Structure: 1,895 structured output records parsed through `json.loads()` with zero malformed schemas.
   - Context Limit: 100% of records $\le 4,096$ tokens; 100% have `supervised_tokens > 0`.
5. **Multi-Turn Context Ingestion**:
   - Ingested all valid multi-turn dialogues from `custom_context_insulation_1200.jsonl` (topic switching, state retention, independent tasks), expanding multi-turn dialogue to **1,424 records and 901,607 supervised tokens**.

---

## 3. Curated Candidate Pool Token Distribution

| Category Identifier | Record Count | Record Share (%) | Supervised Target Tokens | Supervised Token Share (%) | Mean Sup Tokens / Record |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`multi_turn_dialogue`** | 1,424 | 8.67% | 901,607 | **35.66%** | 633.15 |
| **`coding_technical`** | 2,990 | 18.21% | 448,499 | **17.74%** | 150.00 |
| **`reasoning_and_mathematics`**| 2,907 | 17.70% | 360,015 | **14.24%** | 123.84 |
| **`general_conversation`** | 1,224 | 7.45% | 268,359 | **10.61%** | 219.25 |
| **`retrieval_and_structured_output`**| 3,095 | 18.85% | 240,302 | **9.50%** | 77.64 |
| **`instruction_following`** | 1,401 | 8.53% | 195,412 | **7.73%** | 139.48 |
| **`knowledge_and_explanations`**| 3,381 | 20.59% | 114,182 | **4.52%** | 33.77 |
| **Total Curated Pool** | **16,422** | **100.00%** | **2,528,376** | **100.00%** | **153.96** |

---

## 4. Deliverables Created in Phase 1B.2

### In `modern_gpt/data/sft_4k_v4/reports/`:
1. [`source_curation.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/reports/source_curation.md) — Detailed curation actions, pruned volumes, and selection criteria.
2. [`category_mapping.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/reports/category_mapping.md) — Reclassification matrix for all 12,774 remapped records.
3. [`deduplication_report.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/reports/deduplication_report.md) — Duplicate hash purge and programmatic quality gates (AST, JSON).
4. [`license_provenance.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/reports/license_provenance.md) — Complete record-level license accounting (MIT, Apache-2.0, CC-BY, CC-BY-SA).
5. [`token_accounting.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/reports/token_accounting.md) — Exact token distributions and deficit analysis relative to target.
6. [`construction_plan.md`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/reports/construction_plan.md) — Technical roadmap for targeted supplementation and assembly.

### In `modern_gpt/data/sft_4k_v4/manifests/`:
- [`curation_manifest.json`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/manifests/curation_manifest.json) — Master machine-readable curation manifest.
- [`source_registry.json`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/manifests/source_registry.json) — Source-by-source registry of records, tokens, and licenses.
- [`category_mapping.json`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/sft_4k_v4/manifests/category_mapping.json) — Formal mapping rules from v3 to v4 categories.
- `curated_candidate_pool.sha256` — Checksum (`f6d5cccf...`).

### In `modern_gpt/data/sft_4k_v4/raw/`:
- `curated_candidate_pool.jsonl` (16,422 records, 47.38 MB).

---

## 5. Compliance & Boundaries Statement

- **Zero new conversations were generated**: All 16,422 records were curated strictly from existing historical source data.
- **Zero training started**: No models, optimizers, or training scripts were executed.
- **SFT v3 is 100% intact**: All files in `modern_gpt/data/sft_4k_v3/` remain untouched and match their historical SHA-256 hashes.
- **Tokenizer fingerprint**: Confirmed bitwise identical (`be6c3d60c5baf217`).

