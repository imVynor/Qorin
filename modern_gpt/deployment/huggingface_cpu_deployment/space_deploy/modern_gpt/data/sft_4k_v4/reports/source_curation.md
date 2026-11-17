# ModernGPT-S — Phase 1B.2: Source Curation Report

**Dataset Target:** SFT v4 Curated Candidate Pool  
**Status:** CURATED & AUDITED  
**Date:** 2026-11-09  
**Candidate Pool Path:** `modern_gpt/data/sft_4k_v4/raw/curated_candidate_pool.jsonl`  
**Candidate Pool SHA-256:** `f6d5cccf21d4df476c42fe78fbf0520de49c43387a58f9532df28b75606a192e`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

Phase 1B.2 executed the systematic curation of existing source datasets to construct the **SFT v4 candidate pool**.

In accordance with strict phase boundaries:
- **Zero synthetic examples were generated**.
- **Zero model training was initiated**.
- **SFT v3 and historical checkpoints were preserved byte-for-byte**.
- Existing records were audited, pruned of massive template redundancy, reclassified out of inappropriate categories, and deduplicated.

The curated pool comprises **16,422 high-quality candidate records** containing **2,528,376 supervised target tokens**, reduced from the initial 35,226-record SFT v3 corpus.

---

## 2. Quantitative Curation Summary

| Curation Action | Upstream Sources Affected | Initial Records (v3) | Retained Records (v4 Pool) | Pruned / Excluded Records | Rationale & Selection Criteria |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Pruning Multiple-Choice Overrepresentation** | `tau/commonsense_qa:train` | 8,269 | **1,500** | **-6,769 (-81.9%)** | Deterministic quality-based subsampling across distinct premise topics (Seed=42). Eliminated 6,769 instances of `"The correct answer is [Choice]"`. |
| **Pruning Synthetic Retrieval Bloat** | `synthetic_long_retrieval_v3` | 9,781 | **1,200** | **-8,581 (-87.7%)** | Stratified sampling of 300 records each across 512, 1024, 2048, and 3072 context lengths. Eliminated over 8,500 repetitive citation templates. |
| **Reclassifying Instruction Following** | `smol-rewrite`, `dolly_instruction` | 1,401 | **1,401** | **0 (Reclassified)** | Relocated text transformation and task-directed drafting from `general_conversation` to `instruction_following`. |
| **Reclassifying Factual Knowledge** | `squad`, `explore-instruct`, `direct_qa` | 1,881 | **1,881** | **0 (Reclassified)** | Relocated reading comprehension and domain explanations from `general_conversation` to `knowledge_and_explanations`. |
| **Purging Exact Duplicate Hashes** | Cross-source duplicate check | 493 | **0** | **-493** | Exact SHA-256 canonical hash collision purge. |
| **Multi-Turn Context Insulation Ingestion**| `custom_context_insulation_1200` | 632 | **1,101** | **+469 (Ingested)** | Fully integrated all valid multi-turn dialogues from the 1,200 context insulation suite. |
| **Preserving Coding & Math** | `gsm8k`, `code_alpaca`, `self-oss`, `mbpp` | 5,897 | **5,897** | **0** | Retained with 100% AST syntax parsing validation (`ast.parse`) for Python code. |
| **Preserving Valid Structured JSON** | `apigen-80k`, `json` | 1,895 | **1,895** | **0** | Retained with 100% JSON schema validation (`json.loads`). |
| **Total Curated Pool** | **All 16 Curated Sources** | **31,949** | **16,422** | **-15,527 (-48.6%)** | **Balanced, non-monolithic candidate foundation.** |

---

## 3. Detailed Source-by-Source Breakdown

| Source Identifier | License | Retained Records | Retained Supervised Tokens | SFT v4 Candidate Category | Quality Gate Passed |
| :--- | :---: | :---: | :---: | :--- | :--- |
| `coding_technical` (`code_alpaca`, `self-oss`, `mbpp`) | CC-BY / Apache | 2,990 | 448,499 | `coding_technical` | 100% AST syntax parse |
| `reasoning_and_mathematics` (`gsm8k`) | MIT | 2,907 | 360,015 | `reasoning_and_mathematics` | Grade-school math (train split only) |
| `retrieval_and_structured_output` (`synthetic_retrieval`, `json`, `apigen`) | MIT / Apache | 3,095 | 240,302 | `retrieval_and_structured_output` | 100% JSON schema parse + length stratified |
| `knowledge_and_explanations` (`commonsense_qa`, `squad`, `explore-instruct`, `direct_qa`) | MIT / CC-BY-SA | 3,381 | 114,182 | `knowledge_and_explanations` | Pruned MC (1,500) + factual reading QA |
| `multi_turn_dialogue` (`context_insulation`, `magpie`, `systemchats`) | MIT / Apache | 1,424 | 901,607 | `multi_turn_dialogue` | ≥4 message multi-turn chains |
| `instruction_following` (`dolly_instruction`, `smol-rewrite`) | CC-BY-SA / Apache | 1,401 | 195,412 | `instruction_following` | Reclassified task instructions |
| `general_conversation` (`dolly_qa`, `openhermes`, `curated_diverse`) | MIT / CC-BY-SA | 1,224 | 268,359 | `general_conversation` | Conversational dialogue & casual QA |
| **Total** | — | **16,422** | **2,528,376** | — | **Zero syntax or serialization errors** |

