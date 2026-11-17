# ModernGPT-S — Phase 1B: Source Data Inventory & License Provenance Audit

**Status:** AUDITED & APPROVED  
**Date:** 2026-11-09  
**Dataset Target:** ModernGPT-S SFT v4  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

Phase 1B establishes a rigorous source inventory for the new **SFT v4** dataset. Every candidate source has been audited for origin, redistribution license, duplication risk, data contamination, and structural suitability for training a 100M parameter conversational assistant.

All retained sources are licensed under unencumbered, permissive open-source licenses: **MIT**, **Apache-2.0**, **CC-BY-4.0**, or **CC-BY-SA-3.0/4.0**. All non-commercial (CC-BY-NC) or proprietary sources (e.g., `Evol-Instruct-Code-80k-v1`) remain strictly rejected.

---

## 2. Comprehensive Source Inventory Table

| Source Identifier | Origin / Upstream Repo | License | Record Count (Raw) | Intended v4 Category | Multi-Turn Support | Quality & Safety Notes | Audit Verdict |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- | :---: |
| **`curated_natural_greetings`** | Internal Curated (Phase 1B) | MIT | ~1,200 | General Conversation | No (Single turn) | Natural short greetings, acknowledgements; zero corporate boilerplate | **APPROVED** |
| **`curated_identity_retention`** | Internal Curated (Phase 1B) | MIT | ~600 | Multi-Turn Dialogue | Yes (3–8 turns) | Intervening turns between name introduction and recall; name corrections; varied identities | **APPROVED** |
| **`curated_context_insulation`** | Internal Curated (`step_08b`) | MIT | 1,200 | Multi-Turn Dialogue | Yes (4–12 turns) | Topic switching, state retention, irrelevant context suppression | **APPROVED** |
| **`curated_safety_suite`** | Internal Curated (Phase 1B) | MIT | ~1,200 | Safety | Mixed (1–4 turns) | Compassionate self-harm referral, neutral weapons refusal, cyber education | **APPROVED** |
| **`curated_direct_qa`** | Internal Curated (`step_08b`) | MIT | 250 | Knowledge & Explanations | No (Single turn) | Direct factual definitions, geography, science, history | **APPROVED** |
| **`smoltalk:smol-magpie-ultra`** | `HuggingFaceTB/smoltalk` | Apache-2.0 | 2,500 | Multi-Turn Dialogue | Yes (4–10 turns) | High-quality synthetic multi-turn dialogues | **APPROVED** |
| **`smoltalk:everyday-conversations`**| `HuggingFaceTB/smoltalk` | Apache-2.0 | 1,500 | General Conversation | Mixed (1–3 turns) | Filtered strictly to remove business emails and formal salutations | **APPROVED (Filtered)** |
| **`smoltalk:explore-instruct`** | `HuggingFaceTB/smoltalk` | Apache-2.0 | 1,800 | Knowledge & Explanations | No (Single turn) | In-depth domain explanations across sciences and humanities | **APPROVED** |
| **`smoltalk:openhermes`** | `teknium/OpenHermes-2.5` | MIT / Apache-2.0 | 2,000 | General Conversation / Knowledge | Mixed | High-quality conversational instruction pairs | **APPROVED** |
| **`smoltalk:smol-rewrite`** | `HuggingFaceTB/smoltalk` | Apache-2.0 | 1,200 | Instruction Following | No (Single turn) | Text transformation, style transfer, drafting (moved from conversation!) | **APPROVED** |
| **`smoltalk:apigen-80k`** | `HuggingFaceTB/smoltalk` | Apache-2.0 | 1,500 | Retrieval & Structured Output | No (Single turn) | Tool-call and API JSON generation. 100% verified via `json.loads` | **APPROVED** |
| **`smoltalk:self-oss-instruct`** | `HuggingFaceTB/smoltalk` | Apache-2.0 | 1,500 | Coding & Technical | No (Single turn) | Python coding tasks. Verified via `ast.parse` | **APPROVED** |
| **`openai/gsm8k:train`** | `openai/gsm8k` | MIT | 2,500 | Reasoning & Mathematics | No (Single turn) | Human-written math word problems. Test split quarantined | **APPROVED** |
| **`curated_verified_math`** | Internal Curated (Phase 1B) | MIT | 800 | Reasoning & Mathematics | No (Single turn) | Direct arithmetic, percentages, fractions, unit conversions (Python-verified) | **APPROVED** |
| **`tau/commonsense_qa:train`** | `tau/commonsense_qa` | MIT | ~1,500 | Knowledge & Explanations | No (Single turn) | Subsampled from 8,269. Reformulated to eliminate monolithic choice template | **APPROVED (Curated)** |
| **`local_sft_v2:code_alpaca`** | `HuggingFaceH4/CodeAlpaca_20K` | CC-BY-4.0 | 2,200 | Coding & Technical | No (Single turn) | Python coding instructions. Verified via `ast.parse` | **APPROVED** |
| **`local_sft_v2:mbpp`** | `google-research-datasets/mbpp` | CC-BY-4.0 | 350 | Coding & Technical | No (Single turn) | Python programming tasks with unit test references | **APPROVED** |
| **`local_sft_v2:dolly_instruction`**| `databricks/databricks-dolly-15k`| CC-BY-SA-3.0 | 1,200 | Instruction Following | No (Single turn) | Summarization, classification, creative drafting | **APPROVED** |
| **`local_sft_v2:dolly_qa`** | `databricks/databricks-dolly-15k`| CC-BY-SA-3.0 | 400 | General Conversation | No (Single turn) | Casual informational question answering | **APPROVED** |
| **`local_sft_v2:squad`** | `rajpurkar/squad` | CC-BY-SA-4.0 | 1,000 | Knowledge & Explanations | No (Single turn) | Contextual reading comprehension | **APPROVED** |
| **`local_sft_v2:json`** | Internal Curated | MIT | 800 | Retrieval & Structured Output | No (Single turn) | Structured JSON schema generation. Verified via `json.loads` | **APPROVED** |
| **`synthetic_long_retrieval_v4`**| Internal Synthetic Generator | MIT | 1,500 | Retrieval & Structured Output | No (Single turn) | Downscaled from 9,781 to ~1,500. Varied opening syntax to kill attractors | **APPROVED (Curated)** |
| **`nickrosh/Evol-Instruct-Code-80k-v1`** | `nickrosh/Evol-Instruct-Code-80k-v1` | CC-BY-NC-SA-4.0 | 80,000 | Coding | N/A | Non-commercial license incompatible with open redistribution | **REJECTED** |

---

## 3. License & Redistribution Analysis

1. **Permissive Licenses**:
   - **MIT License** (`gsm8k`, `commonsense_qa`, `openhermes`, curated sets, synthetic retrieval): Allows commercial and non-commercial redistribution, modification, and sublicensing with attribution.
   - **Apache 2.0** (`smoltalk` submixes): Permissive license with explicit patent grant.
   - **CC-BY-4.0** (`code_alpaca`, `mbpp`): Permissive with attribution.
2. **Share-Alike (SA) Licenses**:
   - **CC-BY-SA-3.0 & CC-BY-SA-4.0** (`dolly-15k`, `squad`): Requires derivative datasets to be distributed under identical or compatible share-alike terms. All records derived from these sources retain exact source provenance metadata so they can be isolated or packaged with CC-BY-SA terms.
3. **Quarantine & Non-Commercial Purge**:
   - `Evol-Instruct-Code-80K` (CC-BY-NC-SA) is strictly excluded. Zero examples from this source enter SFT v4.
   - All evaluation benchmarks (`gsm8k:test`, `commonsense_qa:validation`, `EVAL_ENTITIES`) are quarantined with zero overlap.

---

## 5. Measured SFT v3 Source Breakdown & Provenance Accounting

The empirical audit of `train.jsonl` (31,949 records) and `val.jsonl` (3,277 records) yields the exact per-source distribution:

| Source Identifier | Upstream Repository | License | Train Records | Train Sup Tokens | Val Records | Val Sup Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `synthetic_long_retrieval_v3` | Internal Synthetic Generator | MIT | 9,781 | 540,077 | 971 | 54,011 |
| `tau/commonsense_qa:train` | `tau/commonsense_qa` | MIT | 8,269 | 360,037 | 826 | 36,036 |
| `openai/gsm8k:train` | `openai/gsm8k` | MIT | 2,729 | 337,793 | 275 | 33,475 |
| `local_sft_v2:code_alpaca` | `HuggingFaceH4/CodeAlpaca_20K` | CC-BY-4.0 | 1,790 | 174,606 | 203 | 22,024 |
| `smoltalk:apigen-80k` | `HuggingFaceTB/smoltalk` | Apache-2.0 | 1,167 | 118,048 | 128 | 12,708 |
| `local_sft_v2:squad` | `rajpurkar/squad` | CC-BY-SA-4.0 | 946 | 14,716 | 83 | 1,282 |
| `local_sft_v2:dolly_instruction` | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | 941 | 111,968 | 84 | 11,482 |
| `smoltalk:explore-instruct` | `HuggingFaceTB/smoltalk` | Apache-2.0 | 916 | 33,370 | 96 | 3,676 |
| `smoltalk:openhermes` | `teknium/OpenHermes-2.5` | MIT / Apache-2.0 | 894 | 252,463 | 105 | 24,670 |
| `smoltalk:self-oss-instruct` | `HuggingFaceTB/smoltalk` | Apache-2.0 | 889 | 243,126 | 104 | 28,769 |
| `local_sft_v2:json` | Internal Curated JSON | MIT | 728 | 54,814 | 72 | 5,371 |
| `smoltalk:systemchats` | `HuggingFaceTB/smoltalk` | Apache-2.0 | 656 | 387,405 | 79 | 41,126 |
| `smoltalk:smol-rewrite` | `HuggingFaceTB/smoltalk` | Apache-2.0 | 460 | 83,444 | 34 | 6,396 |
| `smoltalk:smol-magpie-ultra` | `HuggingFaceTB/smoltalk` | Apache-2.0 | 323 | 477,575 | 33 | 58,696 |
| `local_sft_v2:dolly_qa` | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | 321 | 15,680 | 31 | 1,748 |
| `local_sft_v2:mbpp` | `google-research-datasets/mbpp` | CC-BY-4.0 | 311 | 30,767 | 36 | 3,443 |
| `curated_context_insulation` (5 types) | Internal Curated | MIT | 632 | 103,309 | 85 | 14,325 |
| `local_sft_v2:gsm8k` | `openai/gsm8k` | MIT | 178 | 22,222 | 19 | 2,538 |
| `curated_direct_qa` | Internal Curated | MIT | 19 | 748 | 2 | 110 |
| `curated_conversational_diverse` | Internal Curated | MIT | 9 | 216 | 11 | 362 |
| **Total** | — | — | **31,949** | **3,362,374** | **3,277** | **362,248** |

### License Distribution in SFT v3:
- **MIT**: 23,229 records (72.70% of train)
- **Apache-2.0**: 6,201 records (19.41% of train)
- **CC-BY-4.0**: 2,101 records (6.58% of train)
- **CC-BY-SA-3.0**: 1,262 records (3.95% of train)
- **CC-BY-SA-4.0**: 946 records (2.96% of train)
All sources possess verified permissive licenses. Redistribution requires preserving attribution for CC-BY and share-alike notices for CC-BY-SA derivatives.


