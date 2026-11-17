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

## 4. Personal Information & Privacy Audit

- No records contain private personal identifiable information (PII), credentials, private phone numbers, physical residential addresses, or secret API keys.
- Fictional names used in conversation and identity-retention probes ("Anish", "Carlos", "Elena", "Sarah", "David", "Priya", "Aisha", "Kenji", "Marcus", "Fatima") are purely illustrative and used across diverse fictional contexts.
