# ModernGPT-S — Phase 1B.2: Category Reclassification & Mapping Report

**Dataset Target:** SFT v4 Curated Candidate Pool  
**Status:** RECLASSIFIED & AUDITED  
**Date:** 2026-11-09  
**Manifest Reference:** `modern_gpt/data/sft_4k_v4/manifests/category_mapping.json`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

In SFT v3, the category `general_conversation` was severely overloaded, containing everything from corporate email rewriting to formal reading comprehension tasks. Conversely, task-oriented instruction following was unrepresented as an independent category.

Phase 1B.2 executed a formal category remapping, reorganizing candidate sources into 7 distinct functional categories that reflect the true cognitive task required of the model.

---

## 2. Category Transition Matrix

The table below documents every source whose category was modified between SFT v3 and the SFT v4 candidate pool:

| Source Identifier | SFT v3 Category | SFT v4 Curated Category | Records Reclassified | Supervised Tokens Reclassified | Technical Rationale |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **`smoltalk:smol-rewrite`** | `general_conversation` | `instruction_following` | 460 | 83,444 | Text rewriting, tone adjustment, and professional drafting are explicit instructions, not casual conversation. |
| **`local_sft_v2:dolly_instruction`** | `general_conversation` | `instruction_following` | 941 | 111,968 | Summarization, entity extraction, and classification are instruction-following tasks. |
| **`local_sft_v2:squad`** | `general_conversation` | `knowledge_and_explanations` | 946 | 14,716 | Reading comprehension questions testing factual recall from Wikipedia belong in knowledge. |
| **`smoltalk:explore-instruct`** | `general_conversation` | `knowledge_and_explanations` | 916 | 33,370 | Deep conceptual explanations across science, geography, and humanities belong in knowledge. |
| **`curated_direct_qa`** | `general_conversation` | `knowledge_and_explanations` | 19 | 748 | Direct factual science/history definitions belong in knowledge. |
| **`tau/commonsense_qa:train`** | `science_commonsense` | `knowledge_and_explanations` | 1,500 | 65,293 | Everyday commonsense reasoning consolidated under general knowledge. |
| **`synthetic_long_retrieval_v3`** | `long_context_retrieval` | `retrieval_and_structured_output` | 1,200 | 67,440 | Consolidated with structured JSON under the unified 5% retrieval/structured target. |
| **`local_sft_v2:json`** | `structured_json` | `retrieval_and_structured_output` | 728 | 54,814 | Consolidated under unified structured output. |
| **`smoltalk:apigen-80k`** | `structured_json` | `retrieval_and_structured_output` | 1,167 | 118,048 | Consolidated under unified structured output. |
| **`local_sft_v2:code_alpaca`** | `coding_python` | `coding_technical` | 1,790 | 174,606 | Broadened category naming convention. |
| **`smoltalk:self-oss-instruct`** | `coding_python` | `coding_technical` | 889 | 243,126 | Broadened category naming convention. |
| **`local_sft_v2:mbpp`** | `coding_python` | `coding_technical` | 311 | 30,767 | Broadened category naming convention. |
| **`openai/gsm8k:train`** | `mathematics` | `reasoning_and_mathematics` | 2,729 | 337,793 | Word problems mapped to reasoning and mathematics. |
| **`local_sft_v2:gsm8k`** | `mathematics` | `reasoning_and_mathematics` | 178 | 22,222 | Word problems mapped to reasoning and mathematics. |
| **Total Reclassified Records** | — | — | **12,774** | **1,357,555** | **Restores clear informational semantics to each category.** |

---

## 3. Impact on General Conversation Purity

Before reclassification:
- `general_conversation` in SFT v3 contained 5,162 records and 900,010 tokens.
- **63.2% of its records** were actually email rewrites, reading comprehension tasks, or domain essays.
- Real human conversation was a minority trace.

After reclassification:
- `general_conversation` in the candidate pool is purified down to **1,224 genuinely conversational records** (268,359 supervised tokens) from `dolly_qa`, `openhermes`, and curated dialogue.
- This creates clean semantic space for the addition of the new conversational greeting and social exchange pool in Phase 1B.3.

