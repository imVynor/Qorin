# ModernGPT-S — Phase 1B.2: License & Provenance Manifest Report

**Dataset Target:** SFT v4 Curated Candidate Pool  
**Status:** PROVENANCE TRACEABLE & PERMISSIVELY LICENSED  
**Date:** 2026-11-09  
**Candidate Pool Path:** `modern_gpt/data/sft_4k_v4/raw/curated_candidate_pool.jsonl`  
**Manifest Reference:** `modern_gpt/data/sft_4k_v4/manifests/source_registry.json`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report establishes complete, record-level license and provenance tracking for all 16,422 candidate records assembled in the SFT v4 candidate pool.

Every record in `curated_candidate_pool.jsonl` retains an explicit `"license"` and `"source"` attribute. All included sources operate under verified permissive open-source licenses: **MIT**, **Apache-2.0**, **CC-BY-4.0**, or **CC-BY-SA-3.0/4.0**.

---

## 2. License Breakdown Across Candidate Records

| License Identifier | Record Count | Share of Pool (%) | Supervised Target Tokens | Share of Tokens (%) | Upstream Sources | Redistribution Compliance |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **MIT** | 7,070 | **43.05%** | 895,496 | **35.42%** | `commonsense_qa`, `synthetic_retrieval`, `gsm8k`, `json`, `direct_qa`, `curated_context_insulation` | Permissive: Retain copyright notice. |
| **Apache-2.0** | 4,149 | **25.26%** | 981,598 | **38.82%** | `smoltalk` (`self-oss`, `apigen`, `explore`, `rewrite`, `magpie`) | Permissive: Notice of modifications and patent grant. |
| **CC-BY-4.0** | 2,101 | **12.79%** | 205,373 | **8.12%** | `code_alpaca`, `mbpp` | Permissive with attribution. |
| **CC-BY-SA-3.0** | 1,262 | **7.68%** | 127,648 | **5.05%** | `databricks-dolly-15k` (`dolly_instruction`, `dolly_qa`) | Share-Alike: Derivative datasets must provide compatible SA terms. |
| **CC-BY-SA-4.0** | 946 | **5.76%** | 14,716 | **0.58%** | `squad` | Share-Alike: Derivative datasets must provide compatible SA terms. |
| **MIT / Apache-2.0** | 894 | **5.44%** | 252,463 | **9.99%** | `smoltalk:openhermes` | Dual permissive. |
| **Total** | **16,422** | **100.00%** | **2,528,376** | **100.00%** | **16 Distinct Sources** | **Zero non-commercial or proprietary licenses.** |

---

## 3. Share-Alike (CC-BY-SA) Isolation Analysis

A critical requirement of open-source stewardship is properly isolating Share-Alike data:
- **Total CC-BY-SA Records**: 2,208 records (13.44% of records, 5.63% of supervised tokens).
- **Sources**: Databricks Dolly (`dolly_instruction`, `dolly_qa`) and Stanford SQuAD (`squad`).
- **Isolation Policy**: Because each record retains its explicit `"source"` and `"license"` metadata, the CC-BY-SA portion can be cleanly identified or segregated if a purely MIT/Apache release artifact is desired in downstream packaging.

---

## 4. Exclusion of Non-Commercial / Questionable Datasets

In accordance with Section 1:
- `Evol-Instruct-Code-80K` (CC-BY-NC-SA-4.0) was strictly rejected during screening. Zero records were ingested.
- No scraped or unvetted web-scrapes were added.
- All evaluation splits (`gsm8k:test`, `commonsense_qa:validation`, `EVAL_ENTITIES`) remain strictly quarantined.

