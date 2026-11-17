# ModernGPT-S 500M — Production Tokenizer 3-Way Regression Benchmark

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **PASSED — ZERO DRIFT AGAINST PHASE 1B.1 CANDIDATE**  
**Comparison:**
1. Baseline: `bpe_16k_v2` (Vocab: 16,000, Fingerprint: `be6c3d60c5baf217`)
2. Candidate: `bpe_32k_candidate` (Vocab: 32,768, Fingerprint: `a1274e6fd57b5b63`)
3. Production: `bpe_32k_production` (Vocab: 32,768, Fingerprint: `3559c7435e1d9330`)  
**Data Artifact:** `modern_gpt/reports/phase1b_2_production_data_pipeline/tokenizer_regression_results.json`  

---

## 1. Executive Summary

To verify that reserving the 10 atomic special tokens did not degrade or alter the compression efficiency of the 32K vocabulary, an exhaustive **3-Way Regression Benchmark** was conducted across the 365 held-out documents from Phase 1B.1 (totaling 433,699 raw bytes).

### Key Result: **PERFECT REPRODUCIBILITY ($0.000\%$ VARIANCE)**
Across all 8 evaluation domains, the production tokenizer generated **exactly identical token counts** to the candidate tokenizer:
- `bpe_16k_v2`: **126,138 tokens**
- `bpe_32k_candidate`: **101,572 tokens** ($-19.48\%$ vs 16K)
- `bpe_32k_production`: **101,572 tokens** ($-19.48\%$ vs 16K)
- **Variance between Candidate and Production:** **$\pm 0.000\%$**

---

## 2. Domain-by-Domain 3-Way Comparison Table

| Category | Docs | Raw Bytes | 16K Tokens | 32K Cand Tokens | 32K Prod Tokens | Prod Reduction vs. 16K | Prod vs. Cand Variance |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. General Web Prose** | 50 | 89,091 | 22,091 | 19,176 | **19,176** | **$-13.20\%$** | **$+0.000\%$** |
| **2. Educational / Textbook** | 50 | 99,526 | 22,995 | 20,106 | **20,106** | **$-12.56\%$** | **$+0.000\%$** |
| **3. Source Code** | 50 | 92,224 | 40,563 | 27,184 | **27,184** | **$-32.98\%$** | **$+0.000\%$** |
| **4. Mathematics / Tech** | 50 | 24,302 | 8,327 | 7,300 | **7,300** | **$-12.33\%$** | **$+0.000\%$** |
| **5. Wikipedia / Ref** | 50 | 16,483 | 4,213 | 3,895 | **3,895** | **$-7.55\%$** | **$+0.000\%$** |
| **6. Book-Style Prose** | 50 | 98,575 | 22,870 | 19,803 | **19,803** | **$-13.41\%$** | **$+0.000\%$** |
| **7. Conversational Dialogue**| 50 | 10,661 | 2,876 | 2,525 | **2,525** | **$-12.20\%$** | **$+0.000\%$** |
| **8. Multilingual** | 15 | 2,837 | 2,203 | 1,583 | **1,583** | **$-28.14\%$** | **$+0.000\%$** |
| **Total Benchmark** | **365** | **433,699** | **126,138** | **101,572** | **101,572** | **$-19.48\%$** | **$+0.000\%$** |

---

## 3. Programming Identifier Fragmentation Parity

Testing the 15 standard programming identifiers showed exact subword count preservation between candidate and production:

| Identifier | `bpe_16k_v2` | `bpe_32k_candidate` | `bpe_32k_production` | Subword Delta vs. 16K |
| :--- | :---: | :---: | :---: | :---: |
| `getUserProfileById` | 7 | 6 | **6** | -1 |
| `calculateMovingAverage` | 8 | 7 | **7** | -1 |
| `XMLHttpRequest` | 7 | 4 | **4** | -3 |
| `std::vector<int>` | 10 | 6 | **6** | -4 |
| `ConcurrentHashMap` | 8 | 5 | **5** | -3 |
| `parse_serialized_json` | 9 | 6 | **6** | -3 |
| `numpy.linalg.inv` | 9 | 5 | **5** | -4 |
| `torch.cuda.is_available` | 11 | 9 | **9** | -2 |
| `handle_connection_timeout` | 9 | 5 | **5** | -4 |
| `DATABASE_CONNECTION_POOL_MAX_SIZE` | 26 | 15 | **15** | -11 |
| `AbstractSingletonProxyFactoryBean` | 11 | 10 | **10** | -1 |
| `read_uncommitted_isolation_level` | 10 | 10 | **10** | 0 |
| `AES_256_GCM_SHA384` | 16 | 12 | **12** | -4 |
| `__init_subclass__` | 8 | 6 | **6** | -2 |
| `async_generator_context_manager` | 12 | 7 | **7** | -5 |
| **Mean Subwords per Identifier** | **10.73** | **7.53** | **7.53** | **$-29.81\%$** |

---

## 4. Multilingual Compression Parity

| Language | Raw Bytes | 16K Tokens | 32K Cand Tokens | 32K Prod Tokens | Prod B/T | Prod Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Russian (Cyrillic)** | 833 | 833 | 425 | **425** | **1.96** | **$-48.98\%$** |
| **Japanese** | 303 | 302 | 229 | **229** | **1.32** | **$-24.17\%$** |
| **Arabic** | 430 | 430 | 358 | **358** | **1.20** | **$-16.74\%$** |
| **German** | 315 | 136 | 116 | **116** | **2.72** | **$-14.71\%$** |
| **Spanish** | 316 | 112 | 97 | **97** | **3.26** | **$-13.39\%$** |
| **French** | 370 | 122 | 106 | **106** | **3.49** | **$-13.11\%$** |
| **Chinese (Simplified)**| 270 | 268 | 252 | **252** | **1.07** | **$-5.97\%$** |

---

## 5. Regression Gate Approval

The production tokenizer (`bpe_32k_production`, fingerprint `3559c7435e1d9330`) demonstrates **100% equivalence** with the candidate while cleanly incorporating atomic tool tokens.

**Regression Status: PASSED WITHOUT CAVEATS.** Tokenizer is officially cleared for corpus sharding.

