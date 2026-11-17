# ModernGPT-S — Phase 1B.1: SFT v3 Category Distribution Report

**Dataset Version:** SFT 4K v3  
**Status:** AUDITED & VERIFIED  
**Date:** 2026-11-09  
**Directory:** `modern_gpt/data/sft_4k_v3/processed/`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report provides an in-depth empirical breakdown of the content categories represented in the SFT v3 dataset across both the training and validation splits.

Crucially, this analysis demonstrates the severe divergence between **Record Count percentages** and **Supervised Target-Token percentages**, highlighting why training budgets and objective functions must be governed strictly by supervised tokens.

Key structural observations:
1. **Two Dominant Formats Make Up 56.5% of All Records**: `long_context_retrieval` (30.61%) and `science_commonsense` (25.88%) comprise 18,050 of the 31,949 training records.
2. **Multi-Turn Dialogue is Severely Token-Starved by Count**: While `multi_turn_dialogue` accounts for 17.28% of supervised target tokens, it constitutes only **2.96% of training records** (945 records).
3. **Total Absence of Safety & Policy Data**: SFT v3 contains **zero records (0.0%) and zero tokens (0.0%)** dedicated to safety, responsible refusal, crisis support, or cybersecurity education.

---

## 2. Training Split Category Distribution

The training split contains **31,949 records** and **3,362,374 supervised target tokens**.

| Category | Record Count | Record Share (%) | Supervised Target Tokens | Supervised Token Share (%) | Mean Sup Tokens / Record | Total Input Tokens | Input Token Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Conversation** | 5,162 | 16.16% | 900,010 | **26.77%** | 174.35 | 2,987,412 | 12.41% |
| **Multi-Turn Dialogue** | 945 | 2.96% | 580,874 | **17.28%** | 614.68 | 1,546,235 | 6.42% |
| **Long Context Retrieval** | 9,781 | 30.61% | 540,077 | **16.06%** | 55.22 | 13,874,210 | 57.61% |
| **Coding (Python)** | 2,990 | 9.36% | 448,499 | **13.34%** | 150.00 | 1,842,105 | 7.65% |
| **Science & Commonsense** | 8,269 | 25.88% | 360,037 | **10.71%** | 43.54 | 1,421,980 | 5.90% |
| **Mathematics & Reasoning**| 2,907 | 9.10% | 360,015 | **10.71%** | 123.84 | 1,289,321 | 5.35% |
| **Structured JSON** | 1,895 | 5.93% | 172,862 | **5.14%** | 91.22 | 1,120,440 | 4.65% |
| **Safety & Responsible** | **0** | **0.00%** | **0** | **0.00%** | **0.00** | **0** | **0.00%** |
| **Total** | **31,949** | **100.00%** | **3,362,374** | **100.00%** | **105.24** | **24,081,703** | **100.00%** |

---

## 3. Validation Split Category Distribution

The validation split contains **3,277 records** and **362,248 supervised target tokens** (representing a 9.7% stratified validation holdout).

| Category | Record Count | Record Share (%) | Supervised Target Tokens | Supervised Token Share (%) | Mean Sup Tokens / Record | Total Input Tokens | Input Token Share (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **General Conversation** | 525 | 16.02% | 90,852 | **25.08%** | 173.05 | 301,420 | 12.33% |
| **Multi-Turn Dialogue** | 118 | 3.60% | 73,021 | **20.16%** | 618.82 | 196,512 | 8.04% |
| **Coding (Python)** | 343 | 10.47% | 54,236 | **14.97%** | 158.12 | 210,480 | 8.61% |
| **Long Context Retrieval** | 971 | 29.63% | 54,011 | **14.91%** | 55.62 | 1,378,940 | 56.43% |
| **Science & Commonsense** | 826 | 25.21% | 36,036 | **10.95%** | 43.63 | 142,310 | 5.82% |
| **Mathematics & Reasoning**| 294 | 8.97% | 36,013 | **10.94%** | 122.49 | 131,240 | 5.37% |
| **Structured JSON** | 200 | 6.10% | 18,079 | **4.99%** | 90.40 | 82,744 | 3.39% |
| **Safety & Responsible** | **0** | **0.00%** | **0** | **0.00%** | **0.00** | **0** | **0.00%** |
| **Total** | **3,277** | **100.00%** | **362,248** | **100.00%** | **110.54** | **2,443,646** | **100.00%** |

---

## 4. Record Percentage vs. Supervised Token Share Divergence

The data reveals two massive distortions when comparing record share to supervised target-token share:

1. **The Multiple-Choice Dilution Effect (`science_commonsense`)**:
   - `science_commonsense` represents **25.88% of all records** in the training set (8,269 records).
   - However, because each answer is an ultra-concise choice explanation (mean 43.54 tokens), it accounts for only **10.71% of supervised tokens**.
   - If an unweighted microbatch objective is used (as was done prior to Phase 1A), these 8,269 records exert an outsized $2.5\times$ gradient impact relative to their actual information content.

2. **The Retrieval Context Bloat Effect (`long_context_retrieval`)**:
   - `long_context_retrieval` represents **30.61% of all records** and a staggering **57.61% of all context window input tokens** (13.87 million tokens).
   - Yet the actual assistant target output represents only **16.06% of supervised tokens** (mean 55.22 tokens).
   - More than half of the GPU compute budget during forward passes was spent attending to massive context documents, training the model primarily to ingest documents rather than generate fluid conversational text.

3. **The Multi-Turn Dialogue Scarcity (`multi_turn_dialogue`)**:
   - Despite accounting for **17.28% of supervised tokens**, `multi_turn_dialogue` contains only **945 unique conversations** across the entire training corpus.
   - The model saw only 945 multi-turn interaction opportunities across 32,000 training steps, severely limiting its exposure to turn transitions, user corrections, and conversational pivot dynamics.

