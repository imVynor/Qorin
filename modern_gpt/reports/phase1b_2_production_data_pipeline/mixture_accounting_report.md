# ModernGPT-S 500M — Pretraining Mixture Accounting & Domain Allocation Report

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **ACCOUNTING AUDITED & PRODUCTION TARGETS ESTABLISHED**  
**Target Mixture:** 5.00 Billion Tokens across 7 Domains  

---

## 1. Executive Summary

This report establishes the production mixture accounting framework for ModernGPT-S 500M's 5.00-Billion-Token pretraining corpus.

To guarantee zero epoch repetition and prevent domain imbalance, the sampling engine draws from seven vetted permissive source streams according to fixed probability weights. Attrition rates observed during Pilot G1 and G2 have been incorporated into raw document requisition targets.

---

## 2. Production Domain Mixture Targets

| Domain ID | Domain Name | Target Token Share | Target Tokens | Observed Attrition Rate | Required Raw Tokens | Target Raw Documents (Avg 1k tok/doc) | Primary Permissive Source |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **D1** | **Quality Web Text** | $40.0\%$ | **2,000,000,000** | $1.7\%$ | $2,034,000,000$ | $\sim 2,034,000$ | FineWeb-Edu ($score \ge 4.0$) |
| **D2** | **Educational Material** | $15.0\%$ | **750,000,000** | $1.2\%$ | $759,000,000$ | $\sim 759,000$ | Cosmopedia v2 |
| **D3** | **Source Code** | $15.0\%$ | **750,000,000** | $4.8\%$ | $787,000,000$ | $\sim 787,000$ | The Stack v2 Permissive / CodeParrot |
| **D4** | **Wikipedia / Reference**| $7.5\%$ | **375,000,000** | $2.1\%$ | $383,000,000$ | $\sim 383,000$ | English Wikipedia 2024 (clean articles) |
| **D5** | **Math & Technical** | $7.5\%$ | **375,000,000** | $3.5\%$ | $388,500,000$ | $\sim 388,500$ | OpenWebMath / Proof-Pile-2 |
| **D6** | **Books & Prose** | $10.0\%$ | **500,000,000** | $5.2\%$ | $527,000,000$ | $\sim 5,200\text{ books}$ | Standard Ebooks / Project Gutenberg |
| **D7** | **Curated Dialogue** | $5.0\%$ | **250,000,000** | $2.8\%$ | $257,000,000$ | $\sim 320,000\text{ convs}$ | UltraChat 200k / OpenAssistant |
| **Total**| — | **100.0%** | **5,000,000,000** | **$\sim 2.5\%$ avg** | **$\sim 5,135,500,000$** | **$\sim 4.68\text{M records}$** | $100\%$ Permissive / Public Domain |

---

## 3. Seeded Stochastic Interleaving Architecture

To ensure homogeneous distribution across all binary shards:
- Documents are drawn from independent domain queues using a weighted multinomial sampler parameterized by the domain target shares:
  $$P(\text{domain}) = \{0.40, 0.15, 0.15, 0.075, 0.075, 0.10, 0.05\}$$
- **RNG Seed:** Fixed seed $42$.
- Every binary shard reflects the identical global domain mixture within $\pm 0.5\%$ variance, preventing late-stage training shock (e.g. encountering code only at the end of pretraining).
- **Zero Document Reuse:** The sampler maintains a persistent set of ingested document IDs, strictly preventing duplicate exposure.

