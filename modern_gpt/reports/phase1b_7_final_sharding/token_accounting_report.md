# Phase 1B.7: Domain Token Accounting & Exact Allocation Reconciliation

## 1. Pretraining Corpus Allocation (5,000,000,000 Tokens)

| Domain | Target Share | Target Tokens | Packed Tokens | Shards Count | Docs Used | Variance |
|---|---:|---:|---:|---:|---:|:---:|
| **Quality Web** | 40.0% | 2,000,000,000 | 2,000,000,000 | 40 | 1,956,428 | **0 (Exact)** |
| **Educational** | 15.0% | 750,000,000 | 750,000,000 | 15 | 1,032,334 | **0 (Exact)** |
| **Source Code** | 15.0% | 750,000,000 | 750,000,000 | 15 | 436,289 | **0 (Exact)** |
| **Wikipedia** | 7.5% | 375,000,000 | 375,000,000 | 8 | 426,787 | **0 (Exact)** |
| **Mathematics** | 7.5% | 375,000,000 | 375,000,000 | 8 | 216,242 | **0 (Exact)** |
| **Books / Prose** | 10.0% | 500,000,000 | 500,000,000 | 10 | 24,554 | **0 (Exact)** |
| **Curated Dialogue** | 5.0% | 250,000,000 | 250,000,000 | 5 | 264,187 | **0 (Exact)** |
| **TOTAL TRAINING** | **100.0%** | **5,000,000,000** | **5,000,000,000** | **101** | — | **0 (Exact)** |

## 2. Validation Corpus Allocation (20,000,000 Tokens)

| Domain | Target Tokens | Realized Tokens | Shards Count | Docs Used | Variance |
|---|---:|---:|---:|---:|:---:|
| **Quality Web** | 8,000,000 | 8,000,000 | 1 | 7,766 | **0 (Exact)** |
| **Educational** | 3,000,000 | 3,000,000 | 1 | 4,138 | **0 (Exact)** |
| **Source Code** | 3,000,000 | 3,000,000 | 1 | 1,863 | **0 (Exact)** |
| **Wikipedia** | 1,500,000 | 1,500,000 | 1 | 222 | **0 (Exact)** |
| **Mathematics** | 1,500,000 | 1,500,000 | 1 | 847 | **0 (Exact)** |
| **Books / Prose** | 2,000,000 | 2,000,000 | 1 | 92 | **0 (Exact)** |
| **Curated Dialogue** | 1,000,000 | 1,000,000 | 1 | 847 | **0 (Exact)** |
| **TOTAL VALIDATION** | **20,000,000** | **20,000,000** | **7** | — | **0 (Exact)** |
