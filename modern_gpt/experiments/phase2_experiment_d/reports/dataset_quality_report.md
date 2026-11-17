# Dataset Quality and Contamination Audit Report: QORIN Experiment D

**Audit Timestamp**: 2026-11-17T02:55:00+05:30  
**Target Dataset**: QORIN Experiment D Stage 1 SFT Dataset  
**Parent Checkpoint**: `ckpt_experiment_c_final.pt` (`6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`)  
**Production Tokenizer**: `bpe_32k_production` (`d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`)  

---

## 1. Executive Summary

A comprehensive quality, schema validity, and contamination audit was conducted on the newly constructed Experiment D dataset across 4,000 total conversation trajectories (3,800 train, 200 validation).

Key Findings:
1. **Zero Contamination**: **0** prompts overlap with the 50 protected held-out evaluation tasks.
2. **Zero Train/Val Leakage**: **0** prompt overlap between `train.jsonl` and `val.jsonl`.
3. **100% Schema & Syntax Validity**: All tool calls conform strictly to canonical schema delimiters; 100% of Python code blocks pass Python AST parsing (`ast.parse`).
4. **Balanced Multi-Tool Distribution**: The severe **28.72:1** Python-to-Web imbalance from Experiment C has been remediated to **0.63:1** (1,000 Python calls to 1,578 Web-involved calls), providing rich exposure to web retrieval across colloquial, question-based, fact-verification, and multi-step scenarios.

---

## 2. Dataset Cryptographic Fingerprints & Checksums

| File | Records | Total Tokens | Assistant Tokens | SHA-256 Checksum |
|---|---|---|---|---|
| `train.jsonl` | 3,800 | 1,455,160 | 344,087 | `90aebbac462bbad01e3239a51ef4b2c33f9d49faf5d23172060d89ef0d66cd2a` |
| `val.jsonl` | 200 | 75,637 | 17,912 | `b7c10807fda6ac5553b1806afe2fb3e25df65949dfd256d13456bf4e3a1af8e6` |
| `dataset_manifest.json` | 1 | - | - | `df85327ed2acbe3066486f000405b67c01f479744d50182a67d39750df21be0c` |

---

## 3. Category & Action Breakdown

### Training Split (3,800 records)
```
Category Breakdown:
├── web_colloquial:                365
├── web_question:                  365
├── web_explicit:                  210
├── web_verification:              200
├── python_core:                   665
├── direct_conceptual:             665
├── selection_discrimination:      475
├── multistep_web_python:          475
└── error_recovery_web:            380

Action Distribution:
├── WEB:                          1,615 (42.50%)
├── PYTHON:                         895 (23.55%)
├── DIRECT:                         815 (21.45%)
└── MULTI_TOOL:                     475 (12.50%)
```

### Validation Split (200 records)
```
Category Breakdown:
├── web_colloquial:                 20
├── web_question:                   20
├── web_explicit:                   10
├── web_verification:               10
├── python_core:                    35
├── direct_conceptual:              35
├── selection_discrimination:       25
├── multistep_web_python:           25
└── error_recovery_web:             20

Action Distribution:
├── WEB:                             85 (42.50%)
├── PYTHON:                          47 (23.50%)
├── DIRECT:                          43 (21.50%)
└── MULTI_TOOL:                      25 (12.50%)
```

---

## 4. Contamination and Leakage Verification

| Verification Dimension | Tested Count | Allowed Max | Measured Overlap | Status |
|---|---|---|---|---|
| **Protected Evaluation Contamination** | 50 prompts | 0 | **0 (0.0%)** | **PASSED** |
| **Train/Validation Split Leakage** | 200 prompts | 0 | **0 (0.0%)** | **PASSED** |
| **Duplicate Prompts in Train** | 3,800 records | 0 | **0 (0.0%)** | **PASSED** |
| **Duplicate Prompts in Val** | 200 records | 0 | **0 (0.0%)** | **PASSED** |
| **Tool Call Schema Conformity** | 4,000 records | 0 failures | **0 failures** | **PASSED** |
| **Python AST Syntax Validation** | 1,460 code snippets | 0 failures | **0 failures** | **PASSED** |

---

## 5. Token Statistics

- **Total Serialized Conversation Tokens (Train)**: 1,455,160 tokens
- **Total Supervised Assistant Tokens (Train)**: 344,087 tokens (loss-masked)
- **Mean Conversation Length**: 382.9 tokens (Max: 672)
- **Mean Assistant Target Length**: 90.5 tokens
