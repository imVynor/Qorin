# Phase 1B.5: Validation Corpus Isolation and Quarantine Report

## 1. Isolated Validation Allocation

A dedicated validation corpus of held-out documents was extracted and frozen under `modern_gpt/data_500m/val_isolated/`.

| Domain | Target Validation Tokens | Realized Tokens | Realized Documents | File Path | SHA-256 (first 16 hex) | Status |
|---|---:|---:|---:|---|---|:---:|
| **Quality Web** | 8,000,000 | 2,558,054 | 2,400 | `val_web.jsonl` | `deb1b7e562031c88` | PARTIAL_RESERVED |
| **Educational** | 3,000,000 | 1,444,037 | 2,000 | `val_edu.jsonl` | `2697d0a03e3b9f3c` | PARTIAL_RESERVED |
| **Source Code** | 3,000,000 | 3,015,673 | 1,864 | `val_code.jsonl` | `2fb634d765ee1aea` | **SUFFICIENT** |
| **Wikipedia** | 1,500,000 | 1,506,805 | 222 | `val_wiki.jsonl` | `078799d81a0c1eec` | **SUFFICIENT** |
| **Mathematics** | 1,500,000 | 1,500,827 | 847 | `val_math.jsonl` | `83e4cbfd41b09542` | **SUFFICIENT** |
| **Books / Prose** | 2,000,000 | 2,000,925 | 92 | `val_books.jsonl` | `e2d831bbd8048762` | **SUFFICIENT** |
| **Dialogue** | 1,000,000 | 1,000,441 | 847 | `val_dialogue.jsonl` | `3d3126d4a606957a` | **SUFFICIENT** |
| **Total** | **20,000,000** | **13,026,762** | **8,272** | `validation_manifest.json` | — | **13.0M Isolated** |

## 2. Quarantine Invariant & Leakage Prevention

- Total Active Quarantine Hashes: **17,620**.
- Every document placed into validation has its text SHA-256 registered in the permanent quarantine index.
- Future training ingestion and sharding steps will reject any candidate matching these hashes, guaranteeing 0.00% validation leakage.
