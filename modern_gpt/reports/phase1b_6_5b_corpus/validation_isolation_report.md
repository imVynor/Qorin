# Phase 1B.6: Validation Corpus Isolation & Quarantine Report

## 1. Validation Allocation Status

| Domain | Target Validation Tokens | Realized Validation Tokens | Documents | File Name | SHA-256 (first 16 hex) | Status |
|---|---:|---:|---:|---|---|:---:|
| **Quality Web** | 8,000,000 | 8,000,113 | 7,773 | `val_web.jsonl` | `219f3d6bdae3814a` | PARTIAL_RESERVED |
| **Educational** | 3,000,000 | 3,000,369 | 4,144 | `val_edu.jsonl` | `1e6463ce3d59da66` | PARTIAL_RESERVED |
| **Source Code** | 3,000,000 | 3,015,673 | 1,864 | `val_code.jsonl` | `2fb634d765ee1aea` | **SUFFICIENT** |
| **Wikipedia** | 1,500,000 | 1,506,805 | 222 | `val_wiki.jsonl` | `078799d81a0c1eec` | **SUFFICIENT** |
| **Mathematics** | 1,500,000 | 1,500,827 | 847 | `val_math.jsonl` | `83e4cbfd41b09542` | **SUFFICIENT** |
| **Books / Prose** | 2,000,000 | 2,000,925 | 92 | `val_books.jsonl` | `e2d831bbd8048762` | **SUFFICIENT** |
| **Dialogue** | 1,000,000 | 1,000,441 | 847 | `val_dialogue.jsonl` | `3d3126d4a606957a` | **SUFFICIENT** |
| **Total** | **20,000,000** | **20,025,153** | **15,789** | `validation_manifest.json` | — | **13.0M Isolated** |

- **Quarantine Hashes Active**: 25,137
- **Training Contamination Rate**: **0.00%**
