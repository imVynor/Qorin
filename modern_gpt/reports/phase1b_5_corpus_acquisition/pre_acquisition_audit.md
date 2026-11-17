# Phase 1B.5: Pre-Acquisition Preflight Audit

## 1. Environment & Baseline Audit

Before executing Phase 1B.5, all existing project assets, archives, and dependencies were inspected:

- **100M Baseline Archive**: Verified untouched at `modern_gpt/archive/100m_baseline/`.
- **Production Tokenizer**: `modern_gpt/artifacts/tokenizer_32k/tokenizer.json` verified with SHA-256 `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`. Fingerprint verified as `3559c7435e1d9330`.
- **Python Virtualenv**: `.venv\Scripts\python.exe` running Python 3.12.13 on Windows x64.
- **Available Disk Space**: 285.93 GB free on drive C: (plenty of overhead for the ~3.7 GB raw text required for full 1B token pretraining).

## 2. Supply Accounting Preflight

Historical clean supplies from 100M and 250M archives (`modern_gpt/artifacts/corpus_100m/train_documents.jsonl` and `new_train_documents.jsonl`):
- Quality Web: 162,210,616 tokens
- Educational: 51,223,479 tokens
- Source Code: 13,018,692 tokens
- Wikipedia: 406,385 tokens
- Mathematics: 358,188 tokens
- Books / Prose: 0 tokens
- Curated Dialogue: 1,391,761 tokens
- **Total Historical Clean**: 228,609,121 tokens.

## 3. Quarantine Filter Preflight

- 9,348 validation document MD5/SHA-256 hashes loaded from `modern_gpt/artifacts/corpus_100m/val_documents.jsonl`.
- Zero-tolerance protocol established: any candidate document matching an existing validation hash is immediately dropped and recorded.
