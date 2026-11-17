# ModernGPT-S 500M — Phase 1B.4: Reproducibility & Provenance Integrity Report

## Executive Summary

Phase 1B.4 ensures complete, auditable reproducibility for all acquired dataset files. Every acquired file in `modern_gpt/data_500m/acquired/` is pinned to explicit upstream repository versions, stable document identifiers, and immutable SHA-256 digests.

---

## 1. Upstream Dataset Pinned Revisions & Fingerprints

| Domain | Source Repository | Snapshot Revision / Date | Split Name | Upstream Data Format |
|---|---|---|:---:|---|
| **Quality Web** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` (Commit `2a88478`) | `train` | Parquet RowGroups |
| **Educational** | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` (Commit `3ba9d60`) | `train` | Parquet RowGroups |
| **Source Code** | `codeparrot/codeparrot-clean-valid`| Default (Commit `b57cf6b`) | `train` | Parquet RowGroups |
| **Wikipedia** | `wikimedia/wikipedia` | `20231101.en` (Commit `f2e7f3c`) | `train` | Parquet RowGroups |
| **Mathematics** | `open-web-math/open-web-math` | Default (Commit `fde8ef8`) | `train` | Parquet RowGroups |
| **Books / Prose** | `storytracer/LoC-PD-Books` | Default (Commit `2ec1a1e`) | `train` | Parquet RowGroups |
| **Dialogue** | `HuggingFaceH4/ultrachat_200k` | Default (Commit `8049631`) | `train_sft` | Parquet RowGroups |

---

## 2. Acquired Data Cryptographic Artifact Hashes

All newly acquired data files reside in `modern_gpt/data_500m/acquired/`:

| File Name | Domain | Documents | Size (Bytes) | Full SHA-256 Digest |
|---|---|---:|---:|---|
| `acquired_web.jsonl` | Quality Web | 1,200 | 6,636,181 | `39bfab4d174828a55151e8a00d5abd3cb2d2b0bae44ee18942ff3ec374d0befb` |
| `acquired_edu.jsonl` | Educational | 1,000 | 4,283,934 | `289413bdaa131feb1991753940623eb7af45e972a7d73d01227a7daf4b1c0640` |
| `acquired_code.jsonl` | Source Code | 1,000 | 6,453,444 | `4d829bd4597b384d2990dabdd78e1c1619e20f94ca5a28f62941a881db873126` |
| `acquired_wiki.jsonl` | Wikipedia | 1,000 | 23,589,959 | `ac562f7b0d7ae8a33bb8e9897ecc104e7bcc85810339db4bc313dcd0f4b0fb0f` |
| `acquired_math.jsonl` | Mathematics | 1,000 | 6,781,000 | `f2a86e1b479f4f843844cbc2e612fef76d12b82250ff22cb654454830ac2d08f` |
| `acquired_books.jsonl` | Books / Prose | 80 | 6,276,070 | `43c755c59a92c2face2b09f0cb49a1655d00461c569cf82a69ac71a18182ed56` |
| `acquired_dialogue.jsonl` | Dialogue | 1,000 | 6,217,599 | `888cbd0f1f4554b0cb20a583976aa8126d5dfe2afce81f060aa4b65b22ffa8f2` |
| `acquisition_master_summary.json` | Master Manifest | — | 8,689 | `4c8d500472479fca1f3b35eb65c62df5a39626e254ff99e289bf502b48eefb22` |

---

## 3. Production Tokenizer Verification

- **Artifact Path**: `modern_gpt/artifacts/tokenizer_32k/tokenizer.json`
- **Short Fingerprint**: `3559c7435e1d9330`
- **Full SHA-256 Digest**: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`
- **Vocabulary Size**: Exactly 32,768 entries.
- **Immutability Status**: **CONFIRMED UNCHANGED**.

