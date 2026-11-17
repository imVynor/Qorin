# ModernGPT-S 500M — Phase 1B.4: Dataset Acquisition & Download Log

## Executive Summary

This log records the streaming acquisition runs executed across the seven pretraining domains from official Hugging Face Hub repositories. All acquisition jobs were performed using streaming HTTP ingestion, bounded buffering, real-time SHA-256 hashing, and quarantine filtering.

---

## 1. Acquisition Execution Log

| Job ID | Domain | Upstream Dataset | Config / Split | Documents Ingested | Raw Download Size (Bytes) | Ingestion Elapsed Time | Bandwidth (KB/s) | Destination File | Checksum (SHA-256) |
|---|---|---|:---:|---:|---:|---:|---:|---|---|
| **ACQ-01** | **Quality Web** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` / `train` | 1,200 | 6,606,145 | 29.92s | 215.6 | `acquired_web.jsonl` | `39bfab4d174828a55151e8a00d5abd3cb2d2b0bae44ee18942ff3ec374d0befb` |
| **ACQ-02** | **Educational** | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` / `train` | 1,000 | 4,196,285 | 22.69s | 180.6 | `acquired_edu.jsonl` | `289413bdaa131feb1991753940623eb7af45e972a7d73d01227a7daf4b1c0640` |
| **ACQ-03** | **Source Code** | `codeparrot/codeparrot-clean-valid` | `default` / `train` | 1,000 | 16,763,350 | 12.87s | 1,271.9 | `acquired_code.jsonl` | `4d829bd4597b384d2990dabdd78e1c1619e20f94ca5a28f62941a881db873126` |
| **ACQ-04** | **Wikipedia** | `wikimedia/wikipedia` | `20231101.en` / `train` | 1,000 | 23,283,380 | 29.68s | 766.2 | `acquired_wiki.jsonl` | `ac562f7b0d7ae8a33bb8e9897ecc104e7bcc85810339db4bc313dcd0f4b0fb0f` |
| **ACQ-05** | **Mathematics** | `open-web-math/open-web-math` | `default` / `train` | 1,000 | 7,794,349 | 28.95s | 263.0 | `acquired_math.jsonl` | `f2a86e1b479f4f843844cbc2e612fef76d12b82250ff22cb654454830ac2d08f` |
| **ACQ-06** | **Books / Prose** | `storytracer/LoC-PD-Books` | `default` / `train` | 80 | 12,454,171 | 45.13s | 269.5 | `acquired_books.jsonl` | `43c755c59a92c2face2b09f0cb49a1655d00461c569cf82a69ac71a18182ed56` |
| **ACQ-07** | **Dialogue** | `HuggingFaceH4/ultrachat_200k` | `default` / `train_sft` | 1,000 | 6,158,092 | 20.77s | 289.5 | `acquired_dialogue.jsonl` | `888cbd0f1f4554b0cb20a583976aa8126d5dfe2afce81f060aa4b65b22ffa8f2` |
| **Total** | **All 7 Domains** | — | — | **6,280** | **77,255,772** | **190.39s (~3.2 min)** | **396.6** | — | — |

---

## 2. Ingestion Resilience & Error Handling

- **Connection Resets & Retries**: During the UltraChat acquisition stream, upstream CDN dropped a chunk boundary (`peer closed connection without sending complete message body`). The automatic backoff retry logic recovered after a 1.0s backoff and resumed iteration without data loss or corruption.
- **Unauthenticated Rate Limiting**: Hugging Face Hub issued standard unauthenticated warnings (`HF_TOKEN not set`). Acquisition throttled requests to remain within the public anonymous quota.
- **Stream Interruption Safety**: Each domain's output was written atomically to disk with explicit flushing and UTF-8 encoding.

