# Phase 1B.7: Pre-Packing Inventory & Invariant Audit

## 1. Underlying File Integrity & Checksums

Prior to shard packing, every acquired JSONL record file and baseline artifact was audited:

| Domain | File Path | File Size (Bytes) | Full SHA-256 Checksum | Invariant Status |
|---|---|---:|---|:---:|
| **Quality Web** | `data_500m/acquired/acquired_web.jsonl` | 9,436,993,425 | `805b51d54fa6ca1710072cee3da91787fb1fd48f292c0f9f014b88fb72843f02` | **VERIFIED** |
| **Educational** | `data_500m/acquired/acquired_edu.jsonl` | 4,101,560,378 | `ba6aeb707321e672b174778800b53956e3ca5d4aa4c0f3a28430c9a7161a8de4` | **VERIFIED** |
| **Source Code** | `data_500m/acquired/acquired_code.jsonl` | 2,855,209,312 | `efcb60836370924b0972364b24979fc817447153b6abf06ec197d8eb2f47526b` | **VERIFIED** |
| **Wikipedia** | `data_500m/acquired/acquired_wiki.jsonl` | 1,750,799,821 | `a2616cc763d5db34d19b8b7a9868049bc268db6ef33b57c51dec4cafad8fd9f3` | **VERIFIED** |
| **Mathematics** | `data_500m/acquired/acquired_math.jsonl` | 1,496,526,146 | `51aa38be91c67a3cda9b5d4cf33204d8b570ab787ba97a3913e783b12d66660f` | **VERIFIED** |
| **Books / Prose** | `data_500m/acquired/acquired_books.jsonl` | 1,800,075,759 | `bc9bb412f3774efce8f1df4155a3374b59225615fdc45ddf6c6675025870cbfe` | **VERIFIED** |
| **Dialogue** | `data_500m/acquired/acquired_dialogue.jsonl` | 1,331,068,306 | `55ace771387d8b996c82045e5782df731157339255631d4b2c5a782d1269feb2` | **VERIFIED** |

## 2. Invariant Verification

1. **Production Tokenizer**: Matches SHA-256 `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` and fingerprint `3559c7435e1d9330` with 0 variance.
2. **100M Baseline Archive**: `modern_gpt/archive/100m_baseline/` verified intact (47 files/directories untouched).
3. **Pilot Shards**: `modern_gpt/data_500m/pilot_shards/` (G1, G2, G3) preserved intact without modification.
4. **Validation Separation**: 25,137 validation document SHA-256 hashes loaded into active quarantine.
