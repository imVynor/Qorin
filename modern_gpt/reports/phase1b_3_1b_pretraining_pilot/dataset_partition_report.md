# ModernGPT-S 500M — Phase 1B.3: Dataset Partition and Validation Separation Report

## 1. Partition Strategy and Isolation Principles

ModernGPT-S 500M enforces strict, cryptographically verified separation between the training corpus and the held-out validation corpus.

### 1.1 Strict Isolation Rules
1. **Pre-Generation Split Isolation**: The validation partition is processed and frozen *prior* to training shard generation.
2. **Cryptographic Deduplication Boundary**: Every validation document's text is hashed (SHA-256) into a quarantine filter. Any training candidate document matching a validation hash is strictly discarded.
3. **No Cross-Partition Sequences**: Validation sequences are packed into dedicated validation shards (`pilot_g3_val_shard_*.bin`) and never co-located with training sequences.
4. **Separate Accounting**: Validation tokens are tracked as an independent evaluation budget and are never counted toward the pretraining token totals.

---

## 2. Validation Partition Inventory

The held-out validation partition is sourced from the frozen held-out validation split (`modern_gpt/artifacts/corpus_100m/val_documents.jsonl`), preserving the identical validation distribution used in baseline evaluation.

| Domain | Source Dataset | Document Count | Raw Size (Bytes) | Filtered Clean Docs | Estimated Tokens |
|---|---|---:|---:|---:|---:|
| **Quality Web** | FineWeb-Edu | 6,459 | 33,219,842 | 6,381 | 6,950,000 |
| **Educational** | Cosmopedia-v2 | 2,637 | 13,842,910 | 2,610 | 2,890,000 |
| **Source Code** | CodeParrot-Clean | 252 | 969,591 | 248 | 248,000 |
| **Total Validation** | — | **9,348** | **48,032,343** | **9,239** | **~10,088,000** |

- **Validation Shard**: `modern_gpt/data_500m/pilot_shards/g3_1b/pilot_g3_val_shard_00000.bin`
- **Sequences**: 4,219 sequences (sequence length 2,048)
- **Validation Tokens**: 8,640,512 tokens
- **File Size**: 17,281,104 bytes
- **Payload Bytes**: 17,281,024 bytes (exactly $2 \times 8,640,512$)

---

## 3. Contamination and Leakage Verification

An automated cross-partition audit checked 208,671 training candidate documents against all 9,348 validation documents:
- **Validation Hashes Indexed**: 9,348 unique SHA-256 text digests.
- **Training Candidate Documents Evaluated**: 208,671 documents across `train_documents.jsonl` and `new_train_documents.jsonl`.
- **Exact Hash Collisions**: 0 (Zero).
- **Near-Duplicate Cross-Contamination**: 0 detected above 0.80 Jaccard threshold.
- **Leakage Status**: **CLEAN (0.00% Contamination)**.

---

## 4. Preservation of Source Provenance

Every validation record retains:
- `document_id`: Original source-assigned identifier.
- `source_dataset`: Upstream repository (`fineweb-edu`, `cosmopedia-v2`, `codeparrot-clean`).
- `license_id`: Explicit license identifier (ODC-By 1.0, Apache-2.0, MIT).
- `license_category`: Permissive vs Attribution Required.
- `domain`: Domain categorization for held-out evaluation perplexity tracking.

