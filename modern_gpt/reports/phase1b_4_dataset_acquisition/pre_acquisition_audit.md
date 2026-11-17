# ModernGPT-S 500M — Phase 1B.4: Pre-Acquisition Audit & Supply Reconciliation

## Executive Summary

Phase 1B.4 audits the pretraining corpus inventory following Phase 1B.3. While Phase 1B.3 fully validated the production data pipeline (`modern_gpt/data_500m/`), shard generation halted at 149,999,616 training tokens (3 shards of ~50M tokens each) to uphold strict single-epoch rules and prevent artificial document repetition.

This audit reconciles locally available unique tokens against the approved seven-domain 1B-token mixture, identifies exhausted versus partially staged sources, and establishes exact net acquisition requirements per domain.

---

## 1. Reconciliation of Phase 1B.3 Manifest & Local Inventory

The Phase 1B.3 manifest (`modern_gpt/reports/phase1b_3_1b_pretraining_pilot/phase1b_3_manifest.json`) documents:
- **Generated Training Tokens**: 149,999,616 tokens (73,242 sequences of 2,048) across 3 binary shards (`pilot_g3_train_shard_00000.bin` through `00002.bin`).
- **Generated Validation Tokens**: 8,640,512 tokens (4,219 sequences) in `pilot_g3_val_shard_00000.bin`.
- **Total Local Storage Footprint**: 317,280,576 bytes (~317.28 MB).

### Local Document Source Reconciliation:
1. `modern_gpt/artifacts/corpus_100m/train_documents.jsonl`: 84,144 documents (~92M tokens) — **100% Ingested into G3**.
2. `modern_gpt/artifacts/shards_250m/new_train_documents.jsonl`: 124,527 documents (~134M tokens) — **105,856 documents ingested into G3 (18,671 remaining unconsumed)**.
3. `modern_gpt/artifacts/corpus_100m/val_documents.jsonl`: 9,348 documents (~10.1M raw tokens) — **Dedicated to validation shard; strictly quarantined from training**.
4. Curated SFT Pools (`modern_gpt/data/sft_4k_v4/raw/`): ~3.1M tokens total across Math, Wiki, and Dialogue — **Insufficient for large-scale pretraining**.

---

## 2. Domain-by-Domain Supply Status & Net Shortfall

To generate the full 1,000,000,000-token pretraining mixture (plus a 20,000,000-token reserved validation partition) without repeating documents, the required clean unique tokens are detailed below:

| Domain | 1B Target (Train) | Reserved Val Target | Total Target | Available Local Unique | Realized in G3 | Net Shortfall Required | Source Status |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **Quality Web** | 400,000,000 | 8,000,000 | 408,000,000 | 162,210,616 | 134,217,728 | **273,782,272** | Partially Staged |
| **Educational** | 150,000,000 | 3,000,000 | 153,000,000 | 51,223,479 | 48,154,624 | **104,845,376** | Partially Staged |
| **Source Code** | 150,000,000 | 3,000,000 | 153,000,000 | 13,018,692 | 13,018,692 | **139,981,308** | Exhausted locally |
| **Wikipedia / Reference** | 75,000,000 | 1,500,000 | 76,500,000 | 406,385 | 406,385 | **76,093,615** | Absent / Exhausted |
| **Mathematics / Technical** | 75,000,000 | 1,500,000 | 76,500,000 | 358,188 | 358,188 | **76,141,812** | Absent / Exhausted |
| **Books / Prose** | 100,000,000 | 2,000,000 | 102,000,000 | 0 | 0 | **102,000,000** | Absent (0 bytes on disk) |
| **Curated Dialogue** | 50,000,000 | 1,000,000 | 51,000,000 | 1,391,761 | 1,391,761 | **49,608,239** | Absent / Exhausted |
| **Total Mixture** | **1,000,000,000** | **20,000,000** | **1,020,000,000** | **~228,609,121** | **158,640,128** | **822,452,622** | **Deficit: 80.6%** |

---

## 3. Filtering & Deduplication Yields

Empirical yield measurements from Phase 1B.3 data processing:
- **Quality Filtering Pass Rate**: $98.42\%$ (190,000 clean documents passed of 193,040 examined).
- **Exact Deduplication Loss**: $<1.58\%$ across homogeneous crawls.
- **Estimated Combined Pre-Tokenization Yield**: $\sim 95.0\%$.
- **Required Acquisition Margin**: $+15.0\%$ raw buffer over net token shortfall. For example, to secure 100M clean tokens for Books, the acquisition target is $\sim 115\text{M}$ raw tokens ($\sim 490\text{ MB}$ raw text).

---

## 4. Pre-Acquisition Audit Verdict

1. **Pipeline Immutability**: All existing assets (`modern_gpt/archive/100m_baseline/`, `tokenizer_32k/`, G1/G2/G3 shards, and SFT v4 assets) remain frozen and unmodified.
2. **Net Acquisition Mandate**: Sourcing must focus on the completely absent domains (Books, Wikipedia, OpenWebMath, Dialogue) and heavily deficient domains (Code, Edu, Web) from official open-source Hugging Face repositories.

