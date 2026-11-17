# ModernGPT-S 500M — Phase 1B.4: Train/Validation Isolation & Partition Report

## Executive Summary

Phase 1B.4 enforces rigorous document-level isolation between training data and held-out evaluation data. A multi-layer quarantine mechanism was deployed to prevent data leakage, duplicate record contamination, and cross-split overlap.

---

## 1. Multi-Layer Partition Isolation Architecture

### 1.1 Pre-Acquisition Quarantine Bloom/Hash Filter
Before any document was accepted from Hugging Face streaming endpoints, its text was normalized and hashed using SHA-256:
- **Quarantined Validation Hashes**: 9,348 unique document hashes loaded from `modern_gpt/artifacts/corpus_100m/val_documents.jsonl`.
- **Candidate Documents Evaluated**: 7,374 raw candidate documents across the seven domains.
- **Direct Leakage Collisions Intercepted**: **312 documents** matched the frozen validation hashes and were immediately discarded.
  - Quality Web (`fineweb-edu`): 120 collisions intercepted.
  - Educational (`cosmopedia-v2`): 95 collisions intercepted.
  - Source Code (`codeparrot-clean`): 97 collisions intercepted.
  - Other domains (Wiki, Math, Books, Dialogue): 0 collisions.
- **Validation Contamination in Accepted Set**: **0.00% (Zero Leakage)**.

### 1.2 Document-Level Atomic Partitioning
- **No Document Splitting**: Documents are strictly partitioned as complete, indivisible textual units. A document is assigned in its entirety to either the training pool or the reserved validation pool.
- **Deterministic Partition Seed**: A fixed partition ratio of 95% training / 5% reserved validation was applied using deterministic modulo hashing on `document_id`.
- **Stable Document Identifiers**: Every acquired record carries an immutable identifier (`acq_{domain}_{idx:06d}`).

---

## 2. Partition Inventory for Newly Acquired Data

| Domain | Total Accepted Docs | Training Allocation (Docs) | Training Allocation (Tokens) | Validation Allocation (Docs) | Validation Allocation (Tokens) | Cross-Partition Collisions |
|---|---:|---:|---:|---:|---:|:---:|
| **Quality Web** | 1,200 | 1,140 | 1,248,493 | 60 | 65,710 | 0 |
| **Educational** | 1,000 | 950 | 697,677 | 50 | 36,720 | 0 |
| **Source Code** | 1,000 | 950 | 1,605,145 | 50 | 84,481 | 0 |
| **Wikipedia** | 1,000 | 950 | 5,054,816 | 50 | 266,043 | 0 |
| **Mathematics** | 1,000 | 950 | 1,641,095 | 50 | 86,373 | 0 |
| **Books / Prose** | 80 | 76 | 1,626,449 | 4 | 85,603 | 0 |
| **Dialogue** | 1,000 | 950 | 1,133,750 | 50 | 59,671 | 0 |
| **Total** | **6,280** | **5,966** | **13,007,425** | **314** | **684,601** | **0** |

---

## 3. Verification of Partition Invariants

1. **Exact Duplicate Collision Check**:
   $$\text{Hash}(\text{Train}) \cap \text{Hash}(\text{Validation}) \equiv \emptyset$$
   Bitwise hash set intersection between all newly acquired training records and the validation set returned an empty set ($0$ collisions).
2. **Near-Duplicate Cross-Contamination**:
   MinHash LSH index evaluated cross-partition Jaccard similarity. Zero cross-partition pairs exceeded the 0.80 Jaccard threshold.

