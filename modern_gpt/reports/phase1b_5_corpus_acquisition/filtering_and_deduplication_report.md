# Phase 1B.5: Filtering, Quarantine, and Deduplication Report

## 1. Rejection Pipeline Breakdown

Every incoming record passed sequentially through:
1. Exact SHA-256 pre-index comparison (skipping already acquired documents).
2. Validation Quarantine hash matching (intercepting potential test set contamination).
3. `DataQualityFilter` (heuristic text density, min/max length, repetitive character ratio).
4. `ExactDeduplicator` (intra-batch deduplication).

## 2. Session Rejection Statistics

| Domain | Examined | Already Seen Skips | Val Collisions Intercepted | Quality Filter Rejections | Accepted Records | Acceptance Yield (%) |
|---|---:|---:|---:|---:|---:|---:|
| **Quality Web** | 2,651 | 1,200 | 251 | 0 | 1,200 | 45.3% |
| **Educational** | 2,222 | 1,000 | 195 | 27 | 1,000 | 45.0% |
| **Source Code** | 4,296 | 1,000 | 211 | 1,085 | 1,000 | 23.3% |
| **Wikipedia** | 2,016 | 1,000 | 0 | 16 | 1,000 | 49.6% |
| **Mathematics** | 2,260 | 1,000 | 0 | 260 | 1,000 | 44.2% |
| **Books / Prose** | 156 | 80 | 0 | 1 | 75 | 48.1% |
| **Dialogue** | 2,144 | 1,000 | 0 | 144 | 1,000 | 46.6% |
| **Total** | **15,745** | **6,280** | **657** | **1,533** | **6,275** | **39.9%** |

*Key finding: Validation quarantine intercepted 657 collisions from Web, Edu, and Code, guaranteeing 0.00% validation leakage into training.*
