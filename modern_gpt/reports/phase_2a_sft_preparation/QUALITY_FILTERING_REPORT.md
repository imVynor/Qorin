# ModernGPT-S Phase 2A — Quality Filtering & Hygiene Report

---

## 1. Discard & Reconciliation Summary

During raw ingest of 280,007 candidate records, 41,603 records were quarantined or discarded across five primary rejection stages:

| Rejection Category | Records Quarantined | Share of Discards (%) | Reason & Filter Criterion |
|---|---:|---:|---|
| **Length Violations** | 18,420 | 44.28% | Sequence length > 2,000 tokens (truncation risk) or < 20 tokens |
| **Degenerate Repetition** | 9,845 | 23.66% | Distinct-2 < 0.35, 4-gram repetition loops, synthetic babble |
| **Evaluation Leakage** | 566 | 1.36% | N-gram match with held-out validation domains & evaluation probes |
| **Format & Delimiter Errors** | 7,120 | 17.11% | Broken ChatML tags, unmatched `<|tool_call|>`, missing system prefix |
| **Refusal & Safety Echoing** | 5,652 | 13.59% | "As an AI...", uninformative refusal responses without instruction |
| **Total Discarded** | **41,603** | **100.00%** | **Corpus sanitization rate: 14.86%** |

---

## 2. Split Stratification Methodology

The clean pool of 238,404 records was split using stratified random sampling with fixed seed `42`:
- **Train Split (95.0%):** 226,486 records
- **Validation Split (2.5%):** 5,959 records
- **Test Split (2.5%):** 5,959 records

The split preserves exact category proportions across all three sets with zero overlap.
