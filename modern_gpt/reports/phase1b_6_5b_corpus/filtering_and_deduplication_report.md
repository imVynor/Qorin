# Phase 1B.6: Global Filtering & Deduplication Report

## 1. Deduplication Breakdown

- **Total Acquired Documents on Disk**: 4,143,836
- **Validation Quarantine Collisions Intercepted**: 918 exact collisions prevented across Web, Edu, and Code.
- **Quality Rejections**: 2,425 malformed or low-density documents dropped.
- **Intra-Session Duplicates Dropped**: 11,555 documents skipped via pre-indexed SHA-256 table.

## 2. Global Deduplication Invariant

All documents are uniquely identified by SHA-256 hash of their normalized text:
$$h = \text{SHA-256}(\text{strip}(text))$$
No document is admitted to training if its hash appears in:
1. Historical training documents.
2. Current acquired training documents.
3. Isolated validation documents.
