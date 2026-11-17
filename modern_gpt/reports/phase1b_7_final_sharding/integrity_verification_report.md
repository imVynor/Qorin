# Phase 1B.7: Independent Shard Integrity & Verification Results

## 1. Summary of 18 Verification Invariants

| Test Index | Test Name | Target Invariant | Result | Pass Rate |
|:---:|---|---|:---:|:---:|
| **01** | Tokenizer Identity | Full SHA-256 `d10624a9069ffac5...` | **PASS** | 100.0% |
| **02** | Token ID Range | All IDs $\in [0, 32767]$ | **PASS** | 100.0% |
| **03** | JSONL Record Integrity | Mandatory metadata fields present | **PASS** | 100.0% |
| **04** | Document Uniqueness | Zero cross-shard duplication | **PASS** | 100.0% |
| **05** | Train/Val Leakage | Zero contamination against 25,137 hashes | **PASS** | 100.0% |
| **06** | Domain Allocations | All 7 domain targets exact | **PASS** | 100.0% |
| **07** | Total Token Counts | 5,000,000,000 Train / 20,000,000 Val | **PASS** | 100.0% |
| **08** | Binary 80-Byte Header | Magic `b"MGPT"`, Version 1, Dtype 1 | **PASS** | 100.0% |
| **09** | Payload SHA-256 | Bitwise match for all 108 shards | **PASS** | 100.0% |
| **10** | Token Data Type | uint16 little-endian (2 bytes/token) | **PASS** | 100.0% |
| **11** | Sequence Length | Context length $L = 2048$ tokens | **PASS** | 100.0% |
| **12** | Next-Token Shift | $x[:, 1:] == y[:, :-1]$ | **PASS** | 100.0% |
| **13** | EOT Boundaries | Delimiter 9 preserves document boundaries | **PASS** | 100.0% |
| **14** | DataLoader 1,000 Batches | Zero crashes, empty batches, or errors | **PASS** | 100.0% |
| **15** | Shard Transition | Zero gaps or lost tokens across boundaries | **PASS** | 100.0% |
| **16** | State Checkpoint / Resume | Bitwise identical resumption from state | **PASS** | 100.0% |
| **17** | Multi-Worker Partitioning | Disjoint shard slices per worker | **PASS** | 100.0% |
| **18** | Manifest Verification | Independent re-computation matches manifests | **PASS** | 100.0% |
