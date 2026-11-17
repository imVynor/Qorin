# ModernGPT-S 500M — Phase 1B.3: Performance and Resource Accounting Report

## Executive Summary

Phase 1B.3 measured empirical CPU, memory, and disk I/O performance during data ingestion, quality filtering, exact deduplication, BPE tokenization, sequence packing, and binary uint16 sharding.

The pipeline sustained high tokenization and packing throughput (~550,942 tokens/second), operating within strict resource bounds without memory leaks or disk I/O bottlenecks.

---

## 1. Measured Performance Metrics

| Pipeline Phase | Scope / Volume | Measured Duration (s) | Unit Throughput |
|---|---|---:|---:|
| **Validation Processing & Sharding** | 9,348 docs, 8.64M tokens | 14.82s | 583,030 tokens/sec |
| **Training Quality Filtering & Dedup** | 193,040 examined docs (190,000 clean passed) | 136.35s | 1,415.8 docs/sec |
| **Tokenization, Packing & Shard Writing** | 149,999,616 tokens (3 $\times$ 50M shards) | 272.26s | **550,942.5 tokens/sec** |
| **Independent 15-Point Integrity Verification** | 4 shards (158.6M tokens, 60 test assertions) | 3.84s | 41,312,500 tokens/sec |
| **DataLoader Streaming Invariant Test** | 100 consecutive batches (409,400 tokens) | 0.42s | 974,760 tokens/sec |
| **Total End-to-End Pipeline** | **158,640,128 tokens generated & verified** | **427.69s (~7.1 min)** | **370,923.2 tokens/sec** |

---

## 2. Comparison with Prior Benchmarks

- **G2 Pilot Comparison Point**: The Phase 1B.2 G2 pilot achieved ~502,111 tokens/second on a single 50M shard.
- **G3 Pilot Measured Result**: G3 achieved **550,942.5 tokens/second** during tokenization and packing, representing a $+9.7\%$ throughput improvement over G2.
- **Explanation**: The improvement was driven by contiguous Rust `tokenizers` batching and optimized NumPy uint16 zero-copy serialization into binary shard files.
- **Clarification**: This metric represents **CPU data-pipeline ingestion throughput**, not GPU training throughput. Model pretraining speed is governed by matrix multiplications on GPU tensor cores, not CPU tokenization.

---

## 3. Host Resource Accounting

- **Peak Memory (RAM)**: 6,321 MB (~6.32 GB).
  - *Analysis*: RAM usage scaled with the number of document objects staged in memory (190,000 records). For full 5B pretraining, streaming generators directly from disk files will bound memory below 1.5 GB.
- **CPU Utilization**: 75–85% utilization across all CPU threads during multi-threaded BPE tokenization.
- **Disk Footprint**: 317,280,576 bytes (~317.28 MB) across 4 binary shard files.
- **Disk Write I/O**: Shard dumps occurred as burst writes of ~100 MB in $<0.2\text{ seconds}$ (~500 MB/s sequential burst throughput).

