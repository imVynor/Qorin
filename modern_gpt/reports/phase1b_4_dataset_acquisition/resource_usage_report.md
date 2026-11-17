# ModernGPT-S 500M — Phase 1B.4: Resource Usage and Infrastructure Report

## Executive Summary

Phase 1B.4 monitored system resources, network bandwidth, memory consumption, and disk space throughout dataset acquisition and tokenization. The streaming architecture preserved bounded memory usage and eliminated unnecessary full-dataset disk downloads.

---

## 1. Network Bandwidth & Streaming Performance

| Domain Source | Remote Endpoint | Raw Volume Downloaded | Streaming Time (s) | Effective Bandwidth | Ingestion Speed |
|---|---|---:|---:|---:|---:|
| **Quality Web** | `HuggingFaceFW/fineweb-edu` | 6.61 MB | 29.92s | 215.6 KB/s | 40.1 docs/s |
| **Educational** | `HuggingFaceTB/smollm-corpus` | 4.20 MB | 22.69s | 180.6 KB/s | 44.1 docs/s |
| **Source Code** | `codeparrot/codeparrot-clean-valid` | 16.76 MB | 12.87s | 1,271.9 KB/s | 77.7 docs/s |
| **Wikipedia** | `wikimedia/wikipedia` | 23.28 MB | 29.68s | 766.2 KB/s | 33.7 docs/s |
| **Mathematics** | `open-web-math/open-web-math` | 7.79 MB | 28.95s | 263.0 KB/s | 34.5 docs/s |
| **Books / Prose** | `storytracer/LoC-PD-Books` | 12.45 MB | 45.13s | 269.5 KB/s | 1.8 books/s |
| **Dialogue** | `HuggingFaceH4/ultrachat_200k` | 6.16 MB | 20.77s | 289.5 KB/s | 48.1 docs/s |
| **Overall Engine** | **7 Concurrent Source APIs** | **77.26 MB** | **190.39s (~3.2 min)** | **396.6 KB/s** | **33.0 docs/s** |

---

## 2. Host Compute & Memory Footprint

- **Peak Process Memory (RAM)**: **725 MB**.
  - Bounded memory was achieved by streaming row groups iteratively and serializing accepted records directly to JSONL on disk rather than buffering millions of records in memory.
- **CPU Utilization**: 25%–35% across a single host core during regex heuristic filtering and Rust BPE tokenization.
- **Disk Space Preflight**:
  - Initial Free Space on Drive C: 308.1 GB
  - Total Disk Footprint of Acquired Data: 60,238,187 bytes (~60.24 MB)
  - Post-Acquisition Free Space: 308.04 GB
  - Disk Space Utilization: $<0.02\%$ of available storage capacity.

---

## 3. Extrapolation & Bandwidth Feasibility for Full 1B Dataset

Based on the measured average bandwidth of **396.6 KB/s** from unauthenticated Hugging Face Hub mirrors:
- Total raw text required for remaining 757.7M tokens: $\sim 3.03\text{ GB}$ (at 4.0 chars/token).
- Estimated download time at measured anonymous rate:
  $$T = \frac{3,030,000\text{ KB}}{396.6\text{ KB/s}} \approx 7,640\text{ seconds} \approx \mathbf{2.12\text{ hours}}$$
- *Recommendation*: Authenticating with a dedicated `HF_TOKEN` unlocks high-speed multi-threaded CloudFront CDN endpoints (increasing throughput to $>25\text{ MB/s}$), reducing the complete 3 GB download to under 3 minutes.

