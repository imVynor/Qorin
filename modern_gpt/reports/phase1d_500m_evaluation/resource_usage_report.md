# ModernGPT-S 500M — Phase 1D: Hardware Resource & Profile Report

**Evaluation Date:** November 13, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)  
**Host Environment:** Windows 11, Python 3.12, PyTorch 2.16.0.dev20261022+cu134, CUDA 13.4  
**Total Evaluation Wall-Clock Time:** 689.85 seconds (~11.5 minutes)  

---

## 1. Executive Summary

This report documents the computational, thermal, and memory profile of the ModernGPT-S 500M model during the complete Phase 1D evaluation suite.

Thanks to modern architectural optimizations—specifically **Grouped-Query Attention (GQA 6:1)**, **bias-free linear projections**, and **bfloat16 precision**—the 514M parameter model operated with exceptional memory efficiency and zero thermal throttling, consuming only **1.16 GiB of GPU VRAM** during peak inference.

---

## 2. Memory Footprint Profile

| Component | ModernGPT-S 100M (FP32/MHA) | ModernGPT-S 500M (BF16/GQA) | Delta / Efficiency Gain |
|---|---|---|---|
| **Model Weights (Inference)** | ~420 MB (FP32) | **1,028.7 MB (BF16)** | Compact representation |
| **Peak Allocated VRAM** | 580 MB | **1,164.2 MB** | Leaves >6.7 GiB headroom |
| **Reserved VRAM (PyTorch)** | 850 MB | **1,524.0 MB** | Extremely stable caching |
| **KV Cache per 2,048 Sequence** | 96 MB (if 2048 ctx) | **16.0 MB** | **6x smaller KV footprint via GQA** |
| **System RAM Usage** | ~2.1 GB | **~3.4 GB** | Negligible host footprint |

*Key Takeaway:* With a peak VRAM requirement under 1.2 GiB, ModernGPT-S 500M can comfortably run inference on consumer laptop GPUs, mobile devices, and edge compute nodes.

---

## 3. Computational Throughput & Latency

| Operation | Batch Size | Sequence Length | Mean Latency | Token Throughput |
|---|---|---|---|---|
| **Forward Pass (Validation Loss)** | 1 | 2,048 tokens | 18.2 ms | ~112,500 tokens/sec |
| **Autoregressive Generation ($T=0.3$)** | 1 | 100 tokens | 1,420 ms | **70.4 tokens/sec** |
| **Greedy Generation ($T=0.0$)** | 1 | 100 tokens | 1,380 ms | **72.5 tokens/sec** |
| **Sandbox AST Parsing & Execution** | 1 | 50–150 LOC | 85.0 ms | Subprocess isolation |

---

## 4. Thermal & Duty Cycle Compliance

Operating under strict workstation safety rules (1h 45m active / 15m cooldown):
- **Total Active Compute Window:** 11.5 minutes (10.9% of safety threshold).
- **GPU Core Temperature:** Maintained between 58°C and 64°C throughout the run.
- **Fan Profile:** Low-to-moderate RPM; zero thermal throttling detected.
- **Power Draw:** Average 45–55W (rated TGP: 85W).

---

## 5. Storage Footprint

| Artifact Directory | Storage Size | Notes |
|---|---|---|
| `modern_gpt/checkpoints/checkpoint_501M/` | 3.10 GB | Checkpoint state dict + optimizer states |
| `modern_gpt/reports/phase1d_500m_evaluation/` | ~0.45 MB | Reports, JSONL failure cases, logs |
| `modern_gpt/data/shards/` (101 train + 7 val) | ~10.04 GB | 5.02B pre-tokenized uint16 binary tokens |

---

## 6. Conclusion

The ModernGPT-S 500M architecture demonstrates outstanding hardware ergonomics. The combination of GQA (6:1) and FlashAttention/SDPA delivers 70+ tokens/sec on mobile RTX silicon while consuming less than 15% of the 8GB VRAM pool.

