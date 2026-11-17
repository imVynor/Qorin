# ModernGPT-S 500M — Hardware Inference & Latency Profiling Report

This evaluation benchmarks real runtime performance, VRAM consumption, prefill throughput, and decode latency on local GPU hardware.

## 1. Hardware & Model Configuration
- **Hardware Target:** NVIDIA GeForce RTX 5060 Laptop GPU
- **Precision:** bfloat16
- **Parameters:** 514,389,504 (Weight-tied, 4:1 GQA)
- **Peak VRAM Allocated:** **1220.14 MB** (1.19 GB)
- **Peak VRAM Reserved:** **1352.00 MB**

## 2. Prefill Latency & Throughput (Time to First Token)

| Context Length | Time to First Token (TTFT) | Prefill Throughput |
| :--- | :--- | :--- |
| **128 tokens** | 15.0 ms | 8532.2 tok/s |
| **512 tokens** | 23.85 ms | 21469.7 tok/s |
| **1024 tokens** | 47.58 ms | 21520.2 tok/s |
| **2048 tokens** | 104.04 ms | 19684.5 tok/s |

## 3. Autoregressive Generation Throughput
- **Mean Inter-Token Latency:** **15.63 ms/token**
- **Autoregressive Generation Throughput:** **64.0 tokens/second**

## 4. Hardware SFT Feasibility Assessment
- The 500M model fits comfortably within the 8GB VRAM envelope of the RTX 5060, using <2.8GB during generation.
- Generates at ~30-50 tokens/sec, enabling rapid evaluation and real-time interactive inference.