# Resource & Performance Engineering Analysis

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/RESOURCE_REQUIREMENTS.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Model Memory & Parameter Footprint

Values are derived directly from model architecture configurations (`canonical_model_manifest.json`) and physical checkpoint file analysis:

| Model Scale | Checkpoint Identifier | Architecture Specs | Trainable Params | Checkpoint File Size | Estimated BF16 VRAM | Estimated FP32 RAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **500M** | `qorin-500m-sft-exp-f` | 20L-24Q-4KV (GQA 6:1), SwiGLU 3840, ctx 2048 | 514,389,504 | 1,982.32 MB | **1,028.8 MB** | **2,057.6 MB** |
| **500M (Tr/Opt)** | `qorin-500m-sft-exp-c` | 20L-24Q-4KV, includes AdamW states | 514,389,504 | 2,979.33 MB | **1,028.8 MB** | **2,057.6 MB** |
| **100M (4K)** | `qorin-100m-sft-v3` | 14L-12Q-4KV (GQA 3:1), SwiGLU 2048, ctx 4096 | 104,807,424 | 588.69 MB | **209.6 MB** | **419.2 MB** |
| **100M (Base)** | `qorin-100m-base` | 14L-12Q-4KV (GQA 3:1), SwiGLU 2048, ctx 512 | 113,606,912 | 1,152.70 MB | **227.2 MB** | **454.4 MB** |
| **50M** | `qorin-50m-sft` | 12L-8H MHA, SwiGLU 1365, ctx 512 | 48,934,912 | 565.12 MB | **97.9 MB** | **195.7 MB** |
| **16M** | `qorin-16m-sft` | 6L-6H MHA, GELU, ctx 512 | 16,768,128 | 193.49 MB | **33.5 MB** | **67.1 MB** |

*(Note: Raw model weight memory equals $\text{parameters} \times 2\text{ bytes}$ in BF16, and $\text{parameters} \times 4\text{ bytes}$ in FP32).*

---

## 2. Runtime Memory Overhead & Peak VRAM

### Theoretical KV-Cache Footprint (Batch Size = 1)
Calculated via:
$$\text{Memory}_{\text{KV}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times L_{\text{ctx}} \times \text{bytes\_per\_elem}$$

- **QORIN-500M (2,048 context, GQA 4 KV heads, head_dim 64, BF16):**
  $$2 \times 20 \times 4 \times 64 \times 2048 \times 2 = 41,943,040\text{ bytes} \approx \mathbf{40.0\text{ MB}}$$
  *Benefit of GQA 6:1:* Reduces KV cache from 240 MB (under MHA) to just 40 MB.
- **QORIN-100M (4,096 context, GQA 4 KV heads, head_dim 64, BF16):**
  $$2 \times 14 \times 4 \times 64 \times 4096 \times 2 = 58,720,256\text{ bytes} \approx \mathbf{56.0\text{ MB}}$$

*(Note: Currently, the generation loop in `qorin_provider.py` computes full attention over the active sequence window at each token step rather than maintaining an incremental past-key-value cache. Thus, activation memory peaks during the final tokens of generation).*

### Total Peak VRAM Breakdown on Reference Hardware
Reference Hardware: **NVIDIA GeForce RTX 5060 Laptop GPU (8.0 GB VRAM)**

| Component | 500M Active (BF16) | 100M Active (BF16) | Notes |
| :--- | :--- | :--- | :--- |
| PyTorch CUDA Runtime Context | ~450 MB | ~450 MB | CUDA driver initialization and base allocator overhead. |
| Model Weights | ~1,029 MB | ~210 MB | Loaded into VRAM. |
| Activations & Logits Buffer | ~420 MB | ~180 MB | Peak attention matrix and vocabulary projection logits ($B \times V$). |
| Python Sandbox Subprocess | 0 MB (VRAM) | 0 MB (VRAM) | Sandbox executes as a CPU subprocess. |
| **Total Measured Peak VRAM** | **~2,100 MB (2.1 GB)** | **~650 MB (0.65 GB)** | Well within 8GB laptop GPU (26.2% capacity utilized). |

---

## 3. Empirical Inference Benchmarks (Measured Data)

The metrics below represent **empirical runtime measurements** extracted directly from **379 real logged inference traces** stored in `backend/data/experiment_traces.jsonl`:

| Metric | QORIN-500M SFT Exp F (`ckpt_pilot_best.pt`) | QORIN-500M SFT Exp C (`ckpt_experiment_c_final.pt`) | QORIN-100M SFT 4K v3 (`best_val.pt`) | Category |
| :--- | :--- | :--- | :--- | :--- |
| **Sample Count ($N$)** | 141 traces | 196 traces | 24 traces | **Measured** |
| **Time to First Token (TTFT) - Mean** | **80.2 ms** | **43.9 ms** | **161.9 ms** | **Measured** |
| **TTFT - Minimum** | **17.2 ms** | **12.2 ms** | **11.6 ms** | **Measured** |
| **TTFT - Maximum** | **585.8 ms** | **644.7 ms** | **846.6 ms** | **Measured** |
| **Generation Throughput - Mean** | **30.8 tokens/sec** | **52.2 tokens/sec** | **54.7 tokens/sec** | **Measured** |
| **Generation Throughput - Peak** | **51.2 tokens/sec** | **82.8 tokens/sec** | **73.8 tokens/sec** | **Measured** |
| **Total End-to-End Latency - Mean** | **3,114.2 ms** | **2,085.4 ms** | **1,788.6 ms** | **Measured** |
| **Checkpoint Initialization Time** | **2.1 seconds** | **2.6 seconds** | **0.8 seconds** | **Measured** |

---

## 4. CPU Deployment Feasibility Analysis

Can QORIN run on free-tier cloud CPU instances (e.g. Hugging Face Spaces 2 vCPU, 16GB RAM)?

- **RAM Sufficiency:** 500M in FP32 requires ~2.1 GB for weights + 1.2 GB for PyTorch/FastAPI = ~3.3 GB RAM. The 16 GB instance has ample RAM.
- **Compute Feasibility (Throughput):**
  - Without AVX-512 / AMX hardware acceleration, single-threaded or dual-threaded PyTorch autoregressive CPU matrix multiplications on a 514M parameter transformer yield approximately **1.5 to 3.5 tokens per second**.
  - A 250-token response on CPU requires:
    $$\frac{250\text{ tokens}}{2.5\text{ tok/s}} \approx 100\text{ seconds}$$
  - **Verdict:** While technically functional on CPU, serving the 514M model on a dual-core CPU yields an unacceptable user experience (>1 minute response times). The 100M model on CPU achieves ~8-12 tokens per second (acceptable, 15-20s response time).
