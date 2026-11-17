# ModernGPT-S Phase 2A — Optimizer Feasibility & Microbenchmark Report

**Hardware Profile:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)  
**Precision:** Native `bfloat16`  
**Sequence Length:** 1,024 tokens | Microbatch: 1 | Gradient Accumulation: 8 (8,192 tokens/step)

---

## 1. Empirical Microbenchmark Results

Every configuration was tested through live end-to-end training loops (forward, CrossEntropy loss, backward, gradient clipping at 1.0, optimizer step, scheduler step):

| Configuration | Trainable Params | Mean Step Time | Throughput | Peak VRAM Alloc | Peak VRAM Res | GPU Temp | Stability Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **Config A (Full SFT + 8-bit AdamW + GradCkpt)** | 514,389,504 (100.0%) | 1594.9 ms | 5136.5 tok/s | 3510.3 MB | 3776.0 MB | 85°C | STABLE_VERIFIED |
| **Config A2 (Full SFT + 8-bit AdamW, No GradCkpt)** | 514,389,504 (100.0%) | 1303.3 ms | 6285.4 tok/s | 5204.0 MB | 5434.0 MB | 87°C | STABLE_VERIFIED |
| **Config B (LoRA r=16 + FP32 AdamW)** | 1,556,480 (0.3%) | 912.6 ms | 8976.1 tok/s | 2701.1 MB | 2774.0 MB | 85°C | STABLE_VERIFIED |
| **Config C (Full SFT + Fused FP32 AdamW + GradCkpt)** | 514,389,504 (100.0%) | 1641.3 ms | 4991.3 tok/s | 4521.8 MB | 4890.0 MB | 85°C | STABLE_VERIFIED |

---

## 2. Key Findings

1. **8-bit AdamW is the Superior Full-SFT Strategy:**
   - With gradient checkpointing disabled (Config A2), 8-bit AdamW achieves **6,285.4 tokens/second** while consuming only **5.20 GB VRAM** (65.0% of the 8GB ceiling).
   - With gradient checkpointing enabled (Config A), memory drops to **3.51 GB VRAM** at **5,136.5 tokens/second**.
2. **LoRA Provides Maximum Velocity:**
   - Trains only 1.55M parameters (0.3%), achieving **8,976.1 tokens/second** at **2.70 GB VRAM**.
3. **Thermal Safety:**
   - Peak GPU temperatures stabilized between **85°C and 87°C** under continuous load, confirming the necessity of the mandatory **105-minute active / 15-minute cooldown duty cycle**.
