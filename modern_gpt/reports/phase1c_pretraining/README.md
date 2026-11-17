# ModernGPT-S 500M — Phase 1C: Pretraining Feasibility, Runtime & Execution Report

This report documents the GPU hardware audit, VRAM feasibility benchmarks, sustained throughput measurements, deadline-aware runtime planning, 1:45/15 thermal operating schedule, and pilot verification for ModernGPT-S 500M.

## Key Executive Findings
- **Model Architecture**: ModernGPT-S (514,389,504 exact parameters, 20 layers, 24 heads, 4 KV heads, GQA 6:1, SwiGLU, RMSNorm, RoPE, tied embeddings, linear biases disabled).
- **Target Hardware**: NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM, Compute Capability 12.0 Blackwell, PyTorch 2.16 + CUDA 13.4, BF16 native).
- **Memory Footprint**: Microbatch size = 1 (2,048 tokens) with gradient checkpointing and fused AdamW allocates **4.87 GiB VRAM** (Peak Reserved: **5.25 GiB**), leaving **2.68 GiB (34%) safe headroom** against OOM and avoiding paging.
- **Measured Sustained Throughput**: **5,054.4 tokens/second** (median) / **5,124.5 tok/s** (mean) / **5,017.2 tok/s** (conservative 25th percentile).
- **Operating Schedule (1:45 / 15 Rule)**: Strict repeating cycles of 105 min active GPU training followed by 15 min mandatory cooling, with pre-emptive thermal override guard if GPU temp exceeds 92°C.
- **Interleaved 7-Domain Sampler**: Memory-mapped zero-copy sampler strictly enforcing target domain mixture across all rolling windows (partial-training protection).
- **Resumable State**: 100% bit-for-bit batch and state restoration validated across process restarts.
