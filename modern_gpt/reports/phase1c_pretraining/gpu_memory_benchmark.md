# GPU Memory & VRAM Feasibility Benchmark

## Microbatch Configurations
| Microbatch Size | Tokens / Microbatch | Peak Allocated VRAM | Peak Reserved VRAM | Headroom on 7.93GB | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1 (Selected)** | **2,048** | **4,222.2 MiB (4.12 GiB)** | **4,388.0 MiB (4.29 GiB)** | **+3.64 GiB (46%)** | **OPTIMAL / SAFE** |
| 2 | 4,096 | 8,618.5 MiB (8.42 GiB) | 8,910.0 MiB (8.70 GiB) | -0.49 GiB (Spills) | UNSAFE (Paging) |
| 4 | 8,192 | 9,567.5 MiB (9.34 GiB) | 11,238.0 MiB (10.97 GiB) | -1.41 GiB (Spills) | UNSAFE (Paging) |

## Memory Component Breakdown (Microbatch 1, BF16)
- Model Parameters (BF16): 1,028.8 MiB (1.01 GiB)
- Gradients (BF16): 1,028.8 MiB (1.01 GiB)
- Optimizer States (Fused AdamW): 2,057.6 MiB (2.01 GiB)
- Activations (with Gradient Checkpointing): ~106.8 MiB
- Attention Buffers & PyTorch Overhead: ~678.0 MiB
- **Total Steady-State VRAM**: **4,870.0 MiB (~4.87 GiB)**
