# Technical Decisions and Risk Analysis

1. **Microbatch Size 1 with Gradient Accumulation 16**:
   - *Rationale*: Microbatch 2 resulted in 8.42 GiB VRAM allocation, spilling into shared system RAM. Microbatch 1 stays at a comfortable 4.87 GiB VRAM, guaranteeing peak performance without memory thrashing.
2. **1:45 / 15 Cooling Cycle**:
   - *Rationale*: Protects the RTX 5060 Laptop GPU from thermal degradation over multi-day training runs.
3. **Interleaved Sampling vs Sequential Domain Training**:
   - *Rationale*: Prevents catastrophic forgetting of early domains and allows valid intermediate checkpoints at any milestone.
