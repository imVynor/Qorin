# Training Configuration Specification

- **Architecture**: ModernGPT-S 500M
- **Parameters**: 514,389,504
- **Precision**: BF16 Mixed Precision
- **Optimizer**: AdamW (`betas=(0.9, 0.95)`, `weight_decay=0.1`, `fused=True`)
- **Learning Rate**: Max `3.0e-4`, Min `3.0e-5` with 2,000 warmup updates
- **Gradient Clipping**: Norm 1.0
- **Context Length**: 2,048
- **Microbatch Size**: 1
- **Gradient Accumulation**: 16
- **Effective Batch Size**: 32,768 tokens / update
- **Gradient Checkpointing**: Enabled (reduces activation VRAM by ~75%)
- **Causal Attention**: PyTorch SDPA (Causal Flash/MemEfficient)
