# Checkpoint and Resume Design

## Atomic Write Protocol
1. Checkpoint dictionary serialized to `ckpt_latest.pt.tmp`.
2. Flushed to disk via atomic file replacement (`tmp_path.replace(path)`).
3. Guarantees zero corrupt checkpoints even during power loss.

## State Contents
- Model weights (`bfloat16`)
- Optimizer states (FP32 master parameters, momentum, variance)
- Learning rate scheduler state
- Global update counter and total tokens processed
- Sampler RNG state and per-domain token offsets
- PyTorch CPU, CUDA, and NumPy RNG states
- Architecture configuration and tokenizer fingerprint (`3559c7435e1d9330`)
