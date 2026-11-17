# Validation Protocol

- **Validation Corpus**: 20,000,000 tokens across 7 isolated validation shards.
- **Evaluation Cadence**: Every 50,000,000 training tokens.
- **Execution Mode**: `torch.no_grad()`, `model.eval()`, zero weight updates.
- **Metrics Computed**:
  - Global Cross-Entropy Loss
  - Global Perplexity ($e^{\text{loss}}$)
  - Per-Domain Cross-Entropy Loss for all 7 domains
- **Best Model Retention**: Saves `ckpt_best_val.pt` whenever validation loss reaches a new global minimum.
