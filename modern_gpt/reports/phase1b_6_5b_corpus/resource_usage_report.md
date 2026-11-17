# Phase 1B.6: Resource Usage & Full 5B Storage Projection

## 1. Current Storage Footprint

- **Acquired Data (`modern_gpt/data_500m/acquired/`)**: 21717.29 MB
- **Isolated Validation Data (`modern_gpt/data_500m/val_isolated/`)**: 57.8 MB
- **Available Free Disk Space**: 285.6 GB on drive C:

## 2. Projected Full 5B Corpus Requirements

Based on observed compression ratios (3.92 characters per token) and filtering yields (40.2% acceptance yield):
- **Raw Text Download Required**: ~18.5 GB
- **Filtered JSONL Storage**: ~10.0 GB
- **Packed Binary uint16 Shards**: Exactly 10.00 GB (5,000,000,000 tokens $\times$ 2 bytes/token)
- **Feasibility**: Total projected footprint (~28.5 GB) consumes only 10.0% of available disk space (285.6 GB free).
