# Phase 1B.5: System Resource Usage Report

## 1. Storage Footprint

- **Acquired Data (`modern_gpt/data_500m/acquired/`)**: 110.14 MB
- **Isolated Validation Data (`modern_gpt/data_500m/val_isolated/`)**: ~57.8 MB
- **Free Disk Space**: 285.8 GB remaining on drive C:
- **Projected Storage for Full 1B Corpus**: ~3.71 GB raw JSONL, ~2.00 GB binary uint16 shards. System capacity is $>75	imes$ greater than required.

## 2. Compute and Network Profile

- **RAM Consumption**: Peak 380 MB during batch streaming and exact hash index registration.
- **CPU Utilization**: 8-12% on modern multi-core CPU during parallel JSON parsing and BPE-32K tokenization.
- **Average Network Throughput**: ~850 KB/s average across seven Hugging Face streaming channels.
