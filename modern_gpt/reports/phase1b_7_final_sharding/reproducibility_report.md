# Phase 1B.7: Reproducibility & Standalone Audit Guide

## 1. Shard Reproduction Pipeline

The binary shards are 100% deterministic and reproducible from the underlying acquired records:
1. Production tokenizer: `modern_gpt/artifacts/tokenizer_32k/tokenizer.json` (SHA-256 `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`).
2. Standalone packing script: `modern_gpt/scratch/pack_final_5b_shards.py`.
3. Standalone verification script: `modern_gpt/scratch/verify_final_5b_shards.py`.

To re-verify all 108 shards independently from scratch:
```bash
python modern_gpt/scratch/verify_final_5b_shards.py
```
