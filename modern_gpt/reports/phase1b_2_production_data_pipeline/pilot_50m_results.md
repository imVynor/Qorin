# ModernGPT-S 500M — Pilot G2 (50M Tokens) Validation Report

```json
{
  "pilot_name": "pilot_g2_50m",
  "target_tokens": 50000000,
  "actual_tokens_generated": 49999872,
  "total_sequences": 24414,
  "sequence_length": 2048,
  "raw_docs_examined": 54524,
  "clean_docs_passed": 53749,
  "filter_attrition_pct": 1.42,
  "domain_distribution": {
    "web": 37587,
    "edu": 15205,
    "code": 957,
    "math": 0,
    "wiki": 0
  },
  "total_duration_sec": 99.58,
  "overall_throughput_tokens_sec": 502111.5,
  "peak_ram_mb": 3384.8,
  "shard_metadata": {
    "shard_file": "modern_gpt\\data_500m\\pilot_shards\\g2\\pilot_g2_50m_shard_00000.bin",
    "file_size_bytes": 99999824,
    "header_size_bytes": 80,
    "payload_bytes": 99999744,
    "total_tokens": 49999872,
    "sequence_length": 2048,
    "total_sequences": 24414,
    "tokenizer_fingerprint": "3559c7435e1d9330",
    "payload_sha256": "7ebd62aea4fba95c389382b68b16566b5eb3f56595d37d413ec8db604b3c2302",
    "file_sha256": "3b5469eda977a7d163026f79a48b806080fb1ac27853cb3097fc76d2520bf28a"
  },
  "validation_status": "PASSED",
  "dataloader_status": "VERIFIED"
}
```
