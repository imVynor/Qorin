# ModernGPT-S 500M — Pilot G1 (1M–5M Tokens) Validation Report

```json
{
  "pilot_name": "pilot_g1_small",
  "target_tokens": 2000000,
  "actual_tokens_generated": 1998848,
  "total_sequences": 976,
  "sequence_length": 2048,
  "raw_docs_examined": 2209,
  "clean_docs_passed": 2171,
  "filter_attrition_pct": 1.72,
  "domain_distribution": {
    "web": 1516,
    "edu": 611,
    "code": 44,
    "math": 0,
    "wiki": 0
  },
  "total_duration_sec": 5.89,
  "overall_throughput_tokens_sec": 339472.6,
  "peak_ram_mb": 694.0,
  "shard_metadata": {
    "shard_file": "modern_gpt\\data_500m\\pilot_shards\\g1\\pilot_g1_small_shard_00000.bin",
    "file_size_bytes": 3997776,
    "header_size_bytes": 80,
    "payload_bytes": 3997696,
    "total_tokens": 1998848,
    "sequence_length": 2048,
    "total_sequences": 976,
    "tokenizer_fingerprint": "3559c7435e1d9330",
    "payload_sha256": "2647ffbbfa32404c5156781786e30898d427b6c7e555a29f51b7e90ea794d014",
    "file_sha256": "f23ff6313b014138affc236b80cf59af2322201c35c04b6b059835798f6fa884"
  },
  "validation_status": "PASSED",
  "dataloader_status": "VERIFIED"
}
```
