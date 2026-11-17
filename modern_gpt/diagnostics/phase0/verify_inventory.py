import os
import hashlib
from pathlib import Path

def compute_sha256(filepath):
    p = Path(filepath)
    if not p.exists():
        return None, 0
    h = hashlib.sha256()
    size = p.stat().st_size
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest(), size

artifacts = [
    # Checkpoints
    ("Pretrained 100M Base", "modern_gpt/checkpoints/modern_gpts_100m/best_val.pt"),
    ("Pretrained 100M Step 4 4K Pilot", "modern_gpt/checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt"),
    ("Step 6B SFT v1", "modern_gpt/checkpoints/modern_gpts_100m_sft_4k/best_val.pt"),
    ("Step 7B SFT v2", "modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v2/best_val.pt"),
    ("Step 8D SFT v3", "modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt"),
    # Tokenizer
    ("Tokenizer Config", "modern_gpt/artifacts/tokenizer_16k_v2/tokenizer_config.json"),
    ("Tokenizer JSON", "modern_gpt/artifacts/tokenizer_16k_v2/tokenizer.json"),
    ("Special Tokens Map", "modern_gpt/artifacts/tokenizer_16k_v2/special_tokens_map.json"),
    # Datasets
    ("SFT v3 Train JSONL", "modern_gpt/data/sft_4k_v3/processed/train.jsonl"),
    ("SFT v3 Val JSONL", "modern_gpt/data/sft_4k_v3/processed/val.jsonl"),
    ("Behavioral Eval JSONL", "modern_gpt/data/sft_4k_v3/processed/behavioral_eval.jsonl"),
    ("SFT v2 Train JSONL", "modern_gpt/data/sft_4k_v2/processed/train.jsonl"),
    ("SFT v2 Val JSONL", "modern_gpt/data/sft_4k_v2/processed/val.jsonl"),
    ("SFT v1 Train JSONL", "modern_gpt/data/sft_4k/processed/train.jsonl"),
    ("SFT v1 Val JSONL", "modern_gpt/data/sft_4k/processed/val.jsonl"),
    # Benchmark suites
    ("Coding Benchmark", "modern_gpt/artifacts/benchmarks/coding_pass1_benchmark.jsonl"),
    ("JSON Benchmark", "modern_gpt/artifacts/benchmarks/json_syntax_benchmark.jsonl"),
    ("Retrieval Benchmark", "modern_gpt/artifacts/benchmarks/retrieval_grounding_benchmark.jsonl"),
    # Training logs
    ("Step 8D Training Log", "modern_gpt/artifacts/experiments/modern_gpts_100m_step_08d/step_08d_training_log.jsonl"),
    ("Step 7B Training Log", "modern_gpt/artifacts/experiments/modern_gpts_100m_step_07b/training_log.jsonl"),
    ("Step 6B Training Log", "modern_gpt/artifacts/experiments/modern_gpts_100m_step_06b/training_log.jsonl"),
]

print("| Artifact Name | Path | Size (bytes) | SHA-256 | Status |")
print("| --- | --- | --- | --- | --- |")

for name, path in artifacts:
    sha, size = compute_sha256(path)
    if sha:
        status = "Present"
        print(f"| {name} | `{path}` | {size:,} | `{sha}` | {status} |")
    else:
        status = "MISSING"
        print(f"| {name} | `{path}` | - | - | {status} |")

