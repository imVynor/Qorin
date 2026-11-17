"""Inspect and profile candidate Hugging Face datasets for ModernGPT-S SFT v2.

Profiles:
- google-research-datasets/mbpp
- HuggingFaceH4/CodeAlpaca_20K
- HuggingFaceH4/ultrachat_200k
- databricks/databricks-dolly-15k
- rajpurkar/squad
- interstellarninja/json-mode-singleturn
- openai/gsm8k
"""

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from datasets import load_dataset


def profile_candidate(name, repo, config, split, sample_limit=200):
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")
    print(f"\n{'='*60}")
    print(f"Auditing Candidate: {name} ({repo})")
    print(f"{'='*60}")

    kwargs = {"streaming": True}
    if config:
        ds = load_dataset(repo, config, split=split, **kwargs)
    else:
        ds = load_dataset(repo, split=split, **kwargs)

    lengths = []
    examples = []
    exceeded_512 = 0

    for i, ex in enumerate(ds):
        if i >= sample_limit:
            break
        examples.append(ex)

        # Estimate text based on format
        text = json.dumps(ex, ensure_ascii=False)
        token_count = len(tok.encode(text))
        lengths.append(token_count)
        if token_count > 512:
            exceeded_512 += 1

    lengths.sort()
    avg_len = sum(lengths) / len(lengths)
    med_len = lengths[len(lengths) // 2]
    p90_len = lengths[int(len(lengths) * 0.90)]
    max_len = max(lengths)

    print(f"Inspected sample:     {len(lengths)} examples")
    print(f"Token length stats (full record JSON representation):")
    print(f"  Avg: {avg_len:.1f} | Med: {med_len} | P90: {p90_len} | Max: {max_len}")
    print(f"  Exceeding 512 tokens: {exceeded_512}/{len(lengths)} ({exceeded_512/len(lengths)*100:.1f}%)")
    print(f"Sample keys:          {list(examples[0].keys())}")

    return {
        "name": name,
        "repo": repo,
        "config": config,
        "split": split,
        "sample_size": len(lengths),
        "avg_tokens": round(avg_len, 1),
        "med_tokens": med_len,
        "p90_tokens": p90_len,
        "max_tokens": max_len,
        "pct_exceeding_512": round(exceeded_512 / len(lengths) * 100, 1),
        "keys": list(examples[0].keys()),
        "sample_record": examples[0],
    }


def main():
    candidates = [
        ("mbpp", "google-research-datasets/mbpp", "full", "train"),
        ("CodeAlpaca", "HuggingFaceH4/CodeAlpaca_20K", None, "train"),
        ("dolly_15k", "databricks/databricks-dolly-15k", None, "train"),
        ("squad", "rajpurkar/squad", None, "train"),
        ("json_mode", "interstellarninja/json-mode-singleturn", None, "train"),
        ("gsm8k", "openai/gsm8k", "main", "train"),
        ("ultrachat", "HuggingFaceH4/ultrachat_200k", "default", "train_sft"),
    ]

    results = []
    for c in candidates:
        try:
            res = profile_candidate(c[0], c[1], c[2], c[3], sample_limit=150)
            results.append(res)
        except Exception as e:
            print(f"Error auditing {c[0]}: {e}")

    out_file = Path("artifacts/reports/candidate_profile_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        # Convert sample record for serialization safety
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved profile results to {out_file}")


if __name__ == "__main__":
    main()

