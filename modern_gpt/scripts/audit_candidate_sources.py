"""Inspect candidate SFT dataset sources for ModernGPT-S 100M GQA-4 4K context.

Candidate sources:
1. SmolTalk (HuggingFaceTB/smoltalk)
2. UltraChat 200k (HuggingFaceH4/ultrachat_200k)
3. OpenAssistant OASST1 (OpenAssistant/oasst1)
4. Tulu 3 SFT Mixture (allenai/tulu-3-sft-mixture)
5. Existing SFT v2 dataset (data/sft_v2_public/train.jsonl)
"""

import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from datasets import load_dataset


def inspect_source(name, repo, config, split, sample_limit=20):
    tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
    print(f"\nAuditing: {name} ({repo})")
    try:
        if config:
            ds = load_dataset(repo, config, split=split, streaming=True)
        else:
            ds = load_dataset(repo, split=split, streaming=True)

        samples = []
        token_lengths = []
        turn_counts = []
        for i, item in enumerate(ds):
            if i >= sample_limit:
                break
            samples.append(item)
            # Serialize preview text
            if "messages" in item:
                turns = len(item["messages"])
                turn_counts.append(turns)
                text = "\n".join([f"{m.get('role', '')}: {m.get('content', '')}" for m in item["messages"]])
            elif "prompt" in item and "messages" not in item:
                turn_counts.append(1)
                text = str(item.get("prompt", "")) + "\n" + str(item.get("response", ""))
            elif "text" in item:
                turn_counts.append(1)
                text = str(item.get("text", ""))
            else:
                turn_counts.append(1)
                text = json.dumps(item)

            tokens = tok.encode(text)
            token_lengths.append(len(tokens))

        avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
        avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0

        print(f"  Sampled: {len(samples)} records")
        print(f"  Avg tokens: {avg_tokens:.1f} | Max: {max(token_lengths) if token_lengths else 0}")
        print(f"  Avg turns: {avg_turns:.1f} | Max turns: {max(turn_counts) if turn_counts else 0}")
        print(f"  Keys: {list(samples[0].keys()) if samples else []}")

        return {
            "name": name,
            "repo": repo,
            "config": config,
            "split": split,
            "sample_count": len(samples),
            "keys": list(samples[0].keys()) if samples else [],
            "avg_tokens": avg_tokens,
            "max_tokens": max(token_lengths) if token_lengths else 0,
            "avg_turns": avg_turns,
            "first_sample_keys": list(samples[0].keys()) if samples else [],
            "status": "AVAILABLE"
        }
    except Exception as e:
        print(f"  Failed: {e}")
        return {
            "name": name,
            "repo": repo,
            "config": config,
            "split": split,
            "error": str(e),
            "status": "ERROR"
        }


def main():
    candidates = [
        ("SmolTalk", "HuggingFaceTB/smoltalk", "all", "train"),
        ("UltraChat_200k", "HuggingFaceH4/ultrachat_200k", "default", "train_sft"),
        ("OASST1", "OpenAssistant/oasst1", None, "train"),
        ("Tulu_3_Mixture", "allenai/tulu-3-sft-mixture", None, "train"),
    ]

    results = {}
    for c in candidates:
        res = inspect_source(c[0], c[1], c[2], c[3], sample_limit=25)
        results[c[0]] = res

    # Audit local existing sft_v2_public
    v2_path = REPO_ROOT / "data/sft_v2_public/train.jsonl"
    if v2_path.exists():
        tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
        lines = v2_path.read_text(encoding="utf-8").strip().split("\n")
        lengths = []
        for l in lines[:100]:
            r = json.loads(l)
            txt = r.get("instruction", "") + "\n" + r.get("input", "") + "\n" + r.get("output", "")
            lengths.append(len(tok.encode(txt)))
        results["Existing_SFT_v2"] = {
            "name": "Existing_SFT_v2",
            "repo": "local:data/sft_v2_public/train.jsonl",
            "sample_count": len(lines),
            "avg_tokens": sum(lengths) / len(lengths),
            "max_tokens": max(lengths),
            "avg_turns": 1.0,
            "status": "LOCAL_VERIFIED"
        }

    out_file = REPO_ROOT / "artifacts/experiments/modern_gpts_100m_sft_audit/dataset_source_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved audit results to {out_file}")


if __name__ == "__main__":
    main()
