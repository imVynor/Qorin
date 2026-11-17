"""Prepare a dedicated 50-example validation set for checkpoint selection during 50M SFT.

Ensures zero overlap with midscale_500/train.jsonl and midscale_500/val.jsonl (held-out test suite).
"""

from collections import defaultdict
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    rev_val_file = Path("c:/Users/Anish/Desktop/Projects/ML-course/modern-gpt-instruction-data/data/revised_v2/val.jsonl")
    mid_train_file = root / "data/midscale_500/train.jsonl"
    mid_val_file = root / "data/midscale_500/val.jsonl"

    rev_val = [json.loads(l) for l in rev_val_file.open("r", encoding="utf-8") if l.strip()]
    mid_train = [json.loads(l) for l in mid_train_file.open("r", encoding="utf-8") if l.strip()]
    mid_val = [json.loads(l) for l in mid_val_file.open("r", encoding="utf-8") if l.strip()]

    def norm(s):
        return " ".join(s.lower().strip().split())

    train_prompts = set(norm(r["instruction"]) for r in mid_train)
    val_prompts = set(norm(r["instruction"]) for r in mid_val)

    by_cat = defaultdict(list)
    for r in rev_val:
        p = norm(r["instruction"])
        if p not in train_prompts and p not in val_prompts:
            cat = "json" if r["category"] == "structured" else r["category"]
            by_cat[cat].append(r)

    print("Available clean candidates:", {k: len(v) for k, v in by_cat.items()})

    # Sample 20 python, 20 json, 5 qa, 5 explanation = 50 examples
    selected = []
    for cat, target_n in [("python", 20), ("json", 20), ("qa", 5), ("explanation", 5)]:
        candidates = by_cat[cat][:target_n]
        for c in candidates:
            selected.append({
                "id": f"val_sel_{c['id']}",
                "category": cat,
                "task_family": f"val_ext_{cat}",
                "instruction": c["instruction"],
                "input": c.get("input", ""),
                "output": c["output"],
            })

    print(f"Selected {len(selected)} records for checkpoint selection validation.")
    out_path = root / "data/midscale_500/checkpoint_selection_val.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in selected:
            f.write(json.dumps(item) + "\n")
    print(f"Successfully saved to {out_path}")


if __name__ == "__main__":
    main()

