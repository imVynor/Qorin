"""Audit representative samples from the 5,000 instruction dataset across all categories."""

import ast
from collections import defaultdict
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

dataset_dir = Path("c:/Users/Anish/Desktop/Projects/ML-course/modern-gpt-instruction-data/data/final")
train_path = dataset_dir / "train.jsonl"
val_path = dataset_dir / "val.jsonl"

def audit_file(path: Path):
    print(f"\nAuditing {path.name}...")
    by_cat = defaultdict(list)
    license_counts = defaultdict(int)
    source_counts = defaultdict(int)

    python_valid_ast = 0
    python_total = 0
    structured_valid_json = 0
    structured_total = 0

    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            rec = json.loads(line)
            cat = rec["category"]
            by_cat[cat].append(rec)
            license_counts[rec["license"]] += 1
            source_counts[rec["source_dataset"]] += 1

            if cat == "python":
                python_total += 1
                try:
                    ast.parse(rec["output"])
                    python_valid_ast += 1
                except Exception:
                    pass

            if cat == "structured":
                structured_total += 1
                try:
                    json.loads(rec["output"])
                    structured_valid_json += 1
                except Exception:
                    pass

    print(f"Total records in {path.name}: {sum(len(v) for v in by_cat.values())}")
    for cat, recs in by_cat.items():
        print(f"  Category '{cat}': {len(recs)} records")
    print("Source distribution:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt} records")
    print("License distribution:")
    for lic, cnt in license_counts.items():
        print(f"  {lic}: {cnt} records")

    if python_total > 0:
        print(f"Python AST Syntax Validity: {python_valid_ast}/{python_total} ({python_valid_ast/python_total*100:.1f}%)")
    if structured_total > 0:
        print(f"Structured JSON Validity:   {structured_valid_json}/{structured_total} ({structured_valid_json/structured_total*100:.1f}%)")

    # Inspect 2 representative samples per category
    for cat in ["qa", "explanation", "python", "structured"]:
        print("\n" + "=" * 70)
        print(f"SAMPLE INSPECTION: CATEGORY '{cat.upper()}'")
        print("=" * 70)
        recs = by_cat[cat]
        for s_idx in [0, min(5, len(recs)-1)]:
            r = recs[s_idx]
            print(f"--- Example #{s_idx + 1} (ID: {r['id']}) ---")
            print(f"Source:      {r['source_dataset']} (rev: {r['source_revision'][:8]}...)")
            print(f"License:     {r['license']}")
            print(f"Instruction: {r['instruction'][:120]}...")
            if r.get("input"):
                print(f"Input:       {r['input'][:80]}...")
            print(f"Output:      {r['output'][:160]}...")
            if cat == "python":
                try:
                    tree = ast.parse(r["output"])
                    defs = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
                    print(f"AST Parsed:  PASS (defined functions/classes: {defs})")
                except Exception as e:
                    print(f"AST Parsed:  FAIL ({e})")
            elif cat == "structured":
                try:
                    parsed = json.loads(r["output"])
                    keys = list(parsed.keys()) if isinstance(parsed, dict) else f"list of {len(parsed)}"
                    print(f"JSON Parsed: PASS (top-level schema/keys: {keys})")
                except Exception as e:
                    print(f"JSON Parsed: FAIL ({e})")


if __name__ == "__main__":
    audit_file(train_path)
    audit_file(val_path)
