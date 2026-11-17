"""Deep forensic audit of SFT dataset accounting, categories, and loss masks.

Inspects:
- data/sft_4k/processed/train.jsonl
- data/sft_4k/processed/val.jsonl
- data/sft_4k/intermediate/retrieval/train_retrieval.jsonl
- data/sft_4k/intermediate/retrieval/eval_retrieval.jsonl
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.conversation_serializer import ConversationSerializer


def audit_dataset_accounting():
    tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
    serializer = ConversationSerializer(tok, max_length=4096)

    files = {
        "train": REPO_ROOT / "data/sft_4k/processed/train.jsonl",
        "val": REPO_ROOT / "data/sft_4k/processed/val.jsonl",
        "train_retrieval": REPO_ROOT / "data/sft_4k/intermediate/retrieval/train_retrieval.jsonl",
        "eval_retrieval": REPO_ROOT / "data/sft_4k/intermediate/retrieval/eval_retrieval.jsonl",
    }

    results = {}
    train_hashes = set()
    val_hashes = set()
    train_ids = set()
    val_ids = set()

    for split, p in files.items():
        if not p.exists():
            print(f"File missing: {p}")
            continue

        records = []
        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line.strip():
                    records.append((line_idx, json.loads(line)))

        print(f"\nAuditing {split} ({p.name}): {len(records)} records")

        cat_counts = Counter()
        source_counts = Counter()
        seq_lengths = []
        sup_lengths = []
        masked_lengths = []
        exceeds_4096 = 0
        empty_or_malformed = 0
        duplicate_in_file = 0
        seen_in_split = set()

        cat_total_tokens = Counter()
        cat_sup_tokens = Counter()

        for idx, r in records:
            cat = r.get("category", "unknown")
            cat_counts[cat] += 1
            source_counts[r.get("source", "unknown")] += 1

            # Check duplication via content hash
            msgs = r.get("messages", [])
            content_str = json.dumps(msgs, sort_keys=True)
            chash = hashlib.md5(content_str.encode("utf-8")).hexdigest()

            if chash in seen_in_split:
                duplicate_in_file += 1
            seen_in_split.add(chash)

            if split == "train":
                train_hashes.add(chash)
                train_ids.add(r.get("id", f"train_{idx}"))
            elif split == "val":
                val_hashes.add(chash)
                val_ids.add(r.get("id", f"val_{idx}"))

            # Serialization and loss mask calculation
            try:
                enc = serializer.encode_conversation(msgs, max_length=None)
                total_tok = len(enc["input_ids"])
                labels = enc["labels"]
                sup_tok = (labels != -100).sum().item()
                masked_tok = (labels == -100).sum().item()

                if total_tok > 4096:
                    exceeds_4096 += 1

                seq_lengths.append(total_tok)
                sup_lengths.append(sup_tok)
                masked_lengths.append(masked_tok)

                cat_total_tokens[cat] += total_tok
                cat_sup_tokens[cat] += sup_tok
            except Exception as e:
                empty_or_malformed += 1

        seq_lengths.sort()
        sup_lengths.sort()

        results[split] = {
            "record_count": len(records),
            "category_counts": dict(cat_counts),
            "source_counts": dict(source_counts),
            "duplicate_in_file": duplicate_in_file,
            "empty_or_malformed": empty_or_malformed,
            "exceeds_4096": exceeds_4096,
            "total_tokens": sum(seq_lengths),
            "total_supervised_tokens": sum(sup_lengths),
            "total_masked_tokens": sum(masked_lengths),
            "mean_seq_len": round(sum(seq_lengths) / len(seq_lengths), 2) if seq_lengths else 0,
            "median_seq_len": seq_lengths[len(seq_lengths) // 2] if seq_lengths else 0,
            "min_seq_len": min(seq_lengths) if seq_lengths else 0,
            "max_seq_len": max(seq_lengths) if seq_lengths else 0,
            "pct_gt_512": round(sum(1 for x in seq_lengths if x > 512) / len(seq_lengths) * 100, 2) if seq_lengths else 0,
            "pct_gt_1024": round(sum(1 for x in seq_lengths if x > 1024) / len(seq_lengths) * 100, 2) if seq_lengths else 0,
            "pct_gt_2048": round(sum(1 for x in seq_lengths if x > 2048) / len(seq_lengths) * 100, 2) if seq_lengths else 0,
            "pct_gt_3072": round(sum(1 for x in seq_lengths if x > 3072) / len(seq_lengths) * 100, 2) if seq_lengths else 0,
            "category_total_tokens": dict(cat_total_tokens),
            "category_supervised_tokens": dict(cat_sup_tokens),
        }

    # Cross-split overlap
    cross_overlap = train_hashes.intersection(val_hashes)
    id_overlap = train_ids.intersection(val_ids)

    print("\n--- Cross-Split Analysis ---")
    print(f"Content overlap between train and val: {len(cross_overlap)}")
    print(f"ID overlap between train and val: {len(id_overlap)}")

    results["cross_split"] = {
        "content_overlap_count": len(cross_overlap),
        "id_overlap_count": len(id_overlap),
    }

    # Combined train + val analysis
    total_records = results["train"]["record_count"] + results["val"]["record_count"]
    total_tokens = results["train"]["total_tokens"] + results["val"]["total_tokens"]
    total_supervised = results["train"]["total_supervised_tokens"] + results["val"]["total_supervised_tokens"]
    total_masked = results["train"]["total_masked_tokens"] + results["val"]["total_masked_tokens"]

    print("\n--- Combined Train + Val Summary ---")
    print(f"Total records: {total_records}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Total supervised tokens: {total_supervised:,}")
    print(f"Total masked tokens: {total_masked:,}")

    # Trace category-level supervised tokens sum
    combined_cat_counts = Counter(results["train"]["category_counts"]) + Counter(results["val"]["category_counts"])
    combined_cat_sup = Counter(results["train"]["category_supervised_tokens"]) + Counter(results["val"]["category_supervised_tokens"])
    combined_cat_tot = Counter(results["train"]["category_total_tokens"]) + Counter(results["val"]["category_total_tokens"])

    print("\n--- Combined Category Breakdown (Exact Mask Calculation) ---")
    sum_cat_sup = 0
    sum_cat_tot = 0
    for cat, cnt in combined_cat_counts.items():
        tot = combined_cat_tot[cat]
        sup = combined_cat_sup[cat]
        sum_cat_sup += sup
        sum_cat_tot += tot
        print(f"  {cat:25s}: {cnt:5d} records | {tot:10,d} total tok | {sup:10,d} sup tok ({sup/tot*100:.1f}%)")

    print(f"\nSum of category-level sup tokens: {sum_cat_sup:,}")
    print(f"Discrepancy with total supervised: {total_supervised - sum_cat_sup}")

    results["combined"] = {
        "total_records": total_records,
        "total_tokens": total_tokens,
        "total_supervised_tokens": total_supervised,
        "total_masked_tokens": total_masked,
        "category_counts": dict(combined_cat_counts),
        "category_total_tokens": dict(combined_cat_tot),
        "category_supervised_tokens": dict(combined_cat_sup),
        "sum_of_category_supervised": sum_cat_sup,
    }

    out_file = REPO_ROOT / "artifacts/diagnostics/sft_dataset_accounting_audit.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved accounting audit results to {out_file}")


if __name__ == "__main__":
    audit_dataset_accounting()

