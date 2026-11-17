import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "modern_gpt"))

from data.bpe_tokenizer import HuggingFaceBPEWrapper

DATASETS = {
    "sft_v3_train": "modern_gpt/data/sft_4k_v3/processed/train.jsonl",
    "sft_v3_val": "modern_gpt/data/sft_4k_v3/processed/val.jsonl",
    "sft_v2_train": "modern_gpt/data/sft_4k_v2/processed/train.jsonl",
    "sft_v1_train": "modern_gpt/data/sft_4k/processed/train.jsonl",
}

ATTRACTOR_PATTERNS = {
    "assist_help_open": re.compile(r"how can i (assist|help) you", re.IGNORECASE),
    "email_boilerplate": re.compile(r"i hope this email finds you well|best regards|dear [a-z]+", re.IGNORECASE),
    "ai_disclaimer": re.compile(r"as an ai (language model|assistant)", re.IGNORECASE),
    "feel_free": re.compile(r"feel free to (ask|reach out)", re.IGNORECASE),
    "let_me_know": re.compile(r"let me know if you (have|need|require)", re.IGNORECASE),
    "support_rep": re.compile(r"customer support|customer service|support team", re.IGNORECASE),
    "sure_certainly": re.compile(r"^(sure|certainly|absolutely)[!,\.]", re.IGNORECASE),
}

def analyze_dataset(name: str, path_str: str, tok: HuggingFaceBPEWrapper):
    p = Path(path_str)
    if not p.exists():
        return None

    total_records = 0
    categories = Counter()
    sources = Counter()
    turn_counts = Counter()
    attractor_counts = Counter()
    openings = Counter()

    prompt_lens = []
    resp_lens = []
    total_lens = []

    greeting_examples = []
    email_examples = []

    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_records += 1

            cat = rec.get("category", "unspecified")
            categories[cat] += 1

            src = rec.get("source", "unspecified")
            sources[src] += 1

            messages = rec.get("messages", [])
            user_turns = [m for m in messages if m.get("role") == "user"]
            asst_turns = [m for m in messages if m.get("role") == "assistant"]

            turn_counts[len(user_turns)] += 1

            # Length calculation
            p_len = sum(len(tok.encode(m.get("content", ""))) for m in messages if m.get("role") in ("system", "user"))
            r_len = sum(len(tok.encode(m.get("content", ""))) for m in asst_turns)
            prompt_lens.append(p_len)
            resp_lens.append(r_len)
            total_lens.append(p_len + r_len)

            # Analyze assistant responses
            for asst in asst_turns:
                txt = asst.get("content", "")
                # Opening 5 words
                words = txt.strip().split()
                if words:
                    op = " ".join(words[:min(3, len(words))]).lower()
                    openings[op] += 1

                for att_name, pat in ATTRACTOR_PATTERNS.items():
                    if pat.search(txt):
                        attractor_counts[att_name] += 1

            # Check for greetings
            if user_turns:
                first_user = user_turns[0].get("content", "").strip().lower()
                if first_user in ("hello", "hi", "hey", "good morning", "how are you?", "how are you"):
                    greeting_examples.append({
                        "user": user_turns[0].get("content"),
                        "assistant": asst_turns[0].get("content") if asst_turns else "",
                        "source": src,
                        "category": cat
                    })

            # Check for email boilerplate
            for asst in asst_turns:
                txt = asst.get("content", "")
                if ATTRACTOR_PATTERNS["email_boilerplate"].search(txt) and len(email_examples) < 10:
                    email_examples.append({
                        "user": user_turns[0].get("content") if user_turns else "",
                        "assistant": txt[:300],
                        "source": src
                    })

    prompt_lens.sort()
    resp_lens.sort()
    total_lens.sort()

    def get_stats(arr):
        if not arr:
            return {}
        return {
            "min": arr[0],
            "p25": arr[len(arr) // 4],
            "median": arr[len(arr) // 2],
            "p75": arr[(len(arr) * 3) // 4],
            "p95": arr[(len(arr) * 95) // 100],
            "max": arr[-1],
            "mean": round(sum(arr) / len(arr), 1)
        }

    # Length buckets
    short_resp = sum(1 for r in resp_lens if r < 50)
    med_resp = sum(1 for r in resp_lens if 50 <= r < 200)
    long_resp = sum(1 for r in resp_lens if 200 <= r < 1000)
    huge_resp = sum(1 for r in resp_lens if r >= 1000)

    return {
        "name": name,
        "total_records": total_records,
        "categories": dict(categories.most_common(15)),
        "sources": dict(sources.most_common(15)),
        "single_turn_records": turn_counts[1],
        "single_turn_pct": round(turn_counts[1] / max(1, total_records) * 100, 2),
        "multiturn_records": total_records - turn_counts[1],
        "multiturn_pct": round((total_records - turn_counts[1]) / max(1, total_records) * 100, 2),
        "turn_distribution": dict(turn_counts.most_common()),
        "prompt_tokens_stats": get_stats(prompt_lens),
        "response_tokens_stats": get_stats(resp_lens),
        "total_tokens_stats": get_stats(total_lens),
        "response_length_buckets": {
            "<50_tokens": short_resp,
            "50_to_199_tokens": med_resp,
            "200_to_999_tokens": long_resp,
            ">=1000_tokens": huge_resp,
        },
        "top_openings": dict(openings.most_common(15)),
        "attractor_matches": dict(attractor_counts),
        "greetings_found_count": len(greeting_examples),
        "greeting_samples": greeting_examples[:10],
        "email_samples": email_examples[:5],
    }

def main():
    print("[*] Loading tokenizer for dataset audit...")
    tok = HuggingFaceBPEWrapper.load("modern_gpt/artifacts/tokenizer_16k_v2")

    summary = {}
    for d_name, d_path in DATASETS.items():
        print(f"[*] Analyzing {d_name} ({d_path})...")
        res = analyze_dataset(d_name, d_path, tok)
        if res:
            summary[d_name] = res
            print(f"[+] Done {d_name}: {res['total_records']} records, {res['multiturn_records']} multiturn ({res['multiturn_pct']}%)")

    out_file = Path("modern_gpt/diagnostics/phase0/dataset_audit_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[✓] Finished dataset audit. Saved detailed summary to {out_file}")

if __name__ == "__main__":
    main()

