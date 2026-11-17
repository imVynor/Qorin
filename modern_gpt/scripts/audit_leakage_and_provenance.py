"""Diagnostic audit of SFT v2 training dataset: provenance, duplicates, and leakage.
Outputs: artifacts/diagnostics/sft_v2_audit/sft_v2_leakage_audit.json
"""
from collections import Counter
import json
from pathlib import Path
import re

DATA_SFT_V2 = Path("data/sft_v2_public/train.jsonl")
DATA_HELD_OUT = Path("data/midscale_500/val.jsonl")
DATA_TRANSFER = Path("data/midscale_500/train.jsonl")
DATA_SFT_V1 = Path("data/midscale_500/train.jsonl")  # SFT v1 trained on this 400-example file

def normalize_text(text: str) -> str:
    """Normalize text for near-duplicate and fuzzy overlap checking."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def run_audit():
    print("[*] Auditing SFT v2 dataset integrity and provenance...")
    
    # 1. Load SFT v2 train records
    sft_records = []
    with open(DATA_SFT_V2, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sft_records.append(json.loads(line))
            
    total_records = len(sft_records)
    print(f"Total SFT v2 records: {total_records}")
    
    # Domain / source distribution
    source_counts = Counter(r.get("source", "unknown") for r in sft_records)
    category_counts = Counter(r.get("category", "unknown") for r in sft_records)
    
    # Token length estimates (char count and whitespace token count)
    total_chars = sum(len(r.get("instruction", "")) + len(r.get("input", "")) + len(r.get("output", "")) for r in sft_records)
    prompt_tokens = sum(len((r.get("instruction", "") + " " + r.get("input", "")).split()) for r in sft_records)
    response_tokens = sum(len(r.get("output", "").split()) for r in sft_records)
    
    # 2. Check internal duplicates in SFT v2
    seen_prompts = set()
    exact_duplicate_prompts = []
    seen_norm_prompts = set()
    norm_duplicate_prompts = []
    seen_pairs = set()
    exact_duplicate_pairs = []

    for r in sft_records:
        prompt = (r.get("instruction", "") + "\n" + r.get("input", "")).strip()
        norm_p = normalize_text(prompt)
        pair = (prompt, r.get("output", "").strip())
        
        if prompt in seen_prompts:
            exact_duplicate_prompts.append(r.get("id"))
        else:
            seen_prompts.add(prompt)
            
        if norm_p in seen_norm_prompts:
            norm_duplicate_prompts.append(r.get("id"))
        else:
            seen_norm_prompts.add(norm_p)
            
        if pair in seen_pairs:
            exact_duplicate_pairs.append(r.get("id"))
        else:
            seen_pairs.add(pair)
            
    # 3. Load benchmark datasets
    with open(DATA_HELD_OUT, "r", encoding="utf-8") as f:
        held_out_records = [json.loads(line) for line in f]
    with open(DATA_TRANSFER, "r", encoding="utf-8") as f:
        transfer_records = [json.loads(line) for line in f]
        
    print(f"Loaded {len(held_out_records)} held-out records, {len(transfer_records)} transfer records.")
    
    # 4. Leakage checks against held-out benchmark
    held_out_leakage = []
    held_out_norm_leakage = []
    
    for ho in held_out_records:
        ho_id = ho["id"]
        ho_p = (ho["instruction"] + "\n" + ho.get("input", "")).strip()
        ho_norm = normalize_text(ho_p)
        
        # Check exact prompt match in SFT v2
        if ho_p in seen_prompts:
            held_out_leakage.append({"id": ho_id, "instruction": ho["instruction"], "match": "exact_prompt"})
        elif ho_norm in seen_norm_prompts:
            held_out_norm_leakage.append({"id": ho_id, "instruction": ho["instruction"], "match": "normalized_prompt"})
            
    # 5. Leakage checks against 160 transfer suite / SFT v1 training data (400 examples)
    transfer_leakage = []
    transfer_norm_leakage = []
    
    for tr in transfer_records:
        tr_id = tr["id"]
        tr_p = (tr["instruction"] + "\n" + tr.get("input", "")).strip()
        tr_norm = normalize_text(tr_p)
        
        if tr_p in seen_prompts:
            transfer_leakage.append({"id": tr_id, "category": tr["category"], "instruction": tr["instruction"], "match": "exact_prompt"})
        elif tr_norm in seen_norm_prompts:
            transfer_norm_leakage.append({"id": tr_id, "category": tr["category"], "instruction": tr["instruction"], "match": "normalized_prompt"})

    audit_result = {
        "dataset_path": str(DATA_SFT_V2),
        "total_records": total_records,
        "source_distribution": dict(source_counts),
        "category_distribution": dict(category_counts),
        "token_accounting": {
            "total_estimated_chars": total_chars,
            "approx_prompt_words": prompt_tokens,
            "approx_response_words": response_tokens,
            "approx_total_words": prompt_tokens + response_tokens,
        },
        "internal_duplicates": {
            "exact_duplicate_prompts_count": len(exact_duplicate_prompts),
            "normalized_duplicate_prompts_count": len(norm_duplicate_prompts),
            "exact_duplicate_pairs_count": len(exact_duplicate_pairs),
        },
        "held_out_benchmark_leakage (val.jsonl - 100 tasks)": {
            "exact_prompt_matches": len(held_out_leakage),
            "normalized_prompt_matches": len(held_out_norm_leakage),
            "leakage_records": held_out_leakage + held_out_norm_leakage,
            "contamination_status": "ZERO_LEAKAGE" if len(held_out_leakage) == 0 and len(held_out_norm_leakage) == 0 else "CONTAMINATED"
        },
        "transfer_suite_overlap (train.jsonl - 400 tasks)": {
            "exact_prompt_matches": len(transfer_leakage),
            "normalized_prompt_matches": len(transfer_norm_leakage),
            "transfer_overlap_records": transfer_leakage[:10] + transfer_norm_leakage[:10],
            "note": "400-example transfer suite represents unseen tasks for SFT v2; any overlap indicates shared primitives"
        }
    }
    
    out_file = Path("artifacts/diagnostics/sft_v2_audit/sft_v2_leakage_audit.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit_result, f, indent=2)
        
    print(f"[+] Leakage audit saved to {out_file}")
    print(f"    Held-out exact leakage: {len(held_out_leakage)}, normalized: {len(held_out_norm_leakage)}")
    print(f"    Transfer suite exact overlap: {len(transfer_leakage)}, normalized: {len(transfer_norm_leakage)}")

if __name__ == "__main__":
    run_audit()

