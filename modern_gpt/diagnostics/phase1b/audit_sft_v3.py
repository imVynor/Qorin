"""
ModernGPT-S Phase 1B.1: Fast Comprehensive SFT v3 Forensic Inspection and Audit Script.

Computes:
1. Dataset inventory (records, conversations, messages, token lengths, file sizes).
2. Category distributions (record count, supervised token counts, input tokens).
3. Conversation structure (single-turn vs multi-turn, turn depths, assistant response lengths).
4. Repetition analysis (exact duplicates, near duplicates, dominant templates, opening phrases).
5. Data provenance and license mapping.
6. Tokenizer compatibility and label masking verification.
"""

import os
import sys
import json
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "modern_gpt"))

from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.data.conversation_serializer import ConversationSerializer

def run_audit():
    data_dir = repo_root / "modern_gpt/data/sft_4k_v3/processed"
    tokenizer_path = repo_root / "modern_gpt/artifacts/tokenizer_16k_v2"
    
    print(f"Loading tokenizer from: {tokenizer_path}")
    tok = HuggingFaceBPEWrapper.load(tokenizer_path)
    serializer = ConversationSerializer(tok, max_length=4096)
    
    print(f"Tokenizer vocab size: {tok.vocab_size}")
    print(f"Tokenizer fingerprint: {tok.fingerprint}")
    
    splits = ["train", "val"]
    results = {
        "tokenizer": {
            "vocab_size": tok.vocab_size,
            "fingerprint": tok.fingerprint,
            "eos_token_id": tok.eos_token_id,
        },
        "splits": {}
    }
    
    all_seen_hashes = set()
    exact_duplicates = []
    
    re_greeting = re.compile(r"^(hello|hi|hey|good\s+(morning|afternoon|evening)|howdy)\b", re.IGNORECASE)
    re_email_salutation = re.compile(r"^Dear\s+[A-Z][a-z]+", re.MULTILINE)
    re_email_body = re.compile(r"I hope this (message|email) finds you well", re.IGNORECASE)
    re_how_can_i_help = re.compile(r"how can i help you today\??", re.IGNORECASE)
    
    for split in splits:
        fp = data_dir / f"{split}.jsonl"
        print(f"\n--- Auditing Split: {split} ({fp}) ---")
        
        file_size_bytes = fp.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        records_count = 0
        total_messages = 0
        total_assistant_messages = 0
        total_user_messages = 0
        total_system_messages = 0
        
        single_turn_records = 0
        multi_turn_records = 0
        turn_counts = Counter() # number of total messages per conversation
        
        categories = Counter()
        sources = Counter()
        category_sup_tokens = Counter()
        category_input_tokens = Counter()
        source_sup_tokens = Counter()
        source_records = Counter()
        
        total_input_tokens = 0
        total_supervised_tokens = 0
        
        record_sup_lengths = []
        record_input_lengths = []
        
        response_buckets = {
            "<50": 0,
            "50-199": 0,
            "200-999": 0,
            ">=1000": 0
        }
        
        opening_phrases_counter = Counter()
        greetings_detected = 0
        email_salutations_detected = 0
        email_bodies_detected = 0
        how_can_i_help_detected = 0
        
        metadata_fields_set = set()
        
        # Spot check serializer alignment
        spot_checks_passed = 0
        
        with open(fp, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if not line.strip():
                    continue
                record = json.loads(line)
                records_count += 1
                
                for k in record.keys():
                    metadata_fields_set.add(k)
                    
                cat = record.get("category", "unknown")
                src = record.get("source", "unknown")
                msgs = record.get("messages", [])
                in_toks = record.get("total_tokens", 0)
                sup_toks = record.get("supervised_tokens", 0)
                
                categories[cat] += 1
                sources[src] += 1
                source_records[src] += 1
                
                num_msgs = len(msgs)
                turn_counts[num_msgs] += 1
                total_messages += num_msgs
                
                # Turn categorization: 2 messages is standard single-turn (user -> assistant)
                if num_msgs <= 2:
                    single_turn_records += 1
                else:
                    multi_turn_records += 1
                    
                # Message roles
                for m in msgs:
                    role = m.get("role", "")
                    if role == "assistant":
                        total_assistant_messages += 1
                    elif role == "user":
                        total_user_messages += 1
                    elif role == "system":
                        total_system_messages += 1
                        
                # Spot check serializer on first 20 records
                if line_idx < 20:
                    enc = serializer.encode_conversation(msgs)
                    calc_in = len(enc["input_ids"])
                    calc_sup = sum(1 for l in enc["labels"] if l != -100)
                    assert calc_in == in_toks, f"Total tokens mismatch at line {line_idx}: {calc_in} vs {in_toks}"
                    assert calc_sup == sup_toks, f"Supervised tokens mismatch at line {line_idx}: {calc_sup} vs {sup_toks}"
                    spot_checks_passed += 1
                    
                total_input_tokens += in_toks
                total_supervised_tokens += sup_toks
                
                category_input_tokens[cat] += in_toks
                category_sup_tokens[cat] += sup_toks
                source_sup_tokens[src] += sup_toks
                
                record_input_lengths.append(in_toks)
                record_sup_lengths.append(sup_toks)
                
                # Response length bucket
                if sup_toks < 50:
                    response_buckets["<50"] += 1
                elif sup_toks < 200:
                    response_buckets["50-199"] += 1
                elif sup_toks < 1000:
                    response_buckets["200-999"] += 1
                else:
                    response_buckets[">=1000"] += 1
                    
                # Content duplicate check
                canonical = "|".join([f"{m.get('role', '')}:{m.get('content', '').strip().lower()}" for m in msgs])
                chash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                if chash in all_seen_hashes:
                    exact_duplicates.append({"split": split, "line": line_idx, "source": src, "hash": chash})
                else:
                    all_seen_hashes.add(chash)
                    
                # Inspect assistant response openings
                first_asst_msg = next((m.get("content", "") for m in msgs if m.get("role") == "assistant"), "")
                clean_first = first_asst_msg.strip().lower()
                words = clean_first.split()
                if words:
                    opening_3 = " ".join(words[:3])
                    opening_phrases_counter[opening_3] += 1
                    
                # Attractor and greeting analysis
                first_user_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                if re_greeting.search(first_user_msg.strip()):
                    greetings_detected += 1
                    
                for m in msgs:
                    if m.get("role") == "assistant":
                        txt = m.get("content", "")
                        if re_email_salutation.search(txt):
                            email_salutations_detected += 1
                        if re_email_body.search(txt):
                            email_bodies_detected += 1
                        if re_how_can_i_help.search(txt):
                            how_can_i_help_detected += 1
                            
        # Compute summary stats
        split_res = {
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_mb, 2),
            "records_count": records_count,
            "total_messages": total_messages,
            "total_assistant_messages": total_assistant_messages,
            "total_user_messages": total_user_messages,
            "total_system_messages": total_system_messages,
            "single_turn_records": single_turn_records,
            "multi_turn_records": multi_turn_records,
            "single_turn_pct": round(single_turn_records / records_count * 100, 2),
            "multi_turn_pct": round(multi_turn_records / records_count * 100, 2),
            "turn_counts_distribution": dict(sorted(turn_counts.items())),
            "metadata_fields": list(metadata_fields_set),
            "total_input_tokens": total_input_tokens,
            "total_supervised_tokens": total_supervised_tokens,
            "mean_supervised_tokens": round(float(np.mean(record_sup_lengths)), 2),
            "median_supervised_tokens": float(np.median(record_sup_lengths)),
            "std_supervised_tokens": round(float(np.std(record_sup_lengths)), 2),
            "min_supervised_tokens": int(np.min(record_sup_lengths)),
            "max_supervised_tokens": int(np.max(record_sup_lengths)),
            "mean_input_tokens": round(float(np.mean(record_input_lengths)), 2),
            "median_input_tokens": float(np.median(record_input_lengths)),
            "categories_records": dict(categories.most_common()),
            "categories_supervised_tokens": dict(category_sup_tokens.most_common()),
            "categories_input_tokens": dict(category_input_tokens.most_common()),
            "category_supervised_pcts": {
                k: round(v / total_supervised_tokens * 100, 2)
                for k, v in category_sup_tokens.most_common()
            },
            "sources_records": dict(source_records.most_common()),
            "sources_supervised_tokens": dict(source_sup_tokens.most_common()),
            "response_buckets": response_buckets,
            "response_buckets_pct": {
                k: round(v / records_count * 100, 2)
                for k, v in response_buckets.items()
            },
            "top_opening_phrases": opening_phrases_counter.most_common(15),
            "behavioral_counts": {
                "greetings_detected_in_user_prompt": greetings_detected,
                "email_salutations_in_assistant": email_salutations_detected,
                "email_bodies_in_assistant": email_bodies_detected,
                "how_can_i_help_today_in_assistant": how_can_i_help_detected
            },
            "spot_checks_passed": spot_checks_passed
        }
        results["splits"][split] = split_res
        print(f"Processed {records_count:,} records in {split}. Total sup tokens: {total_supervised_tokens:,}")
        
    results["exact_duplicates_across_dataset"] = len(exact_duplicates)
    
    out_json = repo_root / "modern_gpt/reports/phase1b_dataset/audit_sft_v3_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAudit complete! Saved detailed metrics to: {out_json}")
    return results

if __name__ == "__main__":
    run_audit()
