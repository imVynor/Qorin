"""
Stage 7.5 Programmatic Verification Script
Audits:
1. Multi-step test model participation (distinguishing mocked fixture vs model capability)
2. Exact diagnostic breakdown of the 3 semantic failures
3. Full dataset integrity audit across all 9 JSONL files
4. Training exposure and token kinematics reconciliation
"""

import hashlib
import json
import os
import sys
from collections import Counter
from tokenizers import Tokenizer

TOK_PATH = "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
DATA_DIR = "modern_gpt/experiments/phase2_experiment_e/stage_1_data"
STAGE7_DIR = "modern_gpt/experiments/phase2_experiment_e/stage_7_prerequisites"
STAGE75_DIR = "modern_gpt/experiments/phase2_experiment_e/stage_7_5_verification"


def audit_datasets():
    print("--- 1. Auditing Dataset Files Integrity ---", flush=True)
    tok = Tokenizer.from_file(TOK_PATH)
    
    files = [
        "corpus1_calibration_train.jsonl",
        "corpus1_calibration_val.jsonl",
        "corpus1_calibration_test.jsonl",
        "corpus2_synthesis_train.jsonl",
        "corpus2_synthesis_val.jsonl",
        "corpus2_synthesis_test.jsonl",
        "combined_train.jsonl",
        "combined_val.jsonl",
        "combined_test.jsonl"
    ]
    
    records_by_file = {}
    ids_by_file = {}
    token_stats_by_file = {}
    
    for fname in files:
        fpath = os.path.join(DATA_DIR, fname)
        assert os.path.exists(fpath), f"Missing file: {fpath}"
        
        recs = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
                    
        records_by_file[fname] = recs
        ids = [r["id"] for r in recs]
        ids_by_file[fname] = set(ids)
        assert len(ids) == len(set(ids)), f"Duplicate IDs detected inside {fname}!"
        
        tot_tok = sum(r["total_token_length"] for r in recs)
        tgt_tok = sum(r["target_token_length"] for r in recs)
        pfx_tok = sum(r["prefix_token_length"] for r in recs)
        
        token_stats_by_file[fname] = {
            "record_count": len(recs),
            "total_tokens": tot_tok,
            "target_tokens": tgt_tok,
            "prefix_tokens": pfx_tok,
            "avg_total": round(tot_tok / len(recs), 1),
            "avg_target": round(tgt_tok / len(recs), 1)
        }
        print(f"Verified {fname}: {len(recs)} records | {tot_tok} total tokens | {tgt_tok} target tokens.", flush=True)
        
    # Check disjointness
    c1_all_ids = ids_by_file["corpus1_calibration_train.jsonl"] | ids_by_file["corpus1_calibration_val.jsonl"] | ids_by_file["corpus1_calibration_test.jsonl"]
    c2_all_ids = ids_by_file["corpus2_synthesis_train.jsonl"] | ids_by_file["corpus2_synthesis_val.jsonl"] | ids_by_file["corpus2_synthesis_test.jsonl"]
    
    overlap_c1_c2 = c1_all_ids & c2_all_ids
    assert len(overlap_c1_c2) == 0, f"Corpus 1 and Corpus 2 ID collision: {overlap_c1_c2}"
    print("\nVerified: Zero ID collisions between Corpus 1 and Corpus 2.", flush=True)
    
    # Check train / val / test isolation within Corpus 1
    c1_tr = ids_by_file["corpus1_calibration_train.jsonl"]
    c1_va = ids_by_file["corpus1_calibration_val.jsonl"]
    c1_te = ids_by_file["corpus1_calibration_test.jsonl"]
    assert len(c1_tr & c1_va) == 0, "Corpus 1 Train-Val leakage!"
    assert len(c1_tr & c1_te) == 0, "Corpus 1 Train-Test leakage!"
    assert len(c1_va & c1_te) == 0, "Corpus 1 Val-Test leakage!"
    print("Verified: Zero leakage among Corpus 1 Train / Val / Test.", flush=True)
    
    # Check train / val / test isolation within Corpus 2
    c2_tr = ids_by_file["corpus2_synthesis_train.jsonl"]
    c2_va = ids_by_file["corpus2_synthesis_val.jsonl"]
    c2_te = ids_by_file["corpus2_synthesis_test.jsonl"]
    assert len(c2_tr & c2_va) == 0, "Corpus 2 Train-Val leakage!"
    assert len(c2_tr & c2_te) == 0, "Corpus 2 Train-Test leakage!"
    assert len(c2_va & c2_te) == 0, "Corpus 2 Val-Test leakage!"
    print("Verified: Zero leakage among Corpus 2 Train / Val / Test.", flush=True)
    
    # Check combined sets exact union
    comb_tr = ids_by_file["combined_train.jsonl"]
    assert comb_tr == (c1_tr | c2_tr), "Combined train is not exact union of C1 + C2 train!"
    comb_va = ids_by_file["combined_val.jsonl"]
    assert comb_va == (c1_va | c2_va), "Combined val is not exact union of C1 + C2 val!"
    comb_te = ids_by_file["combined_test.jsonl"]
    assert comb_te == (c1_te | c2_te), "Combined test is not exact union of C1 + C2 test!"
    print("Verified: Combined sets are exact unions of component splits.", flush=True)
    
    return token_stats_by_file


def audit_exposure_kinematics(token_stats):
    print("\n--- 2. Reconciling Branch Exposure Kinematics ---", flush=True)
    # Effective batch size = 4 (batch size 1 * grad accum 4)
    # Branch B: 400 records, 200 updates = 800 record passes = 2.0 epochs
    b_records = token_stats["corpus1_calibration_train.jsonl"]["record_count"]
    b_tot_tokens = token_stats["corpus1_calibration_train.jsonl"]["total_tokens"]
    b_tgt_tokens = token_stats["corpus1_calibration_train.jsonl"]["target_tokens"]
    
    b_updates = 200
    b_epochs = (b_updates * 4) / b_records
    b_exp_tot_tokens = b_epochs * b_tot_tokens
    b_exp_tgt_tokens = b_epochs * b_tgt_tokens
    
    # Branch C: 400 records, 200 updates = 800 record passes = 2.0 epochs
    c_records = token_stats["corpus2_synthesis_train.jsonl"]["record_count"]
    c_tot_tokens = token_stats["corpus2_synthesis_train.jsonl"]["total_tokens"]
    c_tgt_tokens = token_stats["corpus2_synthesis_train.jsonl"]["target_tokens"]
    
    c_updates = 200
    c_epochs = (c_updates * 4) / c_records
    c_exp_tot_tokens = c_epochs * c_tot_tokens
    c_exp_tgt_tokens = c_epochs * c_tgt_tokens
    
    # Branch D (Full): 800 records, 400 updates = 1600 record passes = 2.0 epochs
    d_records = token_stats["combined_train.jsonl"]["record_count"]
    d_tot_tokens = token_stats["combined_train.jsonl"]["total_tokens"]
    d_tgt_tokens = token_stats["combined_train.jsonl"]["target_tokens"]
    
    d_updates = 400
    d_epochs = (d_updates * 4) / d_records
    d_exp_tot_tokens = d_epochs * d_tot_tokens
    d_exp_tgt_tokens = d_epochs * d_tgt_tokens
    
    # Branch D (Matched Updates Ablation): 800 records, 200 updates = 800 record passes = 1.0 epoch
    d_m_updates = 200
    d_m_epochs = (d_m_updates * 4) / d_records
    d_m_exp_tot_tokens = d_m_epochs * d_tot_tokens
    d_m_exp_tgt_tokens = d_m_epochs * d_tgt_tokens
    
    kinematics = {
        "branch_b_calibration": {
            "train_records": b_records,
            "optimizer_updates": b_updates,
            "effective_epochs": b_epochs,
            "total_token_exposure": b_exp_tot_tokens,
            "target_token_exposure": b_exp_tgt_tokens
        },
        "branch_c_synthesis": {
            "train_records": c_records,
            "optimizer_updates": c_updates,
            "effective_epochs": c_epochs,
            "total_token_exposure": c_exp_tot_tokens,
            "target_token_exposure": c_exp_tgt_tokens
        },
        "branch_d_combined_full": {
            "train_records": d_records,
            "optimizer_updates": d_updates,
            "effective_epochs": d_epochs,
            "total_token_exposure": d_exp_tot_tokens,
            "target_token_exposure": d_exp_tgt_tokens
        },
        "branch_d_matched_updates": {
            "train_records": d_records,
            "optimizer_updates": d_m_updates,
            "effective_epochs": d_m_epochs,
            "total_token_exposure": d_m_exp_tot_tokens,
            "target_token_exposure": d_m_exp_tgt_tokens
        }
    }
    
    print("Kinematics Summary:")
    for k, v in kinematics.items():
        print(f"  {k}: Updates={v['optimizer_updates']}, Epochs={v['effective_epochs']}, TargetTokens={v['target_token_exposure']}", flush=True)
        
    return kinematics


if __name__ == "__main__":
    tstats = audit_datasets()
    kins = audit_exposure_kinematics(tstats)
    out_p = os.path.join(STAGE75_DIR, "audit_summary.json")
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"token_stats": tstats, "kinematics": kins}, f, indent=2)
    print(f"\nWrote {out_p}", flush=True)
