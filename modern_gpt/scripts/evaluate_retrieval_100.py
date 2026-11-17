"""Rigorous 100-case retrieval evaluation on ModernGPT-S 100M 4K checkpoint (checkpoint_step_2000.pt).

Evaluates all 100 cases in data/sft_4k/intermediate/retrieval/eval_retrieval.jsonl:
- 20 cases at L=512
- 20 cases at L=1024
- 20 cases at L=2048
- 20 cases at L=3072
- 20 cases at L=4096
- 20 cases at depths [0.10, 0.25, 0.50, 0.75, 0.90]

Computes:
1. Strict Key-Value Match (the ground truth target fact value must appear in prediction)
2. Normalized Exact Match (full sentence equality)
3. Loose Substring Heuristic (the previous flawed metric that matched on question words)
4. Wilson score 95% confidence intervals
"""

from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.generate_retrieval_sft import EVAL_ENTITIES


def wilson_score_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Compute 95% Wilson score confidence interval."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * (total**2)))) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def run_retrieval_100_eval(ckpt_arg: Optional[str] = None, out_arg: Optional[str] = None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_path = REPO_ROOT / (ckpt_arg or "checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt")
    eval_path = REPO_ROOT / "data/sft_4k/intermediate/retrieval/eval_retrieval.jsonl"

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = state.get("model_config", {})
    cfg = ModelConfig(
        vocab_size=cfg_dict.get("vocab_size", 16000),
        context_length=cfg_dict.get("context_length", 4096),
        emb_dim=cfg_dict.get("emb_dim", 768),
        n_heads=cfg_dict.get("n_heads", 12),
        num_kv_heads=cfg_dict.get("num_kv_heads", 4),
        n_layers=cfg_dict.get("n_layers", 14),
        ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 2048),
        dropout=0.0,
        qkv_bias=False,
        rope_theta=10000.0,
        rope_scale=8.0,
        rope_type="linear",
    )
    model = ModernGPT(cfg).to(device=device, dtype=torch.bfloat16)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    tok = HuggingFaceBPEWrapper.load(REPO_ROOT / "artifacts/tokenizer_16k_v2")
    eos_id = tok.eos_token_id or 0

    records = [json.loads(l) for l in open(eval_path, "r", encoding="utf-8").read().splitlines()]
    print(f"Loaded {len(records)} evaluation cases.")

    # Build entity target lookup
    target_vals = {e["entity"]: e["val"] for e in EVAL_ENTITIES}

    key_matches = []
    exact_matches = []
    loose_matches = []

    by_len_key: Dict[int, List[int]] = {}
    by_depth_key: Dict[float, List[int]] = {}
    by_type_key: Dict[str, List[int]] = {}

    detailed_results = []

    print("\nExecuting greedy evaluation across all 100 cases...")
    t0 = time.time()

    for idx, r in enumerate(records):
        t_len = r["target_len"]
        t_depth = r["target_depth"]
        t_type = r.get("task_type", "unknown")
        msgs = r["messages"]
        context_msgs = msgs[:-1]
        expected_answer = msgs[-1]["content"].strip()

        # Format prompt in ChatML
        prompt_text = ""
        for m in context_msgs:
            prompt_text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"

        input_ids = tok.encode(prompt_text)
        if len(input_ids) > 3950:
            input_ids = input_ids[-3950:]

        inp_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)
        gen_tokens = []

        with torch.no_grad():
            for _ in range(64):
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(inp_tensor)
                    next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                tid = next_tok.item()
                if tid == eos_id:
                    break
                gen_tokens.append(tid)
                inp_tensor = torch.cat([inp_tensor, next_tok], dim=1)
                decoded = tok.decode(gen_tokens)
                if "<|im_end|>" in decoded:
                    break

        pred_raw = tok.decode(gen_tokens)
        pred_clean = pred_raw.replace("<|im_end|>", "").strip()

        # 1. Strict Key-Value Match: find the entity's key fact value
        entity_name = None
        for ent in EVAL_ENTITIES:
            if ent["entity"] in expected_answer or ent["entity"] in prompt_text:
                entity_name = ent["entity"]
                break

        target_value = target_vals.get(entity_name, "")
        # For multi-fact, also check dept or year
        key_match = False
        if target_value and target_value.lower() in pred_clean.lower():
            key_match = True
        elif t_type == "unanswerable" and "does not contain" in pred_clean.lower():
            key_match = True

        # 2. Normalized Exact Match
        exact_match = (re.sub(r"\s+", " ", pred_clean).lower() == re.sub(r"\s+", " ", expected_answer).lower())

        # 3. Loose Substring Heuristic (from previous evaluate_sft_4k.py)
        loose_match = False
        for part in expected_answer.split():
            if len(part) >= 4 and part.lower() in pred_clean.lower():
                loose_match = True
                break

        key_int = 1 if key_match else 0
        exact_int = 1 if exact_match else 0
        loose_int = 1 if loose_match else 0

        key_matches.append(key_int)
        exact_matches.append(exact_int)
        loose_matches.append(loose_int)

        by_len_key.setdefault(t_len, []).append(key_int)
        by_depth_key.setdefault(t_depth, []).append(key_int)
        by_type_key.setdefault(t_type, []).append(key_int)

        detailed_results.append({
            "id": r.get("id", f"case_{idx}"),
            "target_len": t_len,
            "target_depth": t_depth,
            "task_type": t_type,
            "entity": entity_name,
            "target_value": target_value,
            "expected_answer": expected_answer,
            "prediction": pred_clean,
            "key_match": key_match,
            "exact_match": exact_match,
            "loose_match": loose_match,
        })

    eval_duration = time.time() - t0
    total_cases = len(records)
    key_acc = sum(key_matches) / total_cases
    exact_acc = sum(exact_matches) / total_cases
    loose_acc = sum(loose_matches) / total_cases

    ci_key = wilson_score_interval(sum(key_matches), total_cases)
    ci_loose = wilson_score_interval(sum(loose_matches), total_cases)

    len_summary = {}
    for l in sorted(by_len_key.keys()):
        succ = sum(by_len_key[l])
        tot = len(by_len_key[l])
        len_summary[l] = {
            "successes": succ,
            "total": tot,
            "accuracy": round(succ / tot, 4),
            "ci_95": wilson_score_interval(succ, tot),
        }

    depth_summary = {}
    for d in sorted(by_depth_key.keys()):
        succ = sum(by_depth_key[d])
        tot = len(by_depth_key[d])
        depth_summary[d] = {
            "successes": succ,
            "total": tot,
            "accuracy": round(succ / tot, 4),
            "ci_95": wilson_score_interval(succ, tot),
        }

    type_summary = {}
    for t in sorted(by_type_key.keys()):
        succ = sum(by_type_key[t])
        tot = len(by_type_key[t])
        type_summary[t] = {
            "successes": succ,
            "total": tot,
            "accuracy": round(succ / tot, 4),
        }

    print("\n=== Rigorous 100-Case Evaluation Results ===")
    print(f"Total Evaluated:                   {total_cases}")
    print(f"Strict Key-Value Accuracy:        {key_acc*100:.1f}% ({sum(key_matches)}/{total_cases}) [95% CI: {ci_key[0]*100:.1f}% - {ci_key[1]*100:.1f}%]")
    print(f"Loose Substring Heuristic:        {loose_acc*100:.1f}% ({sum(loose_matches)}/{total_cases}) [95% CI: {ci_loose[0]*100:.1f}% - {ci_loose[1]*100:.1f}%]")
    print(f"Normalized Exact Match Accuracy:  {exact_acc*100:.1f}% ({sum(exact_matches)}/{total_cases})")
    print(f"Time Taken:                       {eval_duration:.1f} seconds")

    print("\nAccuracy by Context Length (Strict Key-Value Match):")
    for l, stats in len_summary.items():
        print(f"  Length {l:4d}: {stats['accuracy']*100:5.1f}% ({stats['successes']:2d}/{stats['total']:2d}) [95% CI: {stats['ci_95'][0]*100:.1f}% - {stats['ci_95'][1]*100:.1f}%]")

    print("\nAccuracy by Depth (Strict Key-Value Match):")
    for d, stats in depth_summary.items():
        print(f"  Depth {d:4.2f}: {stats['accuracy']*100:5.1f}% ({stats['successes']:2d}/{stats['total']:2d}) [95% CI: {stats['ci_95'][0]*100:.1f}% - {stats['ci_95'][1]*100:.1f}%]")

    print("\nAccuracy by Task Type (Strict Key-Value Match):")
    for t, stats in type_summary.items():
        print(f"  {t:18s}: {stats['accuracy']*100:5.1f}% ({stats['successes']:2d}/{stats['total']:2d})")

    report_data = {
        "checkpoint": str(ckpt_path),
        "total_cases": total_cases,
        "eval_duration_sec": round(eval_duration, 2),
        "strict_key_accuracy": round(key_acc, 4),
        "strict_key_ci_95": ci_key,
        "loose_heuristic_accuracy": round(loose_acc, 4),
        "loose_heuristic_ci_95": ci_loose,
        "exact_match_accuracy": round(exact_acc, 4),
        "by_length": len_summary,
        "by_depth": depth_summary,
        "by_task_type": type_summary,
        "detailed_results": detailed_results,
    }

    out_file = REPO_ROOT / (out_arg or "artifacts/diagnostics/retrieval_100_eval_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\nSaved 100-case evaluation report to {out_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_retrieval_100_eval(args.checkpoint, args.output)

