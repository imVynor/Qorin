import ast
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))

from modern_gpt.model.config import ModelConfig
from modern_gpt.model.gpt import ModernGPT
from modern_gpt.scripts.phase1d_benchmark_data import (
    MATH_TASKS,
    TOOL_TASKS,
    RETRIEVAL_QUERIES,
    ATTRACTORS
)
from modern_gpt.scripts.evaluate_sft_benchmarks import CODING_TASKS, JSON_TASKS

# Destination report directories
EVAL_DIR = REPO_ROOT / "reports" / "phase1d_500m_evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

CKPT_500M_PATH = REPO_ROOT / "checkpoints" / "checkpoint_501M" / "ckpt_501m_tokens_501678080.pt"
TOK_32K_PATH = REPO_ROOT / "artifacts" / "tokenizer_32k" / "tokenizer.json"
VAL_DIR = REPO_ROOT / "data_500m" / "final_5b" / "val"
BEHAVIORAL_PATH = REPO_ROOT / "archive" / "100m_baseline" / "evaluations" / "behavioral_eval_100.jsonl"
CKPT_100M_PATH = REPO_ROOT / "archive" / "100m_baseline" / "checkpoints" / "base_100m_best_val.pt"
TOK_16K_PATH = REPO_ROOT / "archive" / "100m_baseline" / "tokenizer" / "tokenizer.json"


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def generate_text(
    model: nn.Module,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.15,
    stop_tokens: Optional[List[int]] = None,
    device: str = "cuda"
) -> Tuple[str, List[int], float]:
    t0 = time.perf_counter()
    input_ids = tok.encode(prompt).ids
    if len(input_ids) == 0:
        input_ids = [9] # fallback
    
    generated = list(input_ids)
    stop_set = set(stop_tokens or [9, 5]) # <|endoftext|>, <|im_end|>

    for _ in range(max_new_tokens):
        # Truncate context to last 2048 if needed
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
            next_logits = logits[0, -1, :].clone().float()
        
        # Apply repetition penalty
        if repetition_penalty > 1.0:
            for tid in set(generated[-64:]):
                if next_logits[tid] > 0:
                    next_logits[tid] /= repetition_penalty
                else:
                    next_logits[tid] *= repetition_penalty

        if temperature <= 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            scaled = next_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cum_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                scaled[indices_to_remove] = -float("Inf")
            probs = F.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id in stop_set:
            break
        generated.append(next_id)

    new_ids = generated[len(input_ids):]
    decoded = tok.decode(new_ids)
    elapsed = time.perf_counter() - t0
    return decoded, new_ids, elapsed


def evaluate_validation_loss(model: nn.Module, device: str = "cuda") -> Dict[str, Any]:
    print("\n--- 1. Evaluating Validation Loss across 7 Domains ---")
    domains = ["web", "edu", "code", "wiki", "math", "books", "dialogue"]
    domain_weights = {"web": 0.40, "edu": 0.15, "code": 0.15, "wiki": 0.075, "math": 0.075, "books": 0.10, "dialogue": 0.05}
    
    results = {}
    weighted_loss_sum = 0.0
    total_tokens_evaluated = 0

    early_losses = []
    late_losses = []

    for d in domains:
        shard_path = VAL_DIR / d / "shard_00000.bin"
        if not shard_path.exists():
            print(f"Warning: Shard not found for {d}")
            continue
        
        file_size = shard_path.stat().st_size
        num_tokens = (file_size - 80) // 2
        mmap = np.memmap(shard_path, dtype=np.uint16, mode="r", offset=80, shape=(num_tokens,))
        
        # Evaluate 25 contiguous sequences of 2048 tokens
        seq_len = 2048
        num_eval_seqs = min(25, num_tokens // (seq_len + 1))
        losses = []

        with torch.no_grad():
            for i in range(num_eval_seqs):
                offset = i * seq_len
                seq = mmap[offset : offset + seq_len + 1].astype(np.int64)
                x = torch.from_numpy(seq[:-1]).unsqueeze(0).to(device)
                y = torch.from_numpy(seq[1:]).unsqueeze(0).to(device)
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, targets=y)
                
                losses.append(loss.item())

                # Per-token position loss breakdown
                loss_per_token = F.cross_entropy(logits[0], y[0], reduction="none")
                early_losses.append(loss_per_token[:512].mean().item())
                late_losses.append(loss_per_token[-512:].mean().item())

        mean_loss = float(np.mean(losses))
        ppl = float(math.exp(mean_loss))
        tok_count = num_eval_seqs * seq_len
        total_tokens_evaluated += tok_count

        results[d] = {
            "loss": round(mean_loss, 4),
            "perplexity": round(ppl, 2),
            "tokens_evaluated": tok_count,
            "weight": domain_weights.get(d, 0.0),
        }
        weighted_loss_sum += mean_loss * domain_weights.get(d, 0.0)
        print(f"  Domain: {d:<10} | CE Loss: {mean_loss:.4f} | Perplexity: {ppl:8.2f} | Tokens: {tok_count:,}")

    overall_ce = round(weighted_loss_sum, 4)
    overall_ppl = round(math.exp(overall_ce), 2)
    early_ce = round(float(np.mean(early_losses)), 4)
    late_ce = round(float(np.mean(late_losses)), 4)

    print(f"  >> OVERALL WEIGHTED LOSS: {overall_ce:.4f} | OVERALL PERPLEXITY: {overall_ppl:.2f}")
    print(f"  >> Positional Loss: First 512 tokens = {early_ce:.4f} | Last 512 tokens = {late_ce:.4f}")

    return {
        "overall_loss": overall_ce,
        "overall_perplexity": overall_ppl,
        "total_tokens_evaluated": total_tokens_evaluated,
        "early_tokens_loss": early_ce,
        "late_tokens_loss": late_ce,
        "domains": results
    }


def evaluate_behavioral_suite(model: nn.Module, tok: Tokenizer, device: str = "cuda") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print("\n--- 2. Evaluating Behavioral & Multi-Turn Suite (100 Cases) ---")
    raw_lines = [json.loads(l) for l in BEHAVIORAL_PATH.read_text(encoding="utf-8").strip().split("\n")]
    
    total = len(raw_lines)
    isolated_correct = 0
    multiturn_correct = 0
    attractor_hits = 0
    contamination_hits = 0

    category_stats = defaultdict(lambda: {"total": 0, "iso_correct": 0, "multi_correct": 0, "attractor": 0})
    records = []

    for item in raw_lines:
        cid = item["id"]
        cat = item.get("category", "general")
        exp = item.get("expected_answer", "").strip().lower()
        neg_triggers = [t.lower() for t in item.get("negative_triggers", [])]
        req_entities = [e.lower() for e in item.get("required_entities", [])]

        # 1. Isolated evaluation
        iso_val = item.get("isolated_prompt")
        if isinstance(iso_val, list) and len(iso_val) > 0 and isinstance(iso_val[0], dict):
            iso_prompt = iso_val[0].get("content", "")
        elif isinstance(iso_val, str):
            iso_prompt = iso_val
        elif "multi_turn_messages" in item and len(item["multi_turn_messages"]) > 0:
            iso_prompt = item["multi_turn_messages"][-1].get("content", "")
        else:
            iso_prompt = str(iso_val)

        iso_text, _, _ = generate_text(model, tok, iso_prompt, max_new_tokens=64, temperature=0.0, device=device)
        iso_clean = iso_text.lower().strip()

        # Check correctness
        iso_pass = True
        if exp and exp not in iso_clean:
            iso_pass = False
        for neg in neg_triggers:
            if neg in iso_clean:
                iso_pass = False
        for req in req_entities:
            if req not in iso_clean:
                iso_pass = False
        
        # 2. Multi-turn evaluation
        # Format turns into continuous context
        history = ""
        for m in item["multi_turn_messages"]:
            history += f"{m['role'].capitalize()}: {m['content']}\n"
        history += "Assistant: "
        multi_text, _, _ = generate_text(model, tok, history, max_new_tokens=64, temperature=0.0, device=device)
        multi_clean = multi_text.lower().strip()

        multi_pass = True
        if exp and exp not in multi_clean:
            multi_pass = False
        for neg in neg_triggers:
            if neg in multi_clean:
                multi_pass = False
                contamination_hits += 1
        for req in req_entities:
            if req not in multi_clean:
                multi_pass = False

        has_attractor = any(a in iso_clean or a in multi_clean for a in ATTRACTORS)
        if has_attractor:
            attractor_hits += 1

        if iso_pass:
            isolated_correct += 1
            category_stats[cat]["iso_correct"] += 1
        if multi_pass:
            multiturn_correct += 1
            category_stats[cat]["multi_correct"] += 1
        
        category_stats[cat]["total"] += 1
        if has_attractor:
            category_stats[cat]["attractor"] += 1

        records.append({
            "id": cid,
            "category": cat,
            "isolated_prompt": iso_prompt,
            "isolated_output": iso_text,
            "isolated_pass": iso_pass,
            "multiturn_output": multi_text,
            "multiturn_pass": multi_pass,
            "attractor": has_attractor
        })

    cat_summary = {}
    for c, s in category_stats.items():
        tot = s["total"]
        cat_summary[c] = {
            "total": tot,
            "isolated_accuracy_pct": round(s["iso_correct"] / tot * 100, 1),
            "multiturn_accuracy_pct": round(s["multi_correct"] / tot * 100, 1),
            "delta_history_pct": round((s["iso_correct"] - s["multi_correct"]) / tot * 100, 1),
            "attractor_rate_pct": round(s["attractor"] / tot * 100, 1),
        }

    summary = {
        "total_cases": total,
        "overall_isolated_accuracy_pct": round(isolated_correct / total * 100, 1),
        "overall_multiturn_accuracy_pct": round(multiturn_correct / total * 100, 1),
        "overall_delta_history_pct": round((isolated_correct - multiturn_correct) / total * 100, 1),
        "overall_contamination_rate_pct": round(contamination_hits / total * 100, 1),
        "overall_attractor_rate_pct": round(attractor_hits / total * 100, 1),
        "by_category": cat_summary
    }

    print(f"  Isolated Accuracy: {summary['overall_isolated_accuracy_pct']}% ({isolated_correct}/{total})")
    print(f"  Multi-Turn Accuracy: {summary['overall_multiturn_accuracy_pct']}% ({multiturn_correct}/{total})")
    print(f"  Attractor Rate: {summary['overall_attractor_rate_pct']}%")

    return summary, records


def evaluate_coding_suite(model: nn.Module, tok: Tokenizer, device: str = "cuda") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print("\n--- 3. Evaluating Coding Benchmark (50 Tasks) ---")
    ast_valid_count = 0
    functional_pass_count = 0
    records = []

    for task in CODING_TASKS:
        tid = task["id"]
        prompt = f"Write Python code to solve this:\n{task['prompt']}\n```python\n"
        gen_text, _, _ = generate_text(model, tok, prompt, max_new_tokens=160, temperature=0.0, device=device)
        
        # Extract code block if markdown used, or raw text
        code = gen_text
        if "```" in gen_text:
            m = re.search(r"```(?:python)?\s*(.*?)\s*```", gen_text, re.DOTALL)
            if m:
                code = m.group(1)
            else:
                code = gen_text.split("```")[0]

        # 1. AST syntax check
        ast_valid = False
        try:
            ast.parse(code)
            ast_valid = True
            ast_valid_count += 1
        except Exception:
            ast_valid = False

        # 2. Execution check in safe local scope
        func_pass = False
        exec_err = None
        if ast_valid:
            test_stmt = task["test"]
            full_test_code = f"{code}\n\n{test_stmt}\n"
            safe_globals = {"__builtins__": {"range": range, "len": len, "sum": sum, "min": min, "max": max, "abs": abs, "str": str, "int": int, "float": float, "list": list, "dict": dict, "set": set, "bool": bool, "True": True, "False": False, "None": None}}
            try:
                exec(full_test_code, safe_globals)
                func_pass = True
                functional_pass_count += 1
            except Exception as e:
                exec_err = str(e)

        records.append({
            "id": tid,
            "prompt": task["prompt"],
            "generated": gen_text,
            "extracted_code": code,
            "ast_valid": ast_valid,
            "functional_pass": func_pass,
            "error": exec_err
        })

    summary = {
        "total_tasks": len(CODING_TASKS),
        "ast_valid_count": ast_valid_count,
        "ast_valid_pct": round(ast_valid_count / len(CODING_TASKS) * 100, 1),
        "functional_pass_count": functional_pass_count,
        "functional_pass_pct": round(functional_pass_count / len(CODING_TASKS) * 100, 1)
    }

    print(f"  AST Valid: {summary['ast_valid_pct']}% ({ast_valid_count}/{len(CODING_TASKS)})")
    print(f"  Functional Pass@1: {summary['functional_pass_pct']}% ({functional_pass_count}/{len(CODING_TASKS)})")

    return summary, records


def evaluate_json_suite(model: nn.Module, tok: Tokenizer, device: str = "cuda") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print("\n--- 4. Evaluating Structured JSON Benchmark (50 Tasks) ---")
    json_valid_count = 0
    schema_valid_count = 0
    records = []

    for task in JSON_TASKS:
        tid = task["id"]
        prompt = f"Output a valid JSON object only.\n{task['prompt']}\n```json\n"
        gen_text, _, _ = generate_text(model, tok, prompt, max_new_tokens=140, temperature=0.0, device=device)

        # Extract JSON
        json_str = gen_text.strip()
        if "```" in gen_text:
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", gen_text, re.DOTALL)
            if m:
                json_str = m.group(1)
            else:
                m2 = re.search(r"\{.*?\}", gen_text, re.DOTALL)
                if m2:
                    json_str = m2.group(0)
        else:
            m = re.search(r"\{.*?\}", gen_text, re.DOTALL)
            if m:
                json_str = m.group(0)

        json_valid = False
        schema_valid = False
        parsed_obj = None
        err = None

        try:
            parsed_obj = json.loads(json_str)
            json_valid = True
            json_valid_count += 1
            
            # Check required keys
            req_keys = task.get("required_keys", [])
            if all(k in parsed_obj for k in req_keys):
                schema_valid = True
                schema_valid_count += 1
        except Exception as e:
            err = str(e)

        records.append({
            "id": tid,
            "prompt": task["prompt"],
            "generated": gen_text,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "error": err
        })

    summary = {
        "total_tasks": len(JSON_TASKS),
        "json_valid_count": json_valid_count,
        "json_valid_pct": round(json_valid_count / len(JSON_TASKS) * 100, 1),
        "schema_valid_count": schema_valid_count,
        "schema_valid_pct": round(schema_valid_count / len(JSON_TASKS) * 100, 1)
    }

    print(f"  JSON Syntax Valid: {summary['json_valid_pct']}% ({json_valid_count}/{len(JSON_TASKS)})")
    print(f"  Schema Conforming: {summary['schema_valid_pct']}% ({schema_valid_count}/{len(JSON_TASKS)})")

    return summary, records


def evaluate_math_suite(model: nn.Module, tok: Tokenizer, device: str = "cuda") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print("\n--- 5. Evaluating Mathematical Reasoning (50 Tasks) ---")
    correct_count = 0
    records = []
    by_type = defaultdict(lambda: {"total": 0, "correct": 0})

    for task in MATH_TASKS:
        tid = task["id"]
        mtype = task["type"]
        exp = task["expected"]
        prompt = task["prompt"]

        gen_text, _, _ = generate_text(model, tok, prompt, max_new_tokens=48, temperature=0.0, device=device)
        
        # Check if expected number appears in answer
        is_correct = False
        exp_str = str(exp)
        # Search numbers in output
        numbers_found = re.findall(r"[-+]?\d*\.\d+|\d+", gen_text)
        if numbers_found:
            first_num = numbers_found[0]
            if first_num == exp_str or (abs(float(first_num) - float(exp)) < 1e-4 if '.' in exp_str else False):
                is_correct = True
            elif exp_str in gen_text[:20]:
                is_correct = True
        
        if is_correct:
            correct_count += 1
            by_type[mtype]["correct"] += 1
        by_type[mtype]["total"] += 1

        records.append({
            "id": tid,
            "type": mtype,
            "prompt": prompt,
            "expected": exp,
            "generated": gen_text,
            "correct": is_correct
        })

    type_stats = {}
    for t, s in by_type.items():
        type_stats[t] = {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy_pct": round(s["correct"] / s["total"] * 100, 1)
        }

    summary = {
        "total_tasks": len(MATH_TASKS),
        "correct_count": correct_count,
        "accuracy_pct": round(correct_count / len(MATH_TASKS) * 100, 1),
        "by_type": type_stats
    }

    print(f"  Math Accuracy: {summary['accuracy_pct']}% ({correct_count}/{len(MATH_TASKS)})")
    return summary, records


def evaluate_tool_calling(model: nn.Module, tok: Tokenizer, device: str = "cuda") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    print("\n--- 6. Evaluating Tool-Calling & Intent Recognition (20 Tasks) ---")
    tool_triggered_count = 0
    correct_intent_count = 0
    records = []

    for task in TOOL_TASKS:
        tid = task["id"]
        prompt = f"<|im_start|>user\n{task['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        gen_text, _, _ = generate_text(model, tok, prompt, max_new_tokens=64, temperature=0.0, device=device)

        has_tool_tag = "<|tool_call|>" in gen_text or "<tool_call>" in gen_text or "tool:" in gen_text.lower()
        if has_tool_tag:
            tool_triggered_count += 1

        expected = task["expected_tool"]
        matched_intent = False
        if expected is None:
            if not has_tool_tag:
                matched_intent = True
        else:
            if expected in gen_text.lower():
                matched_intent = True

        if matched_intent:
            correct_intent_count += 1

        records.append({
            "id": tid,
            "prompt": task["prompt"],
            "expected_tool": expected,
            "generated": gen_text,
            "has_tool_tag": has_tool_tag,
            "matched_intent": matched_intent
        })

    summary = {
        "total_tasks": len(TOOL_TASKS),
        "tool_tag_count": tool_triggered_count,
        "tool_tag_pct": round(tool_triggered_count / len(TOOL_TASKS) * 100, 1),
        "intent_accuracy_pct": round(correct_intent_count / len(TOOL_TASKS) * 100, 1),
        "note": "ModernGPT-S 500M is a base pretrained checkpoint; formal tool invocation syntax is learned during supervised instruction fine-tuning."
    }

    print(f"  Tool Intent Recognition: {summary['intent_accuracy_pct']}% | Tool Tags Emitted: {tool_triggered_count}")
    return summary, records


def evaluate_generation_diagnostics(model: nn.Module, tok: Tokenizer, device: str = "cuda") -> Dict[str, Any]:
    print("\n--- 7. Evaluating Generation Diagnostics across Decoding Regimes ---")
    prompts = [
        "In artificial intelligence, deep learning models",
        "The history of modern computing began with",
        "To implement a binary search tree in Python,",
        "Quantum mechanics is the branch of physics that",
        "The primary advantage of grouped-query attention is",
    ]

    configs = [
        {"name": "greedy", "temp": 0.0, "top_p": 1.0},
        {"name": "conservative", "temp": 0.3, "top_p": 0.9},
        {"name": "creative", "temp": 0.7, "top_p": 0.9},
    ]

    regime_results = {}

    for cfg in configs:
        cname = cfg["name"]
        lengths = []
        distinct_1_list = []
        distinct_2_list = []
        repetition_loop_count = 0

        for p in prompts:
            text, ids, _ = generate_text(
                model, tok, p,
                max_new_tokens=100,
                temperature=cfg["temp"],
                top_p=cfg["top_p"],
                repetition_penalty=1.0 if cname == "greedy" else 1.15,
                device=device
            )
            lengths.append(len(ids))

            words = text.split()
            if len(words) > 0:
                d1 = len(set(words)) / len(words)
                distinct_1_list.append(d1)
            if len(words) > 1:
                bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
                d2 = len(set(bigrams)) / len(bigrams)
                distinct_2_list.append(d2)

            # Check for repetition loops (e.g. phrase repeated 3+ times)
            if any(words[i:i+4] == words[i+4:i+8] and len(words) > 12 for i in range(len(words)-8)):
                repetition_loop_count += 1

        regime_results[cname] = {
            "mean_length": round(float(np.mean(lengths)), 1),
            "distinct_1": round(float(np.mean(distinct_1_list)), 3) if distinct_1_list else 0.0,
            "distinct_2": round(float(np.mean(distinct_2_list)), 3) if distinct_2_list else 0.0,
            "repetition_loop_rate_pct": round(repetition_loop_count / len(prompts) * 100, 1),
        }
        print(f"  Regime: {cname:<12} | Mean Len: {regime_results[cname]['mean_length']} | Distinct-1: {regime_results[cname]['distinct_1']} | Distinct-2: {regime_results[cname]['distinct_2']}")

    return regime_results


def evaluate_retrieval_matrix(
    model_500m: nn.Module,
    tok_32k: Tokenizer,
    model_100m: nn.Module,
    tok_16k: Tokenizer,
    device: str = "cuda"
) -> Dict[str, Any]:
    print("\n--- 8. Evaluating 4-Way Retrieval Augmentation Matrix (A, B, C, D) ---")
    from backend.retrieval.engine import RetrievalEngine
    engine = RetrievalEngine()

    matrix_scores = {"A_100M_Off": 0, "B_100M_On": 0, "C_500M_Off": 0, "D_500M_On": 0}
    records = []

    for item in RETRIEVAL_QUERIES:
        qid = item["id"]
        q = item["query"]
        terms = item["key_terms"]

        # Retrieve context
        trace = engine.retrieve(query=q, mode="always", top_k=2)
        ctx = trace.injected_context

        # Condition A: 100M Off
        out_a, _, _ = generate_text(model_100m, tok_16k, f"Question: {q}\nAnswer:", max_new_tokens=48, temperature=0.0, device=device)
        score_a = any(t.lower() in out_a.lower() for t in terms)

        # Condition B: 100M On
        prompt_b = f"Context:\n{ctx}\nQuestion: {q}\nAnswer based on context:"
        out_b, _, _ = generate_text(model_100m, tok_16k, prompt_b, max_new_tokens=48, temperature=0.0, device=device)
        score_b = any(t.lower() in out_b.lower() for t in terms)

        # Condition C: 500M Off
        out_c, _, _ = generate_text(model_500m, tok_32k, f"Question: {q}\nAnswer:", max_new_tokens=48, temperature=0.0, device=device)
        score_c = any(t.lower() in out_c.lower() for t in terms)

        # Condition D: 500M On
        prompt_d = f"Context:\n{ctx}\nQuestion: {q}\nAnswer based on context:"
        out_d, _, _ = generate_text(model_500m, tok_32k, prompt_d, max_new_tokens=48, temperature=0.0, device=device)
        score_d = any(t.lower() in out_d.lower() for t in terms)

        if score_a: matrix_scores["A_100M_Off"] += 1
        if score_b: matrix_scores["B_100M_On"] += 1
        if score_c: matrix_scores["C_500M_Off"] += 1
        if score_d: matrix_scores["D_500M_On"] += 1

        records.append({
            "id": qid,
            "query": q,
            "100m_off": {"out": out_a, "score": score_a},
            "100m_on": {"out": out_b, "score": score_b},
            "500m_off": {"out": out_c, "score": score_c},
            "500m_on": {"out": out_d, "score": score_d},
        })

    total_q = len(RETRIEVAL_QUERIES)
    matrix_pct = {k: round(v / total_q * 100, 1) for k, v in matrix_scores.items()}

    print(f"  Matrix Results (Accuracy %):")
    print(f"    Condition A (100M Baseline, RAG Off): {matrix_pct['A_100M_Off']}% ({matrix_scores['A_100M_Off']}/{total_q})")
    print(f"    Condition B (100M Baseline, RAG On) : {matrix_pct['B_100M_On']}% ({matrix_scores['B_100M_On']}/{total_q})")
    print(f"    Condition C (500M Model,    RAG Off): {matrix_pct['C_500M_Off']}% ({matrix_scores['C_500M_Off']}/{total_q})")
    print(f"    Condition D (500M Model,    RAG On) : {matrix_pct['D_500M_On']}% ({matrix_scores['D_500M_On']}/{total_q})")

    return {
        "total_queries": total_q,
        "matrix_scores": matrix_scores,
        "matrix_accuracy_pct": matrix_pct,
        "records": records
    }


def main():
    print("=" * 70)
    print(" MODERN-GPT-S 500M — PHASE 1D COMPREHENSIVE EVALUATION SUITE")
    print("=" * 70)
    start_time = datetime.now()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Target: {device.upper()}")
    if device == "cuda":
        print(f"Device Name: {torch.cuda.get_device_name(0)}")

    # 1. Load Tokenizer & Model 500M
    print(f"\nLoading Tokenizer 32K from {TOK_32K_PATH}...")
    tok_32k = Tokenizer.from_file(str(TOK_32K_PATH))
    tok_32k_sha256 = compute_sha256(TOK_32K_PATH)
    print(f"Tokenizer SHA-256: {tok_32k_sha256}")

    print(f"\nLoading 500M Checkpoint from {CKPT_500M_PATH}...")
    ckpt_500m_sha256 = compute_sha256(CKPT_500M_PATH)
    print(f"Checkpoint SHA-256: {ckpt_500m_sha256}")

    ckpt_500m_data = torch.load(CKPT_500M_PATH, map_location="cpu", weights_only=False)
    cfg_500m = ModelConfig(**ckpt_500m_data["config"])
    model_500m = ModernGPT(cfg_500m).to(device, dtype=torch.bfloat16)
    model_500m.load_state_dict({k: v.to(device, dtype=torch.bfloat16) for k, v in ckpt_500m_data["model_state_dict"].items()})
    model_500m.eval()
    print("500M Model successfully loaded into evaluation mode.")

    # 2. Load Tokenizer & Model 100M Baseline
    print(f"\nLoading 100M Baseline from {CKPT_100M_PATH}...")
    tok_16k = Tokenizer.from_file(str(TOK_16K_PATH))
    ckpt_100m_data = torch.load(CKPT_100M_PATH, map_location="cpu", weights_only=False)
    
    # 100M config
    cfg_100m = ModelConfig.get_100m_gqa4_config(vocab_size=16000, context_length=512)
    model_100m = ModernGPT(cfg_100m).to(device, dtype=torch.bfloat16)
    st_100m = ckpt_100m_data.get("model", ckpt_100m_data.get("model_state_dict", ckpt_100m_data))
    clean_st_100m = {k: v.to(device, dtype=torch.bfloat16) for k, v in st_100m.items() if isinstance(v, torch.Tensor)}
    model_100m.load_state_dict(clean_st_100m, strict=False)
    model_100m.eval()
    print("100M Baseline successfully loaded.")

    # Execute Evaluations
    val_loss_results = evaluate_validation_loss(model_500m, device=device)
    beh_results, beh_records = evaluate_behavioral_suite(model_500m, tok_32k, device=device)
    code_results, code_records = evaluate_coding_suite(model_500m, tok_32k, device=device)
    json_results, json_records = evaluate_json_suite(model_500m, tok_32k, device=device)
    math_results, math_records = evaluate_math_suite(model_500m, tok_32k, device=device)
    tool_results, tool_records = evaluate_tool_calling(model_500m, tok_32k, device=device)
    gen_diag_results = evaluate_generation_diagnostics(model_500m, tok_32k, device=device)
    rag_results = evaluate_retrieval_matrix(model_500m, tok_32k, model_100m, tok_16k, device=device)

    # Master summary compilation
    end_time = datetime.now()
    duration_sec = (end_time - start_time).total_seconds()

    master_results = {
        "evaluation_name": "ModernGPT-S 500M — Phase 1D Evaluation Suite",
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "duration_seconds": round(duration_sec, 2),
        "target_model": {
            "name": "ModernGPT-S 500M (500M Pretrain Milestone)",
            "checkpoint_path": str(CKPT_500M_PATH),
            "checkpoint_sha256": ckpt_500m_sha256,
            "parameters_trainable": 514389504,
            "parameters_total_tensor_elements": 569964032,
            "architecture": "20L-24Q-4KV GQA (6:1), SwiGLU, RMSNorm, RoPE",
            "context_length": 2048,
            "vocab_size": 32768,
            "tokenizer_sha256": tok_32k_sha256,
            "tokens_trained": ckpt_500m_data.get("total_tokens_processed", 501678080),
            "global_step": ckpt_500m_data.get("global_update", 15310)
        },
        "baseline_100m": {
            "name": "Base ModernGPT-S 100M Pretrained",
            "checkpoint_path": str(CKPT_100M_PATH),
            "parameters": 104807424,
            "context_length": 512,
            "vocab_size": 16000,
            "tokens_trained": 1499987968
        },
        "validation_loss": val_loss_results,
        "behavioral_evaluation": beh_results,
        "coding_benchmark": code_results,
        "json_benchmark": json_results,
        "math_benchmark": math_results,
        "tool_calling_benchmark": tool_results,
        "generation_diagnostics": gen_diag_results,
        "retrieval_matrix": {
            "scores": rag_results["matrix_scores"],
            "accuracy_pct": rag_results["matrix_accuracy_pct"]
        }
    }

    # Write results.json
    results_json_path = EVAL_DIR / "results.json"
    results_json_path.write_text(json.dumps(master_results, indent=2), encoding="utf-8")
    print(f"\nWrote master results to {results_json_path}")

    # Write evaluation_config.json
    eval_cfg = {
        "checkpoint": str(CKPT_500M_PATH),
        "checkpoint_sha256": ckpt_500m_sha256,
        "tokenizer": str(TOK_32K_PATH),
        "tokenizer_sha256": tok_32k_sha256,
        "device": device,
        "precision": "bfloat16",
        "decoding": {"max_new_tokens": 128, "temperature": 0.0, "repetition_penalty": 1.15},
        "benchmarks": ["validation_loss", "behavioral_100", "coding_50", "json_50", "math_50", "tools_20", "diagnostics", "retrieval_matrix"]
    }
    (EVAL_DIR / "evaluation_config.json").write_text(json.dumps(eval_cfg, indent=2), encoding="utf-8")

    # Write raw_generations.jsonl
    raw_path = EVAL_DIR / "raw_generations.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for r in beh_records:
            f.write(json.dumps({"suite": "behavioral", **r}) + "\n")
        for r in code_records:
            f.write(json.dumps({"suite": "coding", **r}) + "\n")
        for r in json_records:
            f.write(json.dumps({"suite": "json", **r}) + "\n")
        for r in math_records:
            f.write(json.dumps({"suite": "math", **r}) + "\n")
        for r in tool_records:
            f.write(json.dumps({"suite": "tools", **r}) + "\n")
        for r in rag_results["records"]:
            f.write(json.dumps({"suite": "retrieval", **r}) + "\n")
    print(f"Wrote raw generations to {raw_path}")

    # Write failure_cases.jsonl
    failure_path = EVAL_DIR / "failure_cases.jsonl"
    with open(failure_path, "w", encoding="utf-8") as f:
        for r in beh_records:
            if not r["isolated_pass"] or not r["multiturn_pass"]:
                f.write(json.dumps({"suite": "behavioral", **r}) + "\n")
        for r in code_records:
            if not r["functional_pass"]:
                f.write(json.dumps({"suite": "coding", **r}) + "\n")
        for r in json_records:
            if not r["schema_valid"]:
                f.write(json.dumps({"suite": "json", **r}) + "\n")
        for r in math_records:
            if not r["correct"]:
                f.write(json.dumps({"suite": "math", **r}) + "\n")
    print(f"Wrote failure cases to {failure_path}")

    # Write results.csv
    csv_path = EVAL_DIR / "results.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("benchmark,metric,value,unit\n")
        f.write(f"validation_loss,overall_loss,{val_loss_results['overall_loss']},ce_loss\n")
        f.write(f"validation_loss,overall_ppl,{val_loss_results['overall_perplexity']},ppl\n")
        f.write(f"behavioral,isolated_accuracy,{beh_results['overall_isolated_accuracy_pct']},percent\n")
        f.write(f"behavioral,multiturn_accuracy,{beh_results['overall_multiturn_accuracy_pct']},percent\n")
        f.write(f"behavioral,contamination_rate,{beh_results['overall_contamination_rate_pct']},percent\n")
        f.write(f"behavioral,attractor_rate,{beh_results['overall_attractor_rate_pct']},percent\n")
        f.write(f"coding,ast_valid,{code_results['ast_valid_pct']},percent\n")
        f.write(f"coding,functional_pass,{code_results['functional_pass_pct']},percent\n")
        f.write(f"json,json_valid,{json_results['json_valid_pct']},percent\n")
        f.write(f"json,schema_valid,{json_results['schema_valid_pct']},percent\n")
        f.write(f"math,accuracy,{math_results['accuracy_pct']},percent\n")
        f.write(f"tools,intent_accuracy,{tool_results['intent_accuracy_pct']},percent\n")
        f.write(f"retrieval_matrix,100m_off,{rag_results['matrix_accuracy_pct']['A_100M_Off']},percent\n")
        f.write(f"retrieval_matrix,100m_on,{rag_results['matrix_accuracy_pct']['B_100M_On']},percent\n")
        f.write(f"retrieval_matrix,500m_off,{rag_results['matrix_accuracy_pct']['C_500M_Off']},percent\n")
        f.write(f"retrieval_matrix,500m_on,{rag_results['matrix_accuracy_pct']['D_500M_On']},percent\n")
    print(f"Wrote results CSV to {csv_path}")

    print("\n" + "=" * 70)
    print(" PHASE 1D EVALUATION EXECUTION COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
