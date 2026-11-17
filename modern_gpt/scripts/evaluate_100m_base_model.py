"""Comprehensive Base-Model Evaluation for ModernGPT-S 100M GQA-4.

Evaluates:
1. Held-Out 100-Task Benchmark (Python 40, JSON 40, QA 10, Explanation 10)
2. 160-Task Algorithmic Transfer Benchmark (Python tasks from midscale_500/train.jsonl)
3. Inference Efficiency Profiling (TTFT, Decode throughput, KV cache footprint)
4. Comprehensive Failure Classification & Error Extraction
"""

import ast
import gc
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import get_100m_gqa4_config
from model.gpt import ModernGPT
from scripts.evaluate_sft import (
    generate_sft_completion,
    extract_code_block,
    extract_json_block,
    compute_diversity,
)
from scripts.run_diagnostic_evaluation import (
    sandboxed_python_test,
    evaluate_json_four_outcomes,
    JSON_SCHEMA_CONSTRAINTS,
)

DATA_DIR = REPO_ROOT / "data/midscale_500"
EVAL_DIR = REPO_ROOT / "artifacts/evaluation/modern_gpts_100m"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def load_100m_model(ckpt_path: Path, device: torch.device) -> ModernGPT:
    """Loads 100M GQA-4 model with frozen weights on specified device."""
    print(f"Loading checkpoint {ckpt_path.name} on {device}...")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = get_100m_gqa4_config()
    model = ModernGPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model


def evaluate_held_out_100(
    model: ModernGPT,
    tokenizer: HuggingFaceBPEWrapper,
    val_records: List[Dict[str, Any]],
    unit_tests: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates 100 held-out tasks (Python 40, JSON 40, QA 10, Explanation 10)."""
    py_tests = unit_tests["python_tests"]
    results = []

    # Counters
    py_total = 0
    py_syntax_pass = 0
    py_ast_valid = 0
    py_func_pass = 0

    json_total = 0
    json_raw_valid = 0
    json_extracted_valid = 0
    json_schema_valid = 0
    json_semantic_correct = 0

    qa_total = 0
    qa_exact_match = 0
    qa_containment = 0

    exp_total = 0
    exp_distinct_1 = []
    exp_distinct_2 = []

    print(f"Running Held-Out 100-Task Evaluation ({len(val_records)} items)...")
    t0 = time.time()

    for idx, r in enumerate(val_records):
        cat = r["category"]
        rid = r["id"]
        inst = r["instruction"]
        inp = r.get("input", "")
        target = r["output"].strip()
        fam = r.get("task_family", "unknown")

        resp, meta = generate_sft_completion(
            model, tokenizer, inst, inp, max_new_tokens=128, temperature=0.0, device=device
        )
        gen_text = resp.strip()

        item_result = {
            "id": rid,
            "category": cat,
            "task_family": fam,
            "instruction": inst,
            "input": inp,
            "target": target,
            "generated_text": gen_text,
            "tokens_generated": meta["tokens_generated"],
            "stopped_on_eos": meta["stopped_on_eos"],
        }

        if cat == "python":
            py_total += 1
            code = extract_code_block(gen_text)
            item_result["extracted_code"] = code

            # Syntax and AST check
            try:
                ast.parse(code)
                is_ast = True
                py_ast_valid += 1
                py_syntax_pass += 1
            except Exception:
                is_ast = False

            spec = py_tests.get(rid, {})
            tests = spec.get("tests", [])
            passed, fail_cat, fail_msg = sandboxed_python_test(code, tests)
            if passed:
                py_func_pass += 1

            item_result.update({
                "ast_valid": is_ast,
                "functional_passed": passed,
                "failure_category": fail_cat if not passed else None,
                "failure_message": fail_msg if not passed else None,
            })

        elif cat == "json":
            json_total += 1
            schema_info = JSON_SCHEMA_CONSTRAINTS.get(fam, {})
            eval_res = evaluate_json_four_outcomes(gen_text, fam, schema_info)

            if eval_res.get("raw_json_valid", False):
                json_raw_valid += 1
            if eval_res.get("extracted_json_valid", False):
                json_extracted_valid += 1
            if eval_res.get("schema_valid", False):
                json_schema_valid += 1
            if eval_res.get("semantic_correct", False):
                json_semantic_correct += 1

            item_result.update({
                "raw_json_valid": eval_res.get("raw_json_valid", False),
                "extracted_json_valid": eval_res.get("extracted_json_valid", False),
                "schema_valid": eval_res.get("schema_valid", False),
                "semantic_correct": eval_res.get("semantic_correct", False),
                "schema_reason": eval_res.get("schema_reason"),
                "has_python_wrapper": eval_res.get("has_python_wrapper", False),
                "extracted_object": eval_res.get("extracted_object"),
            })

        elif cat == "qa":
            qa_total += 1
            is_exact = gen_text.strip().lower() == target.strip().lower()
            is_contained = (target.lower() in gen_text.lower()) or (gen_text.lower() in target.lower() and len(gen_text) > 0)
            if is_exact:
                qa_exact_match += 1
            if is_contained:
                qa_containment += 1

            item_result.update({
                "exact_match": is_exact,
                "containment": is_contained,
            })

        elif cat == "explanation":
            exp_total += 1
            div = compute_diversity(gen_text)
            exp_distinct_1.append(div.get("distinct_1", 0.0))
            exp_distinct_2.append(div.get("distinct_2", 0.0))
            item_result.update({
                "diversity": div,
            })

        results.append(item_result)
        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/100 held-out tasks ({time.time() - t0:.1f}s)...")

    summary = {
        "benchmark": "held_out_100_task",
        "total_tasks": len(val_records),
        "duration_seconds": round(time.time() - t0, 2),
        "python": {
            "total": py_total,
            "syntax_valid_count": py_syntax_pass,
            "syntax_valid_pct": round(py_syntax_pass / max(1, py_total) * 100, 2),
            "ast_valid_count": py_ast_valid,
            "ast_valid_pct": round(py_ast_valid / max(1, py_total) * 100, 2),
            "functional_pass_count": py_func_pass,
            "functional_pass_pct": round(py_func_pass / max(1, py_total) * 100, 2),
        },
        "json": {
            "total": json_total,
            "raw_syntax_valid_count": json_raw_valid,
            "raw_syntax_valid_pct": round(json_raw_valid / max(1, json_total) * 100, 2),
            "extracted_syntax_valid_count": json_extracted_valid,
            "extracted_syntax_valid_pct": round(json_extracted_valid / max(1, json_total) * 100, 2),
            "schema_valid_count": json_schema_valid,
            "schema_valid_pct": round(json_schema_valid / max(1, json_total) * 100, 2),
            "semantic_correct_count": json_semantic_correct,
            "semantic_correct_pct": round(json_semantic_correct / max(1, json_total) * 100, 2),
        },
        "qa": {
            "total": qa_total,
            "exact_match_count": qa_exact_match,
            "exact_match_pct": round(qa_exact_match / max(1, qa_total) * 100, 2),
            "containment_count": qa_containment,
            "containment_pct": round(qa_containment / max(1, qa_total) * 100, 2),
        },
        "explanation": {
            "total": exp_total,
            "mean_distinct_1": round(sum(exp_distinct_1) / max(1, len(exp_distinct_1)), 4),
            "mean_distinct_2": round(sum(exp_distinct_2) / max(1, len(exp_distinct_2)), 4),
        },
    }

    return summary, results


def evaluate_transfer_160(
    model: ModernGPT,
    tokenizer: HuggingFaceBPEWrapper,
    train_py_records: List[Dict[str, Any]],
    unit_tests: Dict[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluates 160 Python algorithmic transfer tasks."""
    py_tests = unit_tests["python_tests"]
    results = []

    total_tasks = len(train_py_records)
    ast_valid_count = 0
    func_pass_count = 0

    print(f"Running 160-Task Transfer Benchmark ({total_tasks} tasks)...")
    t0 = time.time()

    for idx, r in enumerate(train_py_records):
        rid = r["id"]
        inst = r["instruction"]
        inp = r.get("input", "")
        fam = r.get("task_family", "python_algo")

        resp, meta = generate_sft_completion(
            model, tokenizer, inst, inp, max_new_tokens=128, temperature=0.0, device=device
        )
        gen_text = resp.strip()
        code = extract_code_block(gen_text)

        # AST test
        is_ast = False
        try:
            ast.parse(code)
            is_ast = True
            ast_valid_count += 1
        except Exception:
            is_ast = False

        # Functional test
        spec = py_tests.get(rid, {})
        tests = spec.get("tests", [])
        passed, fail_cat, fail_msg = sandboxed_python_test(code, tests)
        if passed:
            func_pass_count += 1

        results.append({
            "id": rid,
            "task_family": fam,
            "instruction": inst,
            "generated_text": gen_text,
            "extracted_code": code,
            "ast_valid": is_ast,
            "functional_passed": passed,
            "failure_category": fail_cat if not passed else None,
            "failure_message": fail_msg if not passed else None,
            "tokens_generated": meta["tokens_generated"],
        })

        if (idx + 1) % 40 == 0:
            print(f"  Processed {idx + 1}/160 transfer tasks ({time.time() - t0:.1f}s)...")

    summary = {
        "benchmark": "transfer_160_task",
        "total_tasks": total_tasks,
        "duration_seconds": round(time.time() - t0, 2),
        "ast_valid_count": ast_valid_count,
        "ast_valid_pct": round(ast_valid_count / max(1, total_tasks) * 100, 2),
        "functional_pass_count": func_pass_count,
        "functional_pass_pct": round(func_pass_count / max(1, total_tasks) * 100, 2),
    }

    return summary, results


def profile_inference_100m(model: ModernGPT, device: torch.device) -> Dict[str, Any]:
    """Profiles TTFT, decoding tokens/sec, and calculates KV-cache footprint."""
    print("Profiling inference efficiency on GPU...")
    torch.cuda.empty_cache()
    gc.collect()

    config = model.config
    head_dim = config.emb_dim // config.n_heads  # 64
    kv_heads = config.num_kv_heads  # 4
    bytes_per_elem = 2  # bf16

    # KV-cache footprint
    kv_b1 = (2 * config.n_layers * kv_heads * head_dim * config.context_length * 1 * bytes_per_elem) / (1024 ** 2)
    kv_b4 = (2 * config.n_layers * kv_heads * head_dim * config.context_length * 4 * bytes_per_elem) / (1024 ** 2)
    kv_b16 = (2 * config.n_layers * kv_heads * head_dim * config.context_length * 16 * bytes_per_elem) / (1024 ** 2)

    # TTFT
    ttft_results = {}
    for L in [128, 256, 512]:
        x = torch.randint(0, config.vocab_size, (1, L), device=device)
        for _ in range(3):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(x)
        torch.cuda.synchronize()

        num_iters = 15
        start = time.perf_counter()
        for _ in range(num_iters):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    _ = model(x)
        torch.cuda.synchronize()
        ttft_results[f"ttft_{L}_ms"] = round(((time.perf_counter() - start) / num_iters) * 1000.0, 2)

    # Decode throughput
    decode_results = {}
    for bs in [1, 4]:
        prompt = torch.randint(0, config.vocab_size, (bs, 64), device=device)
        # Warmup
        for _ in range(2):
            curr = prompt.clone()
            for _ in range(8):
                with torch.no_grad():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = model(curr[:, -config.context_length:])
                    next_tok = logits[:, -1:].argmax(dim=-1)
                    curr = torch.cat([curr, next_tok], dim=1)
        torch.cuda.synchronize()

        gen_tokens = 128
        curr = prompt.clone()
        start = time.perf_counter()
        for _ in range(gen_tokens):
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(curr[:, -config.context_length:])
                next_tok = logits[:, -1:].argmax(dim=-1)
                curr = torch.cat([curr, next_tok], dim=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        decode_results[f"decode_tok_sec_b{bs}"] = round((gen_tokens * bs) / elapsed, 1)

    peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_res_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    return {
        "kv_cache_mb_batch1": round(kv_b1, 2),
        "kv_cache_mb_batch4": round(kv_b4, 2),
        "kv_cache_mb_batch16": round(kv_b16, 2),
        **ttft_results,
        **decode_results,
        "inference_peak_allocated_mb": round(peak_alloc_mb, 1),
        "inference_peak_reserved_mb": round(peak_res_mb, 1),
    }


def analyze_failure_patterns(
    ho_results: List[Dict[str, Any]], tr_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Categorizes representative failure patterns across 100-task and 160-task benchmarks."""
    cat_counts = {}
    examples_by_cat = {}

    for item in ho_results:
        cat = item["category"]
        rid = item["id"]
        inst = item["instruction"]
        gen = item["generated_text"]
        target = item["target"]

        if cat == "python" and not item["functional_passed"]:
            fail_type = item["failure_category"] or "syntax_error"
            cat_counts[fail_type] = cat_counts.get(fail_type, 0) + 1
            if fail_type not in examples_by_cat:
                examples_by_cat[fail_type] = []
            if len(examples_by_cat[fail_type]) < 3:
                examples_by_cat[fail_type].append({
                    "benchmark": "held_out_100",
                    "id": rid,
                    "prompt": inst,
                    "generated": gen[:200],
                    "target": target[:200],
                    "message": item.get("failure_message"),
                })

        elif cat == "json" and not item["schema_valid"]:
            fail_type = "json_formatting_or_schema_violation"
            cat_counts[fail_type] = cat_counts.get(fail_type, 0) + 1
            if fail_type not in examples_by_cat:
                examples_by_cat[fail_type] = []
            if len(examples_by_cat[fail_type]) < 3:
                examples_by_cat[fail_type].append({
                    "benchmark": "held_out_100",
                    "id": rid,
                    "prompt": inst,
                    "generated": gen[:200],
                    "target": target[:200],
                    "message": item.get("schema_reason"),
                })

        elif cat == "qa" and not item["containment"]:
            fail_type = "qa_factual_or_instruction_drift"
            cat_counts[fail_type] = cat_counts.get(fail_type, 0) + 1
            if fail_type not in examples_by_cat:
                examples_by_cat[fail_type] = []
            if len(examples_by_cat[fail_type]) < 3:
                examples_by_cat[fail_type].append({
                    "benchmark": "held_out_100",
                    "id": rid,
                    "prompt": inst,
                    "generated": gen[:200],
                    "target": target[:200],
                    "message": "Target not contained in response",
                })

    for item in tr_results:
        if not item["functional_passed"]:
            fail_type = f"transfer_{item['failure_category'] or 'unknown'}"
            cat_counts[fail_type] = cat_counts.get(fail_type, 0) + 1
            if fail_type not in examples_by_cat:
                examples_by_cat[fail_type] = []
            if len(examples_by_cat[fail_type]) < 3:
                examples_by_cat[fail_type].append({
                    "benchmark": "transfer_160",
                    "id": item["id"],
                    "prompt": item["instruction"],
                    "generated": item["generated_text"][:200],
                    "message": item.get("failure_message"),
                })

    return {
        "failure_counts": cat_counts,
        "representative_examples": examples_by_cat,
    }


def main():
    print("=" * 70)
    print("MODERNGPT-S 100M: BASE-MODEL EVALUATION AND AUDIT")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Tokenizer
    tok_path = REPO_ROOT / "artifacts/tokenizer_16k_v2"
    tokenizer = HuggingFaceBPEWrapper.load(str(tok_path))
    print(f"Loaded Tokenizer: {tok_path.name} (fingerprint: {tokenizer.fingerprint})")

    # Load Benchmark Datasets
    val_file = DATA_DIR / "val.jsonl"
    with open(val_file, "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]
    print(f"Loaded {len(val_records)} held-out tasks from {val_file}")

    train_file = DATA_DIR / "train.jsonl"
    with open(train_file, "r", encoding="utf-8") as f:
        train_py_records = [json.loads(line) for line in f if json.loads(line)["category"] == "python"]
    print(f"Loaded {len(train_py_records)} transfer python tasks from {train_file}")

    unit_test_file = DATA_DIR / "unit_tests.json"
    with open(unit_test_file, "r", encoding="utf-8") as f:
        unit_tests = json.load(f)
    print(f"Loaded unit tests from {unit_test_file}")

    # 1. Evaluate Primary Checkpoint: best_val.pt
    best_val_path = REPO_ROOT / "checkpoints/modern_gpts_100m/best_val.pt"
    model_best = load_100m_model(best_val_path, device)

    best_ho_summary, best_ho_results = evaluate_held_out_100(
        model_best, tokenizer, val_records, unit_tests, device
    )
    best_tr_summary, best_tr_results = evaluate_transfer_160(
        model_best, tokenizer, train_py_records, unit_tests, device
    )
    inference_profile = profile_inference_100m(model_best, device)
    failure_analysis = analyze_failure_patterns(best_ho_results, best_tr_results)

    del model_best
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Evaluate Secondary Checkpoint: latest.pt on held-out 100
    latest_path = REPO_ROOT / "checkpoints/modern_gpts_100m/latest.pt"
    model_latest = load_100m_model(latest_path, device)

    latest_ho_summary, latest_ho_results = evaluate_held_out_100(
        model_latest, tokenizer, val_records, unit_tests, device
    )

    del model_latest
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Save Structured Machine-Readable Results
    with open(EVAL_DIR / "held_out_100_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_step_45000": {"summary": best_ho_summary, "results": best_ho_results},
            "latest_step_45770": {"summary": latest_ho_summary, "results": latest_ho_results},
        }, f, indent=2)

    with open(EVAL_DIR / "transfer_160_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_val_step_45000": {"summary": best_tr_summary, "results": best_tr_results},
        }, f, indent=2)

    with open(EVAL_DIR / "inference_profile_results.json", "w", encoding="utf-8") as f:
        json.dump(inference_profile, f, indent=2)

    with open(EVAL_DIR / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(failure_analysis, f, indent=2)

    summary_manifest = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": "ModernGPT-S-100M-GQA4",
        "primary_checkpoint": {
            "path": str(best_val_path),
            "step": 45000,
            "held_out_100_summary": best_ho_summary,
            "transfer_160_summary": best_tr_summary,
            "inference_profile": inference_profile,
        },
        "secondary_checkpoint": {
            "path": str(latest_path),
            "step": 45770,
            "held_out_100_summary": latest_ho_summary,
        },
    }

    with open(EVAL_DIR / "base_model_eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EVALUATIONS SUCCESSFULLY COMPLETED")
    print(f"Artifacts saved to: {EVAL_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

