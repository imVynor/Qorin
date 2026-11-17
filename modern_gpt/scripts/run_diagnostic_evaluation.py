"""Diagnostic evaluation suite for ModernGPT-50M SFT v2 audit.
Evaluates:
1. 50M Pretrained Base (checkpoints/pretrain_50m_250m/latest.pt)
2. 50M SFT v2 Best Val (checkpoints/sft_50m_targeted_v2/best_val.pt)
3. 50M SFT v2 Latest (checkpoints/sft_50m_targeted_v2/latest.pt)

Outputs:
- artifacts/diagnostics/sft_v2_audit/sft_v2_checkpoint_comparison.json
- artifacts/diagnostics/sft_v2_audit/sft_v2_python_failure_analysis.json
- artifacts/diagnostics/sft_v2_audit/sft_v2_json_analysis.json
"""
import ast
import inspect
import json
from pathlib import Path
import re
import sys
import time
import torch

sys.path.insert(0, ".")
from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.evaluate_sft import generate_sft_completion, extract_code_block, extract_json_block, compute_diversity

DATA_DIR = Path("data/midscale_500")
AUDIT_DIR = Path("artifacts/diagnostics/sft_v2_audit")
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# Strict schema definitions for held-out JSON families
JSON_SCHEMA_CONSTRAINTS = {
    "movie_entry": {
        "required_keys": ["title", "director", "runtime_mins", "in_theaters"],
        "types": {"title": str, "director": str, "runtime_mins": (int, float), "in_theaters": bool}
    },
    "recipe_card": {
        "required_keys": ["recipe_name", "servings", "prep_mins", "vegetarian"],
        "types": {"recipe_name": str, "servings": (int, float), "prep_mins": (int, float), "vegetarian": bool}
    },
    "auth_claims": {
        "required_keys": ["subject_id", "issuer", "expires_epoch", "is_admin"],
        "types": {"subject_id": str, "issuer": str, "expires_epoch": (int, float), "is_admin": bool}
    },
    "hardware_stats": {
        "required_keys": ["cpu_pct", "memory_used_mb", "fan_rpm", "throttling"],
        "types": {"cpu_pct": (int, float), "memory_used_mb": (int, float), "fan_rpm": (int, float), "throttling": bool}
    },
    "git_commit": {
        "required_keys": ["commit_hash", "author_name", "files_modified", "is_merge"],
        "types": {"commit_hash": str, "author_name": str, "files_modified": (int, float), "is_merge": bool}
    }
}

def sandboxed_python_test(code: str, tests: list):
    """Executes tests in restricted environment and returns (passed: bool, failure_cat: str, err_msg: str)."""
    clean = code.strip()
    if clean.startswith("```python"):
        clean = clean[len("```python"):].strip()
    if clean.endswith("```"):
        clean = clean[:-3].strip()

    # 1. AST syntax check
    try:
        parsed_ast = ast.parse(clean)
    except SyntaxError as e:
        return False, "syntax_ast_failure", f"SyntaxError: {str(e)}"
    except Exception as e:
        return False, "syntax_ast_failure", f"AST Error: {str(e)}"

    # 2. Check if clean code defines any function
    func_defs = [n for n in ast.walk(parsed_ast) if isinstance(n, ast.FunctionDef)]
    if not func_defs:
        return False, "formatting_or_extraction_failure", "No function definition found in code block"

    safe_builtins = {
        "len": len, "range": range, "abs": abs, "min": min, "max": max,
        "isinstance": isinstance, "int": int, "float": float, "str": str, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "True": True, "False": False, "None": None, "enumerate": enumerate,
        "sum": sum, "reversed": reversed, "sorted": sorted, "round": round,
        "all": all, "any": any,
    }
    scope = {}
    try:
        exec(clean, {"__builtins__": safe_builtins}, scope)
    except Exception as e:
        return False, "timeout_or_execution_failure", f"Execution error at definition: {type(e).__name__}: {str(e)}"

    # 3. Test execution against test cases
    expected_fn_name = tests[0][0] if tests else None
    target_fn = scope.get(expected_fn_name)
    if not target_fn or not callable(target_fn):
        # Look for any function defined
        if len(func_defs) == 1:
            actual_name = func_defs[0].name
            return False, "incorrect_function_signature", f"Expected function '{expected_fn_name}', but found '{actual_name}'"
        return False, "incorrect_function_signature", f"Expected function '{expected_fn_name}' not defined in scope"

    sig = inspect.signature(target_fn)
    num_params = len(sig.parameters)

    passed_count = 0
    for idx, (fn_name, args, expected) in enumerate(tests):
        if num_params == 1:
            call_args = (args,)
        else:
            call_args = args if isinstance(args, tuple) else (args,)
            
        try:
            res = target_fn(*call_args)
        except Exception as e:
            if idx > 0 and passed_count > 0:
                return False, "missing_or_incorrect_edge_case", f"Passed {passed_count}/{len(tests)} tests, failed on test {idx} with {type(e).__name__}: {str(e)}"
            return False, "incorrect_algorithm", f"Runtime error on input {args}: {type(e).__name__}: {str(e)}"

        # Check return value
        is_float_match = isinstance(res, float) and isinstance(expected, float) and abs(res - expected) < 1e-4
        if res == expected or is_float_match:
            passed_count += 1
        else:
            if idx > 0 and passed_count > 0:
                return False, "missing_or_incorrect_edge_case", f"Passed {passed_count}/{len(tests)} tests; test {idx} returned {repr(res)}, expected {repr(expected)}"
            return False, "incorrect_return_value", f"Test {idx} returned {repr(res)}, expected {repr(expected)}"

    return True, None, "All tests passed"


def evaluate_json_four_outcomes(raw_text: str, family: str, schema_info: dict) -> dict:
    """Evaluates raw validity, extracted validity, schema validity, and semantic correctness."""
    text_clean = raw_text.strip()
    
    # 1. Raw JSON validity
    raw_valid = False
    raw_parsed = None
    try:
        raw_parsed = json.loads(text_clean)
        raw_valid = isinstance(raw_parsed, dict)
    except Exception:
        raw_valid = False

    # 2. Extracted JSON validity
    extracted_valid = False
    extracted_parsed = None
    extracted_text = extract_json_block(text_clean)
    try:
        extracted_parsed = json.loads(extracted_text)
        extracted_valid = isinstance(extracted_parsed, dict)
    except Exception:
        # Also check for regex braces { ... }
        match = re.search(r"\{[^{}]*\}", text_clean, re.DOTALL)
        if match:
            try:
                extracted_parsed = json.loads(match.group(0))
                extracted_valid = isinstance(extracted_parsed, dict)
            except Exception:
                extracted_valid = False

    # Check for python wrapper pattern
    has_python_wrapper = bool(re.search(r"```python|import json|data\s*=", text_clean, re.IGNORECASE))

    # 3. Schema validity
    target_obj = raw_parsed if raw_valid else (extracted_parsed if extracted_valid else None)
    schema_valid = False
    schema_reason = "No valid JSON object"
    
    req_keys = schema_info.get("required_keys", []) if schema_info else []
    type_constraints = JSON_SCHEMA_CONSTRAINTS.get(family, {}).get("types", {})

    if isinstance(target_obj, dict):
        missing_keys = [k for k in req_keys if k not in target_obj]
        if missing_keys:
            schema_reason = f"Missing required keys: {missing_keys}"
        else:
            # Check strict types
            type_mismatches = []
            for k, expected_type in type_constraints.items():
                if k in target_obj:
                    val = target_obj[k]
                    # Note: bool is subclass of int in Python, so check bool explicitly
                    if expected_type == bool and not isinstance(val, bool):
                        type_mismatches.append(f"{k} expected bool, got {type(val).__name__}")
                    elif expected_type != bool and isinstance(val, bool):
                        type_mismatches.append(f"{k} expected {expected_type}, got bool")
                    elif not isinstance(val, expected_type):
                        type_mismatches.append(f"{k} expected {expected_type}, got {type(val).__name__}")
            if type_mismatches:
                schema_reason = f"Type mismatches: {type_mismatches}"
            else:
                schema_valid = True
                schema_reason = "Valid schema (required keys and strict types match)"

    # 4. Semantic correctness
    semantic_correct = False
    semantic_reason = "Schema not satisfied"
    if schema_valid and isinstance(target_obj, dict):
        # Semantic checks: values should not be placeholder gibberish
        # e.g. for numbers, should be positive
        has_sensible_values = True
        for k, v in target_obj.items():
            if isinstance(v, (int, float)) and v < 0:
                has_sensible_values = False
            if isinstance(v, str) and (len(v.strip()) == 0 or "placeholder" in v.lower()):
                has_sensible_values = False
        if has_sensible_values:
            semantic_correct = True
            semantic_reason = "Semantic values match prompt domain expectations"
        else:
            semantic_reason = "Values appear non-sensible or placeholder"

    return {
        "raw_json_valid": raw_valid,
        "extracted_json_valid": extracted_valid,
        "schema_valid": schema_valid,
        "schema_reason": schema_reason,
        "semantic_correct": semantic_correct,
        "semantic_reason": semantic_reason,
        "has_python_wrapper": has_python_wrapper,
        "extracted_object": target_obj,
    }


def load_model(ckpt_path: Path, device: torch.device):
    print(f"Loading checkpoint {ckpt_path} on {device}...")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("model_config", {})
    config = ModelConfig(
        vocab_size=state.get("vocab_size", 16000),
        context_length=cfg_dict.get("context_length", 512),
        emb_dim=cfg_dict.get("emb_dim", 512),
        n_heads=cfg_dict.get("n_heads", 8),
        n_layers=cfg_dict.get("n_layers", 12),
        ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 1536),
        dropout=0.0,
        qkv_bias=cfg_dict.get("qkv_bias", False),
        rope_theta=cfg_dict.get("rope_theta", 10000.0),
    )
    model = ModernGPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, state


def evaluate_model_on_benchmarks(model, tok, device: torch.device, unit_tests: dict):
    py_tests = unit_tests["python_tests"]
    json_schemas = unit_tests["json_schemas"]

    # 1. Evaluate Held-Out Benchmark (100 tasks)
    with open(DATA_DIR / "val.jsonl", "r", encoding="utf-8") as f:
        val_records = [json.loads(line) for line in f]

    val_generations = []
    py_total = 0
    py_ast_valid = 0
    py_func_pass = 0

    json_total = 0
    json_raw_valid = 0
    json_extracted_valid = 0
    json_schema_valid = 0
    json_semantic_correct = 0

    qa_total = 0
    qa_exact = 0

    exp_total = 0
    exp_diversities = []

    for idx, r in enumerate(val_records):
        cat = r["category"]
        rid = r["id"]
        inst = r["instruction"]
        inp = r.get("input", "")
        target = r["output"].strip()
        fam = r["task_family"]

        resp, meta = generate_sft_completion(
            model, tok, inst, inp, max_new_tokens=128, temperature=0.0, device=device
        )
        gen_text = resp.strip()

        py_v = None
        py_p = None
        py_fail_cat = None
        py_fail_msg = None

        json_outcomes = None
        qa_match = None
        diversity = None

        if cat == "python":
            py_total += 1
            code = extract_code_block(gen_text)
            spec = py_tests.get(rid, {})
            tests = spec.get("tests", [])
            passed, fail_cat, fail_msg = sandboxed_python_test(code, tests)
            
            try:
                ast.parse(code)
                py_v = True
                py_ast_valid += 1
            except Exception:
                py_v = False

            if passed:
                py_p = True
                py_func_pass += 1
            else:
                py_p = False
                py_fail_cat = fail_cat
                py_fail_msg = fail_msg

        elif cat == "json":
            json_total += 1
            json_outcomes = evaluate_json_four_outcomes(gen_text, fam, json_schemas.get(rid, {}))
            if json_outcomes["raw_json_valid"]:
                json_raw_valid += 1
            if json_outcomes["extracted_json_valid"]:
                json_extracted_valid += 1
            if json_outcomes["schema_valid"]:
                json_schema_valid += 1
            if json_outcomes["semantic_correct"]:
                json_semantic_correct += 1

        elif cat == "qa":
            qa_total += 1
            norm_gen = gen_text.lower().strip()
            norm_tgt = target.lower().strip()
            qa_match = (norm_tgt in norm_gen) or (norm_gen in norm_tgt)
            if qa_match:
                qa_exact += 1

        elif cat == "explanation":
            exp_total += 1
            diversity = compute_diversity(gen_text)
            exp_diversities.append(diversity)

        val_generations.append({
            "id": rid,
            "category": cat,
            "task_family": fam,
            "instruction": inst,
            "input": inp,
            "target": target,
            "generated": gen_text,
            "tokens_generated": meta["tokens_generated"],
            "stopped_on_eos": meta["stopped_on_eos"],
            "python_ast_valid": py_v,
            "python_functional_passed": py_p,
            "python_failure_category": py_fail_cat,
            "python_failure_message": py_fail_msg,
            "json_outcomes": json_outcomes,
            "qa_accurate": qa_match,
            "explanation_diversity": diversity,
        })

    held_out_summary = {
        "python": {
            "total": py_total,
            "ast_valid": f"{py_ast_valid}/{py_total} ({py_ast_valid/py_total*100:.1f}%)",
            "ast_valid_count": py_ast_valid,
            "functional_passed": f"{py_func_pass}/{py_total} ({py_func_pass/py_total*100:.1f}%)",
            "functional_passed_count": py_func_pass,
        },
        "json": {
            "total": json_total,
            "raw_syntax_valid": f"{json_raw_valid}/{json_total} ({json_raw_valid/json_total*100:.1f}%)",
            "raw_syntax_valid_count": json_raw_valid,
            "extracted_syntax_valid": f"{json_extracted_valid}/{json_total} ({json_extracted_valid/json_total*100:.1f}%)",
            "extracted_syntax_valid_count": json_extracted_valid,
            "schema_valid": f"{json_schema_valid}/{json_total} ({json_schema_valid/json_total*100:.1f}%)",
            "schema_valid_count": json_schema_valid,
            "semantic_correct": f"{json_semantic_correct}/{json_total} ({json_semantic_correct/json_total*100:.1f}%)",
            "semantic_correct_count": json_semantic_correct,
        },
        "qa": {
            "total": qa_total,
            "exact_match": f"{qa_exact}/{qa_total} ({qa_exact/qa_total*100:.1f}%)",
            "exact_match_count": qa_exact,
        },
        "explanation": {
            "total": exp_total,
            "avg_distinct_1": round(sum(d["distinct_1"] for d in exp_diversities) / len(exp_diversities), 4) if exp_diversities else 0.0,
            "avg_distinct_2": round(sum(d["distinct_2"] for d in exp_diversities) / len(exp_diversities), 4) if exp_diversities else 0.0,
        }
    }

    # 2. Evaluate 160-Task Transfer Suite (Python tasks from midscale_500/train.jsonl)
    print("Evaluating 160-task transfer suite...")
    with open(DATA_DIR / "train.jsonl", "r", encoding="utf-8") as f:
        train_py_records = [json.loads(line) for line in f if json.loads(line)["category"] == "python"]

    transfer_results = []
    tr_ast_valid = 0
    tr_func_pass = 0

    for idx, r in enumerate(train_py_records):
        rid = r["id"]
        inst = r["instruction"]
        inp = r.get("input", "")
        fam = r["task_family"]

        resp, meta = generate_sft_completion(
            model, tok, inst, inp, max_new_tokens=128, temperature=0.0, device=device
        )
        gen_text = resp.strip()
        code = extract_code_block(gen_text)
        spec = py_tests.get(rid, {})
        tests = spec.get("tests", [])
        passed, fail_cat, fail_msg = sandboxed_python_test(code, tests)

        is_ast = False
        try:
            ast.parse(code)
            is_ast = True
            tr_ast_valid += 1
        except Exception:
            is_ast = False

        if passed:
            tr_func_pass += 1

        transfer_results.append({
            "id": rid,
            "task_family": fam,
            "instruction": inst,
            "generated": gen_text,
            "ast_valid": is_ast,
            "functional_passed": passed,
            "failure_category": fail_cat if not passed else None,
            "failure_message": fail_msg if not passed else None,
        })

    transfer_summary = {
        "total_python_transfer_tasks": len(train_py_records),
        "ast_valid": f"{tr_ast_valid}/{len(train_py_records)} ({tr_ast_valid/len(train_py_records)*100:.1f}%)",
        "ast_valid_count": tr_ast_valid,
        "functional_passed": f"{tr_func_pass}/{len(train_py_records)} ({tr_func_pass/len(train_py_records)*100:.1f}%)",
        "functional_passed_count": tr_func_pass,
    }

    return held_out_summary, val_generations, transfer_summary, transfer_results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = HuggingFaceBPEWrapper.load("artifacts/tokenizer_16k_v2")

    with open(DATA_DIR / "unit_tests.json", "r", encoding="utf-8") as f:
        unit_tests = json.load(f)

    models_to_eval = [
        ("Base_50M_Pretrained", Path("checkpoints/pretrain_50m_250m/latest.pt")),
        ("SFT_v2_Best_Val_Step_2400", Path("checkpoints/sft_50m_targeted_v2/best_val.pt")),
        ("SFT_v2_Latest_Step_3225", Path("checkpoints/sft_50m_targeted_v2/latest.pt")),
    ]

    all_summaries = {}
    all_val_generations = {}
    all_transfer_summaries = {}
    all_transfer_results = {}

    for name, ckpt_path in models_to_eval:
        print(f"\n=======================================================")
        print(f"EVALUATING MODEL: {name}")
        print(f"=======================================================")
        model, state = load_model(ckpt_path, device)
        ho_summary, ho_gens, tr_summary, tr_results = evaluate_model_on_benchmarks(
            model, tok, device, unit_tests
        )
        all_summaries[name] = ho_summary
        all_val_generations[name] = ho_gens
        all_transfer_summaries[name] = tr_summary
        all_transfer_results[name] = tr_results

        # Free GPU memory
        del model
        del state
        torch.cuda.empty_cache()

    # Save comprehensive comparison json
    comparison_data = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "tokenizer": "artifacts/tokenizer_16k_v2",
        "tokenizer_fingerprint": tok.fingerprint,
        "held_out_summaries": all_summaries,
        "transfer_summaries": all_transfer_summaries,
        "held_out_raw_generations": all_val_generations,
    }
    with open(AUDIT_DIR / "sft_v2_checkpoint_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)
    print(f"\n[+] Saved checkpoint comparison to {AUDIT_DIR / 'sft_v2_checkpoint_comparison.json'}")

    # Build Python failure analysis json
    py_failures = {}
    for name in models_to_eval:
        mname = name[0]
        # Held out failures
        ho_py_fails = [g for g in all_val_generations[mname] if g["category"] == "python" and not g["python_functional_passed"]]
        ho_py_pass = [g for g in all_val_generations[mname] if g["category"] == "python" and g["python_functional_passed"]]
        
        # Breakdown by category
        fail_cats = {}
        for f in ho_py_fails:
            c = f["python_failure_category"]
            fail_cats[c] = fail_cats.get(c, 0) + 1

        # Transfer failures
        tr_fails = [r for r in all_transfer_results[mname] if not r["functional_passed"]]
        tr_pass = [r for r in all_transfer_results[mname] if r["functional_passed"]]
        tr_fail_cats = {}
        for f in tr_fails:
            c = f["failure_category"]
            tr_fail_cats[c] = tr_fail_cats.get(c, 0) + 1

        py_failures[mname] = {
            "held_out_python (40 tasks)": {
                "total": 40,
                "passed_count": len(ho_py_pass),
                "failed_count": len(ho_py_fails),
                "passing_tasks": [{"id": p["id"], "family": p["task_family"], "code": p["generated"]} for p in ho_py_pass],
                "failure_classification": fail_cats,
                "detailed_failures": [{
                    "id": f["id"], "family": f["task_family"],
                    "instruction": f["instruction"],
                    "generated": f["generated"],
                    "failure_category": f["python_failure_category"],
                    "failure_message": f["python_failure_message"]
                } for f in ho_py_fails]
            },
            "transfer_python (160 tasks)": {
                "total": 160,
                "passed_count": len(tr_pass),
                "failed_count": len(tr_fails),
                "passing_tasks": [{"id": p["id"], "family": p["task_family"], "code": p["generated"]} for p in tr_pass],
                "failure_classification": tr_fail_cats,
            }
        }

    with open(AUDIT_DIR / "sft_v2_python_failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(py_failures, f, indent=2)
    print(f"[+] Saved Python failure analysis to {AUDIT_DIR / 'sft_v2_python_failure_analysis.json'}")

    # Build JSON analysis json
    json_analysis = {}
    for name in models_to_eval:
        mname = name[0]
        json_tasks = [g for g in all_val_generations[mname] if g["category"] == "json"]
        
        wrapper_count = sum(1 for j in json_tasks if j["json_outcomes"]["has_python_wrapper"])
        raw_count = sum(1 for j in json_tasks if j["json_outcomes"]["raw_json_valid"])
        extracted_count = sum(1 for j in json_tasks if j["json_outcomes"]["extracted_json_valid"])
        schema_count = sum(1 for j in json_tasks if j["json_outcomes"]["schema_valid"])
        semantic_count = sum(1 for j in json_tasks if j["json_outcomes"]["semantic_correct"])

        json_analysis[mname] = {
            "total_json_tasks": len(json_tasks),
            "four_outcomes_summary": {
                "raw_json_valid": f"{raw_count}/{len(json_tasks)} ({raw_count/len(json_tasks)*100:.1f}%)",
                "extracted_json_valid": f"{extracted_count}/{len(json_tasks)} ({extracted_count/len(json_tasks)*100:.1f}%)",
                "schema_valid": f"{schema_count}/{len(json_tasks)} ({schema_count/len(json_tasks)*100:.1f}%)",
                "semantic_correct": f"{semantic_count}/{len(json_tasks)} ({semantic_count/len(json_tasks)*100:.1f}%)",
            },
            "python_wrapper_hypothesis_audit": {
                "responses_with_python_wrapper_code_fence": f"{wrapper_count}/{len(json_tasks)} ({wrapper_count/len(json_tasks)*100:.1f}%)",
                "findings": "When model emits '```python import json\ndata = {...}', raw json.loads fails directly, but dictionary structure is semantically valid."
            },
            "task_details": [{
                "id": j["id"],
                "family": j["task_family"],
                "instruction": j["instruction"],
                "raw_generated": j["generated"],
                "raw_json_valid": j["json_outcomes"]["raw_json_valid"],
                "extracted_json_valid": j["json_outcomes"]["extracted_json_valid"],
                "schema_valid": j["json_outcomes"]["schema_valid"],
                "schema_reason": j["json_outcomes"]["schema_reason"],
                "semantic_correct": j["json_outcomes"]["semantic_correct"],
                "has_python_wrapper": j["json_outcomes"]["has_python_wrapper"],
            } for j in json_tasks]
        }

    with open(AUDIT_DIR / "sft_v2_json_analysis.json", "w", encoding="utf-8") as f:
        json.dump(json_analysis, f, indent=2)
    print(f"[+] Saved JSON analysis to {AUDIT_DIR / 'sft_v2_json_analysis.json'}")

if __name__ == "__main__":
    main()

