import json
import math
import re
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

API_URL = "http://127.0.0.1:8000/api/chat/stream"
MODEL_ID = "qorin-500m-sft-exp-c"

BENCHMARK_TASKS = [
    # Python Required (5 tasks)
    {"id": "task_01_arithmetic", "type": "python", "prompt": "Calculate (24 * 35) + (1024 / 32) - (15 ** 2) using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": 647.0},
    {"id": "task_02_factorial", "type": "python", "prompt": "Compute the value of 12 factorial using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": 479001600},
    {"id": "task_03_stats", "type": "python", "prompt": "Given the dataset [15, 25, 35, 45, 55], calculate the sample mean and variance using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": {"mean": 35.0, "variance": 250.0}},
    {"id": "task_04_sorting", "type": "python", "prompt": "Sort the list [89, 12, 45, 67, 23, 90, 34] in ascending order and find the median using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": 45},
    {"id": "task_05_dict_recovery", "type": "python", "prompt": "In Python, safely access the value of key 'port' from {'host': 'localhost'} using .get('port', 8080).", "tool_expected": True, "expected_tool": "python", "ground_truth": 8080},

    # Web Required (4 tasks)
    {"id": "task_06_transformer_paper", "type": "web", "prompt": "Search the web for the authors and year of the Transformer paper Attention Is All You Need.", "tool_expected": True, "expected_tool": "web_research", "ground_truth": ["Vaswani", "2017"]},
    {"id": "task_07_python_release", "type": "web", "prompt": "Search the web for the release date and major features of Python 3.12.", "tool_expected": True, "expected_tool": "web_research", "ground_truth": ["Python 3.12", "2023"]},
    {"id": "task_08_pytorch_org", "type": "web", "prompt": "Search the web for what organization developed the PyTorch deep learning framework.", "tool_expected": True, "expected_tool": "web_research", "ground_truth": ["Meta", "Facebook"]},
    {"id": "task_09_unavail_query", "type": "web", "prompt": "Search the web for the teleportation crystal formula discovered under the Antarctic ice sheet in 1720.", "tool_expected": True, "expected_tool": "web_research", "ground_truth": "graceful_handling"},

    # No-Tool Negative Controls (6 tasks)
    {"id": "task_10_capital_france", "type": "no_tool", "prompt": "What is the capital city of France?", "tool_expected": False, "expected_tool": None, "ground_truth": "Paris"},
    {"id": "task_11_mitochondria", "type": "no_tool", "prompt": "What is the function of mitochondria in eukaryotic cells?", "tool_expected": False, "expected_tool": None, "ground_truth": ["energy", "ATP", "powerhouse"]},
    {"id": "task_12_hamlet", "type": "no_tool", "prompt": "Who wrote the play Hamlet?", "tool_expected": False, "expected_tool": None, "ground_truth": "Shakespeare"},
    {"id": "task_13_water_formula", "type": "no_tool", "prompt": "What is the chemical formula for water?", "tool_expected": False, "expected_tool": None, "ground_truth": "H2O"},
    {"id": "task_14_translation", "type": "no_tool", "prompt": "Translate 'Thank you very much' into Spanish.", "tool_expected": False, "expected_tool": None, "ground_truth": ["Muchas gracias", "gracias"]},
    {"id": "task_15_utilitarianism", "type": "no_tool", "prompt": "Define the philosophical concept of utilitarianism in one concise sentence.", "tool_expected": False, "expected_tool": None, "ground_truth": ["greatest happiness", "greatest good", "utility"]},

    # Multi-step & Error Recovery (4 tasks)
    {"id": "task_16_multistep_primes", "type": "python", "prompt": "Find all prime numbers between 20 and 40, and compute their sum using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": 120}, # 23 + 29 + 31 + 37 = 120
    {"id": "task_17_vowel_count", "type": "python", "prompt": "Count the number of vowels in 'supercalifragilisticexpialidocious' using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": 16},
    {"id": "task_18_zerodiv_catch", "type": "python", "prompt": "In Python, write a try-except block to divide 50 by 0, catching ZeroDivisionError and returning -1.", "tool_expected": True, "expected_tool": "python", "ground_truth": -1},
    {"id": "task_19_file_catch", "type": "python", "prompt": "In Python, catch FileNotFoundError when opening 'absent_log.txt' and return 'Not Found'.", "tool_expected": True, "expected_tool": "python", "ground_truth": "Not Found"},

    # Primality Verification (1 task)
    {"id": "task_20_primality_91", "type": "python", "prompt": "Is the number 91 a prime number? Compute its smallest divisor greater than 1 using Python.", "tool_expected": True, "expected_tool": "python", "ground_truth": 7} # 91 = 7 * 13 (composite!)
]

def query_harness(prompt: str, web_enabled: bool, py_enabled: bool) -> Dict[str, Any]:
    payload = {
        "model_id": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "web_research_enabled": web_enabled,
        "python_sandbox_enabled": py_enabled,
        "temperature": 0.0,
        "max_new_tokens": 120
    }
    
    tool_starts = []
    tool_results = []
    tokens = []
    done_data = None
    
    try:
        resp = requests.post(API_URL, json=payload, stream=True, timeout=35)
        curr_event = None
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8")
            if line.startswith("event:"):
                curr_event = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str = line.replace("data:", "").strip()
                try:
                    obj = json.loads(data_str)
                except Exception:
                    obj = data_str
                
                if curr_event == "tool_start":
                    tool_starts.append(obj)
                elif curr_event == "tool_result":
                    tool_results.append(obj)
                elif curr_event == "token":
                    if isinstance(obj, dict) and "text" in obj:
                        tokens.append(obj["text"])
                elif curr_event == "done":
                    done_data = obj
                    
        return {
            "success": True,
            "tool_starts": tool_starts,
            "tool_results": tool_results,
            "final_text": "".join(tokens),
            "done_data": done_data
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tool_starts": tool_starts,
            "tool_results": tool_results,
            "final_text": "".join(tokens),
            "done_data": None
        }

def evaluate_task_correctness(task: Dict[str, Any], final_text: str, tool_results: List[Dict[str, Any]]) -> bool:
    gt = task["ground_truth"]
    if gt is None:
        return True
    
    # Check numeric
    if isinstance(gt, (int, float)):
        # Check in final text or tool stdout
        tool_stdout = ""
        for tr in tool_results:
            p = tr.get("output_payload", {})
            if isinstance(p, dict):
                tool_stdout += str(p.get("stdout", ""))
        
        target_str = str(int(gt)) if isinstance(gt, int) or gt.is_integer() else str(gt)
        if target_str in final_text or target_str in tool_stdout:
            return True
        return False
        
    # Check string or list of facts
    if isinstance(gt, str):
        return gt.lower() in final_text.lower()
        
    if isinstance(gt, list):
        matches = sum(1 for item in gt if item.lower() in final_text.lower())
        return matches >= max(1, len(gt) // 2)
        
    if isinstance(gt, dict):
        matches = sum(1 for k, v in gt.items() if str(v) in final_text)
        return matches >= len(gt) // 2

    return False

def run_evaluation_suite():
    print("=" * 60)
    print("Running Final 8-Hour Sprint Controlled Evaluation Across 4 Modes")
    print(f"Total benchmark tasks: {len(BENCHMARK_TASKS)}")
    print("=" * 60)

    modes = [
        ("standalone", False, False, "Standalone (Both Tools OFF)"),
        ("python_only", False, True, "Python Execution Only"),
        ("web_only", True, False, "Web Retrieval Only"),
        ("both_tools", True, True, "Two-Tool Orchestrator (Config B)")
    ]

    all_results = {
        "metadata": {
            "title": "QORIN Final Sprint 4-Condition Evaluation Audit",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": MODEL_ID,
            "total_tasks": len(BENCHMARK_TASKS),
            "tool_expected_count": sum(1 for t in BENCHMARK_TASKS if t["tool_expected"]),
            "negative_control_count": sum(1 for t in BENCHMARK_TASKS if not t["tool_expected"])
        },
        "conditions": {}
    }

    for mode_key, web_en, py_en, mode_name in modes:
        print(f"\n>>> EVALUATING CONDITION: {mode_name} <<<")
        condition_record = {
            "mode_key": mode_key,
            "mode_name": mode_name,
            "web_enabled": web_en,
            "python_enabled": py_en,
            "tasks": []
        }

        tp = 0
        fp = 0
        tn = 0
        fn = 0
        tool_selection_correct = 0
        schema_valid_count = 0
        dispatch_count = 0
        execution_success_count = 0
        interpretation_count = 0
        e2e_success_count = 0

        for t in BENCHMARK_TASKS:
            tid = t["id"]
            prompt = t["prompt"]
            tool_expected = t["tool_expected"]
            exp_tool = t["expected_tool"]

            res = query_harness(prompt, web_enabled=web_en, py_enabled=py_en)
            
            tool_starts = res["tool_starts"]
            tool_results = res["tool_results"]
            final_text = res["final_text"]

            called_tool = len(tool_starts) > 0
            called_tool_name = tool_starts[0].get("tool_name") if called_tool else None

            # Confusion counts
            if tool_expected and called_tool:
                tp += 1
            elif not tool_expected and called_tool:
                fp += 1
            elif not tool_expected and not called_tool:
                tn += 1
            elif tool_expected and not called_tool:
                fn += 1

            # Tool selection correctness
            is_tool_correct = (called_tool_name == exp_tool) if (called_tool and exp_tool) else False
            if is_tool_correct:
                tool_selection_correct += 1

            # Schema valid
            is_schema_valid = any(tr.get("status") != "failed" or "syntax" not in str(tr.get("error", "")).lower() for tr in tool_results) if tool_results else False
            if is_schema_valid:
                schema_valid_count += 1

            # Dispatch success
            is_dispatched = len(tool_results) > 0
            if is_dispatched:
                dispatch_count += 1

            # Execution success (exit code 0 / completed)
            is_exec_ok = any(tr.get("status") == "completed" for tr in tool_results)
            if is_exec_ok:
                execution_success_count += 1

            # Interpretation (model produced synthesized final response after tool)
            has_synthesis = bool(called_tool and len(final_text.strip()) > 10)
            if has_synthesis:
                interpretation_count += 1

            # End to end task success
            is_task_correct = evaluate_task_correctness(t, final_text, tool_results)
            if is_task_correct:
                e2e_success_count += 1

            condition_record["tasks"].append({
                "id": tid,
                "type": t["type"],
                "prompt": prompt,
                "tool_expected": tool_expected,
                "expected_tool": exp_tool,
                "called_tool": called_tool,
                "called_tool_name": called_tool_name,
                "tool_selection_correct": is_tool_correct,
                "schema_valid": is_schema_valid,
                "dispatched": is_dispatched,
                "execution_success": is_exec_ok,
                "result_interpreted": has_synthesis,
                "end_to_end_success": is_task_correct,
                "final_snippet": final_text[:120],
                "tool_starts": tool_starts,
                "tool_results": tool_results
            })

            status_str = "OK" if is_task_correct else "FAIL"
            print(f"  [{tid}] called={called_tool} ({called_tool_name}), exec_ok={is_exec_ok}, e2e={status_str}")

        # Compute summary metrics
        total_tasks = len(BENCHMARK_TASKS)
        tool_req_tasks = sum(1 for t in BENCHMARK_TASKS if t["tool_expected"])
        no_tool_tasks = sum(1 for t in BENCHMARK_TASKS if not t["tool_expected"])

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fp == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        unnecessary_rate = fp / no_tool_tasks if no_tool_tasks > 0 else 0.0

        condition_record["metrics"] = {
            "total_tasks": total_tasks,
            "tool_required_tasks": tool_req_tasks,
            "negative_control_tasks": no_tool_tasks,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tool_selection_accuracy": round((tool_selection_correct / max(1, tp)) * 100, 2),
            "schema_validity_pct": round((schema_valid_count / max(1, tp)) * 100, 2),
            "dispatch_success_pct": round((dispatch_count / max(1, tp)) * 100, 2),
            "execution_success_pct": round((execution_success_count / max(1, dispatch_count)) * 100, 2),
            "result_interpretation_pct": round((interpretation_count / max(1, execution_success_count)) * 100, 2),
            "end_to_end_success_count": e2e_success_count,
            "end_to_end_success_pct": round((e2e_success_count / total_tasks) * 100, 2),
            "unnecessary_tool_call_rate": round(unnecessary_rate * 100, 2)
        }

        print(f"  --> Condition F1: {f1:.4f} | Recall: {rec*100:.1f}% | Precision: {prec*100:.1f}% | E2E Correct: {e2e_success_count}/{total_tasks} ({e2e_success_count/total_tasks*100:.1f}%)")
        all_results["conditions"][mode_key] = condition_record

    out_file = Path("modern_gpt/reports/final_8hour_sprint/updated_held_out_evaluation.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[DONE] Saved comprehensive evaluation to {out_file} ({out_file.stat().st_size} bytes)")

if __name__ == "__main__":
    run_evaluation_suite()
