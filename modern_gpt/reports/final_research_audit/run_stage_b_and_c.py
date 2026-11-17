import json
import re
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

API_URL = "http://127.0.0.1:8000/api/chat/stream"
MODEL_ID = "qorin-500m-sft-exp-c"

# Stage B: Independently Held-Out Benchmark
PYTHON_TASKS = [
    {
        "id": "py_heldout_01",
        "category": "arithmetic",
        "prompt": "Calculate (144 * 12) + (864 / 16) - (25 ** 2) using Python.",
        "ground_truth": 1157.0,
        "eval_type": "numerical"
    },
    {
        "id": "py_heldout_02",
        "category": "statistics",
        "prompt": "Given the dataset [18, 24, 30, 36, 42, 48, 54], compute the sample mean and variance using Python.",
        "ground_truth": {"mean": 36.0, "variance": 168.0},
        "eval_type": "statistical"
    },
    {
        "id": "py_heldout_03",
        "category": "data_analysis",
        "prompt": "Count the frequency of each vowel in 'artificialintelligence' using Python.",
        "ground_truth": {"a": 2, "e": 3, "i": 5, "o": 0, "u": 1},
        "eval_type": "data"
    },
    {
        "id": "py_heldout_04",
        "category": "error_recovery",
        "prompt": "In Python, safely access the value of key 'user_id' from {'session': 'active'} defaulting to -1 if missing.",
        "ground_truth": -1,
        "eval_type": "recovery"
    },
    {
        "id": "py_heldout_05",
        "category": "numerical_exact",
        "prompt": "What is 18 squared plus 24 squared? Compute the square root of the sum using Python.",
        "ground_truth": 30.0,
        "eval_type": "numerical"
    }
]

WEB_TASKS = [
    {
        "id": "web_live_01",
        "category": "live_retrieval",
        "prompt": "Search the web for the primary author and year of the Transformer paper 'Attention Is All You Need'.",
        "expected_facts": ["Vaswani", "2017"],
        "is_fixture": False
    },
    {
        "id": "web_live_02",
        "category": "live_retrieval",
        "prompt": "Search the web for what organization or benevolent dictator created Python programming language.",
        "expected_facts": ["Guido van Rossum", "Python"],
        "is_fixture": False
    },
    {
        "id": "web_fixture_03",
        "category": "controlled_fixture",
        "prompt": "Search the web for the exact parameter count and pretraining token count of ModernGPT-S architecture.",
        "fixture_sources": [
            {
                "title": "ModernGPT-S: Architecture Specification",
                "url": "https://research.moderngpt.internal/spec-500m",
                "snippet": "ModernGPT-S (QORIN) is an open 514M parameter decoder-only Transformer pretrained on 1.12 billion tokens with 32k vocab."
            }
        ],
        "expected_facts": ["514M", "1.12 billion"],
        "is_fixture": True
    },
    {
        "id": "web_unavailable_04",
        "category": "unavailable_query",
        "prompt": "Search the web for the secret warp-drive formula discovered on Mars by rover Curiosity in 1842.",
        "expected_behavior": "graceful_handling_no_hallucination",
        "is_fixture": True,
        "fixture_sources": []
    }
]

# Diverse No-Tool Tasks for Stage C Negative Control Generalization
DIVERSE_NO_TOOL_TASKS = [
    {"id": "ctl_01", "prompt": "What is photosynthesis in plant biology?"},
    {"id": "ctl_02", "prompt": "Define the concept of recursion in computer science."},
    {"id": "ctl_03", "prompt": "Who composed the Four Seasons?"},
    {"id": "ctl_04", "prompt": "Translate 'Good morning, how are you?' into Spanish."},
    {"id": "ctl_05", "prompt": "Summarize the law of supply and demand in economics."},
    {"id": "ctl_06", "prompt": "What are the primary colors in subtractive color theory?"},
    {"id": "ctl_07", "prompt": "Explain the difference between a virus and a bacterium."},
    {"id": "ctl_08", "prompt": "What is the capital of Australia?"},
    {"id": "ctl_09", "prompt": "Write a short haiku about autumn leaves."},
    {"id": "ctl_10", "prompt": "Explain what Occam's razor means."}
]

def query_sse_stream(prompt: str, python_enabled: bool, web_enabled: bool, temperature: float = 0.0) -> Dict[str, Any]:
    payload = {
        "model_id": MODEL_ID,
        "messages": [{"role": "user", "content": prompt}],
        "python_sandbox_enabled": python_enabled,
        "web_research_enabled": web_enabled,
        "temperature": temperature,
        "max_new_tokens": 120
    }
    
    events = []
    tool_starts = []
    tool_results = []
    tokens = []
    done_data = None
    
    try:
        resp = requests.post(API_URL, json=payload, stream=True, timeout=40)
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
                    data_obj = json.loads(data_str)
                except Exception:
                    data_obj = data_str
                
                if curr_event == "tool_start":
                    tool_starts.append(data_obj)
                elif curr_event == "tool_result":
                    tool_results.append(data_obj)
                elif curr_event == "token":
                    if isinstance(data_obj, dict) and "text" in data_obj:
                        tokens.append(data_obj["text"])
                elif curr_event == "done":
                    done_data = data_obj
                    
        final_text = "".join(tokens)
        return {
            "success": True,
            "tool_starts": tool_starts,
            "tool_results": tool_results,
            "final_text": final_text,
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

def run_stage_b_evaluation():
    print("Starting Stage B: Held-Out Tool Evaluation Suite across 4 Modes...")
    results = {
        "metadata": {
            "title": "Stage B Held-Out Tool Validation Suite",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": MODEL_ID,
            "conditions": ["standalone", "python_only", "web_only", "both_tools"]
        },
        "python_evaluation": {},
        "web_researcher_evaluation": {},
        "no_tool_control_generalization": {},
        "comparative_matrix": {}
    }

    modes = [
        ("standalone", False, False),
        ("python_only", True, False),
        ("web_only", False, True),
        ("both_tools", True, True)
    ]

    # 1. Run Python Tasks across all 4 modes
    print("\n--- Evaluating Python Tasks ---")
    results["python_evaluation"]["tasks"] = []
    for task in PYTHON_TASKS:
        task_record = {"task": task, "mode_results": {}}
        print(f"Executing Python task: {task['id']} - '{task['prompt']}'")
        for mode_name, py_en, web_en in modes:
            res = query_sse_stream(task["prompt"], python_enabled=py_en, web_enabled=web_en)
            tool_called = len(res["tool_starts"]) > 0
            tool_success = any(tr.get("status") == "completed" for tr in res["tool_results"])
            
            # Check numerical correctness
            final_ans = res["final_text"]
            gt = task["ground_truth"]
            is_num_correct = False
            if isinstance(gt, (int, float)):
                if str(gt) in final_ans or str(int(gt)) in final_ans:
                    is_num_correct = True
            elif isinstance(gt, dict):
                # check if keys/values present
                matches = sum(1 for k, v in gt.items() if str(v) in final_ans)
                is_num_correct = (matches >= len(gt) // 2)

            task_record["mode_results"][mode_name] = {
                "tool_called": tool_called,
                "tool_starts": res["tool_starts"],
                "tool_results": res["tool_results"],
                "tool_success": tool_success,
                "final_text": res["final_text"],
                "numerical_correct": is_num_correct
            }
            print(f"  [{mode_name}] called={tool_called}, exec_ok={tool_success}, ans_ok={is_num_correct}")
        results["python_evaluation"]["tasks"].append(task_record)

    # 2. Run Web Tasks across all 4 modes
    print("\n--- Evaluating Web Researcher Tasks ---")
    results["web_researcher_evaluation"]["tasks"] = []
    for task in WEB_TASKS:
        task_record = {"task": task, "mode_results": {}}
        print(f"Executing Web task: {task['id']} - '{task['prompt']}'")
        for mode_name, py_en, web_en in modes:
            res = query_sse_stream(task["prompt"], python_enabled=py_en, web_enabled=web_en)
            tool_called = len(res["tool_starts"]) > 0
            tool_name_used = res["tool_starts"][0].get("tool_name") if res["tool_starts"] else None
            tool_success = any(tr.get("status") == "completed" for tr in res["tool_results"])
            final_ans = res["final_text"]
            
            facts_found = []
            if "expected_facts" in task:
                for f in task["expected_facts"]:
                    if f.lower() in final_ans.lower():
                        facts_found.append(f)
            
            task_record["mode_results"][mode_name] = {
                "tool_called": tool_called,
                "tool_name_used": tool_name_used,
                "tool_starts": res["tool_starts"],
                "tool_results": res["tool_results"],
                "tool_success": tool_success,
                "final_text": final_ans,
                "facts_found": facts_found
            }
            print(f"  [{mode_name}] called={tool_called} ({tool_name_used}), exec_ok={tool_success}, facts={facts_found}")
        results["web_researcher_evaluation"]["tasks"].append(task_record)

    # 3. Run Diverse No-Tool Tasks on both_tools mode to test Stage C generalization
    print("\n--- Evaluating Diverse No-Tool Controls (Config B) ---")
    results["no_tool_control_generalization"]["tasks"] = []
    false_positives = 0
    for task in DIVERSE_NO_TOOL_TASKS:
        res = query_sse_stream(task["prompt"], python_enabled=True, web_enabled=True)
        tool_called = len(res["tool_starts"]) > 0
        if tool_called:
            false_positives += 1
        rec = {
            "id": task["id"],
            "prompt": task["prompt"],
            "tool_called": tool_called,
            "tool_starts": res["tool_starts"],
            "response_snippet": res["final_text"][:120]
        }
        results["no_tool_control_generalization"]["tasks"].append(rec)
        print(f"  [{task['id']}] Tool called: {tool_called}")

    fp_rate = false_positives / len(DIVERSE_NO_TOOL_TASKS)
    results["no_tool_control_generalization"]["metrics"] = {
        "total_tasks": len(DIVERSE_NO_TOOL_TASKS),
        "false_positive_count": false_positives,
        "false_positive_rate": fp_rate,
        "accuracy": 1.0 - fp_rate
    }
    print(f"\nNo-Tool False Positive Rate on Diverse Benchmark: {fp_rate * 100:.1f}%")

    out_path = Path("modern_gpt/reports/final_research_audit/live_tool_evaluation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path} ({out_path.stat().st_size} bytes)")

if __name__ == "__main__":
    run_stage_b_evaluation()
