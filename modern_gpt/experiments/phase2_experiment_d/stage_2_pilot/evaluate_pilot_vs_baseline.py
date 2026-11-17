"""
Held-out Evaluation Benchmark: Experiment C Baseline vs Experiment D Pilot (Step 50)
Evaluates 30 standardized test tasks across 6 capability categories:
1. Colloquial Web Intent Recognition (5 tasks)
2. Explicit Web Requests (5 tasks)
3. Python Selection & Execution (5 tasks)
4. Direct Conceptual / Retention (5 tasks)
5. Selection Discrimination Matched Pairs (5 tasks)
6. Multi-Step Tool Use (5 tasks)
"""

import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from modern_gpt.model import ModernGPT, ModelConfig

PILOT_DIR = Path(__file__).resolve().parent
EXP_C_CKPT_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt"
PILOT_CKPT_PATH = PILOT_DIR / "checkpoints/ckpt_pilot_step50.pt"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
OUTPUT_JSON = PILOT_DIR / "logs/pilot_evaluation_results.json"

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Execute Python code in an isolated sandbox.
2. `web_research`: Query real-time search engine for current facts, news, and live information.
To call a tool, generate:
<|tool_call|>{"name": "<tool_name>", "arguments": {<args>}}"""

BENCHMARK_TASKS = [
    # 1. Colloquial Web Intent Recognition (5 tasks)
    {
        "id": "web_colloquial_01",
        "category": "web_colloquial",
        "prompt": "grab info on elon musk",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "musk" in json.dumps(args).lower()
    },
    {
        "id": "web_colloquial_02",
        "category": "web_colloquial",
        "prompt": "who is the current ceo of microsoft?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "microsoft" in json.dumps(args).lower()
    },
    {
        "id": "web_colloquial_03",
        "category": "web_colloquial",
        "prompt": "what is the latest status of the artemis moon mission?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "artemis" in json.dumps(args).lower()
    },
    {
        "id": "web_colloquial_04",
        "category": "web_colloquial",
        "prompt": "who won the 2024 premier league title?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("premier league" in json.dumps(args).lower() or "2024" in json.dumps(args))
    },
    {
        "id": "web_colloquial_05",
        "category": "web_colloquial",
        "prompt": "find recent news on deepmind releases",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "deepmind" in json.dumps(args).lower()
    },

    # 2. Explicit Web Requests (5 tasks)
    {
        "id": "web_explicit_01",
        "category": "web_explicit",
        "prompt": "Search the web for python 3.13 key features and release date.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "python" in json.dumps(args).lower()
    },
    {
        "id": "web_explicit_02",
        "category": "web_explicit",
        "prompt": "Look up the current population of Tokyo in 2024.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "tokyo" in json.dumps(args).lower()
    },
    {
        "id": "web_explicit_03",
        "category": "web_explicit",
        "prompt": "Search online for nvidia rtx 5090 specifications and architecture details.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "rtx 5090" in json.dumps(args).lower()
    },
    {
        "id": "web_explicit_04",
        "category": "web_explicit",
        "prompt": "Search the web for the current stock price of Apple.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("apple" in json.dumps(args).lower() or "aapl" in json.dumps(args).lower())
    },
    {
        "id": "web_explicit_05",
        "category": "web_explicit",
        "prompt": "Look up information about the James Webb Space Telescope latest discoveries.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("james webb" in json.dumps(args).lower() or "jwst" in json.dumps(args).lower())
    },

    # 3. Python Selection & Execution (5 tasks)
    {
        "id": "python_core_01",
        "category": "python_core",
        "prompt": "Calculate 25 * 40 using python.",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and "25" in json.dumps(args) and "40" in json.dumps(args)
    },
    {
        "id": "python_core_02",
        "category": "python_core",
        "prompt": "Compute the standard deviation of [12, 18, 24, 30, 36, 42] using python.",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and "std" in json.dumps(args).lower() or "statistics" in json.dumps(args)
    },
    {
        "id": "python_core_03",
        "category": "python_core",
        "prompt": "Calculate the factorial of 14 using python.",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and ("14" in json.dumps(args) or "factorial" in json.dumps(args))
    },
    {
        "id": "python_core_04",
        "category": "python_core",
        "prompt": "Calculate the greatest common divisor of 123456 and 7890 using python math.gcd.",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and "gcd" in json.dumps(args)
    },
    {
        "id": "python_core_05",
        "category": "python_core",
        "prompt": "Is 104729 a prime number? Write a python script to test it.",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and "104729" in json.dumps(args)
    },

    # 4. Direct Conceptual / Retention (5 tasks)
    {
        "id": "direct_conceptual_01",
        "category": "direct_conceptual",
        "prompt": "Hello! How are you doing today?",
        "expected_intent": False,
        "expected_tool": None,
        "check_fn": lambda tool, args, ans: tool is None and len(ans.strip()) > 5
    },
    {
        "id": "direct_conceptual_02",
        "category": "direct_conceptual",
        "prompt": "Explain the difference between a stack and a queue in data structures.",
        "expected_intent": False,
        "expected_tool": None,
        "check_fn": lambda tool, args, ans: tool is None and "lifo" in ans.lower() or "fifo" in ans.lower()
    },
    {
        "id": "direct_conceptual_03",
        "category": "direct_conceptual",
        "prompt": "What is backpropagation in neural networks and how does it compute gradients?",
        "expected_intent": False,
        "expected_tool": None,
        "check_fn": lambda tool, args, ans: tool is None and ("chain rule" in ans.lower() or "gradient" in ans.lower())
    },
    {
        "id": "direct_conceptual_04",
        "category": "direct_conceptual",
        "prompt": "Define what an NP-complete problem is in computational complexity theory.",
        "expected_intent": False,
        "expected_tool": None,
        "check_fn": lambda tool, args, ans: tool is None and ("np" in ans.lower() or "polynomial" in ans.lower())
    },
    {
        "id": "direct_conceptual_05",
        "category": "direct_conceptual",
        "prompt": "Write a short haiku about artificial intelligence.",
        "expected_intent": False,
        "expected_tool": None,
        "check_fn": lambda tool, args, ans: tool is None and len(ans.strip()) > 10
    },

    # 5. Selection Discrimination Matched Pairs (5 tasks)
    {
        "id": "selection_discrim_01",
        "category": "selection_discrimination",
        "prompt": "Calculate the compound interest on $10,000 invested at 5% annual interest compounded monthly for 10 years.",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and "10000" in json.dumps(args)
    },
    {
        "id": "selection_discrim_02",
        "category": "selection_discrimination",
        "prompt": "Who was the historical mathematician that first published compound interest tables?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and "compound interest" in json.dumps(args).lower()
    },
    {
        "id": "selection_discrim_03",
        "category": "selection_discrimination",
        "prompt": "Explain what gradient descent optimization is mathematically.",
        "expected_intent": False,
        "expected_tool": None,
        "check_fn": lambda tool, args, ans: tool is None and ("gradient" in ans.lower() or "learning rate" in ans.lower())
    },
    {
        "id": "selection_discrim_04",
        "category": "selection_discrimination",
        "prompt": "What is the latest published benchmark score of Gemini 1.5 Pro on MATH benchmark?",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("gemini" in json.dumps(args).lower() or "math" in json.dumps(args).lower())
    },
    {
        "id": "selection_discrim_05",
        "category": "selection_discrimination",
        "prompt": "Sort the following array of integers using Python: [45, 12, 89, 34, 7, 23]",
        "expected_intent": True,
        "expected_tool": "python",
        "check_fn": lambda tool, args, ans: tool == "python" and "sort" in json.dumps(args).lower()
    },

    # 6. Multi-Step Tool Use (5 tasks)
    {
        "id": "multistep_01",
        "category": "multistep_web_python",
        "prompt": "Find the annual revenue of Apple and Microsoft for fiscal year 2023, then compute the difference in billions.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("apple" in json.dumps(args).lower() or "microsoft" in json.dumps(args).lower() or "revenue" in json.dumps(args).lower())
    },
    {
        "id": "multistep_02",
        "category": "multistep_web_python",
        "prompt": "Look up the closing stock price of Tesla on December 29, 2023 and compute the percentage change from $200.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("tesla" in json.dumps(args).lower() or "tsla" in json.dumps(args).lower())
    },
    {
        "id": "multistep_03",
        "category": "multistep_web_python",
        "prompt": "Find the distance in kilometers between Paris and Rome, then calculate travel time driving at 85 km/h.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("paris" in json.dumps(args).lower() and "rome" in json.dumps(args).lower())
    },
    {
        "id": "multistep_04",
        "category": "multistep_web_python",
        "prompt": "Look up the population of Australia and Canada in 2023, then calculate the ratio between them.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("australia" in json.dumps(args).lower() or "canada" in json.dumps(args).lower())
    },
    {
        "id": "multistep_05",
        "category": "multistep_web_python",
        "prompt": "Find the atomic weight of Titanium and Vanadium, then compute the average of both using Python.",
        "expected_intent": True,
        "expected_tool": "web_research",
        "check_fn": lambda tool, args, ans: tool == "web_research" and ("titanium" in json.dumps(args).lower() or "vanadium" in json.dumps(args).lower() or "atomic" in json.dumps(args).lower())
    }
]

def generate_greedy(model, tok, prompt_text: str, max_new_tokens: int = 256, device: str = "cuda") -> str:
    input_ids = tok.encode(prompt_text).ids
    generated = list(input_ids)
    stop_ids = {5, 9}  # <|im_end|>=5, <|endoftext|>=9

    for _ in range(max_new_tokens):
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inp)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if next_id in stop_ids:
            break
        generated.append(next_id)

    new_ids = generated[len(input_ids):]
    return tok.decode(new_ids)

def parse_model_output(output_text: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]], bool]:
    has_tool_call = "<|tool_call|>" in output_text or "<tool_call>" in output_text
    if not has_tool_call:
        return False, None, None, True

    # Extract JSON inside or after tool_call tag
    tag = "<|tool_call|>" if "<|tool_call|>" in output_text else "<tool_call>"
    parts = output_text.split(tag, 1)
    call_str = parts[1].strip()
    if "</tool_call>" in call_str:
        call_str = call_str.split("</tool_call>")[0].strip()

    try:
        data = json.loads(call_str)
        name = data.get("name")
        args = data.get("arguments", {})
        return True, name, args, True
    except Exception:
        # Check if valid JSON substring exists
        brace_start = call_str.find("{")
        brace_end = call_str.rfind("}")
        if brace_start != -1 and brace_end != -1:
            try:
                sub = call_str[brace_start:brace_end+1]
                data = json.loads(sub)
                return True, data.get("name"), data.get("arguments", {}), True
            except Exception:
                pass
        return True, None, None, False

def evaluate_model_on_benchmark(model, tok, device: str, model_name: str) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Evaluating {model_name} on {len(BENCHMARK_TASKS)} Held-out Tasks...")
    print(f"=======================================================")

    results = []
    tp = fp = tn = fn = 0
    correct_tool_selection = 0
    valid_syntax = 0
    total_tool_expected = sum(1 for t in BENCHMARK_TASKS if t["expected_intent"])
    total_direct_expected = sum(1 for t in BENCHMARK_TASKS if not t["expected_intent"])
    web_expected_count = sum(1 for t in BENCHMARK_TASKS if t["expected_tool"] == "web_research")
    web_correct_count = 0
    python_expected_count = sum(1 for t in BENCHMARK_TASKS if t["expected_tool"] == "python")
    python_correct_count = 0
    semantic_success_count = 0

    for idx, task in enumerate(BENCHMARK_TASKS):
        prompt_formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{task['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        raw_output = generate_greedy(model, tok, prompt_formatted, max_new_tokens=256, device=device)
        has_tool, tool_name, args, syntax_ok = parse_model_output(raw_output)

        # Confusion Matrix
        expected_intent = task["expected_intent"]
        if expected_intent:
            if has_tool:
                tp += 1
            else:
                fn += 1
        else:
            if has_tool:
                fp += 1
            else:
                tn += 1

        # Tool selection
        tool_correct = False
        if expected_intent and has_tool:
            if tool_name == task["expected_tool"]:
                correct_tool_selection += 1
                tool_correct = True
                if task["expected_tool"] == "web_research":
                    web_correct_count += 1
                elif task["expected_tool"] == "python":
                    python_correct_count += 1

        # Syntax
        if has_tool and syntax_ok:
            valid_syntax += 1

        # Semantic check
        try:
            semantic_pass = task["check_fn"](tool_name if has_tool else None, args, raw_output)
        except Exception:
            semantic_pass = False

        if semantic_pass:
            semantic_success_count += 1

        res = {
            "id": task["id"],
            "category": task["category"],
            "prompt": task["prompt"],
            "expected_intent": expected_intent,
            "detected_intent": has_tool,
            "expected_tool": task["expected_tool"],
            "selected_tool": tool_name if has_tool else None,
            "tool_correct": tool_correct,
            "syntax_valid": syntax_ok if has_tool else "N/A",
            "semantic_success": semantic_pass,
            "raw_output": raw_output[:300]
        }
        results.append(res)
        print(f"[{idx+1:02d}/{len(BENCHMARK_TASKS)}] {task['id']:<22} | Exp: {str(task['expected_tool']):<12} | Act: {str(tool_name):<12} | Intent: {'OK' if has_tool == expected_intent else 'FAIL'} | Semantic: {'PASS' if semantic_pass else 'FAIL'}")

    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-8, (precision + recall))

    summary = {
        "model_name": model_name,
        "total_tasks": len(BENCHMARK_TASKS),
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "intent_accuracy": round((tp + tn) / len(BENCHMARK_TASKS), 4)
        },
        "tool_selection": {
            "overall_accuracy": round(correct_tool_selection / max(1, total_tool_expected), 4),
            "web_recall": round(web_correct_count / max(1, web_expected_count), 4),
            "web_correct": web_correct_count,
            "web_expected": web_expected_count,
            "python_recall": round(python_correct_count / max(1, python_expected_count), 4),
            "python_correct": python_correct_count,
            "python_expected": python_expected_count
        },
        "syntax_validity": {
            "valid_calls": valid_syntax,
            "total_calls": (tp + fp),
            "syntax_rate": round(valid_syntax / max(1, (tp + fp)), 4)
        },
        "end_to_end_semantic_success": {
            "success_count": semantic_success_count,
            "total_tasks": len(BENCHMARK_TASKS),
            "rate": round(semantic_success_count / len(BENCHMARK_TASKS), 4)
        },
        "tasks": results
    }
    return summary

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = ModelConfig.get_500m_config()

    # 1. Evaluate Experiment C Baseline
    print("Loading Experiment C Baseline Model...")
    model_c = ModernGPT(cfg).to(device)
    ckpt_c = torch.load(EXP_C_CKPT_PATH, map_location=device, weights_only=False)
    model_c.load_state_dict(ckpt_c["model_state_dict"], strict=True)
    model_c.eval()
    eval_c = evaluate_model_on_benchmark(model_c, tok, device, "Experiment C Baseline")
    del model_c, ckpt_c
    torch.cuda.empty_cache()

    # 2. Evaluate Experiment D Pilot
    print("\nLoading Experiment D Pilot Model (Step 50)...")
    model_d = ModernGPT(cfg).to(device)
    ckpt_d = torch.load(PILOT_CKPT_PATH, map_location=device, weights_only=False)
    model_d.load_state_dict(ckpt_d["model_state_dict"], strict=True)
    model_d.eval()
    eval_d = evaluate_model_on_benchmark(model_d, tok, device, "Experiment D Pilot (Step 50)")
    del model_d, ckpt_d
    torch.cuda.empty_cache()

    comparison = {
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_benchmark_tasks": len(BENCHMARK_TASKS),
        "experiment_c_baseline": eval_c,
        "experiment_d_pilot": eval_d,
        "comparative_deltas": {
            "tool_intent_f1_delta": round(eval_d["confusion_matrix"]["f1"] - eval_c["confusion_matrix"]["f1"], 4),
            "web_recall_delta": round(eval_d["tool_selection"]["web_recall"] - eval_c["tool_selection"]["web_recall"], 4),
            "python_recall_delta": round(eval_d["tool_selection"]["python_recall"] - eval_c["tool_selection"]["python_recall"], 4),
            "semantic_success_delta": round(eval_d["end_to_end_semantic_success"]["rate"] - eval_c["end_to_end_semantic_success"]["rate"], 4)
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nComparative evaluation results written to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
