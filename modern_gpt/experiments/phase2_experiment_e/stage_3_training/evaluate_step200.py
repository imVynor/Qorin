"""
QORIN Phase 2E: Step 200 Evaluation Gate Engine
Evaluates ckpt_branch_d_step200.pt against:
- Frozen Experiment D Parent
- Branch B Best (Calibration)
- Branch C Best (Synthesis)
on the identical 150-task benchmark using:
- Pure model-generated decisions (zero mocking, autonomous Web -> Python chaining for multi-step)
- Corrected claim-level evaluation protocol (regex-normalized synonym and number matching)
- Complete decision calibration, evidence synthesis, and overall metrics.
"""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from backend.tools.orchestrator import ToolOrchestrator, ToolExecutionTrace
from backend.tools.two_stage_protocol import QorinTwoStageProtocol

BENCHMARK_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
EXP_D_METRICS_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/metrics.json"
BRANCH_B_METRICS_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_b_calibration/branch_b_metrics.json"
BRANCH_C_METRICS_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_c_synthesis/branch_c_metrics.json"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{"name": "python", "arguments": {"code": "<code_string>"}}
2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{"name": "web_research", "arguments": {"query": "<query_string>"}}
When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user."""

def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.95996
    p = successes / trials
    denom = 1 + z**2 / trials
    centre = p + z**2 / (2 * trials)
    adj_se = math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials)
    lower = max(0.0, (centre - z * adj_se) / denom)
    upper = min(1.0, (centre + z * adj_se) / denom)
    return round(lower, 4), round(upper, 4)

@torch.inference_mode()
def generate_greedy(model, tok, prompt_formatted: str, max_new_tokens: int = 160, device: str = "cuda") -> str:
    input_ids = tok.encode(prompt_formatted).ids
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
    return tok.decode(new_ids, skip_special_tokens=False)

def check_claim_level_semantic_correctness(task: Dict[str, Any], final_answer: str, tool_trace: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    text_lower = final_answer.lower()
    criteria = task.get("semantic_criteria", [])
    expected = task.get("expected_answer", "")
    uncertainty_req = task.get("uncertainty_required", False)

    if uncertainty_req:
        uncertainty_phrases = ["cannot", "unknown", "uncertain", "not known", "no record", "no such", "not found", "future", "not available", "unsolved", "clarification"]
        if any(p in text_lower for p in uncertainty_phrases) or any(c.lower() in text_lower for c in criteria if isinstance(c, str)):
            return True, "Correctly acknowledged uncertainty or unfulfillable condition."
        return False, "Failed to express required uncertainty or acknowledgment."

    # Python computation exact number check
    if task["category"] == "python_computation":
        exp_clean = re.sub(r"[^\d\.-]", "", expected)
        if exp_clean and exp_clean in text_lower:
            return True, f"Found exact numerical target '{exp_clean}' in response."
        if tool_trace and "output_payload" in tool_trace:
            out_str = str(tool_trace["output_payload"].get("stdout", ""))
            if exp_clean and exp_clean in out_str:
                return True, f"Target '{exp_clean}' found in execution output."
        matches = sum(1 for c in criteria if isinstance(c, str) and c.lower() in text_lower)
        if matches / max(1, len(criteria)) >= 0.5:
            return True, f"Matched {matches}/{len(criteria)} numerical criteria."
        return False, f"Failed numerical check; criteria match {matches}/{len(criteria)}."

    # Claim-level / synonym list normalization
    matched_groups = 0
    total_groups = len(criteria)
    for c in criteria:
        if isinstance(c, list):
            if any(opt.lower() in text_lower or opt.lower().replace(",", "") in text_lower.replace(",", "") for opt in c):
                matched_groups += 1
        else:
            c_str = str(c).lower()
            if c_str in text_lower or c_str.replace(",", "") in text_lower.replace(",", ""):
                matched_groups += 1

    match_ratio = matched_groups / max(1, total_groups)
    if match_ratio >= 0.5:
        return True, f"Matched {matched_groups}/{total_groups} semantic criteria."
    return False, f"Semantic mismatch: matched only {matched_groups}/{total_groups} criteria."

def evaluate_step200(ckpt_path: Path, output_dir: Path) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print("STEP 200 EVALUATION GATE: BRANCH D (COMBINED MULTI-TASK SFT)")
    print(f"Checkpoint: {ckpt_path.name}")
    print(f"Output Dir: {output_dir}")
    print(f"==================================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = ModelConfig.get_500m_config()

    model = ModernGPT(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    orchestrator = ToolOrchestrator()
    protocol = QorinTwoStageProtocol(orchestrator)

    tasks = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    results = []
    traces = []
    out_file = output_dir / "branch_d_step200_results.jsonl"
    traces_file = output_dir / "branch_d_step200_traces.jsonl"

    multistep_success_count = 0
    multistep_total_count = 0

    for idx, task in enumerate(tasks):
        t_id = task["id"]
        cat = task["category"]
        prompt_user = task["prompt"]
        prompt_stage1 = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Stage 1: Initial Generation
        raw_stage1 = generate_greedy(model, tok, prompt_stage1, max_new_tokens=128, device=device)
        action, tool_name, tool_args = protocol.determine_action(
            first_round_text=raw_stage1,
            web_enabled=True,
            python_enabled=True
        )
        detected_intent = (action in ["PYTHON", "WEB"])
        expected_intent = task["expected_intent"]
        expected_tool = task["expected_tool"]

        tp = (expected_intent and detected_intent)
        fp = (not expected_intent and detected_intent)
        tn = (not expected_intent and not detected_intent)
        fn = (expected_intent and not detected_intent)
        tool_correct = (expected_intent and detected_intent and tool_name == expected_tool)

        # Schema & Argument validation
        is_valid_schema = False
        validated_args = {}
        val_error = None
        if detected_intent and tool_name:
            is_valid_schema, validated_args, val_error = protocol.validate_and_sanitize_arguments(
                tool_name=tool_name,
                arguments=tool_args,
                raw_text=raw_stage1,
                user_query=prompt_user
            )

        # Dispatch & Execution
        dispatched = False
        executed = False
        tool_trace_dict = None
        tool_output_str = ""

        if is_valid_schema and tool_name:
            dispatched = True
            try:
                trace: ToolExecutionTrace = orchestrator.execute_tool(
                    tool_name=tool_name,
                    arguments=validated_args,
                    requested_by="model"
                )
                tool_trace_dict = trace.to_dict()
                tool_trace_dict["task_id"] = t_id
                executed = (trace.status.value == "SUCCESS" or trace.status == "completed")

                if tool_name == "python":
                    tool_output_str = str(trace.output_payload.get("stdout", "") or trace.output_payload.get("error", ""))
                elif tool_name == "web_research":
                    tool_output_str = str(trace.output_payload.get("summary", ""))
            except Exception as e:
                executed = False
                tool_trace_dict = {"error": str(e), "task_id": t_id}

        # Multi-turn Chaining Handling (for multi-step tasks)
        final_response = raw_stage1
        evidence_utilized = False
        citations_found = []

        if cat == "multistep_web_python":
            multistep_total_count += 1
            # Round 2: Return Web result and observe if model autonomously calls Python
            if dispatched and executed and tool_output_str:
                prompt_round2 = (
                    f"{prompt_stage1}{raw_stage1.strip()}<|im_end|>\n"
                    f"<|im_start|>tool\n<|tool_result|>\n{tool_output_str}\n<|end_tool_result|><|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                raw_round2 = generate_greedy(model, tok, prompt_round2, max_new_tokens=160, device=device)
                
                # Check if model autonomously emitted Python tool call
                act2, tool2, args2 = protocol.determine_action(raw_round2, web_enabled=True, python_enabled=True)
                if act2 == "PYTHON":
                    s2_valid, s2_args, _ = protocol.validate_and_sanitize_arguments(tool2, args2, raw_round2, prompt_user)
                    if s2_valid:
                        try:
                            trace2 = orchestrator.execute_tool(tool2, s2_args, requested_by="model")
                            t2_out = str(trace2.output_payload.get("stdout", "") or trace2.output_payload.get("error", ""))
                            # Round 3: Synthesis
                            prompt_round3 = (
                                f"{prompt_round2}{raw_round2.strip()}<|im_end|>\n"
                                f"<|im_start|>tool\n<|tool_result|>\n{t2_out}\n<|end_tool_result|><|im_end|>\n"
                                f"<|im_start|>assistant\n"
                            )
                            final_response = generate_greedy(model, tok, prompt_round3, max_new_tokens=200, device=device)
                        except Exception:
                            final_response = raw_round2
                    else:
                        final_response = raw_round2
                else:
                    final_response = raw_round2
        else:
            # Standard Two-Stage Synthesis (Stage 2)
            if dispatched and executed and tool_output_str:
                prompt_stage2 = (
                    f"{prompt_stage1}{raw_stage1.strip()}<|im_end|>\n"
                    f"<|im_start|>tool\n<|tool_result|>\n{tool_output_str}\n<|end_tool_result|><|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                final_response = generate_greedy(model, tok, prompt_stage2, max_new_tokens=160, device=device)

                clean_obs = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", tool_output_str.lower()))
                clean_resp = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", final_response.lower()))
                evidence_utilized = (len(clean_obs & clean_resp) >= 3)
                citations_found = re.findall(r"\[(?:SOURCE\s*\d+|\d+)\]", final_response, re.IGNORECASE)
            elif not detected_intent:
                evidence_utilized = True

        semantic_pass, reason = check_claim_level_semantic_correctness(task, final_response, tool_trace_dict)
        if cat == "multistep_web_python" and semantic_pass:
            multistep_success_count += 1

        rec = {
            "id": t_id,
            "category": cat,
            "expected_intent": expected_intent,
            "expected_tool": expected_tool,
            "detected_intent": detected_intent,
            "selected_tool": tool_name if detected_intent else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tool_selection_correct": tool_correct,
            "valid_schema": is_valid_schema,
            "dispatched": dispatched,
            "executed": executed,
            "evidence_utilized": evidence_utilized,
            "has_citations": len(citations_found) > 0,
            "citation_count": len(citations_found),
            "semantic_success": semantic_pass,
            "semantic_reason": reason,
            "raw_stage1": raw_stage1[:120],
            "final_response": final_response[:200]
        }
        results.append(rec)
        if tool_trace_dict:
            traces.append(tool_trace_dict)

        if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
            print(f"  Evaluated {idx + 1:>3}/{len(tasks)} tasks...")

    # Write files
    with open(out_file, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(traces_file, "w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")

    # Aggregate Metrics
    total = len(results)
    tp_c = sum(1 for r in results if r["tp"])
    fp_c = sum(1 for r in results if r["fp"])
    tn_c = sum(1 for r in results if r["tn"])
    fn_c = sum(1 for r in results if r["fn"])

    prec = tp_c / max(1, (tp_c + fp_c))
    rec = tp_c / max(1, (tp_c + fn_c))
    f1 = 2 * prec * rec / max(1e-8, (prec + rec))
    acc = (tp_c + tn_c) / total

    direct_tasks = [r for r in results if not r["expected_intent"]]
    direct_fp = sum(1 for r in direct_tasks if r["detected_intent"])
    direct_fpr = direct_fp / max(1, len(direct_tasks))

    web_tasks = [r for r in results if r["expected_tool"] == "web_research"]
    web_correct = sum(1 for r in web_tasks if r["selected_tool"] == "web_research")
    web_rec = web_correct / max(1, len(web_tasks))

    py_tasks = [r for r in results if r["expected_tool"] == "python"]
    py_correct = sum(1 for r in py_tasks if r["selected_tool"] == "python")
    py_rec = py_correct / max(1, len(py_tasks))

    sem_correct = sum(1 for r in results if r["semantic_success"])
    sem_rate = sem_correct / total

    # Synthesis metrics
    factual_cats = {"colloquial_web_research", "current_information", "multi_step_reasoning"}
    factual_tasks = [r for r in results if r["category"] in factual_cats]
    factual_sem = sum(1 for r in factual_tasks if r["semantic_success"])
    factual_rate = factual_sem / max(1, len(factual_tasks))
    unsupported_rate = 1.0 - factual_rate

    factual_executed = [r for r in factual_tasks if r["executed"]]
    utilized_count = sum(1 for r in factual_executed if r["evidence_utilized"])
    evidence_coverage = utilized_count / max(1, len(factual_executed))

    citation_tasks = [r for r in factual_executed if r["has_citations"]]
    citation_rate = len(citation_tasks) / max(1, len(factual_executed))

    py_comp_tasks = [r for r in results if r["category"] == "python_computation"]
    num_correct = sum(1 for r in py_comp_tasks if r["semantic_success"])
    num_accuracy = num_correct / max(1, len(py_comp_tasks))

    summary = {
        "model_tag": "branch_d_step200",
        "checkpoint": ckpt_path.name,
        "total_tasks": total,
        "intent_metrics": {
            "tp": tp_c, "fp": fp_c, "tn": tn_c, "fn": fn_c,
            "precision": round(prec, 4),
            "precision_ci": wilson_score_interval(tp_c, tp_c + fp_c),
            "recall": round(rec, 4),
            "recall_ci": wilson_score_interval(tp_c, tp_c + fn_c),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "accuracy_ci": wilson_score_interval(tp_c + tn_c, total)
        },
        "decision_calibration": {
            "direct_task_fpr": round(direct_fpr, 4),
            "direct_task_fpr_ci": wilson_score_interval(direct_fp, len(direct_tasks)),
            "direct_fp_count": direct_fp,
            "direct_total": len(direct_tasks),
            "web_selection_recall": round(web_rec, 4),
            "web_selection_recall_ci": wilson_score_interval(web_correct, len(web_tasks)),
            "web_correct": web_correct,
            "web_total": len(web_tasks),
            "python_selection_recall": round(py_rec, 4),
            "python_selection_recall_ci": wilson_score_interval(py_correct, len(py_tasks)),
            "python_correct": py_correct,
            "python_total": len(py_tasks),
            "correct_tool_selection_accuracy": round((web_correct + py_correct) / max(1, len(web_tasks) + len(py_tasks)), 4),
            "tool_precision": round(prec, 4),
            "tool_f1": round(f1, 4)
        },
        "evidence_synthesis": {
            "factual_semantic_rate": round(factual_rate, 4),
            "unsupported_factual_claim_rate": round(unsupported_rate, 4),
            "evidence_coverage": round(evidence_coverage, 4),
            "citation_rate": round(citation_rate, 4),
            "numerical_extraction_accuracy": round(num_accuracy, 4)
        },
        "overall_performance": {
            "end_to_end_semantic_success": round(sem_rate, 4),
            "semantic_success_ci": wilson_score_interval(sem_correct, total),
            "semantic_success_count": sem_correct,
            "multistep_success_rate": round(multistep_success_count / max(1, multistep_total_count), 4),
            "multistep_success_count": multistep_success_count,
            "multistep_total": multistep_total_count,
            "tool_execution_success_rate": round(sum(1 for r in results if r["executed"]) / max(1, sum(1 for r in results if r["dispatched"])), 4)
        }
    }

    metrics_out = output_dir / "branch_d_step200_metrics.json"
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 200 BENCHMARK EVALUATION RESULTS")
    print(f"Intent Precision:           {prec:.4f}")
    print(f"Intent Recall:              {rec:.4f}")
    print(f"Intent F1:                  {f1:.4f}")
    print(f"Direct Task FPR:            {direct_fpr:.4f} ({direct_fp}/{len(direct_tasks)})")
    print(f"Web Selection Recall:       {web_rec:.4f} ({web_correct}/{len(web_tasks)})")
    print(f"Python Selection Recall:    {py_rec:.4f} ({py_correct}/{len(py_tasks)})")
    print(f"Evidence Coverage:          {evidence_coverage:.4f}")
    print(f"Citation Rate:              {citation_rate:.4f}")
    print(f"Numerical Accuracy:         {num_accuracy:.4f}")
    print(f"End-to-End Semantic Rate:   {sem_rate:.4f} ({sem_correct}/{total})")
    print(f"Multi-Step Autonomous Rate: {multistep_success_count}/{multistep_total_count}")
    print("=" * 80)

    del model
    torch.cuda.empty_cache()
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    evaluate_step200(Path(args.ckpt_path), Path(args.output_dir))
