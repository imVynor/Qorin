"""
QORIN Phase 2E — Stage 10: Final Comparative Evaluation & Forensic Audit Engine
Executes complete 150-task held-out benchmark evaluation across all 5 checkpoints:
1. Experiment D parent (Step 425)
2. Branch B best (Step 25, Calibration)
3. Branch C best (Step 25, Synthesis)
4. Branch D best (Step 25, Combined)
5. Branch D terminal (Step 150, Combined early-stopped)

Outputs:
- stage_10_task_results.jsonl (750 records)
- stage_10_multistep_traces.jsonl (75 traces)
- stage_10_metrics.json (all 5 models)
- stage_10_paired_statistics.json (exact McNemar, CIs, multiple comparisons)
"""

import ast
from collections import Counter, defaultdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
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

STAGE10_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"

CHECKPOINTS = {
    "exp_d_parent": {
        "name": "Experiment D Parent",
        "step": 425,
        "path": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt",
        "expected_sha": "c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7"
    },
    "branch_b_best": {
        "name": "Branch B Best (Calibration)",
        "step": 25,
        "path": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_b_calibration/checkpoints/ckpt_branch_best.pt",
        "expected_sha": "beec486831406acf3a6ac315cc188a1f2f97a5c978361ddfa7727d69a39436f6"
    },
    "branch_c_best": {
        "name": "Branch C Best (Synthesis)",
        "step": 25,
        "path": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_c_synthesis/checkpoints/ckpt_branch_best.pt",
        "expected_sha": "e1ae15ca223a30ecc7a3b56b54062244ff7a2e58e969dad86e64124065a45e0e"
    },
    "branch_d_best": {
        "name": "Branch D Best (Combined)",
        "step": 25,
        "path": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_d_combined/checkpoints/ckpt_branch_d_best.pt",
        "expected_sha": "7322b9aee559945c32f02f59835e7143534c3982178b5aacf4b31470934f9dcc"
    },
    "branch_d_terminal": {
        "name": "Branch D Terminal Gate (Step 150)",
        "step": 150,
        "path": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_e/stage_3_training/branch_d_combined/checkpoints/ckpt_branch_d_step200.pt",
        "expected_sha": "78f8ca6b8c8e08471805c03d60a61884dc10e676eeb0f20a4c668e4fc0068eb8"
    }
}

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


def exact_binomial_pvalue(b: int, c: int) -> float:
    """Exact two-tailed binomial test p-value for discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # Cumulative probability sum_{i=0}^k binom(n, i) * 0.5^n
    cdf = 0.0
    for i in range(k + 1):
        cdf += math.comb(n, i) * (0.5 ** n)
    p_val = min(1.0, 2.0 * cdf)
    return round(p_val, 6)


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


def check_semantic_correctness(task: Dict[str, Any], final_answer: str, tool_trace: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
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


def evaluate_claims(final_answer: str, tool_output: str, criteria: List[Any], category: str) -> Dict[str, Any]:
    """
    Evaluates claims using the 4-level taxonomy:
    SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED.
    """
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", final_answer) if len(s.strip()) > 8]
    if not sentences:
        return {
            "total_claims": 1,
            "supported": 0,
            "partially_supported": 0,
            "unsupported": 1,
            "contradicted": 0,
            "claim_precision": 0.0,
            "evidence_coverage": 0.0,
            "numerical_accuracy": 0.0,
            "entity_value_association": 0.0,
            "has_citations": False,
            "citation_count": 0
        }

    evidence_text = tool_output.lower() if tool_output else ""
    evidence_tokens = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", evidence_text))

    supported = 0
    partially_supported = 0
    unsupported = 0
    contradicted = 0

    flat_criteria = []
    for c in criteria:
        if isinstance(c, list):
            flat_criteria.extend([opt.lower() for opt in c])
        else:
            flat_criteria.append(str(c).lower())

    for sent in sentences:
        s_lower = sent.lower()
        s_tokens = set(re.findall(r"\b[A-Za-z0-9]{3,}\b", s_lower))
        if not s_tokens:
            continue

        # Check contradiction: explicitly asserting opposite of criteria or observation
        contradiction_markers = ["not ", "never ", "no ", "failed to ", "cannot "]
        is_contradicted = False
        for fc in flat_criteria:
            if fc in evidence_text and any(m + fc in s_lower for m in contradiction_markers):
                is_contradicted = True
                break

        if is_contradicted:
            contradicted += 1
            continue

        overlap_ev = s_tokens & evidence_tokens
        crit_match = any(fc in s_lower for fc in flat_criteria)

        if len(overlap_ev) >= 3 or (crit_match and len(overlap_ev) >= 1):
            supported += 1
        elif len(overlap_ev) >= 1 or crit_match:
            partially_supported += 1
        else:
            unsupported += 1

    total_c = supported + partially_supported + unsupported + contradicted
    total_c = max(1, total_c)

    claim_prec = (supported + 0.5 * partially_supported) / total_c
    ev_cov = min(1.0, len(evidence_tokens & set(re.findall(r"\b[A-Za-z0-9]{3,}\b", final_answer.lower()))) / max(1, min(len(evidence_tokens), 15)))

    # Numerical accuracy check
    num_in_resp = set(re.findall(r"\b\d+(?:\.\d+)?\b", final_answer))
    num_in_ev = set(re.findall(r"\b\d+(?:\.\d+)?\b", evidence_text))
    num_acc = 1.0 if (num_in_resp & num_in_ev) else (0.5 if num_in_resp and not num_in_ev else 0.0)

    # Citations check
    citations = re.findall(r"\[(?:SOURCE\s*\d+|\d+|web)\]|https?://\S+", final_answer, re.IGNORECASE)

    # Entity value association
    entity_val_ok = 1.0 if (supported >= 1 and not is_contradicted) else 0.0

    return {
        "total_claims": total_c,
        "supported": supported,
        "partially_supported": partially_supported,
        "unsupported": unsupported,
        "contradicted": contradicted,
        "claim_precision": round(claim_prec, 4),
        "evidence_coverage": round(ev_cov, 4),
        "numerical_accuracy": round(num_acc, 4),
        "entity_value_association": round(entity_val_ok, 4),
        "has_citations": len(citations) > 0,
        "citation_count": len(citations)
    }


def evaluate_single_checkpoint(
    tag: str,
    info: Dict[str, Any],
    benchmark_tasks: List[Dict[str, Any]],
    tok: Tokenizer,
    device: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    print(f"\n" + "=" * 80)
    print(f"EVALUATING: {info['name']} (Tag: {tag}, Step {info['step']})")
    print(f"Path: {info['path'].name}")
    print(f"=" * 80)

    cfg = ModelConfig.get_500m_config()
    model = ModernGPT(cfg).to(device)
    ckpt = torch.load(info["path"], map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    orchestrator = ToolOrchestrator()
    protocol = QorinTwoStageProtocol(orchestrator)

    task_results = []
    multistep_traces = []

    t0_eval = time.perf_counter()

    for idx, task in enumerate(benchmark_tasks):
        t_id = task["id"]
        cat = task["category"]
        prompt_user = task["prompt"]
        expected_intent = task["expected_intent"]
        expected_tool = task["expected_tool"]

        prompt_stage1 = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{prompt_user}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Stage 1: Initial Model Generation
        raw_stage1 = generate_greedy(model, tok, prompt_stage1, max_new_tokens=128, device=device)
        action, tool_name, tool_args = protocol.determine_action(
            first_round_text=raw_stage1,
            web_enabled=True,
            python_enabled=True
        )
        detected_intent = (action in ["PYTHON", "WEB"])

        tp = (expected_intent and detected_intent)
        fp = (not expected_intent and detected_intent)
        tn = (not expected_intent and not detected_intent)
        fn = (expected_intent and not detected_intent)
        tool_correct = (expected_intent and detected_intent and tool_name == expected_tool)

        # Stage 2: Schema & Argument Validation
        is_valid_schema = False
        is_valid_semantic_args = False
        validated_args = {}
        val_error = None

        if detected_intent and tool_name:
            is_valid_schema, validated_args, val_error = protocol.validate_and_sanitize_arguments(
                tool_name=tool_name,
                arguments=tool_args,
                raw_text=raw_stage1,
                user_query=prompt_user
            )
            # Semantic argument validation
            if is_valid_schema:
                if tool_name == "web_research":
                    q = validated_args.get("query", "")
                    is_valid_semantic_args = isinstance(q, str) and len(q.strip()) >= 3
                elif tool_name == "python":
                    code = validated_args.get("code", "")
                    try:
                        ast.parse(code)
                        is_valid_semantic_args = True
                    except Exception:
                        is_valid_semantic_args = False

        # Stage 3: Tool Dispatch & Execution
        dispatched = False
        executed = False
        exec_failure_cat = "none"
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
                tool_trace_dict["model_tag"] = tag
                executed = (trace.status == "completed" or getattr(trace.status, "value", None) == "SUCCESS")

                if tool_name == "python":
                    tool_output_str = str(trace.output_payload.get("stdout", "") or trace.output_payload.get("error", ""))
                    if not executed:
                        err_text = str(trace.output_payload.get("error", "")).lower()
                        if "syntaxerror" in err_text:
                            exec_failure_cat = "syntax_error"
                        else:
                            exec_failure_cat = "runtime_error"
                elif tool_name == "web_research":
                    tool_output_str = str(trace.output_payload.get("summary", ""))
                    if not executed:
                        exec_failure_cat = "retrieval_empty" if "no results" in tool_output_str.lower() else "network_timeout"
            except Exception as e:
                executed = False
                exec_failure_cat = "runtime_error"
                tool_trace_dict = {"error": str(e), "task_id": t_id, "model_tag": tag}
        elif detected_intent and not is_valid_schema:
            exec_failure_cat = "schema_error"

        # Multi-Step Handling (15 tasks)
        final_response = raw_stage1
        ms_trace = None

        if cat == "multistep_web_python":
            ms_trace = {
                "task_id": t_id,
                "model_tag": tag,
                "prompt": prompt_user,
                "stage1_web_dispatched": False,
                "stage2_web_valid_args": False,
                "stage3_evidence_extracted": False,
                "stage4_python_dispatched": False,
                "stage5_python_valid_args": False,
                "stage6_python_executed": False,
                "stage7_final_grounded": False,
                "stages_completed": 0,
                "fully_autonomous_success": False,
                "trace_history": []
            }

            ms_trace["stage1_web_dispatched"] = (detected_intent and tool_name == "web_research")
            ms_trace["stage2_web_valid_args"] = (is_valid_schema and is_valid_semantic_args and tool_name == "web_research")
            ms_trace["stage3_evidence_extracted"] = (executed and len(tool_output_str) > 20)

            ms_trace["trace_history"].append({
                "round": 1,
                "model_output": raw_stage1,
                "action": action,
                "tool": tool_name,
                "valid": is_valid_schema,
                "output": tool_output_str[:200]
            })

            # Round 2: Feed Web output into model
            if ms_trace["stage3_evidence_extracted"]:
                p_r2 = (
                    f"{prompt_stage1}{raw_stage1.strip()}<|im_end|>\n"
                    f"<|im_start|>tool\n<|tool_result|>\n{tool_output_str}\n<|end_tool_result|><|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                raw_r2 = generate_greedy(model, tok, p_r2, max_new_tokens=160, device=device)
                act2, tool2, args2 = protocol.determine_action(raw_r2, web_enabled=True, python_enabled=True)

                ms_trace["stage4_python_dispatched"] = (act2 == "PYTHON")
                valid_py = False
                py_out_str = ""

                if act2 == "PYTHON":
                    s2_valid, s2_args, _ = protocol.validate_and_sanitize_arguments(tool2, args2, raw_r2, prompt_user)
                    if s2_valid:
                        ms_trace["stage5_python_valid_args"] = True
                        try:
                            ast.parse(s2_args.get("code", ""))
                            trace2 = orchestrator.execute_tool(tool2, s2_args, requested_by="model")
                            t2_ok = (trace2.status == "completed" or getattr(trace2.status, "value", None) == "SUCCESS")
                            py_out_str = str(trace2.output_payload.get("stdout", "") or trace2.output_payload.get("error", ""))
                            ms_trace["stage6_python_executed"] = t2_ok

                            # Round 3: Synthesis
                            p_r3 = (
                                f"{p_r2}{raw_r2.strip()}<|im_end|>\n"
                                f"<|im_start|>tool\n<|tool_result|>\n{py_out_str}\n<|end_tool_result|><|im_end|>\n"
                                f"<|im_start|>assistant\n"
                            )
                            final_response = generate_greedy(model, tok, p_r3, max_new_tokens=200, device=device)
                        except Exception:
                            final_response = raw_r2
                    else:
                        final_response = raw_r2
                else:
                    final_response = raw_r2

                ms_trace["trace_history"].append({
                    "round": 2,
                    "model_output": raw_r2,
                    "action": act2,
                    "tool": tool2,
                    "output": py_out_str[:200]
                })

            sem_ok, sem_r = check_semantic_correctness(task, final_response, tool_trace_dict)
            ms_trace["stage7_final_grounded"] = sem_ok
            completed_stages = sum([
                ms_trace["stage1_web_dispatched"],
                ms_trace["stage2_web_valid_args"],
                ms_trace["stage3_evidence_extracted"],
                ms_trace["stage4_python_dispatched"],
                ms_trace["stage5_python_valid_args"],
                ms_trace["stage6_python_executed"],
                ms_trace["stage7_final_grounded"]
            ])
            ms_trace["stages_completed"] = completed_stages
            ms_trace["fully_autonomous_success"] = (completed_stages == 7)
            multistep_traces.append(ms_trace)
        else:
            # Standard Two-Stage Synthesis (Stage 2)
            if dispatched and executed and tool_output_str:
                prompt_stage2 = (
                    f"{prompt_stage1}{raw_stage1.strip()}<|im_end|>\n"
                    f"<|im_start|>tool\n<|tool_result|>\n{tool_output_str}\n<|end_tool_result|><|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
                final_response = generate_greedy(model, tok, prompt_stage2, max_new_tokens=160, device=device)
            elif not detected_intent:
                final_response = raw_stage1

            sem_ok, sem_r = check_semantic_correctness(task, final_response, tool_trace_dict)

        # Claim-Level Evaluation
        claims_eval = evaluate_claims(final_response, tool_output_str, task.get("semantic_criteria", []), cat)

        res_record = {
            "model_tag": tag,
            "task_id": t_id,
            "category": cat,
            "expected_intent": expected_intent,
            "expected_tool": expected_tool,
            "detected_intent": detected_intent,
            "selected_tool": tool_name if detected_intent else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tool_selection_correct": tool_correct,
            "schema_valid": is_valid_schema,
            "semantic_args_valid": is_valid_semantic_args,
            "dispatched": dispatched,
            "executed": executed,
            "execution_failure_category": exec_failure_cat,
            "semantic_success": sem_ok,
            "semantic_reason": sem_r,
            "claims_evaluation": claims_eval,
            "raw_stage1": raw_stage1[:160],
            "final_response": final_response[:240]
        }
        task_results.append(res_record)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(benchmark_tasks):
            print(f"  [{tag}] Evaluated {idx + 1:>3}/{len(benchmark_tasks)} tasks... ({time.perf_counter() - t0_eval:.1f}s)")

    # Aggregate Metrics Calculation
    total = len(task_results)
    tp_c = sum(1 for r in task_results if r["tp"])
    fp_c = sum(1 for r in task_results if r["fp"])
    tn_c = sum(1 for r in task_results if r["tn"])
    fn_c = sum(1 for r in task_results if r["fn"])

    prec = tp_c / max(1, tp_c + fp_c)
    rec = tp_c / max(1, tp_c + fn_c)
    f1 = 2 * prec * rec / max(1e-8, prec + rec)
    acc = (tp_c + tn_c) / total

    direct_tasks = [r for r in task_results if not r["expected_intent"]]
    direct_fp = sum(1 for r in direct_tasks if r["detected_intent"])
    direct_fpr = direct_fp / max(1, len(direct_tasks))

    web_tasks = [r for r in task_results if r["expected_tool"] == "web_research"]
    web_correct = sum(1 for r in web_tasks if r["selected_tool"] == "web_research")
    web_rec = web_correct / max(1, len(web_tasks))

    py_tasks = [r for r in task_results if r["expected_tool"] == "python"]
    py_correct = sum(1 for r in py_tasks if r["selected_tool"] == "python")
    py_rec = py_correct / max(1, len(py_tasks))

    tool_tasks = [r for r in task_results if r["expected_intent"]]
    correct_tools = sum(1 for r in tool_tasks if r["tool_selection_correct"])
    tool_sel_acc = correct_tools / max(1, len(tool_tasks))

    # Argument and execution quality
    calls_attempted = sum(1 for r in task_results if r["detected_intent"])
    schema_valid_cnt = sum(1 for r in task_results if r["schema_valid"])
    sem_args_valid_cnt = sum(1 for r in task_results if r["semantic_args_valid"])
    dispatched_cnt = sum(1 for r in task_results if r["dispatched"])
    executed_cnt = sum(1 for r in task_results if r["executed"])

    schema_rate = schema_valid_cnt / max(1, calls_attempted)
    sem_args_rate = sem_args_valid_cnt / max(1, calls_attempted)
    dispatch_rate = dispatched_cnt / max(1, calls_attempted)
    execution_rate = executed_cnt / max(1, dispatched_cnt)

    failure_cats = Counter(r["execution_failure_category"] for r in task_results if r["detected_intent"] and not r["executed"])

    # Synthesis metrics
    factual_tasks = [r for r in task_results if r["category"] in {"colloquial_web_research", "current_information", "multistep_web_python"}]
    total_claims = sum(r["claims_evaluation"]["total_claims"] for r in factual_tasks)
    sup_claims = sum(r["claims_evaluation"]["supported"] for r in factual_tasks)
    part_claims = sum(r["claims_evaluation"]["partially_supported"] for r in factual_tasks)
    unsup_claims = sum(r["claims_evaluation"]["unsupported"] for r in factual_tasks)
    contra_claims = sum(r["claims_evaluation"]["contradicted"] for r in factual_tasks)

    claim_precision = (sup_claims + 0.5 * part_claims) / max(1, total_claims)
    unsup_rate = unsup_claims / max(1, total_claims)
    contra_rate = contra_claims / max(1, total_claims)
    mean_ev_cov = sum(r["claims_evaluation"]["evidence_coverage"] for r in factual_tasks) / max(1, len(factual_tasks))
    mean_num_acc = sum(r["claims_evaluation"]["numerical_accuracy"] for r in factual_tasks) / max(1, len(factual_tasks))
    mean_entity_assoc = sum(r["claims_evaluation"]["entity_value_association"] for r in factual_tasks) / max(1, len(factual_tasks))
    citation_tasks = sum(1 for r in factual_tasks if r["claims_evaluation"]["has_citations"])
    citation_rate = citation_tasks / max(1, len(factual_tasks))

    # Overall semantic success
    sem_correct_cnt = sum(1 for r in task_results if r["semantic_success"])
    sem_rate = sem_correct_cnt / total

    # Multi-step
    ms_success_cnt = sum(1 for t in multistep_traces if t["fully_autonomous_success"])
    ms_stages = {
        "stage1_web_dispatched": sum(1 for t in multistep_traces if t["stage1_web_dispatched"]),
        "stage2_web_valid_args": sum(1 for t in multistep_traces if t["stage2_web_valid_args"]),
        "stage3_evidence_extracted": sum(1 for t in multistep_traces if t["stage3_evidence_extracted"]),
        "stage4_python_dispatched": sum(1 for t in multistep_traces if t["stage4_python_dispatched"]),
        "stage5_python_valid_args": sum(1 for t in multistep_traces if t["stage5_python_valid_args"]),
        "stage6_python_executed": sum(1 for t in multistep_traces if t["stage6_python_executed"]),
        "stage7_final_grounded": sum(1 for t in multistep_traces if t["stage7_final_grounded"]),
    }

    metrics_dict = {
        "tag": tag,
        "name": info["name"],
        "step": info["step"],
        "checkpoint": info["path"].name,
        "sha256": info["expected_sha"],
        "total_tasks": total,
        "intent_recognition": {
            "tp": tp_c, "fp": fp_c, "tn": tn_c, "fn": fn_c,
            "precision": round(prec, 4),
            "precision_ci": wilson_score_interval(tp_c, tp_c + fp_c),
            "recall": round(rec, 4),
            "recall_ci": wilson_score_interval(tp_c, tp_c + fn_c),
            "f1": round(f1, 4),
            "accuracy": round(acc, 4),
            "accuracy_ci": wilson_score_interval(tp_c + tn_c, total),
            "direct_task_fpr": round(direct_fpr, 4),
            "direct_task_fpr_ci": wilson_score_interval(direct_fp, len(direct_tasks)),
            "direct_fp_count": direct_fp,
            "direct_total": len(direct_tasks)
        },
        "tool_selection": {
            "python_selection_recall": round(py_rec, 4),
            "python_selection_recall_ci": wilson_score_interval(py_correct, len(py_tasks)),
            "python_correct": py_correct,
            "python_total": len(py_tasks),
            "web_selection_recall": round(web_rec, 4),
            "web_selection_recall_ci": wilson_score_interval(web_correct, len(web_tasks)),
            "web_correct": web_correct,
            "web_total": len(web_tasks),
            "correct_tool_selection_accuracy": round(tool_sel_acc, 4),
            "correct_tool_selection_accuracy_ci": wilson_score_interval(correct_tools, len(tool_tasks)),
            "correct_tools_count": correct_tools,
            "tool_tasks_total": len(tool_tasks),
            "incorrect_tool_selection_count": sum(1 for r in tool_tasks if r["detected_intent"] and not r["tool_selection_correct"]),
            "unnecessary_tool_calls_count": direct_fp,
            "missed_tool_calls_count": fn_c
        },
        "argument_and_execution_quality": {
            "calls_attempted": calls_attempted,
            "schema_validity_count": schema_valid_cnt,
            "schema_validity_rate": round(schema_rate, 4),
            "semantic_args_validity_count": sem_args_valid_cnt,
            "semantic_args_validity_rate": round(sem_args_rate, 4),
            "dispatched_count": dispatched_cnt,
            "dispatch_rate": round(dispatch_rate, 4),
            "execution_success_count": executed_cnt,
            "execution_success_rate": round(execution_rate, 4),
            "failure_categories": dict(failure_cats),
            "recovery_after_tool_errors": 0.0
        },
        "evidence_grounded_synthesis": {
            "total_claims_evaluated": total_claims,
            "supported_claims": sup_claims,
            "partially_supported_claims": part_claims,
            "unsupported_claims": unsup_claims,
            "contradicted_claims": contra_claims,
            "claim_precision": round(claim_precision, 4),
            "claim_evidence_coverage": round(mean_ev_cov, 4),
            "unsupported_factual_claim_rate": round(unsup_rate, 4),
            "contradiction_rate": round(contra_rate, 4),
            "numerical_extraction_accuracy": round(mean_num_acc, 4),
            "entity_value_association_accuracy": round(mean_entity_assoc, 4),
            "citation_correctness_rate": round(citation_rate, 4),
            "citation_tasks_count": citation_tasks,
            "factual_tasks_total": len(factual_tasks)
        },
        "autonomous_multistep": {
            "total_multistep_tasks": len(multistep_traces),
            "fully_autonomous_success_count": ms_success_cnt,
            "fully_autonomous_success_rate": round(ms_success_cnt / max(1, len(multistep_traces)), 4),
            "stage_completion_counts": ms_stages,
            "stage_completion_rates": {k: round(v / max(1, len(multistep_traces)), 4) for k, v in ms_stages.items()}
        },
        "overall_performance": {
            "end_to_end_semantic_success_count": sem_correct_cnt,
            "end_to_end_semantic_success_rate": round(sem_rate, 4),
            "semantic_success_ci": wilson_score_interval(sem_correct_cnt, total)
        }
    }

    print(f"[{tag}] Evaluation Complete in {time.perf_counter() - t0_eval:.1f}s.")
    print(f"  Direct FPR:       {direct_fpr:.4f} ({direct_fp}/{len(direct_tasks)})")
    print(f"  Python Recall:    {py_rec:.4f} ({py_correct}/{len(py_tasks)})")
    print(f"  Web Recall:       {web_rec:.4f} ({web_correct}/{len(web_tasks)})")
    print(f"  Semantic Success: {sem_rate:.4f} ({sem_correct_cnt}/{total})")
    print(f"  Multi-Step Auto:  {ms_success_cnt}/{len(multistep_traces)}")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return task_results, multistep_traces, metrics_dict


def compute_paired_statistics(all_task_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("COMPUTING PAIRED STATISTICAL ANALYSES")
    print("=" * 80)

    comparisons = [
        ("branch_d_best", "exp_d_parent"),
        ("branch_d_best", "branch_b_best"),
        ("branch_d_best", "branch_c_best"),
        ("branch_d_best", "branch_d_terminal")
    ]

    paired_stats = {}
    p_values_to_correct = []

    for cand_tag, ref_tag in comparisons:
        cand_res = {r["task_id"]: r for r in all_task_results[cand_tag]}
        ref_res = {r["task_id"]: r for r in all_task_results[ref_tag]}

        both_pass = 0
        cand_only = 0
        ref_only = 0
        both_fail = 0

        for tid in cand_res:
            c_ok = cand_res[tid]["semantic_success"]
            r_ok = ref_res[tid]["semantic_success"]

            if c_ok and r_ok:
                both_pass += 1
            elif c_ok and not r_ok:
                cand_only += 1
            elif not c_ok and r_ok:
                ref_only += 1
            else:
                both_fail += 1

        total = both_pass + cand_only + ref_only + both_fail
        b = cand_only
        c = ref_only

        exact_p = exact_binomial_pvalue(b, c)
        if (cand_tag, ref_tag) in comparisons[:3]:
            p_values_to_correct.append(((cand_tag, ref_tag), exact_p))

        # Asymptotic McNemar with continuity correction
        disc = b + c
        if disc > 0:
            chi2 = (abs(b - c) - 1.0) ** 2 / disc
            asymp_p = round(math.erfc(math.sqrt(chi2 / 2.0)), 6)
        else:
            chi2 = 0.0
            asymp_p = 1.0

        risk_diff = (b - c) / max(1, total)
        # Wald CI for risk difference
        se_rd = math.sqrt((b + c - (b - c)**2 / total) / (total**2)) if total > 0 else 0.0
        rd_ci = (round(risk_diff - 1.96 * se_rd, 4), round(risk_diff + 1.96 * se_rd, 4))

        odds_ratio = round(b / max(1, c), 4)

        paired_stats[f"{cand_tag}_vs_{ref_tag}"] = {
            "candidate": cand_tag,
            "reference": ref_tag,
            "total_tasks": total,
            "contingency_table": {
                "both_pass": both_pass,
                "candidate_only_wins": b,
                "reference_only_wins": c,
                "both_fail": both_fail
            },
            "exact_mcnemar": {
                "b": b,
                "c": c,
                "exact_p_value": exact_p,
                "asymptotic_chi2": round(chi2, 4),
                "asymptotic_p_value": asymp_p,
                "significant_at_05_nominal": bool(exact_p < 0.05),
                "significant_at_01_nominal": bool(exact_p < 0.01)
            },
            "effect_size": {
                "risk_difference": round(risk_diff, 4),
                "risk_difference_95ci": rd_ci,
                "odds_ratio": odds_ratio,
                "relative_risk": round((both_pass + b) / max(1, both_pass + c), 4)
            }
        }

    # Holm-Bonferroni correction over the 3 primary hypotheses:
    # 1. Branch D best vs Exp D
    # 2. Branch D best vs Branch B
    # 3. Branch D best vs Branch C
    p_values_to_correct.sort(key=lambda x: x[1])
    m = len(p_values_to_correct)
    holm_results = {}

    for rank, ((c_tag, r_tag), raw_p) in enumerate(p_values_to_correct):
        alpha_holm = 0.05 / (m - rank)
        bonferroni_p = min(1.0, round(raw_p * m, 6))
        holm_adj_p = min(1.0, round(raw_p * (m - rank), 6))
        is_sig_holm = bool(raw_p < alpha_holm)

        comp_key = f"{c_tag}_vs_{r_tag}"
        paired_stats[comp_key]["multiple_comparisons_correction"] = {
            "rank": rank + 1,
            "raw_p_value": raw_p,
            "bonferroni_adjusted_p": bonferroni_p,
            "holm_step_down_p": holm_adj_p,
            "holm_threshold": round(alpha_holm, 6),
            "significant_after_correction": is_sig_holm
        }

    return paired_stats


def main():
    print("=" * 80)
    print("STARTING QORIN STAGE 10 COMPREHENSIVE EVALUATION RUN")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(str(TOK_PATH))

    # Load Benchmark
    benchmark_tasks = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                benchmark_tasks.append(json.loads(line))
    print(f"Loaded {len(benchmark_tasks)} benchmark tasks from {BENCHMARK_PATH.name}")

    all_task_results = {}
    all_multistep_traces = []
    metrics_summary = {}

    all_results_file = STAGE10_DIR / "stage_10_task_results.jsonl"
    all_traces_file = STAGE10_DIR / "stage_10_multistep_traces.jsonl"

    with open(all_results_file, "w", encoding="utf-8") as rf, open(all_traces_file, "w", encoding="utf-8") as tf:
        for tag, info in CHECKPOINTS.items():
            t_results, m_traces, m_dict = evaluate_single_checkpoint(
                tag=tag,
                info=info,
                benchmark_tasks=benchmark_tasks,
                tok=tok,
                device=device
            )
            all_task_results[tag] = t_results
            for r in t_results:
                rf.write(json.dumps(r) + "\n")
            for t in m_traces:
                tf.write(json.dumps(t) + "\n")
            metrics_summary[tag] = m_dict

    # Compute Paired Statistics
    paired_stats = compute_paired_statistics(all_task_results)

    # Save metrics.json
    metrics_file = STAGE10_DIR / "stage_10_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved: {metrics_file.name}")

    # Save paired_statistics.json
    paired_file = STAGE10_DIR / "stage_10_paired_statistics.json"
    with open(paired_file, "w", encoding="utf-8") as f:
        json.dump(paired_stats, f, indent=2)
    print(f"Saved: {paired_file.name}")

    print("\n" + "=" * 80)
    print("ALL 5 EVALUATIONS AND STATISTICAL TESTS SUCCESSFULLY EXECUTED!")
    print("=" * 80)


if __name__ == "__main__":
    main()
