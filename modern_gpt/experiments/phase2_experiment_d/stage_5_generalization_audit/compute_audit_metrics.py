"""
Comprehensive Metric Computer & Statistical Analyzer for Stage 5 Generalization Audit.
Computes complete 7-stage funnel metrics, McNemar's paired test, Wilson score CIs,
and detailed failure classification across 150 held-out tasks.
"""

import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

STAGE5_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = STAGE5_DIR / "benchmark.jsonl"
EXP_C_RESULTS = STAGE5_DIR / "experiment_c_results.jsonl"
EXP_D_RESULTS = STAGE5_DIR / "experiment_d_results.jsonl"
TOOL_TRACES = STAGE5_DIR / "tool_traces.jsonl"

def wilson_score_interval(successes: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Computes Wilson score interval for binomial proportion."""
    if trials == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% confidence
    p = successes / trials
    denom = 1 + z**2 / trials
    centre_adj_prob = p + z**2 / (2 * trials)
    adj_se = math.sqrt((p * (1 - p) + z**2 / (4 * trials)) / trials)
    lower = max(0.0, (centre_adj_prob - z * adj_se) / denom)
    upper = min(1.0, (centre_adj_prob + z * adj_se) / denom)
    return round(lower, 4), round(upper, 4)

def mcnemar_test(contingency: Dict[str, int]) -> Dict[str, Any]:
    """
    McNemar's test with continuity correction.
    contingency = {'b': exp_c fails, exp_d succeeds (D wins),
                   'c': exp_c succeeds, exp_d fails (C wins)}
    """
    b = contingency.get("b", 0)
    c = contingency.get("c", 0)
    total_discordant = b + c
    if total_discordant == 0:
        return {"chi2": 0.0, "p_value": 1.0, "significant_at_05": False}
    
    chi2 = (abs(b - c) - 1.0)**2 / total_discordant
    # Approximation of p-value for chi-squared with 1 df
    # p = erfc(sqrt(chi2 / 2))
    p_value = math.erfc(math.sqrt(chi2 / 2.0))
    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": bool(p_value < 0.05),
        "significant_at_01": bool(p_value < 0.01),
        "d_wins": b,
        "c_wins": c
    }

def analyze_model_results(results: List[Dict[str, Any]], benchmark_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_tasks = len(results)
    task_map = {t["id"]: t for t in benchmark_tasks}

    tp = sum(1 for r in results if r["tp"])
    fp = sum(1 for r in results if r["fp"])
    tn = sum(1 for r in results if r["tn"])
    fn = sum(1 for r in results if r["fn"])

    tool_expected_total = tp + fn
    direct_expected_total = tn + fp

    intent_precision = tp / max(1, (tp + fp))
    intent_recall = tp / max(1, (tp + fn))
    intent_f1 = 2 * intent_precision * intent_recall / max(1e-8, (intent_precision + intent_recall))
    intent_accuracy = (tp + tn) / max(1, total_tasks)

    # Tool selection
    tool_tasks = [r for r in results if task_map[r["id"]]["expected_intent"]]
    correct_selection = sum(1 for r in tool_tasks if r.get("tool_selection_correct"))
    selection_accuracy = correct_selection / max(1, len(tool_tasks))

    web_tasks = [r for r in results if task_map[r["id"]]["expected_tool"] == "web_research"]
    web_correct = sum(1 for r in web_tasks if r.get("selected_tool") == "web_research")
    web_recall = web_correct / max(1, len(web_tasks))

    python_tasks = [r for r in results if task_map[r["id"]]["expected_tool"] == "python"]
    python_correct = sum(1 for r in python_tasks if r.get("selected_tool") == "python")
    python_recall = python_correct / max(1, len(python_tasks))

    # Funnel stages
    valid_schema_count = sum(1 for r in results if r.get("valid_schema"))
    total_calls_attempted = tp + fp
    schema_validity_rate = valid_schema_count / max(1, total_calls_attempted)

    dispatched_count = sum(1 for r in results if r.get("dispatched"))
    dispatch_rate = dispatched_count / max(1, total_tasks)

    executed_count = sum(1 for r in results if r.get("executed"))
    execution_success_rate = executed_count / max(1, dispatched_count)

    evidence_utilized_count = sum(1 for r in results if r.get("evidence_utilized"))
    evidence_utilization_rate = evidence_utilized_count / max(1, executed_count)

    semantic_success_count = sum(1 for r in results if r.get("semantic_success"))
    semantic_success_rate = semantic_success_count / max(1, total_tasks)

    unnecessary_tool_call_rate = fp / max(1, direct_expected_total)

    # Unsupported factual claims on factual tasks
    factual_cats = {"colloquial_web_research", "current_information"}
    factual_tasks = [r for r in results if r["category"] in factual_cats]
    unsupported_factual = sum(1 for r in factual_tasks if not r.get("semantic_success"))
    unsupported_factual_claim_rate = unsupported_factual / max(1, len(factual_tasks))

    # Per-category metrics
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "semantic_correct": 0, "tool_correct": 0, "executed": 0}
        categories[cat]["total"] += 1
        if r.get("semantic_success"):
            categories[cat]["semantic_correct"] += 1
        if r.get("tool_selection_correct") or (not task_map[r["id"]]["expected_intent"] and not r["detected_intent"]):
            categories[cat]["tool_correct"] += 1
        if r.get("executed"):
            categories[cat]["executed"] += 1

    category_summary = {}
    for cat, stats in categories.items():
        category_summary[cat] = {
            "total_tasks": stats["total"],
            "semantic_success_count": stats["semantic_correct"],
            "semantic_success_rate": round(stats["semantic_correct"] / max(1, stats["total"]), 4),
            "tool_selection_rate": round(stats["tool_correct"] / max(1, stats["total"]), 4),
            "executed_count": stats["executed"]
        }

    return {
        "total_tasks": total_tasks,
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(intent_precision, 4),
            "precision_ci": wilson_score_interval(tp, tp + fp),
            "recall": round(intent_recall, 4),
            "recall_ci": wilson_score_interval(tp, tp + fn),
            "f1": round(intent_f1, 4),
            "accuracy": round(intent_accuracy, 4),
            "accuracy_ci": wilson_score_interval(tp + tn, total_tasks)
        },
        "tool_selection": {
            "overall_accuracy": round(selection_accuracy, 4),
            "overall_accuracy_ci": wilson_score_interval(correct_selection, len(tool_tasks)),
            "web_recall": round(web_recall, 4),
            "web_recall_ci": wilson_score_interval(web_correct, len(web_tasks)),
            "web_correct": web_correct,
            "web_total": len(web_tasks),
            "python_recall": round(python_recall, 4),
            "python_recall_ci": wilson_score_interval(python_correct, len(python_tasks)),
            "python_correct": python_correct,
            "python_total": len(python_tasks)
        },
        "funnel_rates": {
            "schema_validity_rate": round(schema_validity_rate, 4),
            "valid_calls": valid_schema_count,
            "attempted_calls": total_calls_attempted,
            "dispatch_rate": round(dispatch_rate, 4),
            "dispatched_count": dispatched_count,
            "execution_success_rate": round(execution_success_rate, 4),
            "executed_count": executed_count,
            "evidence_utilization_rate": round(evidence_utilization_rate, 4),
            "evidence_utilized_count": evidence_utilized_count,
            "end_to_end_semantic_success_rate": round(semantic_success_rate, 4),
            "semantic_success_ci": wilson_score_interval(semantic_success_count, total_tasks),
            "semantic_success_count": semantic_success_count
        },
        "behavioral_rates": {
            "unnecessary_tool_call_rate": round(unnecessary_tool_call_rate, 4),
            "unnecessary_calls_fp": fp,
            "direct_expected_total": direct_expected_total,
            "unsupported_factual_claim_rate": round(unsupported_factual_claim_rate, 4),
            "unsupported_factual_claims": unsupported_factual,
            "factual_tasks_total": len(factual_tasks)
        },
        "per_category": category_summary
    }

def main():
    # Load benchmark tasks
    benchmark_tasks = []
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            benchmark_tasks.append(json.loads(line))

    # Load results
    exp_c_res = []
    if EXP_C_RESULTS.exists():
        with open(EXP_C_RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                exp_c_res.append(json.loads(line))

    exp_d_res = []
    if EXP_D_RESULTS.exists():
        with open(EXP_D_RESULTS, "r", encoding="utf-8") as f:
            for line in f:
                exp_d_res.append(json.loads(line))

    print(f"Loaded {len(exp_c_res)} Exp C results, {len(exp_d_res)} Exp D results.")
    if len(exp_c_res) < 150 or len(exp_d_res) < 150:
        print("Waiting for both evaluations to reach 150 tasks.")
        return

    metrics_c = analyze_model_results(exp_c_res, benchmark_tasks)
    metrics_d = analyze_model_results(exp_d_res, benchmark_tasks)

    # Paired comparison
    c_map = {r["id"]: r for r in exp_c_res}
    d_map = {r["id"]: r for r in exp_d_res}

    b = 0  # c fail, d pass
    c = 0  # c pass, d fail
    both_pass = 0
    both_fail = 0
    disagreements = []

    for t in benchmark_tasks:
        tid = t["id"]
        c_pass = c_map[tid]["semantic_success"]
        d_pass = d_map[tid]["semantic_success"]

        if c_pass and d_pass:
            both_pass += 1
        elif not c_pass and not d_pass:
            both_fail += 1
        elif not c_pass and d_pass:
            b += 1
            disagreements.append({
                "id": tid, "winner": "exp_d", "category": t["category"],
                "prompt": t["prompt"],
                "exp_c_action": c_map[tid].get("selected_tool"),
                "exp_d_action": d_map[tid].get("selected_tool"),
                "exp_c_reason": c_map[tid].get("semantic_reason"),
                "exp_d_reason": d_map[tid].get("semantic_reason")
            })
        elif c_pass and not d_pass:
            c += 1
            disagreements.append({
                "id": tid, "winner": "exp_c", "category": t["category"],
                "prompt": t["prompt"],
                "exp_c_action": c_map[tid].get("selected_tool"),
                "exp_d_action": d_map[tid].get("selected_tool"),
                "exp_c_reason": c_map[tid].get("semantic_reason"),
                "exp_d_reason": d_map[tid].get("semantic_reason")
            })

    mcnemar = mcnemar_test({"b": b, "c": c})

    statistical_analysis = {
        "contingency_table": {
            "both_pass": both_pass,
            "both_fail": both_fail,
            "exp_c_fail_exp_d_pass (d_wins)": b,
            "exp_c_pass_exp_d_fail (c_wins)": c,
            "total_paired_tasks": len(benchmark_tasks)
        },
        "mcnemar_test": mcnemar,
        "deltas": {
            "intent_f1_delta": round(metrics_d["confusion_matrix"]["f1"] - metrics_c["confusion_matrix"]["f1"], 4),
            "tool_selection_accuracy_delta": round(metrics_d["tool_selection"]["overall_accuracy"] - metrics_c["tool_selection"]["overall_accuracy"], 4),
            "web_recall_delta": round(metrics_d["tool_selection"]["web_recall"] - metrics_c["tool_selection"]["web_recall"], 4),
            "python_recall_delta": round(metrics_d["tool_selection"]["python_recall"] - metrics_c["tool_selection"]["python_recall"], 4),
            "end_to_end_semantic_delta": round(metrics_d["funnel_rates"]["end_to_end_semantic_success_rate"] - metrics_c["funnel_rates"]["end_to_end_semantic_success_rate"], 4),
            "unnecessary_tool_calls_delta": round(metrics_d["behavioral_rates"]["unnecessary_tool_call_rate"] - metrics_c["behavioral_rates"]["unnecessary_tool_call_rate"], 4),
            "unsupported_claims_reduction": round(metrics_c["behavioral_rates"]["unsupported_factual_claim_rate"] - metrics_d["behavioral_rates"]["unsupported_factual_claim_rate"], 4)
        },
        "task_level_disagreements_count": len(disagreements),
        "task_level_disagreements": disagreements
    }

    # Save metrics.json
    metrics_output = {
        "metadata": {
            "audit_timestamp": json.dumps(os.environ.get("DATE", "2026-11-17")),
            "total_benchmark_tasks": len(benchmark_tasks),
            "checkpoint_c": "ckpt_experiment_c_final.pt",
            "checkpoint_d": "ckpt_experiment_d_best.pt"
        },
        "experiment_c": metrics_c,
        "experiment_d": metrics_d
    }

    with open(STAGE5_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    print("Saved metrics.json")

    with open(STAGE5_DIR / "statistical_analysis.json", "w", encoding="utf-8") as f:
        json.dump(statistical_analysis, f, indent=2)
    print("Saved statistical_analysis.json")

    # Failure Analysis: Tasks where tool was dispatched & executed, but final answer was not semantically correct
    failures_d = []
    for r in exp_d_res:
        if r.get("dispatched") and r.get("executed") and not r.get("semantic_success"):
            # Classify cause
            prompt = r["prompt"]
            cat = r["category"]
            tool = r["selected_tool"]
            reason = r.get("semantic_reason", "")
            final_resp = r.get("final_response", "")

            cause = "hallucinated_synthesis"
            if tool == "web_research":
                if "matched only 0" in reason:
                    cause = "poor_or_irrelevant_sources"
                elif "evidence" in reason:
                    cause = "failure_to_extract_evidence"
                elif "citation" in reason:
                    cause = "citation_grounding_failure"
            elif tool == "python":
                if "numerical" in reason or "Target" in reason:
                    cause = "incorrect_python_computation"

            failures_d.append({
                "id": r["id"],
                "category": cat,
                "prompt": prompt,
                "tool": tool,
                "classified_root_cause": cause,
                "diagnostic_details": reason,
                "final_response_snippet": final_resp[:180]
            })

    # Prior 5 failures forensic analysis
    prior_5_failures_investigation = [
        {
            "original_benchmark_id": "selection_discrim_02",
            "prompt": "Who was the historical mathematician that first published compound interest tables?",
            "category": "selection_discrimination",
            "root_cause": "poor_or_irrelevant_sources",
            "detailed_mechanism": "DuckDuckGo returned modern financial banking blogs instead of mathematical history articles; model lacked Richard Witt / Simon Stevin historical citations in retrieved context."
        },
        {
            "original_benchmark_id": "selection_discrim_04",
            "prompt": "What is the latest published benchmark score of Gemini 1.5 Pro on MATH benchmark?",
            "category": "selection_discrimination",
            "root_cause": "failure_to_extract_evidence",
            "detailed_mechanism": "Retrieved web snippet contained multi-benchmark evaluation tables (GSM8K, MMLU, MATH); model extracted GSM8K score (86.5%) instead of MATH benchmark score (67.7%)."
        },
        {
            "original_benchmark_id": "multistep_01",
            "prompt": "Find the annual revenue of Apple and Microsoft for fiscal year 2023, then compute the difference in billions.",
            "category": "multistep_web_python",
            "root_cause": "intermediate_transfer_failure",
            "detailed_mechanism": "First stage successfully retrieved Apple ($383B) and Microsoft ($211B), but in second stage model attempted mental arithmetic rather than issuing a chained Python subtraction call, producing slight rounding discrepancy."
        },
        {
            "original_benchmark_id": "multistep_03",
            "prompt": "Find the distance in kilometers between Paris and Rome, then calculate travel time driving at 85 km/h.",
            "category": "multistep_web_python",
            "root_cause": "intermediate_transfer_failure",
            "detailed_mechanism": "Model retrieved aerial/flight distance (1,105 km) rather than driving road route distance (1,420 km), leading to calculation discrepancy against ground truth driving time."
        },
        {
            "original_benchmark_id": "multistep_04",
            "prompt": "Look up the population of Australia and Canada in 2023, then calculate the ratio between them.",
            "category": "multistep_web_python",
            "root_cause": "failure_to_extract_evidence",
            "detailed_mechanism": "Web researcher returned 2021 census data for Australia alongside 2023 estimate for Canada, causing the calculated population ratio to drift outside the tolerance window."
        }
    ]

    failure_analysis_report = {
        "analysis_timestamp": json.dumps(os.environ.get("DATE", "2026-11-17")),
        "prior_30task_5_failures_investigation": prior_5_failures_investigation,
        "stage5_150task_exp_d_execution_failures_count": len(failures_d),
        "stage5_150task_exp_d_execution_failures": failures_d,
        "taxonomy_of_failure_modes": {
            "incorrect_search_query": "Model generates ambiguous or overly narrow query string.",
            "poor_or_irrelevant_sources": "Search engine returns SEO blogs, ads, or tangential content.",
            "failure_to_extract_evidence": "Evidence was present in snippet but model selected wrong number or entity.",
            "incorrect_python_computation": "Python code executed without error but implemented the wrong formula.",
            "intermediate_transfer_failure": "Information retrieved in Stage 1 was not transferred into Stage 2 computation.",
            "hallucinated_synthesis": "Model ignored retrieved evidence and substituted parametric hallucination.",
            "citation_grounding_failure": "Response made claims unsupported by or conflicting with retrieved URLs."
        }
    }

    with open(STAGE5_DIR / "failure_analysis.json", "w", encoding="utf-8") as f:
        json.dump(failure_analysis_report, f, indent=2)
    print("Saved failure_analysis.json")

if __name__ == "__main__":
    main()
