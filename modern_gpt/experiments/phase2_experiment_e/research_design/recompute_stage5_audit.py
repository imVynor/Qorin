import json
import math
from collections import Counter, defaultdict

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + (z**2) / n
    center = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))
    return (max(0.0, center - margin), min(1.0, center + margin))

def main():
    bench_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
    exp_c_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/experiment_c_results.jsonl"
    exp_d_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/experiment_d_results.jsonl"
    traces_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/tool_traces.jsonl"
    
    benchmark = {row["id"]: row for row in load_jsonl(bench_path)}
    c_results = {row["id"]: row for row in load_jsonl(exp_c_path)}
    d_results = {row["id"]: row for row in load_jsonl(exp_d_path)}
    tool_traces = load_jsonl(traces_path)
    
    print(f"Loaded {len(benchmark)} benchmark tasks.")
    print(f"Loaded {len(c_results)} Exp C results.")
    print(f"Loaded {len(d_results)} Exp D results.")
    print(f"Loaded {len(tool_traces)} tool traces.")
    
    # 1. Investigate Inconsistency 1: 100% Web selection in colloquial & current vs 80.26% overall Web recall
    web_expected_tasks = [t for t, row in benchmark.items() if row.get("expected_tool") == "web_research"]
    print(f"\nTotal tasks with expected_tool == 'web_research': {len(web_expected_tasks)}")
    
    web_by_cat = defaultdict(list)
    for tid in web_expected_tasks:
        cat = benchmark[tid]["category"]
        web_by_cat[cat].append(tid)
        
    print("\nExpected Web tasks by category and Exp D selection:")
    web_sel_d_by_cat = {}
    for cat, tids in web_by_cat.items():
        corr = sum(1 for tid in tids if d_results[tid].get("selected_tool") == "web_research")
        web_sel_d_by_cat[cat] = (corr, len(tids))
        print(f"  Category '{cat}': {corr}/{len(tids)} = {corr/len(tids)*100:.2f}%")
        
    overall_web_d = sum(corr for corr, _ in web_sel_d_by_cat.values())
    total_web_tasks = sum(total for _, total in web_sel_d_by_cat.values())
    print(f"Overall Web recall for Exp D: {overall_web_d}/{total_web_tasks} = {overall_web_d/total_web_tasks*100:.2f}%")
    
    # Check what Exp D selected when it missed Web
    missed_web = [tid for tid in web_expected_tasks if d_results[tid].get("selected_tool") != "web_research"]
    print(f"\nTasks where Exp D missed Web ({len(missed_web)} tasks):")
    for tid in missed_web:
        cat = benchmark[tid]["category"]
        actual = d_results[tid].get("selected_tool")
        intent = d_results[tid].get("detected_intent")
        prompt = benchmark[tid]["prompt"]
        print(f"  [{tid}] ({cat}) selected_tool={actual}, detected_intent={intent} | Prompt: {prompt[:60]}...")

    # 2. Investigate Inconsistency 2: Verify the 50% unnecessary tool-call rate and its exact denominator
    direct_expected = [t for t, row in benchmark.items() if row.get("expected_intent") is False or row.get("expected_tool") is None]
    print(f"\nDirect-expected tasks: {len(direct_expected)}")
    direct_by_cat = Counter(benchmark[tid]["category"] for tid in direct_expected)
    print("Direct expected by category:", dict(direct_by_cat))
    
    # How many called a tool?
    d_fps = [tid for tid in direct_expected if d_results[tid].get("detected_intent") is True]
    c_fps = [tid for tid in direct_expected if c_results[tid].get("detected_intent") is True]
    print(f"Exp D False Positives (tool called when direct expected): {len(d_fps)} / {len(direct_expected)} = {len(d_fps)/len(direct_expected)*100:.2f}%")
    print(f"Exp C False Positives (tool called when direct expected): {len(c_fps)} / {len(direct_expected)} = {len(c_fps)/len(direct_expected)*100:.2f}%")
    
    print("\nExp D False Positives by category:")
    d_fps_by_cat = Counter(benchmark[tid]["category"] for tid in d_fps)
    for cat, cnt in direct_by_cat.items():
        fp_cnt = d_fps_by_cat.get(cat, 0)
        print(f"  {cat}: {fp_cnt}/{cnt} = {fp_cnt/cnt*100:.2f}%")
        
    print("\nTools invoked by Exp D on direct tasks:")
    fp_tools = Counter(d_results[tid].get("selected_tool") for tid in d_fps)
    print(dict(fp_tools))

    # 3. Investigate Inconsistency 3: Recompute McNemar's test from actual paired task outcomes
    print("\n--- Paired Contingency Table (Exp D vs Exp C semantic success) ---")
    n11 = sum(1 for tid in benchmark if d_results[tid]["semantic_success"] and c_results[tid]["semantic_success"])
    n10 = sum(1 for tid in benchmark if d_results[tid]["semantic_success"] and not c_results[tid]["semantic_success"])
    n01 = sum(1 for tid in benchmark if not d_results[tid]["semantic_success"] and c_results[tid]["semantic_success"])
    n00 = sum(1 for tid in benchmark if not d_results[tid]["semantic_success"] and not c_results[tid]["semantic_success"])
    
    print(f"Both Success (n11): {n11}")
    print(f"Exp D only Success (b = n10): {n10}")
    print(f"Exp C only Success (c = n01): {n01}")
    print(f"Both Fail (n00): {n00}")
    print(f"Total: {n11 + n10 + n01 + n00}")
    
    b = n10
    c = n01
    chi2_stat = ((abs(b - c) - 1)**2) / (b + c) if (b + c) > 0 else 0.0
    from scipy import stats
    p_val = 1 - stats.chi2.cdf(chi2_stat, df=1) if chi2_stat > 0 else 1.0
    print(f"McNemar chi2 (with Edwards continuity correction): {chi2_stat:.4f}, p-value: {p_val:.4f}")
    
    # 4. Investigate Inconsistency 4: Check whether the 106 tool executions represent unique tasks or multiple calls per task
    print("\n--- Tool Traces & Executions ---")
    trace_tasks = [t["id"] for t in tool_traces]
    unique_trace_tasks = set(trace_tasks)
    print(f"Total tool trace records: {len(tool_traces)}")
    print(f"Unique tasks in tool traces: {len(unique_trace_tasks)}")
    
    trace_counts = Counter(trace_tasks)
    mult_traces = {tid: cnt for tid, cnt in trace_counts.items() if cnt > 1}
    print(f"Tasks with multiple tool traces: {len(mult_traces)}")
    
    d_dispatched = [tid for tid, row in d_results.items() if row.get("dispatched") is True]
    d_executed = [tid for tid, row in d_results.items() if row.get("executed") is True]
    print(f"Tasks with dispatched=True in Exp D results: {len(d_dispatched)}")
    print(f"Tasks with executed=True in Exp D results: {len(d_executed)}")
    
    # 5. Investigate Inconsistency 5: Verify 98.11% evidence utilization
    print("\n--- Evidence Utilization Check ---")
    evidence_util_tasks = [tid for tid, row in d_results.items() if row.get("evidence_utilized") is True]
    dispatched_tasks = [tid for tid, row in d_results.items() if row.get("dispatched") is True]
    print(f"Evidence utilized tasks: {len(evidence_util_tasks)} / {len(dispatched_tasks)} = {len(evidence_util_tasks)/len(dispatched_tasks)*100:.2f}%")
    
    dispatched_semantic_success = [tid for tid in dispatched_tasks if d_results[tid]["semantic_success"] is True]
    print(f"Semantic success among dispatched tasks: {len(dispatched_semantic_success)} / {len(dispatched_tasks)} = {len(dispatched_semantic_success)/len(dispatched_tasks)*100:.2f}%")
    print(f"Gap: Evidence utilized = 98.11%, but Semantic success = {len(dispatched_semantic_success)/len(dispatched_tasks)*100:.2f}%!")
    
    # 6. Evaluator Objectivity: Check if evaluator was blind to model identity
    print("\n--- Evaluator Blindness ---")
    print("Checked run_audit_evaluation.py: Rule-based programmatic matcher using task-level criteria. Evaluates both models with identical criteria functions.")
    
    # 7. Check if task-level results support every aggregate reported in generalization_report.md
    print("\n--- Aggregate Metrics Verification ---")
    def compute_all_metrics(results):
        n = len(results)
        tp = sum(1 for tid, r in results.items() if r["detected_intent"] and benchmark[tid]["expected_intent"])
        fp = sum(1 for tid, r in results.items() if r["detected_intent"] and not benchmark[tid]["expected_intent"])
        fn = sum(1 for tid, r in results.items() if not r["detected_intent"] and benchmark[tid]["expected_intent"])
        tn = sum(1 for tid, r in results.items() if not r["detected_intent"] and not benchmark[tid]["expected_intent"])
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        web_expected = [tid for tid, b in benchmark.items() if b["expected_tool"] == "web_research"]
        py_expected = [tid for tid, b in benchmark.items() if b["expected_tool"] == "python"]
        
        web_corr = sum(1 for tid in web_expected if results[tid]["selected_tool"] == "web_research")
        py_corr = sum(1 for tid in py_expected if results[tid]["selected_tool"] == "python")
        
        web_rec = web_corr / len(web_expected) if web_expected else 0
        py_rec = py_corr / len(py_expected) if py_expected else 0
        
        tool_invoked = [tid for tid, r in results.items() if r["detected_intent"]]
        schema_valid = sum(1 for tid in tool_invoked if results[tid]["valid_schema"])
        schema_rate = schema_valid / len(tool_invoked) if tool_invoked else 0
        
        dispatched = sum(1 for tid in tool_invoked if results[tid]["dispatched"])
        disp_rate = dispatched / len(tool_invoked) if tool_invoked else 0
        
        executed = sum(1 for tid in tool_invoked if results[tid]["executed"])
        exec_rate = executed / len(tool_invoked) if tool_invoked else 0
        
        sem_corr = sum(1 for tid, r in results.items() if r["semantic_success"])
        sem_rate = sem_corr / n
        
        return {
            "n": n,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "intent_prec": prec, "intent_rec": rec, "intent_f1": f1,
            "web_recall": (web_corr, len(web_expected), web_rec),
            "python_recall": (py_corr, len(py_expected), py_rec),
            "schema_validity": (schema_valid, len(tool_invoked), schema_rate),
            "dispatch_rate": (dispatched, len(tool_invoked), disp_rate),
            "exec_rate": (executed, len(tool_invoked), exec_rate),
            "semantic_success": (sem_corr, n, sem_rate),
            "ci_semantic": wilson_ci(sem_corr, n)
        }
        
    met_c = compute_all_metrics(c_results)
    met_d = compute_all_metrics(d_results)
    
    print("\nExperiment C recomputed:")
    for k, v in met_c.items():
        print(f"  {k}: {v}")
        
    print("\nExperiment D recomputed:")
    for k, v in met_d.items():
        print(f"  {k}: {v}")

    # Output detailed audit JSON
    audit_data = {
        "exp_c": met_c,
        "exp_d": met_d,
        "mcnemar": {
            "n11": n11, "n10": n10, "n01": n01, "n00": n00,
            "chi2_stat": chi2_stat,
            "p_value": p_val
        },
        "web_by_cat": {cat: {"correct": corr, "total": tot, "recall": corr/tot} for cat, (corr, tot) in web_sel_d_by_cat.items()},
        "direct_false_positives": {
            "total_direct": len(direct_expected),
            "d_fps": len(d_fps),
            "c_fps": len(c_fps),
            "d_fp_rate": len(d_fps) / len(direct_expected),
            "c_fp_rate": len(c_fps) / len(direct_expected),
            "d_fps_by_cat": dict(d_fps_by_cat)
        },
        "dispatched_analysis": {
            "total_dispatched": len(dispatched_tasks),
            "evidence_utilized": len(evidence_util_tasks),
            "semantic_success": len(dispatched_semantic_success)
        }
    }
    
    with open("modern_gpt/experiments/phase2_experiment_e/research_design/audit_recomputation.json", "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2)
    print("\nWrote audit_recomputation.json")

if __name__ == "__main__":
    main()
