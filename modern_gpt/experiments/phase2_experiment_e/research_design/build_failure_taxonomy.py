import json
import re
from collections import Counter, defaultdict

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    bench_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
    exp_c_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/experiment_c_results.jsonl"
    exp_d_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/experiment_d_results.jsonl"
    traces_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/tool_traces.jsonl"
    
    benchmark = {row["id"]: row for row in load_jsonl(bench_path)}
    c_results = {row["id"]: row for row in load_jsonl(exp_c_path)}
    d_results = {row["id"]: row for row in load_jsonl(exp_d_path)}
    traces = load_jsonl(traces_path)
    d_traces = {t["task_id"]: t for t in traces if t.get("model_tag") == "exp_d"}
    
    failures = []
    category_counts = Counter()
    origin_counts = Counter()
    
    for tid, r in d_results.items():
        if r["semantic_success"]:
            continue
            
        b = benchmark[tid]
        cat = b["category"]
        expected_intent = b["expected_intent"]
        expected_tool = b["expected_tool"]
        detected_intent = r["detected_intent"]
        selected_tool = r["selected_tool"]
        valid_schema = r["valid_schema"]
        dispatched = r["dispatched"]
        executed = r["executed"]
        sem_reason = r["semantic_reason"]
        trace = d_traces.get(tid)
        
        # Determine earliest point of origin and failure class
        origin = "model"
        failure_class = "unknown"
        details = ""
        
        if not expected_intent and detected_intent:
            # Over-triggering tool on direct question
            origin = "model"
            failure_class = "unnecessary_tool_invocation"
            details = f"Model invoked {selected_tool} on direct {cat} question."
        elif expected_intent and not detected_intent:
            origin = "model"
            failure_class = "missed_tool_intent"
            details = f"Model produced direct answer when {expected_tool} was expected."
        elif expected_intent and detected_intent and selected_tool != expected_tool:
            if cat == "multistep_web_python":
                # Harness limitation: model chose python directly because harness doesn't support interactive chaining
                origin = "harness_orchestrator"
                failure_class = "multistep_orchestration_absence"
                details = "Benchmark required Web->Python chaining, but harness only supports single-turn dispatch; model chose python directly with ungrounded inputs."
            else:
                origin = "model"
                failure_class = "incorrect_tool_selection"
                details = f"Selected {selected_tool} instead of {expected_tool}."
        elif not valid_schema:
            origin = "model"
            failure_class = "schema_syntax_malformation"
            details = f"Tool call schema invalid: {r.get('schema_error')}"
        elif dispatched and not executed:
            # Check why execution failed
            if trace and trace.get("error"):
                err = trace.get("error", "")
                if "duckduckgo" in err.lower() or "search" in err.lower() or "connection" in err.lower() or "timeout" in err.lower() or "no results" in err.lower():
                    origin = "retrieval_engine"
                    failure_class = "retrieval_service_failure"
                    details = f"DuckDuckGo search error: {err[:100]}"
                elif "syntaxerror" in err.lower() or "zerodivision" in err.lower() or "nameerror" in err.lower():
                    origin = "model"
                    failure_class = "invalid_code_generation"
                    details = f"Python runtime exception: {err[:100]}"
                else:
                    origin = "retrieval_engine"
                    failure_class = "tool_execution_error"
                    details = f"Tool error: {err[:100]}"
            else:
                origin = "harness_orchestrator"
                failure_class = "dispatch_failure"
                details = "Tool dispatched but execution status was false without specific error."
        elif executed:
            # Tool executed cleanly, but semantic success failed
            # Distinguish between retrieval snippet quality, evidence extraction, parametric hallucination, formula error, and evaluator string matching
            trace_out = trace.get("output_payload", {}) if trace else {}
            out_str = str(trace_out)
            final_resp = r["final_response"].lower()
            
            # Check evaluator false negative first
            expected_ans = str(b.get("expected_answer", "")).lower()
            criteria = [c.lower() for c in b.get("semantic_criteria", [])]
            
            # Check if answer is actually correct but missed strict criteria
            if expected_ans and expected_ans in final_resp:
                origin = "evaluator"
                failure_class = "evaluator_string_criteria_miss"
                details = f"Expected answer '{expected_ans}' appeared in response, but criteria failed: {sem_reason}"
            elif selected_tool == "python":
                # Python executed cleanly but failed semantic check
                origin = "model"
                failure_class = "incorrect_computational_logic"
                details = f"Python executed cleanly, but logic or formula was incorrect: {sem_reason}"
            elif selected_tool == "web_research":
                # Check if search returned zero or irrelevant results
                summary = trace_out.get("summary", "")
                results_list = trace_out.get("results", [])
                if not summary and not results_list:
                    origin = "retrieval_engine"
                    failure_class = "retrieval_empty_results"
                    details = "Search engine returned no snippets for the generated query."
                elif len(summary) < 50:
                    origin = "retrieval_engine"
                    failure_class = "retrieval_insufficient_depth"
                    details = f"Search engine returned sparse snippet: {summary}"
                else:
                    # Search returned content. Did model extract facts or hallucinate?
                    # Check if model answer contradicts evidence or hallucinated parametric data
                    origin = "evidence_representation" if "dense" in cat or len(summary) > 500 else "model"
                    # Did it copy text from snippet or hallucinate?
                    words_in_summary = set(re.findall(r"\b[a-z0-9]{4,}\b", summary.lower()))
                    words_in_resp = set(re.findall(r"\b[a-z0-9]{4,}\b", final_resp))
                    overlap_ratio = len(words_in_summary & words_in_resp) / max(1, len(words_in_resp))
                    
                    if overlap_ratio < 0.15:
                        origin = "model"
                        failure_class = "parametric_override_hallucination"
                        details = "Model ignored retrieved web snippets and output parametric hallucination."
                    else:
                        origin = "evidence_representation"
                        failure_class = "evidence_extraction_failure"
                        details = "Raw text snippets lacked structured claim/entity format; model extracted wrong numerical or entity values."
        else:
            # Direct response failed
            origin = "model"
            failure_class = "parametric_direct_inaccuracy"
            details = f"Direct answer failed semantic criteria: {sem_reason}"
            
        record = {
            "id": tid,
            "category": cat,
            "prompt": b["prompt"],
            "expected_intent": expected_intent,
            "expected_tool": expected_tool,
            "selected_tool": selected_tool,
            "origin": origin,
            "failure_class": failure_class,
            "details": details,
            "semantic_reason": sem_reason,
            "final_response_snippet": r["final_response"][:180]
        }
        failures.append(record)
        category_counts[cat] += 1
        origin_counts[origin] += 1
        
    print(f"Total analyzed failures: {len(failures)}")
    print("\nFailures by earliest point of origin:")
    for orig, cnt in origin_counts.most_common():
        print(f"  {orig}: {cnt} ({cnt/len(failures)*100:.1f}%)")
        
    print("\nFailures by class:")
    class_counts = Counter(f["failure_class"] for f in failures)
    for cls, cnt in class_counts.most_common():
        print(f"  {cls}: {cnt} ({cnt/len(failures)*100:.1f}%)")
        
    taxonomy = {
        "metadata": {
            "total_benchmark_tasks": len(benchmark),
            "exp_d_semantic_passes": len(benchmark) - len(failures),
            "exp_d_semantic_failures": len(failures),
            "origins_breakdown": dict(origin_counts),
            "classes_breakdown": dict(class_counts)
        },
        "origin_definitions": {
            "model": "Failure rooted in model weights: over-triggering tools on direct questions, incorrect query formulation, formula errors, or hallucinating over retrieved evidence.",
            "harness_orchestrator": "Failure caused by harness architecture: lack of multi-turn chaining loop forcing multi-step Web->Python tasks into single-shot dispatch.",
            "retrieval_engine": "Failure caused by external search engine: DuckDuckGo timeouts, 403 blocks, or zero/low-quality search results.",
            "evidence_representation": "Failure caused by input structure: unstructured, noisy raw text blobs causing 500M model to miss subtle numerical facts or dates.",
            "evaluator": "Failure caused by evaluation harness: rigid keyword matching rejecting valid paraphrases or equivalent representations."
        },
        "failures": failures
    }
    
    out_path = "modern_gpt/experiments/phase2_experiment_e/research_design/failure_taxonomy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, indent=2)
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    main()
