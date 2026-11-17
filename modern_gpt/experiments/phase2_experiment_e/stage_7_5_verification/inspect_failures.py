import json

res = json.load(open("modern_gpt/experiments/phase2_experiment_e/stage_7_prerequisites/multistep_test_results.json"))
bench = {r["id"]: r for r in [json.loads(line) for line in open("modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl")]}

failed_ids = ["multistep_02", "multistep_06", "multistep_15"]
failed_traces = [t for t in res["task_traces"] if t["task_id"] in failed_ids]

print(f"Found {len(failed_traces)} failed traces:")
for t in failed_traces:
    tid = t["task_id"]
    b = bench[tid]
    print("=" * 80)
    print(f"TASK: {tid}")
    print(f"Prompt: {b['prompt']}")
    print(f"Expected Answer: {b.get('expected_answer')}")
    print(f"Semantic Criteria: {b.get('semantic_criteria')}")
    print(f"Web Query: {t['web_query']} | Status: {t['web_status']}")
    print(f"Extracted Metrics: {t['extracted_metrics']}")
    print(f"Python Code:\n{t['python_code']}")
    print(f"Python Output:\n{t['python_output']}")
    print(f"Final Synthesis:\n{t['final_synthesis']}")
    print(f"Semantic Pass: {t['semantic_pass']}")
