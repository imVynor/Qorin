import json

bench_path = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
with open(bench_path, "r", encoding="utf-8") as f:
    bench = [json.loads(line) for line in f]

multistep = [r for r in bench if r.get("category") == "multistep_web_python"]
print(f"Total multistep tasks: {len(multistep)}")
for m in multistep:
    print(f"[{m['id']}] {m['prompt']}")
    print(f"  Expected: {m.get('expected_answer')}")
    print(f"  Criteria: {m.get('semantic_criteria')}\n")
