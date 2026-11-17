import json

with open("modern_gpt/artifacts/diagnostics/sft_50m_targeted_v2_latest_eval.json", encoding="utf-8") as f:
    d = json.load(f)

passed_py = [r for r in d['val_details'] if r.get('python_functional_passed')]
print(f"=== HELD-OUT VAL FUNCTIONAL PASS CASES (LATEST.PT) (Total: {len(passed_py)}) ===")
for p in passed_py:
    print(f"ID: {p['id']} | Family: {p['task_family']}")
    print(f"Instruction: {p['instruction']}")
    print(f"Target:\n{p['target']}")
    print(f"Generated:\n{p['generated']}")
    print("-" * 50)

