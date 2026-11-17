import json

with open("modern_gpt/artifacts/diagnostics/sft_50m_targeted_v2_best_val_eval.json", encoding="utf-8") as f:
    d = json.load(f)

passed_py = [r for r in d['train_details'] if r.get('python_functional_passed')]
print(f"=== FUNCTIONAL PASS CASES (Total: {len(passed_py)}) ===")
for p in passed_py:
    print(f"ID: {p['id']} | Family: {p['task_family']}")
    print(f"Instruction: {p['instruction']}")
    print(f"Target: {p['target']}")
    print(f"Generated:\n{p['generated']}")
    print("-" * 50)

print("\n=== SAMPLE JSON GENERATIONS (VAL) ===")
json_cases = [r for r in d['val_details'] if r['category'] == 'json']
for r in json_cases[:4]:
    print(f"ID: {r['id']} | Syntax: {r['json_syntax_valid']} | Schema: {r['json_schema_valid']}")
    print(f"Instruction: {r['instruction']}")
    print(f"Generated:\n{repr(r['generated'][:150])}")
    print("-" * 50)

print("\n=== SAMPLE QA GENERATIONS (VAL) ===")
qa_cases = [r for r in d['val_details'] if r['category'] == 'qa']
for r in qa_cases[:4]:
    print(f"ID: {r['id']}")
    print(f"Instruction: {r['instruction']}")
    print(f"Target: {r['target']}")
    print(f"Generated: {r['generated']}")
    print("-" * 50)

