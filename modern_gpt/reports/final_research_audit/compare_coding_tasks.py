import json

# Load Exp B coding tasks
b_tasks = {}
with open('modern_gpt/reports/phase_2c/experiment_b_evaluation/individual_task_results.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('suite') == 'coding' and item.get('branch') == 'experiment_b':
            b_tasks[item['id']] = item

# Load Exp C coding tasks
with open('modern_gpt/experiments/phase2_experiment_c/reports/coding_evaluation.json', 'r', encoding='utf-8') as f:
    c_eval = json.load(f)
c_tasks = {t['id']: t for t in c_eval['task_details']}

# Load raw generations for prompts and code
c_gens = {}
with open('modern_gpt/experiments/phase2_experiment_c/reports/raw_generations.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('id', '').startswith('code_'):
            c_gens[item['id']] = item

b_gens = {}
with open('modern_gpt/reports/phase_2c/experiment_b_evaluation/raw_generations.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if item.get('id', '').startswith('code_') and item.get('branch') == 'experiment_b':
            b_gens[item['id']] = item

print(f"Exp B total passed: {sum(1 for t in b_tasks.values() if t['passed'])} / {len(b_tasks)} (42.0%)")
print(f"Exp C total passed: {sum(1 for t in c_tasks.values() if t['passed'])} / {len(c_tasks)} (38.0%)")

print("\n=== Tasks passed in Exp B (42%) but FAILED in Exp C (38%) ===")
for tid, bt in sorted(b_tasks.items()):
    ct = c_tasks.get(tid)
    if bt['passed'] and ct and not ct['passed']:
        prompt = b_gens.get(tid, {}).get('prompt', '') or c_gens.get(tid, {}).get('prompt', '')
        b_code = b_gens.get(tid, {}).get('generated', '')
        c_code = c_gens.get(tid, {}).get('generated', '')
        print(f"\nTask ID: {tid}")
        print(f"  Prompt: {repr(prompt.strip()[:100])}")
        print(f"  Exp B Code: {repr(b_code[:120])}")
        print(f"  Exp C Code: {repr(c_code[:120])}")

print("\n=== Tasks passed in Exp C but FAILED in Exp B ===")
for tid, ct in sorted(c_tasks.items()):
    bt = b_tasks.get(tid)
    if ct['passed'] and bt and not bt['passed']:
        prompt = c_gens.get(tid, {}).get('prompt', '') or b_gens.get(tid, {}).get('prompt', '')
        b_code = b_gens.get(tid, {}).get('generated', '')
        c_code = c_gens.get(tid, {}).get('generated', '')
        print(f"\nTask ID: {tid}")
        print(f"  Prompt: {repr(prompt.strip()[:100])}")
        print(f"  Exp B Code: {repr(b_code[:120])}")
        print(f"  Exp C Code: {repr(c_code[:120])}")
