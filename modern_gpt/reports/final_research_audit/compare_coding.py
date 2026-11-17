import json

with open('modern_gpt/reports/phase_2c/experiment_b_evaluation/coding_evaluation.json', 'r', encoding='utf-8') as f:
    b_data = json.load(f)
    b = b_data['experiment_b']
with open('modern_gpt/experiments/phase2_experiment_c/reports/coding_evaluation.json', 'r', encoding='utf-8') as f:
    c = json.load(f)

print('Exp B Pass@1:', b.get('pass_at_1_pct', b.get('pass_at_1')), 'Syntax:', b.get('syntax_valid_pct'))
print('Exp C Pass@1:', c.get('pass_at_1_pct', c.get('pass_at_1')), 'Syntax:', c.get('syntax_valid_pct'))

b_tasks = {t['id']: t for t in b.get('task_details', [])}
c_tasks = {t['id']: t for t in c.get('task_details', [])}

print('\nTasks where Exp B passed but Exp C failed:')
for tid, bt in b_tasks.items():
    ct = c_tasks.get(tid)
    if bt.get('passed') and ct and not ct.get('passed'):
        print(f"\nTask {tid}: prompt={bt.get('prompt', '')[:80]}")
        print(f"  B code: {repr(bt.get('generated_code', '')[:120])}")
        print(f"  C code: {repr(ct.get('generated_code', '')[:120])}")

print('\nTasks where Exp C passed but Exp B failed:')
for tid, ct in c_tasks.items():
    bt = b_tasks.get(tid)
    if ct.get('passed') and bt and not bt.get('passed'):
        print(f"\nTask {tid}: prompt={ct.get('prompt', '')[:80]}")
        print(f"  B code: {repr(bt.get('generated_code', '')[:120]) if bt else 'None'}")
        print(f"  C code: {repr(ct.get('generated_code', '')[:120])}")
