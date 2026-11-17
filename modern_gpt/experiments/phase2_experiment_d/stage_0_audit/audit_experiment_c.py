import json
from pathlib import Path
from collections import Counter, defaultdict
import re

c_train_path = Path('modern_gpt/experiments/phase2_experiment_c/stage_1_data/train.jsonl')
c_val_path = Path('modern_gpt/experiments/phase2_experiment_c/stage_1_data/val.jsonl')

def audit_file(path):
    counts = Counter()
    user_prompts = []
    web_prompts = []
    py_prompts = []
    direct_prompts = []
    multi_tool = 0
    schemas = Counter()
    
    with open(path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            msgs = d.get('messages', [])
            u_msg = next((m['content'] for m in msgs if m['role'] == 'user'), '')
            user_prompts.append(u_msg)
            
            tool_calls = []
            for m in msgs:
                if m['role'] == 'assistant' and '<|tool_call|>' in m['content']:
                    c = m['content']
                    matches = re.findall(r'<\|tool_call\|>(.*?)(?:<\|im_end\|>|$)', c, re.DOTALL)
                    for match in matches:
                        try:
                            t_data = json.loads(match.strip())
                            name = t_data.get('name')
                            args = t_data.get('arguments', {})
                            tool_calls.append((name, args))
                            schemas[f'{name}_valid'] += 1
                        except Exception:
                            schemas['malformed_json'] += 1
                            tool_calls.append(('malformed', {}))
            
            has_py = any('python' in str(t[0]) for t in tool_calls)
            has_web = any('web' in str(t[0]) for t in tool_calls)
            
            if len(tool_calls) > 1:
                multi_tool += 1
                
            if has_web and has_py:
                counts['both_tools'] += 1
            elif has_web:
                counts['web_only'] += 1
                web_prompts.append(u_msg)
            elif has_py:
                counts['python_only'] += 1
                py_prompts.append(u_msg)
            else:
                counts['direct'] += 1
                direct_prompts.append(u_msg)
                
    return {
        'total': len(user_prompts),
        'counts': dict(counts),
        'multi_tool': multi_tool,
        'schemas': dict(schemas),
        'web_prompts': web_prompts,
        'user_prompts': user_prompts
    }

train_audit = audit_file(c_train_path)
val_audit = audit_file(c_val_path)

print('=== TRAIN AUDIT ===')
print('Total:', train_audit['total'])
print('Counts:', train_audit['counts'])
print('Multi-tool:', train_audit['multi_tool'])
print('Schemas:', train_audit['schemas'])

py_total = train_audit['counts'].get('python_only', 0) + train_audit['counts'].get('both_tools', 0)
web_total = train_audit['counts'].get('web_only', 0) + train_audit['counts'].get('both_tools', 0)
ratio = py_total / max(1, web_total)
print(f'Actual Python / Web ratio: {ratio:.2f}:1 (Python: {py_total}, Web: {web_total})')

web_prefixes = Counter()
for p in train_audit['web_prompts']:
    prefix = p[:30]
    web_prefixes[prefix] += 1
print('Web prompt prefix distribution:')
for k, v in web_prefixes.most_common(10):
    print(f'  "{k}": {v}')

print('\n=== VAL AUDIT ===')
print('Total:', val_audit['total'])
print('Counts:', val_audit['counts'])
print('Multi-tool:', val_audit['multi_tool'])
print('Schemas:', val_audit['schemas'])

# Check train/val prompt overlap
train_set = set(p.strip().lower() for p in train_audit['user_prompts'])
val_set = set(p.strip().lower() for p in val_audit['user_prompts'])
overlap = train_set.intersection(val_set)
print(f'Exact Prompt Overlap between Train and Val: {len(overlap)}')

# Save structured audit results
out_data = {
    'train': {
        'total': train_audit['total'],
        'counts': train_audit['counts'],
        'multi_tool': train_audit['multi_tool'],
        'schemas': train_audit['schemas'],
        'python_to_web_ratio': round(ratio, 2)
    },
    'val': {
        'total': val_audit['total'],
        'counts': val_audit['counts'],
        'multi_tool': val_audit['multi_tool'],
        'schemas': val_audit['schemas']
    },
    'train_val_exact_overlap_count': len(overlap)
}
with open('modern_gpt/experiments/phase2_experiment_d/stage_0_audit/exp_c_dataset_quantification.json', 'w', encoding='utf-8') as f:
    json.dump(out_data, f, indent=2)
