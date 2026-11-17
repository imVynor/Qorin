import json
import re
from pathlib import Path

path = Path('modern_gpt/experiments/phase2_experiment_c/reports/tool_use_evaluation.json')
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

cond = data['cond_4_both_tools']
tasks = cond['task_details']

print("=== DETAILED AUDIT OF ALL 30 TASKS IN COND 4 ===")
for i, t in enumerate(tasks, 1):
    tid = t['id']
    ttype = t['type']
    exp = t['tool_expected']
    called = t['called_tool']
    syntax = t['syntax_ok']
    exec_ok = t['exec_ok']
    task_ok = t['task_ok']
    r1 = t['round1_text']
    r2 = t.get('round2_text')
    
    # Check what tool was selected in r1
    selected_tool = None
    has_tag = False
    json_valid = False
    parse_err = None
    if r1:
        if "<|tool_call|>" in r1 or "<tool_call>" in r1:
            has_tag = True
            m = re.search(r"<\|tool_call\|>\s*(.*?)\s*(?:<\|im_end\|>|<\|endoftext\|>|</tool_call>|$)", r1, re.DOTALL)
            if not m:
                m = re.search(r"<tool_call>(.*?)</tool_call>", r1, re.DOTALL)
            if m:
                raw_json = m.group(1).strip()
                try:
                    p = json.loads(raw_json)
                    json_valid = True
                    selected_tool = p.get('name')
                except Exception as e:
                    parse_err = str(e)
            else:
                parse_err = "No regex match"
                
    print(f"\nTask {i:02d} [{tid}] ({ttype})")
    print(f"  Expected={exp} | HasTag={has_tag} | JSONValid={json_valid} | Tool={selected_tool} | SyntaxOk={syntax} | ExecOk={exec_ok} | TaskOk={task_ok}")
    if parse_err:
        print(f"  Parse Error: {parse_err}")
    print(f"  R1 Text: {repr(r1[:150]) if r1 else 'None'}")
    if r2:
        print(f"  R2 Text: {repr(r2[:150])}")
