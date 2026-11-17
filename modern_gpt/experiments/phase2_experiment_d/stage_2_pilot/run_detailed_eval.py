"""
Test all 30 tasks with skip_special_tokens=False
"""
import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tokenizers import Tokenizer
from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.experiments.phase2_experiment_d.stage_2_pilot.evaluate_pilot_vs_baseline import BENCHMARK_TASKS, SYSTEM_PROMPT

tok = Tokenizer.from_file('modern_gpt/artifacts/tokenizer_32k/tokenizer.json')
cfg = ModelConfig.get_500m_config()

def evaluate_tasks(ckpt_path, name):
    print(f"\n==========================================")
    print(f"Evaluating {name}")
    print(f"==========================================")
    m = ModernGPT(cfg).to('cuda')
    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'], strict=True)
    m.eval()

    results = []
    for idx, t in enumerate(BENCHMARK_TASKS):
        prompt_formatted = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{t['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        input_ids = tok.encode(prompt_formatted).ids
        generated = list(input_ids)
        for _ in range(128):
            ctx = generated[-2048:]
            inp = torch.tensor([ctx], dtype=torch.long, device='cuda')
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = m(inp)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            if next_id in [5, 9]: break
            generated.append(next_id)
        
        new_ids = generated[len(input_ids):]
        raw_output = tok.decode(new_ids, skip_special_tokens=False)
        
        has_tool = ("<|tool_call|>" in raw_output) or ("<tool_call>" in raw_output) or (len(new_ids) > 0 and new_ids[0] == 6)
        
        # parse tool name
        tool_name = None
        args = None
        if has_tool:
            text_after = raw_output.replace("<|tool_call|>", "").strip()
            try:
                # find json
                s = text_after.find("{")
                e = text_after.rfind("}")
                if s != -1 and e != -1:
                    j = json.loads(text_after[s:e+1])
                    tool_name = j.get("name")
                    args = j.get("arguments")
            except Exception:
                pass
        
        print(f"[{idx+1:02d}/30] {t['id']:<22} | Exp: {str(t['expected_tool']):<12} | Act: {str(tool_name):<12} | Intent: {'OK' if has_tool == t['expected_intent'] else 'FAIL'}")
        results.append({
            "id": t["id"],
            "category": t["category"],
            "expected_intent": t["expected_intent"],
            "has_tool": has_tool,
            "expected_tool": t["expected_tool"],
            "selected_tool": tool_name,
            "output": raw_output[:120]
        })
    del m, ckpt
    torch.cuda.empty_cache()
    return results

res_c = evaluate_tasks('modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt', 'Exp C Baseline')
res_d = evaluate_tasks('modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/checkpoints/ckpt_pilot_step50.pt', 'Exp D Pilot (Step 50)')

with open('modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/logs/detailed_30task_results.json', 'w', encoding='utf-8') as f:
    json.dump({'exp_c': res_c, 'exp_d': res_d}, f, indent=2)
print("Saved to detailed_30task_results.json")
