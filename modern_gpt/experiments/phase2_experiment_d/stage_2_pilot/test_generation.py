import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tokenizers import Tokenizer
from modern_gpt.model import ModernGPT, ModelConfig

tok = Tokenizer.from_file('modern_gpt/artifacts/tokenizer_32k/tokenizer.json')
cfg = ModelConfig.get_500m_config()

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Execute Python code in an isolated sandbox.
2. `web_research`: Query real-time search engine for current facts, news, and live information.
To call a tool, generate:
<|tool_call|>{"name": "<tool_name>", "arguments": {<args>}}"""

prompt = "grab info on elon musk"
prompt_formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

def test_model(path, name):
    m = ModernGPT(cfg).to('cuda')
    ckpt = torch.load(path, map_location='cuda', weights_only=False)
    m.load_state_dict(ckpt['model_state_dict'], strict=True)
    m.eval()
    
    input_ids = tok.encode(prompt_formatted).ids
    generated = list(input_ids)
    for _ in range(64):
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device='cuda')
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            logits = m(inp)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if next_id in [5, 9]: break
        generated.append(next_id)
    new_ids = generated[len(input_ids):]
    text_with_spec = tok.decode(new_ids, skip_special_tokens=False)
    print(f"{name} tokens:", new_ids[:10])
    print(f"{name} decoded with specials:", repr(text_with_spec))

test_model('modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt', 'Exp C')
test_model('modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/checkpoints/ckpt_pilot_step50.pt', 'Exp D Pilot')
