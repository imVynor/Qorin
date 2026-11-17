import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tokenizers import Tokenizer
from modern_gpt.model import ModernGPT, ModelConfig
from backend.tools.orchestrator import ToolOrchestrator
from backend.tools.two_stage_protocol import QorinTwoStageProtocol

tok = Tokenizer.from_file('modern_gpt/artifacts/tokenizer_32k/tokenizer.json')
cfg = ModelConfig.get_500m_config()
orchestrator = ToolOrchestrator()
protocol = QorinTwoStageProtocol(orchestrator)

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Execute Python code in an isolated sandbox.
2. `web_research`: Query real-time search engine for current facts, news, and live information.
To call a tool, generate:
<|tool_call|>{"name": "<tool_name>", "arguments": {<args>}}"""

prompt = "grab info on elon musk"
prompt_formatted = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"

m = ModernGPT(cfg).to('cuda')
ckpt = torch.load('modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt', map_location='cuda', weights_only=False)
m.load_state_dict(ckpt['model_state_dict'], strict=True)
m.eval()

input_ids = tok.encode(prompt_formatted).ids
generated = list(input_ids)
stop_ids = {5, 9}

for _ in range(64):
    ctx = generated[-2048:]
    inp = torch.tensor([ctx], dtype=torch.long, device='cuda')
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = m(inp)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    if next_id in stop_ids:
        break
    generated.append(next_id)

new_ids = generated[len(input_ids):]
raw_output = tok.decode(new_ids, skip_special_tokens=False)
print("Stage 1 Raw Output:", raw_output)

action, tool_name, args = protocol.determine_action(raw_output, web_enabled=True, python_enabled=True)
print(f"Protocol Decision: Action={action}, Tool={tool_name}, Args={args}")

is_valid, validated_args, err = protocol.validate_and_sanitize_arguments(tool_name, args, raw_output, user_query=prompt)
print(f"Validation Result: Valid={is_valid}, Args={validated_args}, Error={err}")

trace = orchestrator.execute_tool(tool_name, validated_args, web_enabled=True, python_enabled=True)
print(f"Dispatch Status: {trace.status}, Latency: {trace.latency_ms:.1f}ms")
print(f"Retrieved Sources: {trace.output_payload.get('sources', [])}")
print(f"Observation (preview): {repr(str(trace.output_payload)[:150])}")

# Stage 2: Synthesis
obs_text = json.dumps(trace.output_payload)
s2_prompt = (
    f"{prompt_formatted}{raw_output}<|im_end|>\n"
    f"<|im_start|>tool\n{obs_text}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)
s2_ids = tok.encode(s2_prompt).ids
s2_gen = list(s2_ids)
for _ in range(256):
    ctx = s2_gen[-2048:]
    inp = torch.tensor([ctx], dtype=torch.long, device='cuda')
    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        logits = m(inp)
    next_id = int(torch.argmax(logits[0, -1, :]).item())
    if next_id in stop_ids:
        break
    s2_gen.append(next_id)

s2_raw = s2_gen[len(s2_ids):]
final_answer = tok.decode(s2_raw, skip_special_tokens=True)
print("\nFinal Grounded Synthesis:")
print(final_answer)
