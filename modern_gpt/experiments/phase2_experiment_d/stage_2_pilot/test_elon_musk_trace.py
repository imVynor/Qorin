"""
Authoritative Live Trace Test on 'grab info on elon musk'
Evaluates both Experiment C baseline and Experiment D Pilot (Step 50)
under full Two-Stage Protocol with both tools (web_research and python) active.
"""

import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from tokenizers import Tokenizer
from modern_gpt.model import ModernGPT, ModelConfig
from backend.tools.orchestrator import ToolOrchestrator
from backend.tools.two_stage_protocol import QorinTwoStageProtocol

TOK_PATH = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"
EXP_C_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt"
PILOT_CKPT = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/checkpoints/ckpt_pilot_step50.pt"

SYSTEM_PROMPT = """You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Execute Python code in an isolated sandbox.
2. `web_research`: Query real-time search engine for current facts, news, and live information.
To call a tool, generate:
<|tool_call|>{"name": "<tool_name>", "arguments": {<args>}}"""

PROMPT = "grab info on elon musk"

def run_trace(model, tok, orchestrator, protocol, model_name: str):
    print(f"\n=======================================================")
    print(f"Tracing 'grab info on elon musk' with {model_name}")
    print(f"=======================================================")

    prompt_formatted = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{PROMPT}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # Stage 1: First-round Generation
    input_ids = tok.encode(prompt_formatted).ids
    generated = list(input_ids)
    stop_ids = {5, 9}  # <|im_end|>=5, <|endoftext|>=9

    for _ in range(128):
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device="cuda")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(inp)
        next_id = int(torch.argmax(logits[0, -1, :]).item())
        if next_id in stop_ids:
            break
        generated.append(next_id)

    raw_token_ids = generated[len(input_ids):]
    raw_output_with_specials = tok.decode(raw_token_ids, skip_special_tokens=False)
    raw_output_text = tok.decode(raw_token_ids, skip_special_tokens=True)

    print(f"Stage 1 Raw Token IDs (first 15): {raw_token_ids[:15]}")
    print(f"Stage 1 Output (with specials): {repr(raw_output_with_specials[:120])}")
    print(f"Stage 1 Output (clean): {repr(raw_output_text[:120])}")

    # Stage A Action Determination via Protocol
    action, tool_name, args = protocol.determine_action(
        first_round_text=raw_output_with_specials,
        web_enabled=True,
        python_enabled=True,
        manual_override=None
    )
    print(f"Protocol Decision: Action={action}, Tool={tool_name}, Args={args}")

    dispatch_occurred = False
    execution_success = False
    observation_text = None
    retrieved_sources = []
    final_output = ""

    if action in ["WEB", "PYTHON"] and tool_name:
        dispatch_occurred = True
        print(f"Dispatching to Orchestrator ({tool_name})...")
        trace = orchestrator.execute_tool(tool_name, args if isinstance(args, dict) else {}, web_enabled=True, python_enabled=True)
        execution_success = (trace.status == "completed")
        observation_text = json.dumps(trace.output_payload)
        retrieved_sources = trace.output_payload.get("sources", []) if isinstance(trace.output_payload, dict) else []

        print(f"Execution Success: {execution_success}")
        print(f"Observation (preview): {repr(str(observation_text)[:120])}")
        print(f"Retrieved Sources: {retrieved_sources}")

        # Stage 2: Second-round Synthesis
        second_prompt = (
            f"{prompt_formatted}{raw_output_with_specials}<|im_end|>\n"
            f"<|im_start|>tool\n{observation_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        s2_input_ids = tok.encode(second_prompt).ids
        s2_generated = list(s2_input_ids)
        for _ in range(256):
            ctx = s2_generated[-2048:]
            inp = torch.tensor([ctx], dtype=torch.long, device="cuda")
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inp)
            next_id = int(torch.argmax(logits[0, -1, :]).item())
            if next_id in stop_ids:
                break
            s2_generated.append(next_id)
        s2_raw_ids = s2_generated[len(s2_input_ids):]
        final_output = tok.decode(s2_raw_ids, skip_special_tokens=True)
    else:
        final_output = raw_output_text

    print(f"Final Synthesized Output: {repr(final_output[:200])}")

    return {
        "model_name": model_name,
        "raw_token_ids": raw_token_ids[:30],
        "raw_output_with_specials": raw_output_with_specials,
        "raw_output_text": raw_output_text,
        "action": action,
        "tool_name": tool_name,
        "arguments": args,
        "dispatch_occurred": dispatch_occurred,
        "execution_success": execution_success,
        "observation": observation_text,
        "retrieved_sources": retrieved_sources,
        "final_output": final_output
    }

def main():
    tok = Tokenizer.from_file(str(TOK_PATH))
    cfg = ModelConfig.get_500m_config()
    orchestrator = ToolOrchestrator()
    protocol = QorinTwoStageProtocol(orchestrator)

    # 1. Experiment C
    m_c = ModernGPT(cfg).to("cuda")
    ckpt_c = torch.load(EXP_C_CKPT, map_location="cuda", weights_only=False)
    m_c.load_state_dict(ckpt_c["model_state_dict"], strict=True)
    m_c.eval()
    trace_c = run_trace(m_c, tok, orchestrator, protocol, "Experiment C Baseline")
    del m_c, ckpt_c
    torch.cuda.empty_cache()

    # 2. Experiment D Pilot
    m_d = ModernGPT(cfg).to("cuda")
    ckpt_d = torch.load(PILOT_CKPT, map_location="cuda", weights_only=False)
    m_d.load_state_dict(ckpt_d["model_state_dict"], strict=True)
    m_d.eval()
    trace_d = run_trace(m_d, tok, orchestrator, protocol, "Experiment D Pilot (Step 50)")
    del m_d, ckpt_d
    torch.cuda.empty_cache()

    out_file = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_d/stage_2_pilot/logs/elon_musk_live_trace.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"exp_c": trace_c, "exp_d": trace_d}, f, indent=2)
    print(f"\nTrace results written to {out_file}")

if __name__ == "__main__":
    main()
