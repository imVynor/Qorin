"""
Zero-Shot Model Inference Evaluation across Harness Configurations A, B, and C
Model: ModernGPT-S 500M (ckpt_experiment_b_final.pt)
"""

import json
from pathlib import Path
import sys
import time
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.tools.orchestrator import ToolOrchestrator
from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper
from tokenizers import Tokenizer

STAGE_DIR = Path(__file__).resolve().parent

# Manifest definitions
MANIFEST_FULL = (
    "You are ModernGPT-S, a helpful and precise AI assistant.\n"
    "You have access to the following external tools:\n"
    "1. `python`: Executes Python code in an isolated sandbox. Use this for calculations, math, data processing, and algorithms.\n"
    "   Call format: <|tool_call|>{\"name\": \"python\", \"arguments\": {\"code\": \"<python_code>\"}}<|im_end|>\n"
    "2. `web_research`: Searches the web for current, factual, or verifiable information.\n"
    "   Call format: <|tool_call|>{\"name\": \"web_research\", \"arguments\": {\"query\": \"<search_query>\"}}<|im_end|>\n"
    "When a tool is needed, output ONLY the tool call. Otherwise, respond directly to the user."
)

MANIFEST_PYTHON_ONLY = (
    "You are ModernGPT-S, a helpful and precise AI assistant.\n"
    "You have access to the following external tool:\n"
    "1. `python`: Executes Python code in an isolated sandbox. Use this for calculations, math, data processing, and algorithms.\n"
    "   Call format: <|tool_call|>{\"name\": \"python\", \"arguments\": {\"code\": \"<python_code>\"}}<|im_end|>\n"
    "When a tool is needed, output ONLY the tool call. Otherwise, respond directly to the user."
)

MANIFEST_WEB_ONLY = (
    "You are ModernGPT-S, a helpful and precise AI assistant.\n"
    "You have access to the following external tool:\n"
    "1. `web_research`: Searches the web for current, factual, or verifiable information.\n"
    "   Call format: <|tool_call|>{\"name\": \"web_research\", \"arguments\": {\"query\": \"<search_query>\"}}<|im_end|>\n"
    "When a tool is needed, output ONLY the tool call. Otherwise, respond directly to the user."
)

STANDALONE_PROMPT = "You are ModernGPT-S, a helpful and precise AI assistant."

TEST_PROMPTS = [
    {
        "id": "calc_01",
        "category": "python_math",
        "prompt": "Calculate the square root of 9801 minus 45 using Python."
    },
    {
        "id": "calc_02",
        "category": "python_math",
        "prompt": "Compute the sum of the cubes of integers from 1 to 25."
    },
    {
        "id": "web_01",
        "category": "web_research",
        "prompt": "Who won the 2024 Nobel Prize in Physics and for what discovery?"
    },
    {
        "id": "direct_01",
        "category": "no_tool",
        "prompt": "What is the difference between supervised and unsupervised learning?"
    },
    {
        "id": "direct_02",
        "category": "no_tool",
        "prompt": "What is 2 + 2?"
    }
]

def generate_response(model, tokenizer, system_prompt, user_prompt, device, max_new_tokens=128):
    if system_prompt:
        chat_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    else:
        chat_prompt = f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    input_ids = tokenizer.encode(chat_prompt)
    curr_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    stop_tokens = {5, 9, 3}  # <|im_end|>, <|endoftext|>, <|eos|>
    
    gen_ids = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if curr_ids.shape[1] >= 2048:
                break
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(curr_ids)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            tid = next_tok.item()
            if tid in stop_tokens:
                break
            gen_ids.append(tid)
            curr_ids = torch.cat([curr_ids, next_tok], dim=-1)
            
    return tokenizer.decode(gen_ids).strip()

def main():
    print("Loading Model and Tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = REPO_ROOT / "modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt"
    tok_path = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"

    tok_raw = Tokenizer.from_file(str(tok_path))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")

    cfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
    model = ModernGPT(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    orchestrator = ToolOrchestrator()
    results = []

    configs = [
        ("Config A (Standalone: No Manifest)", None),
        ("Config B (Harness Enabled: Full Two-Tool Manifest)", MANIFEST_FULL),
        ("Config C1 (Partial: Python Only Manifest)", MANIFEST_PYTHON_ONLY),
        ("Config C2 (Partial: Web Only Manifest)", MANIFEST_WEB_ONLY),
    ]

    for cfg_name, sys_prompt in configs:
        print(f"\n=== Evaluating {cfg_name} ===")
        for tp in TEST_PROMPTS:
            out = generate_response(model, tokenizer, sys_prompt, tp["prompt"], device)
            parsed = orchestrator.parse_model_tool_call(out)
            
            rec = {
                "config": cfg_name,
                "task_id": tp["id"],
                "category": tp["category"],
                "prompt": tp["prompt"],
                "model_output": out,
                "tool_called": parsed is not None,
                "parsed_tool": parsed
            }
            results.append(rec)
            print(f"[{tp['id']}] Output snippet: {repr(out[:80])}")
            print(f"  Tool emitted: {parsed is not None} -> {parsed}")

    out_file = STAGE_DIR / "zero_shot_model_behavior.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved zero-shot evaluation behavior to {out_file}")

if __name__ == "__main__":
    main()
