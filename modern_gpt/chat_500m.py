"""
ModernGPT-S 500M — Interactive Chat Terminal
============================================
Chat with the fine-tuned 500M checkpoints:
- Mode 'exp_b' (Default): modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt
- Mode 'exp_a': modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt
- Mode 'base': modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt

Features:
- Seamless in-session model switching: `/mode exp_b`, `/mode exp_a`, `/mode base`
- Canonical ChatML formatting (<|im_start|>role\ncontent<|im_end|>)
- Multi-turn stateful conversational memory
- Dynamic sampling parameters (/temp, /rep)
- Live VRAM and status monitoring (/status)

Usage:
    # Interactive REPL (defaults to Experiment B):
    python modern_gpt/chat_500m.py

    # Interactive REPL with Experiment A:
    python modern_gpt/chat_500m.py --mode exp_a

    # Interactive REPL with Pretrained Base:
    python modern_gpt/chat_500m.py --mode base

    # Single-shot prompt with Experiment B:
    python modern_gpt/chat_500m.py --mode exp_b --prompt "Write a Python function to check palindrome"
"""

import argparse
import gc
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import torch
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modern_gpt.model import ModernGPT, ModelConfig
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper

MODEL_MODES = {
    "exp_c": {
        "name": "Experiment C (Targeted Autonomous Two-Tool SFT)",
        "checkpoint": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt",
        "description": "Targeted autonomous two-tool SFT model fine-tuned for closed-loop Python and Web Research."
    },
    "exp_b": {
        "name": "Experiment B (35M General + Tools SFT)",
        "checkpoint": REPO_ROOT / "modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt",
        "fallback": REPO_ROOT / "modern_gpt/experiments/phase2_experiment_b/run_35m/checkpoints/ckpt_experiment_b_final.pt",
        "description": "Controlled 35M-token SFT model fine-tuned on General Instructions, Reasoning, Coding, and Tool-Use."
    },
    "exp_a": {
        "name": "Experiment A (25M General SFT Pilot)",
        "checkpoint": REPO_ROOT / "modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt",
        "description": "Controlled 25M-token general instruction fine-tuning pilot checkpoint."
    },
    "base": {
        "name": "ModernGPT-S 500M Pretrained Base (1.12B Tokens)",
        "checkpoint": REPO_ROOT / "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
        "description": "Frozen foundational base language model trained on 1,121,648,640 tokens."
    }
}

# Aliases
MODEL_MODES["c"] = MODEL_MODES["exp_c"]
MODEL_MODES["b"] = MODEL_MODES["exp_b"]
MODEL_MODES["a"] = MODEL_MODES["exp_a"]
MODEL_MODES["pretrain"] = MODEL_MODES["base"]

DEFAULT_MODE = "exp_b"
DEFAULT_TOK = REPO_ROOT / "modern_gpt/artifacts/tokenizer_32k/tokenizer.json"


def resolve_checkpoint_path(mode: str, custom_ckpt: Optional[str] = None) -> Tuple[Path, str]:
    if custom_ckpt:
        p = Path(custom_ckpt)
        if not p.is_absolute():
            p = REPO_ROOT / p
        return p, f"Custom Checkpoint ({p.name})"
    
    mode_key = mode.lower().strip()
    if mode_key not in MODEL_MODES:
        print(f"[!] Unknown mode '{mode}'. Falling back to default '{DEFAULT_MODE}'.")
        mode_key = DEFAULT_MODE
        
    info = MODEL_MODES[mode_key]
    primary_p = info["checkpoint"]
    if primary_p.exists():
        return primary_p, info["name"]
    fallback_p = info.get("fallback")
    if fallback_p and fallback_p.exists():
        return fallback_p, info["name"]
    return primary_p, info["name"]


def load_model(checkpoint_path: Path, device: torch.device):
    print(f"[*] Loading ModernGPT-S 500M from {checkpoint_path.name} onto {device}...")
    m_cfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
    model = ModernGPT(m_cfg).to(device=device, dtype=torch.bfloat16 if "cuda" in str(device) else torch.float32)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print("[+] Model loaded successfully.")
    return model


@torch.no_grad()
def generate_reply(
    model: ModernGPT,
    tokenizer: HuggingFaceBPEWrapper,
    messages: List[Dict[str, str]],
    device: torch.device,
    max_new_tokens: int = 250,
    temperature: float = 0.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.18,
    context_length: int = 2048,
) -> str:
    chat_prompt = ""
    for msg in messages:
        chat_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    chat_prompt += "<|im_start|>assistant\n"

    input_ids = tokenizer.encode(chat_prompt)
    if not input_ids:
        input_ids = [2]

    # Enforce maximum prompt window budget
    if len(input_ids) > (context_length - max_new_tokens):
        input_ids = input_ids[-(context_length - max_new_tokens):]

    curr_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_tokens: List[int] = []

    use_autocast = "cuda" in str(device)
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else torch.no_grad()

    for _ in range(max_new_tokens):
        if curr_ids.size(1) >= context_length:
            break

        with autocast_ctx:
            logits = model(curr_ids)[:, -1, :].clone()

        # Repetition penalty
        if repetition_penalty != 1.0 and generated_tokens:
            for token_id in set(generated_tokens):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

        if temperature <= 0.05:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled_logits = logits / max(temperature, 1e-4)
            probs = torch.softmax(scaled_logits, dim=-1)

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                sorted_probs[sorted_indices_to_remove] = 0.0
                probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                next_token_sorted_idx = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, next_token_sorted_idx)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id in [3, 5, 9]:  # <|eos|>, <|im_end|>, <|endoftext|>
            break

        generated_tokens.append(token_id)
        curr_ids = torch.cat([curr_ids, next_token], dim=1)

    return tokenizer.decode(generated_tokens).strip()


from backend.tools.orchestrator import ToolOrchestrator

TOOL_SPEC_PATH = REPO_ROOT / "modern_gpt/experiments/phase2_experiment_c/tool_context_spec.json"
TWO_TOOL_MANIFEST = ""
if TOOL_SPEC_PATH.exists():
    try:
        import json
        with open(TOOL_SPEC_PATH, "r", encoding="utf-8") as f:
            TWO_TOOL_MANIFEST = json.load(f)["compact_manifests"]["two_tool_manifest"]
    except Exception:
        pass


def execute_closed_loop_turn(
    model: ModernGPT,
    tokenizer: HuggingFaceBPEWrapper,
    history: List[Dict[str, str]],
    device: torch.device,
    orchestrator: ToolOrchestrator,
    temperature: float = 0.5,
    repetition_penalty: float = 1.18,
    max_rounds: int = 2
) -> str:
    current_round = 0
    final_reply = ""

    while current_round < max_rounds:
        reply = generate_reply(model, tokenizer, history, device, temperature=temperature, repetition_penalty=repetition_penalty)
        parsed = orchestrator.parse_model_tool_call(reply)

        if parsed:
            tool_name, tool_args = parsed
            print(f"\n[Autonomous Tool Call]: {tool_name}({tool_args})")
            trace = orchestrator.execute_tool(tool_name, tool_args, requested_by="model", web_enabled=True, python_enabled=True)
            print(f"[Tool Execution Result]: {trace.output_payload}")

            obs_tag = "<|tool_result|>" if trace.status == "completed" else "<|tool_error|>"
            import json
            obs_content = f"{obs_tag}{json.dumps(trace.output_payload)}"

            history.append({"role": "assistant", "content": reply})
            history.append({"role": "tool", "content": obs_content})
            current_round += 1
        else:
            final_reply = reply
            break

    if not final_reply and current_round >= max_rounds:
        # Final synthesis round
        final_reply = generate_reply(model, tokenizer, history, device, temperature=temperature, repetition_penalty=repetition_penalty)

    return final_reply


def run_single_prompt(prompt: str, ckpt_path: Path, model_name: str, temp: float, rep_pen: float, current_mode: str = "exp_c"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_raw = Tokenizer.from_file(str(DEFAULT_TOK))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")
    model = load_model(ckpt_path, device)
    orchestrator = ToolOrchestrator()

    system_content = TWO_TOOL_MANIFEST if (current_mode in ["exp_c", "c"] and TWO_TOOL_MANIFEST) else "You are ModernGPT-S, an intelligent, helpful, and concise AI assistant."
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt}
    ]
    print(f"\n[Mode: {model_name}]")
    print(f"User: {prompt}\n")

    if current_mode in ["exp_c", "c"]:
        reply = execute_closed_loop_turn(model, tokenizer, messages, device, orchestrator, temperature=temp, repetition_penalty=rep_pen)
    else:
        reply = generate_reply(model, tokenizer, messages, device, temperature=temp, repetition_penalty=rep_pen)

    print(f"Assistant: {reply}\n")


def print_help():
    print("""
Available Interactive Commands:
  /mode <exp_b|exp_a|base> : Switch active model checkpoint seamlessly
  /status                  : Show active model, checkpoint, and generation settings
  /clear                   : Reset conversation history
  /temp <float>            : Set sampling temperature (e.g., /temp 0.3)
  /rep <float>             : Set repetition penalty (e.g., /rep 1.15)
  /help                    : Display this help menu
  /exit or exit            : Exit interactive chat
""")


def interactive_repl(current_mode: str, custom_ckpt: Optional[str], temp: float, rep_pen: float):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok_raw = Tokenizer.from_file(str(DEFAULT_TOK))
    tokenizer = HuggingFaceBPEWrapper(tok_raw, name="bpe_32k_production")
    orchestrator = ToolOrchestrator()

    ckpt_path, model_name = resolve_checkpoint_path(current_mode, custom_ckpt)
    model = load_model(ckpt_path, device)

    default_system = TWO_TOOL_MANIFEST if (current_mode in ["exp_c", "c"] and TWO_TOOL_MANIFEST) else "You are ModernGPT-S, an intelligent, helpful, and concise AI assistant."
    history = [
        {"role": "system", "content": default_system}
    ]

    print("\n" + "="*68)
    print(" ModernGPT-S 500M Interactive Chat Terminal")
    print(f" Active Mode : {model_name}")
    print(f" Checkpoint  : {ckpt_path.name}")
    print(" Type /help for commands or /mode <exp_c|exp_b|exp_a|base> to switch models.")
    print("="*68 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["/exit", "exit", "quit", ":q"]:
            print("Goodbye!")
            break
        if user_input.lower() in ["/help", "help", "?"]:
            print_help()
            continue
        if user_input.lower() == "/clear":
            history = [{"role": "system", "content": default_system}]
            print("[+] Conversation memory cleared.\n")
            continue
        if user_input.lower() == "/status":
            vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
            print(f"\n[Status] Active Mode: {model_name}")
            print(f"         Checkpoint : {ckpt_path}")
            print(f"         Device     : {device} (VRAM: {vram_mb:.1f} MB)")
            print(f"         Temperature: {temp:.2f} | Repetition Penalty: {rep_pen:.2f}")
            print(f"         Turns      : {len(history)-1}\n")
            continue
        if user_input.lower().startswith("/temp"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    temp = float(parts[1])
                    print(f"[+] Sampling temperature set to {temp:.2f}\n")
                except ValueError:
                    print("Invalid temperature value. Example: /temp 0.4\n")
            else:
                print(f"Current temperature: {temp:.2f}\n")
            continue
        if user_input.lower().startswith("/rep"):
            parts = user_input.split()
            if len(parts) > 1:
                try:
                    rep_pen = float(parts[1])
                    print(f"[+] Repetition penalty set to {rep_pen:.2f}\n")
                except ValueError:
                    print("Invalid penalty value. Example: /rep 1.15\n")
            else:
                print(f"Current repetition penalty: {rep_pen:.2f}\n")
            continue
        if user_input.lower().startswith(("/mode", "/model", "/switch")):
            parts = user_input.split()
            if len(parts) > 1:
                target_mode = parts[1].lower().strip()
                new_ckpt, new_name = resolve_checkpoint_path(target_mode)
                if not new_ckpt.exists():
                    print(f"[!] Checkpoint for mode '{target_mode}' not found at {new_ckpt}\n")
                    continue
                print(f"[*] Switching model mode: {model_name} -> {new_name}...")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                model = load_model(new_ckpt, device)
                ckpt_path, model_name = new_ckpt, new_name
                current_mode = target_mode
                default_system = TWO_TOOL_MANIFEST if (current_mode in ["exp_c", "c"] and TWO_TOOL_MANIFEST) else "You are ModernGPT-S, an intelligent, helpful, and concise AI assistant."
                history = [{"role": "system", "content": default_system}]
                print(f"[+] Successfully switched to: {model_name}\n")
            else:
                print(f"Current mode: {model_name} ({current_mode})")
                print("Available modes: 'exp_c' (Experiment C Two-Tool), 'exp_b' (Experiment B), 'exp_a' (Experiment A), 'base' (Pretrained)\n")
            continue

        history.append({"role": "user", "content": user_input})
        print(f"\n[{model_name.split()[0]}]: ", end="", flush=True)

        if current_mode in ["exp_c", "c"]:
            reply = execute_closed_loop_turn(model, tokenizer, history, device, orchestrator, temperature=temp, repetition_penalty=rep_pen)
        else:
            reply = generate_reply(model, tokenizer, history, device, temperature=temp, repetition_penalty=rep_pen)

        print(reply + "\n")
        history.append({"role": "assistant", "content": reply})


def main():
    parser = argparse.ArgumentParser(description="ModernGPT-S 500M Multi-Mode Chat Terminal")
    parser.add_argument("--mode", "-m", type=str, default="exp_c", choices=["exp_c", "exp_b", "exp_a", "base", "c", "b", "a"],
                        help="Model mode to chat with: 'exp_c' (Experiment C Two-Tool, default), 'exp_b' (Experiment B 35M), 'exp_a' (Experiment A 25M), or 'base' (Pretrained)")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Explicit path to custom checkpoint (overrides --mode)")
    parser.add_argument("--prompt", "-p", type=str, default=None, help="Single-shot prompt mode (non-interactive)")
    parser.add_argument("--temp", "-t", type=float, default=0.5, help="Sampling temperature (default: 0.5)")
    parser.add_argument("--rep-penalty", "-r", type=float, default=1.18, help="Repetition penalty (default: 1.18)")
    args = parser.parse_args()

    ckpt_path, model_name = resolve_checkpoint_path(args.mode, args.checkpoint)

    if args.prompt:
        run_single_prompt(args.prompt, ckpt_path, model_name, args.temp, args.rep_penalty, args.mode)
    else:
        interactive_repl(args.mode, args.checkpoint, args.temp, args.rep_penalty)


if __name__ == "__main__":
    main()
