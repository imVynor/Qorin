"""
ModernGPT-S 500M — Interactive Chat CLI
========================================
A minimal terminal chat interface for the 500M pretrained checkpoint.

IMPORTANT: This is a base *pretrained* model, not an instruction-tuned assistant.
It operates as a text completion / continuation engine. Treat inputs as prompts
it will continue, not questions it will answer.

Usage:
    python modern_gpt/scripts/chat.py

Options (pass as flags):
    --temp FLOAT     Sampling temperature (default: 0.7)
    --top_p FLOAT    Top-p nucleus sampling (default: 0.9)
    --max_new INT    Max new tokens per response (default: 200)
    --no_rep         Disable repetition penalty

Tips for best results:
  - End your prompt with a colon, dash, or newline to get completions
  - Paste a code comment and get a function body
  - Provide article openings to continue them
  - Use /help to see commands during chat
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT.parent))

from modern_gpt.model.config import ModelConfig
from modern_gpt.model.gpt import ModernGPT

CKPT_PATH = REPO_ROOT / "checkpoints" / "checkpoint_501M" / "ckpt_501m_tokens_501678080.pt"
TOK_PATH  = REPO_ROOT / "artifacts" / "tokenizer_32k" / "tokenizer.json"

ANSI_RESET  = "\033[0m"
ANSI_BOLD   = "\033[1m"
ANSI_CYAN   = "\033[96m"
ANSI_GREEN  = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_DIM    = "\033[2m"
ANSI_RED    = "\033[91m"


def supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def c(text: str, code: str) -> str:
    return f"{code}{text}{ANSI_RESET}" if supports_color() else text


def load_model(device: str):
    print(c("  Loading tokenizer...", ANSI_DIM))
    tok = Tokenizer.from_file(str(TOK_PATH))

    print(c("  Loading checkpoint (3.1 GB — may take 10-20 seconds)...", ANSI_DIM))
    t0 = time.perf_counter()
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg  = ModelConfig(**ckpt["config"])
    model = ModernGPT(cfg).to(device, dtype=torch.bfloat16)
    model.load_state_dict(
        {k: v.to(device, dtype=torch.bfloat16) for k, v in ckpt["model_state_dict"].items()}
    )
    model.eval()
    dt = time.perf_counter() - t0

    tokens_trained = ckpt.get("total_tokens_processed", 501_678_080)
    step           = ckpt.get("global_update", 15310)
    print(c(f"  Model loaded in {dt:.1f}s — {cfg.n_layers}L {cfg.n_heads}Q/{cfg.kv_heads}KV, "
            f"{cfg.emb_dim}d, ctx={cfg.context_length}, "
            f"{tokens_trained/1e6:.1f}M tokens, step {step}", ANSI_DIM))
    return model, tok, cfg


@torch.inference_mode()
def generate(
    model,
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    device: str,
    stop_ids: Optional[List[int]] = None,
    stream: bool = True,
) -> str:
    input_ids = tok.encode(prompt).ids or [9]
    generated = list(input_ids)
    stop_set  = set(stop_ids or [9, 5])
    new_text  = ""

    for _ in range(max_new_tokens):
        ctx = generated[-2048:]
        inp = torch.tensor([ctx], dtype=torch.long, device=device)
        with torch.autocast(device_type=device if device != "cpu" else "cpu",
                            dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(inp)
        next_logits = logits[0, -1, :].clone().float()

        # Repetition penalty
        if repetition_penalty > 1.0:
            for tid in set(generated[-64:]):
                if 0 <= tid < next_logits.shape[0]:
                    if next_logits[tid] > 0:
                        next_logits[tid] /= repetition_penalty
                    else:
                        next_logits[tid] *= repetition_penalty

        # Sampling
        if temperature <= 0.0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            scaled = next_logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                scaled[sorted_idx[remove]] = -float("inf")
            probs   = F.softmax(scaled, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())

        if next_id in stop_set:
            break
        generated.append(next_id)

        # Stream decode one token at a time
        token_str = tok.decode([next_id])
        new_text += token_str
        if stream:
            print(c(token_str, ANSI_GREEN), end="", flush=True)

    if stream:
        print()  # newline after stream

    return new_text


HELP_TEXT = """
Commands:
  /help          Show this message
  /clear         Clear conversation history / start fresh context
  /temp N        Set temperature (e.g. /temp 0.3)
  /max N         Set max new tokens (e.g. /max 300)
  /top_p N       Set top-p (e.g. /top_p 0.95)
  /raw           Toggle raw prompt mode (append directly without formatting)
  /ctx           Show current context window (token count)
  /quit or /exit Exit the chat

Tips:
  - This is a base pretrained model. It CONTINUES text, it does not answer questions.
  - For better code completions, start with a function signature and docstring.
  - For prose, start with an opening sentence and let the model continue.
  - Temperature 0.3-0.5 → coherent and focused
  - Temperature 0.7-0.9 → creative and varied
  - Temperature 0.0     → deterministic (may loop)
"""


def print_banner(step: int, tokens_m: float, temp: float, top_p: float, max_new: int):
    print()
    print(c("╔══════════════════════════════════════════════════════╗", ANSI_CYAN))
    print(c("║      ModernGPT-S 500M  —  Interactive Chat          ║", ANSI_CYAN))
    print(c("╚══════════════════════════════════════════════════════╝", ANSI_CYAN))
    print(c(f"  Checkpoint : Step {step:,}  |  {tokens_m:.1f}M tokens trained", ANSI_DIM))
    print(c(f"  Defaults   : temp={temp}  top_p={top_p}  max_new={max_new}", ANSI_DIM))
    print(c("  Type /help for commands. Ctrl-C or /quit to exit.\n", ANSI_DIM))
    print(c("  ⚠  Base pretrained model — it continues text, not an assistant.", ANSI_YELLOW))
    print()


def main():
    parser = argparse.ArgumentParser(description="ModernGPT-S 500M Interactive Chat")
    parser.add_argument("--temp",    type=float, default=0.7,  help="Sampling temperature")
    parser.add_argument("--top_p",   type=float, default=0.9,  help="Top-p nucleus sampling")
    parser.add_argument("--max_new", type=int,   default=200,  help="Max new tokens per reply")
    parser.add_argument("--no_rep",  action="store_true",       help="Disable repetition penalty")
    parser.add_argument("--cpu",     action="store_true",       help="Force CPU inference")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    rep_penalty = 1.0 if args.no_rep else 1.15

    print(c(f"\nInitialising on {device.upper()}...\n", ANSI_CYAN))
    model, tok, cfg = load_model(device)

    ckpt_meta = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    step_num   = ckpt_meta.get("global_update", 15310)
    tokens_m   = ckpt_meta.get("total_tokens_processed", 501_678_080) / 1e6
    del ckpt_meta

    temperature = args.temp
    top_p       = args.top_p
    max_new     = args.max_new
    raw_mode    = False
    context     = ""   # running context window

    print_banner(step_num, tokens_m, temperature, top_p, max_new)

    while True:
        try:
            user_input = input(c("You › ", ANSI_BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print(c("\n\nExiting.", ANSI_DIM))
            break

        if not user_input:
            continue

        # --- Commands ---
        if user_input.startswith("/"):
            cmd_parts = user_input.split()
            cmd = cmd_parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print(c("Goodbye.", ANSI_DIM))
                break

            elif cmd == "/help":
                print(c(HELP_TEXT, ANSI_DIM))

            elif cmd == "/clear":
                context = ""
                print(c("Context cleared.", ANSI_DIM))

            elif cmd == "/raw":
                raw_mode = not raw_mode
                print(c(f"Raw mode {'ON' if raw_mode else 'OFF'}.", ANSI_DIM))

            elif cmd == "/ctx":
                n = len(tok.encode(context).ids) if context else 0
                print(c(f"Context: {n} tokens / {cfg.context_length} max.", ANSI_DIM))

            elif cmd == "/temp" and len(cmd_parts) == 2:
                try:
                    temperature = float(cmd_parts[1])
                    print(c(f"Temperature set to {temperature}", ANSI_DIM))
                except ValueError:
                    print(c("Invalid temperature value.", ANSI_RED))

            elif cmd == "/max" and len(cmd_parts) == 2:
                try:
                    max_new = int(cmd_parts[1])
                    print(c(f"Max new tokens set to {max_new}", ANSI_DIM))
                except ValueError:
                    print(c("Invalid max_new value.", ANSI_RED))

            elif cmd == "/top_p" and len(cmd_parts) == 2:
                try:
                    top_p = float(cmd_parts[1])
                    print(c(f"Top-p set to {top_p}", ANSI_DIM))
                except ValueError:
                    print(c("Invalid top_p value.", ANSI_RED))

            else:
                print(c(f"Unknown command: {cmd}. Type /help.", ANSI_RED))

            continue

        # --- Generate ---
        if raw_mode:
            prompt = context + user_input
        else:
            # Append user turn to rolling context
            context += user_input + "\n"
            prompt = context

        # Trim context to stay within window (keep last ~1800 tokens worth of text)
        ctx_ids = tok.encode(prompt).ids
        if len(ctx_ids) > 1800:
            # Trim from the front, keep latest 1800 tokens
            trimmed_ids = ctx_ids[-1800:]
            prompt = tok.decode(trimmed_ids)

        print(c("Model › ", ANSI_BOLD), end="", flush=True)
        t0 = time.perf_counter()

        response = generate(
            model=model,
            tok=tok,
            prompt=prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            device=device,
            stream=True,
        )

        elapsed = time.perf_counter() - t0
        new_tokens = len(tok.encode(response).ids)
        tps = new_tokens / elapsed if elapsed > 0 else 0
        print(c(f"  [{new_tokens} tokens, {elapsed:.1f}s, {tps:.1f} tok/s]", ANSI_DIM))
        print()

        # Append response to context
        context += response + "\n"


if __name__ == "__main__":
    main()

