"""Interactive chat interface for ModernGPT-S 100M GQA-4 (4K SFT & Pretrained).

Supports:
- ChatML conversational format (<|im_start|>role\ncontent<|im_end|>)
- Multi-turn stateful conversational memory
- Repetition penalty & nucleus (top-p / top-k) sampling
- Single-shot command line prompts
- Interactive REPL with /clear, /system, /history, and /temp commands

Usage:
    # Interactive chat session:
    python chat.py

    # Single-shot query:
    python chat.py --prompt "Write a Python function to reverse a string."
    python chat.py --prompt "What is the capital of Italy?"

    # Custom temperature / max tokens:
    python chat.py --temp 0.5 --max-tokens 150
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F

# Ensure modern_gpt is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from model.config import ModelConfig
from model.gpt import ModernGPT


def resolve_path(rel_or_abs: str) -> str:
    p = Path(rel_or_abs)
    if p.exists():
        return str(p)
    here = Path(__file__).resolve().parent
    if (here / rel_or_abs).exists():
        return str(here / rel_or_abs)
    root = here.parent
    if (root / rel_or_abs).exists():
        return str(root / rel_or_abs)
    return rel_or_abs


def default_checkpoint_path() -> str:
    candidates = [
        "checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt",
        "checkpoints/modern_gpts_100m_sft_4k_v3/checkpoint_epoch_3.pt",
        "checkpoints/modern_gpts_100m_sft_4k_v2/best_val.pt",
        "checkpoints/modern_gpts_100m_sft_4k_v2/checkpoint_epoch_3.pt",
        "checkpoints/modern_gpts_100m_sft_4k/best_val.pt",
        "checkpoints/modern_gpts_100m_sft_4k/checkpoint_epoch_3.pt",
        "checkpoints/modern_gpts_100m_4k_pilot/checkpoint_step_2000.pt",
        "checkpoints/modern_gpts_100m/best_val.pt",
    ]
    for c in candidates:
        resolved = resolve_path(c)
        if Path(resolved).exists():
            return resolved
    return resolve_path(candidates[0])


def load_model_and_tokenizer(
    checkpoint_path: str = None,
    tokenizer_path: str = "artifacts/tokenizer_16k_v2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    if checkpoint_path is None:
        checkpoint_path = default_checkpoint_path()
    else:
        checkpoint_path = resolve_path(checkpoint_path)

    tokenizer_path = resolve_path(tokenizer_path)

    print(f"[*] Loading tokenizer from {tokenizer_path}...")
    tok = HuggingFaceBPEWrapper.load(tokenizer_path)

    print(f"[*] Loading checkpoint from {checkpoint_path} onto {device}...")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = state.get("model_config", {})

    emb_dim = cfg_dict.get("emb_dim", 768)
    n_heads = cfg_dict.get("n_heads", 12)
    n_layers = cfg_dict.get("n_layers", 14)
    ffn_hidden_dim = cfg_dict.get("ffn_hidden_dim", 2048)
    num_kv_heads = cfg_dict.get("num_kv_heads", cfg_dict.get("kv_heads", 4))
    vocab_size = cfg_dict.get("vocab_size", 16000)
    context_length = cfg_dict.get("context_length", 4096)
    qkv_bias = cfg_dict.get("qkv_bias", False)
    rope_theta = cfg_dict.get("rope_theta", 10000.0)
    rope_scale = cfg_dict.get("rope_scale", 8.0 if context_length > 512 else 1.0)
    rope_type = cfg_dict.get("rope_type", "linear")

    arch = f"{emb_dim}d"
    if emb_dim == 768:
        arch = "100M-GQA4"
    elif emb_dim == 512:
        arch = "50M"

    config = ModelConfig(
        vocab_size=vocab_size,
        context_length=context_length,
        emb_dim=emb_dim,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_hidden_dim=ffn_hidden_dim,
        dropout=0.0,
        qkv_bias=qkv_bias,
        rope_theta=rope_theta,
        num_kv_heads=num_kv_heads,
        rope_scale=rope_scale,
        rope_type=rope_type,
    )

    model = ModernGPT(config).to(device=device, dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    ckpt_type = "SFT Fine-Tuned (4K Context)" if "sft" in checkpoint_path.lower() else "Pretrained Base"
    print(f"[+] Successfully loaded {arch} {ckpt_type} ({total_params:,} parameters, {context_length} context length).")
    return model, tok, config, device, ckpt_type, arch


@torch.no_grad()
def generate_response(
    model: ModernGPT,
    tok: HuggingFaceBPEWrapper,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.20,
    context_length: int = 4096,
    device: str = "cuda",
) -> str:
    # Build ChatML prompt
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"

    input_ids = tok.encode(prompt)
    if not input_ids:
        input_ids = [tok.bos_token_id or 1]

    # Enforce maximum prompt window budget
    if len(input_ids) > (context_length - max_new_tokens):
        input_ids = input_ids[-(context_length - max_new_tokens):]

    inp = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated_tokens: List[int] = []
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else 0

    use_autocast = device.startswith("cuda")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else torch.no_grad()

    for _ in range(max_new_tokens):
        with autocast_ctx:
            logits = model(inp[:, -context_length:])[:, -1, :].clone()

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
            logits = logits / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")

            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        token_id = next_token.item()
        if token_id == eos_id:
            break

        generated_tokens.append(token_id)
        inp = torch.cat([inp, next_token], dim=1)

        # Early break if ChatML end delimiter is emitted
        decoded_so_far = tok.decode(generated_tokens)
        if "<|im_end|>" in decoded_so_far:
            break

    response = tok.decode(generated_tokens).replace("<|im_end|>", "").strip()
    return response


def interactive_session(
    model: ModernGPT,
    tok: HuggingFaceBPEWrapper,
    config: ModelConfig,
    device: str,
    ckpt_type: str,
    arch: str,
    args: argparse.Namespace,
):
    system_prompt = args.system
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]

    print("=" * 70)
    print(f" ModernGPT-S 100M Interactive Chat ({arch} - 4,096 Tokens)")
    print("=" * 70)
    print(f"System: {system_prompt}")
    print(f"Params: Temperature={args.temp}, Top-P={args.top_p}, Repetition Penalty={args.rep_penalty}")
    print("Commands:")
    print("  /clear or /reset : Clear dialogue history")
    print("  /system <prompt> : Update system instructions")
    print("  /temp <float>    : Change sampling temperature")
    print("  /history         : View current turn history")
    print("  exit, quit, q    : Exit chat session")
    print("-" * 70)

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Goodbye!")
            break

        if not user_input:
            continue

        lower = user_input.lower()
        if lower in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break
        elif lower in ("/clear", "/reset"):
            messages = [{"role": "system", "content": system_prompt}]
            print("[+] Conversation memory cleared.")
            continue
        elif lower == "/history":
            print(f"\n--- History ({len(messages)} messages) ---")
            for m in messages:
                print(f"[{m['role'].upper()}]: {m['content']}")
            print("--- End History ---")
            continue
        elif lower.startswith("/system "):
            system_prompt = user_input[8:].strip()
            messages[0] = {"role": "system", "content": system_prompt}
            print(f"[+] System prompt updated to: {system_prompt}")
            continue
        elif lower.startswith("/temp "):
            try:
                args.temp = float(user_input[6:].strip())
                print(f"[+] Temperature set to: {args.temp}")
            except ValueError:
                print("[-] Invalid temperature value.")
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate response
        response = generate_response(
            model=model,
            tok=tok,
            messages=messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            context_length=config.context_length,
            device=device,
        )

        messages.append({"role": "assistant", "content": response})
        print(f"\n[ModernGPT-S]\n{response}")


def main():
    parser = argparse.ArgumentParser(description="Chat with ModernGPT-S 100M")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (defaults to best_val.pt in modern_gpts_100m_sft_4k)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="artifacts/tokenizer_16k_v2",
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single-shot prompt to execute",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are ModernGPT-S, a helpful and precise AI assistant.",
        help="System prompt instruction",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6, lower for precision, higher for creativity)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-K sampling cutoff (default: 50)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.90,
        help="Top-P nucleus sampling cutoff (default: 0.90)",
    )
    parser.add_argument(
        "--rep-penalty",
        type=float,
        default=1.20,
        help="Repetition penalty (default: 1.20 to prevent repetition loops)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=160,
        help="Maximum new tokens to generate (default: 160)",
    )
    args = parser.parse_args()

    model, tok, config, device, ckpt_type, arch = load_model_and_tokenizer(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    if args.prompt:
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]
        response = generate_response(
            model=model,
            tok=tok,
            messages=messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temp,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            context_length=config.context_length,
            device=device,
        )
        print("\n" + "=" * 60)
        print("Prompt:", args.prompt)
        print("=" * 60)
        print(f"ModernGPT-S Response:\n{response}")
        print("=" * 60)
    else:
        interactive_session(model, tok, config, device, ckpt_type, arch, args)


if __name__ == "__main__":
    main()
