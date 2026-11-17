"""Evaluation harness for matched validation loss and perplexity across context lengths.

Evaluates on the held-out validation shard (artifacts/shards_250m/val.bin).
Maintains exact token-weighted loss aggregation and supports arbitrary context lengths:
512, 1024, 2048, 4096 tokens.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.sharding import ShardedPackedDataset
from model.config import ModelConfig
from model.gpt import ModernGPT


def evaluate_val_loss(
    model: ModernGPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
    autocast_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> Tuple[float, float, int]:
    """Computes exact token-weighted cross-entropy loss and perplexity."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batches = 0

    use_autocast = (device.type == "cuda" and autocast_dtype is not None)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            num_tokens = targets.numel()

            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    _, loss = model(inputs, targets=targets)
            else:
                _, loss = model(inputs, targets=targets)

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            batches += 1
            if max_batches is not None and batches >= max_batches:
                break

    if was_training:
        model.train()

    if total_tokens == 0:
        return float("nan"), float("nan"), 0

    avg_loss = total_loss / total_tokens
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    return avg_loss, ppl, total_tokens


def run_matched_eval(
    ckpt_path: Path,
    val_bin_path: Path,
    context_lengths: Tuple[int, ...] = (512, 1024, 2048, 4096),
    max_batches_per_len: int = 50,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    device = torch.device(device_str)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = state.get("model_config", {})

    results = {}

    for ctx in context_lengths:
        scale = max(1.0, ctx / 512.0)
        cfg = ModelConfig(
            vocab_size=cfg_dict.get("vocab_size", 16000),
            context_length=ctx,
            emb_dim=cfg_dict.get("emb_dim", 768),
            n_heads=cfg_dict.get("n_heads", 12),
            n_layers=cfg_dict.get("n_layers", 14),
            ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 2048),
            dropout=0.0,
            qkv_bias=cfg_dict.get("qkv_bias", False),
            rope_theta=cfg_dict.get("rope_theta", 10000.0),
            num_kv_heads=cfg_dict.get("num_kv_heads", 4),
            rope_scale=scale,
            rope_type="linear",
        )
        model = ModernGPT(cfg).to(device)
        model.load_state_dict(state["model"], strict=True)

        batch_size = max(1, 4096 // ctx)
        dataset = ShardedPackedDataset(str(val_bin_path), context_length=ctx)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        val_loss, ppl, tokens_evaled = evaluate_val_loss(
            model=model,
            loader=loader,
            device=device,
            max_batches=max_batches_per_len,
            autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
        )

        results[f"ctx_{ctx}"] = {
            "context_length": ctx,
            "rope_scale": scale,
            "batch_size": batch_size,
            "val_loss": round(val_loss, 4),
            "perplexity": round(ppl, 3),
            "tokens_evaluated": tokens_evaled,
            "total_dataset_sequences": len(dataset),
        }

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate validation loss across context lengths")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/modern_gpts_100m/best_val.pt")
    parser.add_argument("--val-bin", type=str, default="artifacts/shards_250m/val.bin")
    parser.add_argument("--max-batches", type=int, default=10)
    args = parser.parse_args()

    ckpt = REPO_ROOT / args.checkpoint
    val = REPO_ROOT / args.val_bin
    res = run_matched_eval(ckpt, val, max_batches_per_len=args.max_batches)
    print(json.dumps(res, indent=2))

