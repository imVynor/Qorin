"""Evaluate pretraining validation loss on artifacts/shards_250m/val.bin across checkpoints.
"""
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from data.sharding import ShardedPackedDataset
from model.config import ModelConfig
from model.gpt import ModernGPT
from evaluate import validation_loss

def evaluate_checkpoint_loss(ckpt_path: Path, val_loader: DataLoader, device: torch.device, max_batches: int = 200):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_dict = state.get("model_config", {})
    config = ModelConfig(
        vocab_size=state.get("vocab_size", 16000),
        context_length=cfg_dict.get("context_length", 512),
        emb_dim=cfg_dict.get("emb_dim", 512),
        n_heads=cfg_dict.get("n_heads", 8),
        n_layers=cfg_dict.get("n_layers", 12),
        ffn_hidden_dim=cfg_dict.get("ffn_hidden_dim", 1536),
        dropout=0.0,
        qkv_bias=cfg_dict.get("qkv_bias", False),
        rope_theta=cfg_dict.get("rope_theta", 10000.0),
    )
    model = ModernGPT(config).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    val_loss, ppl = validation_loss(
        model=model,
        loader=val_loader,
        device=device,
        max_batches=max_batches,
        autocast_dtype=torch.bfloat16 if device.type == "cuda" else None,
    )
    return val_loss, ppl

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_shard = ShardedPackedDataset("artifacts/shards_250m/val.bin", context_length=512)
    val_loader = DataLoader(val_shard, batch_size=8, shuffle=False)

    ckpts = [
        ("50M Pretrained Base (250M tokens)", Path("checkpoints/pretrain_50m_250m/latest.pt")),
        ("50M SFT v1 (Midscale 500)", Path("checkpoints/sft_50m_midscale_500/latest.pt")),
        ("50M SFT v2 (Best Val, Step 2400)", Path("checkpoints/sft_50m_targeted_v2/best_val.pt")),
        ("50M SFT v2 (Latest, Step 3225)", Path("checkpoints/sft_50m_targeted_v2/latest.pt")),
    ]

    print("=" * 70)
    print("GENERAL PRETRAINING VALIDATION LOSS RETENTION (shards_250m/val.bin, 200 batches)")
    print("=" * 70)

    results = {}
    for name, path in ckpts:
        if not path.exists():
            print(f"Skipping {name}: {path} not found")
            continue
        loss, ppl = evaluate_checkpoint_loss(path, val_loader, device, max_batches=200)
        results[name] = {"loss": round(loss, 4), "perplexity": round(ppl, 2)}
        print(f"{name:35s} | Val Loss: {loss:.4f} | Perplexity: {ppl:.2f}")
    print("=" * 70)
