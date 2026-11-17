import glob
import os
import sys
from pathlib import Path
import torch

repo_root = Path(".").resolve()
checkpoints_dir = repo_root / "modern_gpt" / "checkpoints"

print(f"Scanning {checkpoints_dir} for checkpoints...")
pt_files = sorted(glob.glob(str(checkpoints_dir / "**" / "*.pt"), recursive=True))

target_keywords = ["500m", "501m", "1b", "latest", "best_val"]

found = []
for f in pt_files:
    f_path = Path(f)
    if any(k in f.lower() for k in target_keywords):
        size_mb = f_path.stat().st_size / (1024 * 1024)
        mtime = f_path.stat().st_mtime
        found.append((f_path, size_mb, mtime))

print(f"Found {len(found)} candidate checkpoints:\n")
for f_path, size_mb, mtime in found:
    rel_path = f_path.relative_to(repo_root)
    print(f"- {rel_path} ({size_mb:.2f} MB)")
    # Peek metadata without loading full state dict if possible
    try:
        # Load weights_only=False or inspect dict keys
        ckpt = torch.load(f_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            step = ckpt.get("step") or ckpt.get("optimizer_step")
            tokens = ckpt.get("tokens_seen") or ckpt.get("total_tokens") or ckpt.get("tokens")
            val_loss = ckpt.get("val_loss") or ckpt.get("best_val_loss")
            config = ckpt.get("config") or ckpt.get("model_config")
            print(f"    keys: {keys}")
            print(f"    step: {step}, tokens_seen: {tokens}, val_loss: {val_loss}")
            if config:
                print(f"    config type: {type(config)}")
        else:
            print(f"    Type: {type(ckpt)}")
    except Exception as e:
        print(f"    Error reading checkpoint: {e}")
    print()
