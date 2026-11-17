import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import get_100m_gqa4_config
from model.gpt import ModernGPT
from scripts.pretrain_100m import ShardTokenDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 1. Load latest checkpoint
ckpt_path = Path("checkpoints/modern_gpts_100m/latest.pt")
print(f"Checking checkpoint: {ckpt_path} (exists={ckpt_path.exists()})")

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Checkpoint keys: {list(ckpt.keys())}")
print(f"Step: {ckpt.get('step')}, tokens_seen: {ckpt.get('tokens_seen')}")
print(f"Dataset manifest hash in ckpt: {ckpt.get('dataset_manifest_hash')}")
print(f"Model config in ckpt: {ckpt.get('model_config')}")

# 2. Recreate model and load state
cfg = get_100m_gqa4_config()
model = ModernGPT(cfg)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
print(f"Model loaded with strict=True. Missing: {missing}, Unexpected: {unexpected}")

# 3. Test on a fixed batch from Shard 0
shard_path = Path("artifacts/modern_gpts_100m/shards/shard_00000.bin")
ds = ShardTokenDataset(shard_path, context_length=512, start_offset=0)
inputs, targets = ds[0]
inputs = inputs.unsqueeze(0).to(device) # shape (1, 512)
targets = targets.unsqueeze(0).to(device)

print(f"Inputs shape: {inputs.shape}, dtype: {inputs.dtype}, min: {inputs.min().item()}, max: {inputs.max().item()}")
print(f"Targets shape: {targets.shape}, dtype: {targets.dtype}, min: {targets.min().item()}, max: {targets.max().item()}")

# Check shift
print(f"First 10 inputs:  {inputs[0, :10].tolist()}")
print(f"First 10 targets: {targets[0, :10].tolist()}")
# In ds[0], chunk was tokens[0 : 513]. inputs = chunk[:-1], targets = chunk[1:]
# So targets[t] == inputs[t+1]!
shift_ok = torch.equal(inputs[0, 1:10], targets[0, :9])
print(f"Target is exactly next-token shifted inputs: {shift_ok}")

# 4. Forward pass in FP32
model = model.to(device).float()
model.eval()

with torch.no_grad():
    logits_fp32 = model(inputs)
    print(f"\nFP32 Logits shape: {logits_fp32.shape}")
    print(f"Logits min: {logits_fp32.min().item():.4f}, max: {logits_fp32.max().item():.4f}")
    print(f"Logits mean: {logits_fp32.mean().item():.4f}, std: {logits_fp32.std().item():.4f}")
    print(f"Non-finite logits: {torch.isnan(logits_fp32).sum().item()} NaNs, {torch.isinf(logits_fp32).sum().item()} Infs")

    # Loss with reduction='mean'
    loss_mean = F.cross_entropy(logits_fp32.view(-1, cfg.vocab_size), targets.view(-1), reduction="mean")
    # Loss with reduction='sum'
    loss_sum = F.cross_entropy(logits_fp32.view(-1, cfg.vocab_size), targets.view(-1), reduction="sum")
    # Normalized by tokens
    loss_norm = loss_sum / targets.numel()

    print(f"Cross Entropy (reduction='mean'): {loss_mean.item():.6f}")
    print(f"Cross Entropy (reduction='sum'):  {loss_sum.item():.6f}")
    print(f"Loss sum / numel:                 {loss_norm.item():.6f}")

    # Forward pass in BF16
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits_bf16 = model(inputs)
        loss_bf16 = F.cross_entropy(logits_bf16.view(-1, cfg.vocab_size), targets.view(-1))
        print(f"BF16 Cross Entropy:               {loss_bf16.item():.6f}")

    # What about an untrained fresh model?
    fresh_model = ModernGPT(cfg).to(device).float()
    fresh_model.eval()
    fresh_logits = fresh_model(inputs)
    fresh_loss = F.cross_entropy(fresh_logits.view(-1, cfg.vocab_size), targets.view(-1))
    print(f"\nFresh randomly initialized model loss: {fresh_loss.item():.6f}")
    print(f"Theoretical uniform random loss ln(16384): {math.log(16384):.6f}")
    print(f"Fresh model logits std: {fresh_logits.std().item():.4f}, mean: {fresh_logits.mean().item():.4f}")
    print(f"Fresh model logits min: {fresh_logits.min().item():.4f}, max: {fresh_logits.max().item():.4f}")
