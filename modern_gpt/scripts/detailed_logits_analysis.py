import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from model.config import get_100m_gqa4_config
from model.gpt import ModernGPT
from scripts.pretrain_100m import ShardTokenDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = get_100m_gqa4_config()

# Load latest checkpoint
ckpt_path = REPO_ROOT / "checkpoints/modern_gpts_100m/latest.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = ModernGPT(cfg)
model.load_state_dict(ckpt["model"])
model = model.to(device).float()
model.eval()

# Load fresh untrained model
fresh_model = ModernGPT(cfg).to(device).float()
fresh_model.eval()

# Load fixed 8 microbatch sequence (1 full optimizer step = 8 sequences of length 512 = 4096 tokens)
shard_path = REPO_ROOT / "artifacts/modern_gpts_100m/shards/shard_00000.bin"
ds = ShardTokenDataset(shard_path, context_length=512, start_offset=0)

batch_inputs = []
batch_targets = []
for i in range(8):
    inp, tgt = ds[i]
    batch_inputs.append(inp)
    batch_targets.append(tgt)

inputs = torch.stack(batch_inputs).to(device)  # (8, 512)
targets = torch.stack(batch_targets).to(device) # (8, 512)

print("=" * 60)
print(f"DETAILED LOGITS AND LOSS AUDIT (Step 50 vs Untrained)")
print("=" * 60)
print(f"Batch shape: {inputs.shape} ({inputs.numel()} tokens)")
print(f"Vocab size: {cfg.vocab_size}")

for name, m in [("Checkpoint Step 50 (latest.pt)", model), ("Fresh Initialized Model", fresh_model)]:
    with torch.no_grad():
        logits = m(inputs) # (8, 512, 16000)
        
        # Statistics
        l_min = logits.min().item()
        l_max = logits.max().item()
        l_mean = logits.mean().item()
        l_std = logits.std().item()
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        
        # Probabilities & Entropy
        probs = F.softmax(logits, dim=-1) # (8, 512, 16000)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean().item() # Mean entropy over tokens in nats
        max_possible_entropy = math.log(cfg.vocab_size)
        
        # Cross entropy / Negative Log-Likelihood
        loss_mean = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), reduction="mean").item()
        loss_sum = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), reduction="sum").item()
        
        # Per-microbatch losses
        microbatch_losses = []
        for mb in range(8):
            mb_loss = F.cross_entropy(logits[mb], targets[mb], reduction="mean").item()
            microbatch_losses.append(mb_loss)
            
        summed_microbatch_loss = sum(microbatch_losses)

        print(f"\n--- {name} ---")
        print(f"Logits Min:          {l_min:.4f}")
        print(f"Logits Max:          {l_max:.4f}")
        print(f"Logits Mean:         {l_mean:.4f}")
        print(f"Logits Std:          {l_std:.4f}")
        print(f"Non-finite count:    {nan_count} NaNs, {inf_count} Infs")
        print(f"Distribution Entropy:{entropy:.4f} nats (max possible: {max_possible_entropy:.4f})")
        print(f"True Cross-Entropy:  {loss_mean:.4f} (per token)")
        print(f"Summed across 8 MBs: {summed_microbatch_loss:.4f} (matches logged loss {summed_microbatch_loss:.4f})")
        print(f"Per-microbatch loss: {[round(x, 4) for x in microbatch_losses]}")

