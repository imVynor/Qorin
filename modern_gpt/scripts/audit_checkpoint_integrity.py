import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import get_100m_gqa4_config
from model.gpt import ModernGPT

ckpt_path = REPO_ROOT / "checkpoints/modern_gpts_100m/latest.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("=" * 60)
print("CHECKPOINT INTEGRITY AUDIT")
print("=" * 60)
print(f"File Size: {ckpt_path.stat().st_size:,} bytes")
print(f"Step: {ckpt.get('step')}")
print(f"Tokens seen: {ckpt.get('tokens_seen')}")
print(f"Shard index: {ckpt.get('shard_index')}")
print(f"Sequence offset: {ckpt.get('sequence_offset')}")
print(f"Tokenizer fingerprint: {ckpt.get('tokenizer_fingerprint')}")
print(f"Dataset manifest hash: {ckpt.get('dataset_manifest_hash')}")
print(f"Experiment ID: {ckpt.get('experiment_id')}")
print(f"Session ID: {ckpt.get('session_id')}")
print(f"Model config: {ckpt.get('model_config')}")
print(f"Scheduler state present: {'scheduler' in ckpt and ckpt['scheduler'] is not None}")
print(f"Grad scaler state present: {'grad_scaler' in ckpt and ckpt['grad_scaler'] is not None}")
print(f"Optimizer state present: {'optimizer' in ckpt and ckpt['optimizer'] is not None}")
print(f"Data sampler state: {ckpt.get('data_sampler_state')}")
print(f"RNG states present:")
for k in ['python_rng_state', 'numpy_rng_state', 'torch_cpu_rng_state', 'cuda_rng_state']:
    print(f"  {k}: {k in ckpt and ckpt[k] is not None}")

# Check parameter count
cfg = get_100m_gqa4_config()
model = ModernGPT(cfg)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=True)
print(f"\nModel load: missing={missing}, unexpected={unexpected}")
total_params = model.num_parameters()
trainable_params = model.num_parameters(only_trainable=True)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
assert total_params == 100401408, f"Expected 100,401,408, got {total_params}"

# Optimizer state check
opt_state = ckpt["optimizer"]
print(f"\nOptimizer state:")
print(f"  Param groups: {len(opt_state.get('param_groups', []))}")
print(f"  Tracked state tensors: {len(opt_state.get('state', {}))}")
for pg in opt_state.get('param_groups', []):
    print(f"  Param group lr: {pg.get('lr')}, weight_decay: {pg.get('weight_decay')}")

