import torch
from pathlib import Path

ckpt_path = Path("modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt")
state = torch.load(ckpt_path, map_location="cpu")

print("=== CHECKPOINT TOP-LEVEL KEYS ===")
for k in state.keys():
    val = state[k]
    val_type = type(val).__name__
    if isinstance(val, dict):
        print(f"  {k} (dict, len={len(val)})")
    elif isinstance(val, (int, float, str, bool)):
        print(f"  {k} ({val_type}) = {val}")
    else:
        print(f"  {k} ({val_type})")

print("\n=== STORED CONFIGURATION ===")
cfg = state.get("config", {})
if isinstance(cfg, dict):
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
else:
    print(f"  config object: {cfg}")

print("\n=== STORED METADATA ===")
meta = state.get("metadata", {})
if isinstance(meta, dict):
    for k, v in sorted(meta.items()):
        print(f"  {k}: {v}")
else:
    print(f"  metadata object: {meta}")

# State dict inspection
sd = state.get("model_state_dict", state.get("model", state))
print(f"\nTotal tensors in state_dict: {len(sd)}")

layer_indices = set()
tensor_details = []
total_params = 0

for name, tensor in sd.items():
    shape = list(tensor.shape)
    numel = tensor.numel()
    total_params += numel
    tensor_details.append((name, shape, numel, str(tensor.dtype)))
    
    # Check layer index
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i+1].isdigit():
            layer_indices.add(int(parts[i+1]))

print(f"Detected Layer Indices: {sorted(list(layer_indices))}")
print(f"Number of transformer layers: {len(layer_indices)}")
print(f"Total calculated parameters (sum of tensor.numel()): {total_params:,}")

print("\n=== KEY TENSOR SHAPES ===")
key_patterns = [
    "tok_embeddings", "wte", "embed",
    "layers.0.", "layers.1.",
    "norm", "ln_f", "output", "lm_head"
]

for name, shape, numel, dtype in tensor_details:
    # Print embeddings, first layer tensors, and final norm/output
    if any(p in name for p in ["tok_embeddings", "wte", "embed", "layers.0.", "norm", "lm_head", "output"]):
        print(f"  {name:50s} : shape={shape!s:25s} params={numel:12,d} dtype={dtype}")
