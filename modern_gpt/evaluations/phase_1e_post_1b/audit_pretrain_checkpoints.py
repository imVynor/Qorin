import hashlib
import json
import os
import sys
from pathlib import Path
import torch

repo_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from modern_gpt.model import ModernGPT, ModelConfig

ckpts = {
    "ckpt_501m": repo_root / "modern_gpt/checkpoints/checkpoint_501M/ckpt_501m_tokens_501678080.pt",
    "ckpt_best_val": repo_root / "modern_gpt/checkpoints/500m_pretrain/ckpt_best_val.pt",
    "ckpt_latest": repo_root / "modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt",
}

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

print("=" * 80)
print("MODERNGPT-S 500M PRETRAIN CHECKPOINT AUDIT")
print("=" * 80)

audit_results = {}

for name, path in ckpts.items():
    print(f"\nInspecting {name}: {path.name}")
    if not path.exists():
        print(f"  FILE NOT FOUND: {path}")
        continue
    
    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"  File size: {file_size_mb:.2f} MB ({file_size_bytes:,} bytes)")
    
    print("  Computing SHA-256 (this may take a few seconds)...")
    sha256_hash = compute_sha256(path)
    print(f"  SHA-256: {sha256_hash}")
    
    print("  Loading checkpoint dictionary (CPU)...")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    
    meta_keys = [k for k in ckpt.keys() if k != "model_state_dict" and k != "optimizer_state_dict"]
    print(f"  Metadata keys: {meta_keys}")
    
    global_update = ckpt.get("global_update")
    tokens_processed = ckpt.get("total_tokens_processed")
    best_val_loss = ckpt.get("best_val_loss")
    config = ckpt.get("config", {})
    cycle_count = ckpt.get("cycle_count")
    train_sec = ckpt.get("cumulative_cycle_train_sec")
    tok_fingerprint = ckpt.get("tokenizer_fingerprint")
    timestamp = ckpt.get("timestamp")
    
    print(f"  Global Update (Step): {global_update}")
    print(f"  Total Tokens Processed: {tokens_processed:,}" if tokens_processed is not None else "  Total Tokens Processed: None")
    print(f"  Reported Best Val Loss: {best_val_loss}")
    print(f"  Cycle Count: {cycle_count}")
    print(f"  Cumulative Train Seconds: {train_sec}")
    print(f"  Tokenizer Fingerprint: {tok_fingerprint}")
    print(f"  Timestamp: {timestamp}")
    
    # Audit model state dict
    sd = ckpt["model_state_dict"]
    num_tensors = len(sd)
    total_elements = sum(p.numel() for p in sd.values())
    
    # Check for NaNs and Infs
    has_nan = False
    has_inf = False
    for k, v in sd.items():
        if torch.isnan(v).any():
            has_nan = True
            print(f"    WARNING: NaN detected in tensor {k}")
        if torch.isinf(v).any():
            has_inf = True
            print(f"    WARNING: Inf detected in tensor {k}")
            
    print(f"  Tensor count: {num_tensors}")
    print(f"  Total parameters in state_dict: {total_elements:,}")
    print(f"  NaN detected: {has_nan}, Inf detected: {has_inf}")
    
    # Test instantiation against config
    print("  Testing model instantiation against ModelConfig.get_500m_config()...")
    try:
        cfg = ModelConfig.get_500m_config(gradient_checkpointing=False)
        model = ModernGPT(cfg)
        
        # Load state dict strictly
        load_result = model.load_state_dict(sd, strict=True)
        print(f"  Strict state_dict load successful! Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")
        
        model_param_count = sum(p.numel() for p in model.parameters())
        emb_tied = (model.token_embedding.weight.data_ptr() == model.lm_head.weight.data_ptr())
        print(f"  Instantiated model parameters: {model_param_count:,}")
        print(f"  Weight tying (token_embedding == lm_head): {emb_tied}")
    except Exception as e:
        print(f"  Error instantiating model: {e}")
    except Exception as e:
        print(f"  Error instantiating model: {e}")
        
    audit_results[name] = {
        "file_name": path.name,
        "path": str(path.relative_to(repo_root)),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "sha256": sha256_hash,
        "global_update": global_update,
        "total_tokens_processed": tokens_processed,
        "best_val_loss": best_val_loss,
        "cycle_count": cycle_count,
        "cumulative_train_sec": train_sec,
        "tokenizer_fingerprint": tok_fingerprint,
        "timestamp": timestamp,
        "tensor_count": num_tensors,
        "parameter_count": total_elements,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "config": config,
    }

output_json = repo_root / "modern_gpt/evaluations/phase_1e_post_1b/checkpoint_audit/pretrain_checkpoint_audit.json"
output_json.parent.mkdir(parents=True, exist_ok=True)
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)

print("\nAudit saved to:", output_json)
