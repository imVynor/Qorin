import sys
import time
import gc
import json
import math
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model.config import ModelConfig
from model.gpt import ModernGPT
from scripts.monitor_gpu import GPUMonitor

assert torch.cuda.is_available(), "CUDA required for GPU validation"
device = torch.device("cuda:0")

monitor = GPUMonitor()
poll_res = monitor.poll_once()
temp_start = poll_res.get("gpu_temp_c", 0.0)
total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print("=== GPU BENCHMARK & BACKWARD VALIDATION ===")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {total_vram_gb:.2f} GB")
print(f"Initial Temperature: {temp_start}°C")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")

configs = [
    {"context": 512, "batch": 8, "rope_scale": 1.0},
    {"context": 1024, "batch": 4, "rope_scale": 2.0},
    {"context": 2048, "batch": 2, "rope_scale": 4.0},
    {"context": 4096, "batch": 1, "rope_scale": 8.0},
]

results = []

for cfg_item in configs:
    ctx = cfg_item["context"]
    b = cfg_item["batch"]
    s = cfg_item["rope_scale"]
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    cfg = ModelConfig(
        vocab_size=16000,
        context_length=ctx,
        emb_dim=768,
        n_heads=12,
        num_kv_heads=4,
        n_layers=14,
        ffn_hidden_dim=2048,
        dropout=0.0,
        qkv_bias=False,
        rope_theta=10000.0,
        rope_scale=s,
        rope_type="linear",
    )
    model = ModernGPT(cfg).to(device=device, dtype=torch.bfloat16)
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    
    inputs = torch.randint(0, 16000, (b, ctx), device=device)
    targets = torch.randint(0, 16000, (b, ctx), device=device)
    
    # Warmup step
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(inputs, targets=targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    # Timed & measured step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits, loss = model(inputs, targets=targets)
    
    loss.backward()
    
    grads_finite = True
    for p in model.parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():
                grads_finite = False
                break
                
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    
    step_time_ms = (t1 - t0) * 1000.0
    allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    poll_now = monitor.poll_once()
    temp_now = poll_now.get("gpu_temp_c", 0.0)
    loss_val = loss.item()
    loss_finite = math.isfinite(loss_val)
    
    res = {
        "context": ctx,
        "batch_size": b,
        "rope_scale": s,
        "tokens_per_batch": ctx * b,
        "step_time_ms": round(step_time_ms, 2),
        "allocated_mb": round(allocated_mb, 2),
        "allocated_gb": round(allocated_mb / 1024, 3),
        "reserved_mb": round(reserved_mb, 2),
        "reserved_gb": round(reserved_mb / 1024, 3),
        "vram_utilization_pct": round((reserved_mb / 1024) / total_vram_gb * 100, 1),
        "loss": round(loss_val, 4),
        "loss_finite": loss_finite,
        "grads_finite": grads_finite,
        "temp_c": temp_now,
    }
    results.append(res)
    print(f"Ctx {ctx:4d} | Batch {b:2d} | Step {step_time_ms:6.1f}ms | Alloc {res['allocated_gb']:4.2f}GB | Res {res['reserved_gb']:4.2f}GB ({res['vram_utilization_pct']:4.1f}%) | Grads finite: {grads_finite} | Temp {temp_now}C")
    
    del model, optimizer, inputs, targets, logits, loss
    gc.collect()
    torch.cuda.empty_cache()

out_path = REPO_ROOT / "artifacts/diagnostics/context_4k_gpu_benchmark.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nBenchmark saved to:", out_path)
print(json.dumps(results, indent=2))
