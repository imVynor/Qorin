"""GPU smoke test and throughput/VRAM benchmarking for ModernGPT-50M architecture."""

import gc
import json
from pathlib import Path
import sys
import time
import torch

sys.path.insert(0, ".")
from model.config import ModelConfig, MODERN_GPT_50M, MODERN_GPT_50M_WIDE
from model.gpt import ModernGPT


def benchmark_configuration(
    config: ModelConfig,
    config_name: str,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
    batch_size: int = 16,
    num_steps: int = 50,
    warmup_steps: int = 10,
):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    model = ModernGPT(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

    # Synthetic training data: (batch_size, context_length)
    vocab_size = config.vocab_size
    context_length = config.context_length
    tokens_per_step = batch_size * context_length

    # Warmup
    for _ in range(warmup_steps):
        inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    # Timed steps
    for _ in range(num_steps):
        inputs = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            _, loss = model(inputs, targets)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    total_tokens = tokens_per_step * num_steps
    tokens_per_sec = total_tokens / elapsed
    ms_per_step = (elapsed / num_steps) * 1000.0

    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    # Clean up
    del model, optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # Estimates for 250M and 500M tokens
    sec_250m = 250_000_000 / tokens_per_sec
    sec_500m = 500_000_000 / tokens_per_sec
    hours_250m = sec_250m / 3600.0
    hours_500m = sec_500m / 3600.0

    return {
        "config_name": config_name,
        "parameters": config.parameter_count if hasattr(config, "parameter_count") else None,
        "dtype": dtype_name,
        "batch_size": batch_size,
        "context_length": context_length,
        "tokens_per_step": tokens_per_step,
        "tokens_per_second": round(tokens_per_sec, 1),
        "ms_per_step": round(ms_per_step, 2),
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "estimated_hours_250m": round(hours_250m, 2),
        "estimated_hours_500m": round(hours_500m, 2),
    }


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Smoke test aborted.")
        return

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    print(f"Benchmarking on: {gpu_name} (Total VRAM: {total_vram_mb:.0f} MB)")

    configs = [
        ("ModernGPT-50M (12 layers, 512 dim)", MODERN_GPT_50M),
        ("ModernGPT-50M-Wide (8 layers, 640 dim)", MODERN_GPT_50M_WIDE),
    ]

    results = []
    # Test batch sizes and precisions
    for cfg_name, cfg in configs:
        m = ModernGPT(cfg)
        exact_params = m.parameter_count
        del m

        for dtype, dname in [(torch.bfloat16, "bfloat16"), (torch.float16, "float16")]:
            for bs in [8, 16, 32]:
                print(f"Running {cfg_name} | {dname} | batch_size={bs}...")
                try:
                    res = benchmark_configuration(
                        cfg, cfg_name, device, dtype, dname, batch_size=bs, num_steps=50, warmup_steps=10
                    )
                    res["exact_parameters"] = exact_params
                    results.append(res)
                    print(f"  -> {res['tokens_per_second']:,.0f} tokens/s | Peak VRAM: {res['peak_allocated_mb']:.1f} MB | 250M: {res['estimated_hours_250m']:.2f}h | 500M: {res['estimated_hours_500m']:.2f}h")
                except torch.cuda.OutOfMemoryError:
                    print(f"  -> OOM at batch_size={bs}")
                    torch.cuda.empty_cache()

    out_path = Path("artifacts/diagnostics/benchmark_50m_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "gpu": gpu_name,
            "total_vram_mb": round(total_vram_mb, 1),
            "benchmarks": results,
        }, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")


if __name__ == "__main__":
    main()

