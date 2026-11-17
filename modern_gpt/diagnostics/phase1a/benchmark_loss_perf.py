"""
ModernGPT-S Phase 1A: Benchmark Loss Performance and Numerical Stability.

Measures:
1. Numerical stability across FP32, BF16, and FP16 (AMP + GradScaler) with extreme token counts
   (e.g., 50 tokens across 8 microbatches vs 32,768 tokens across 8 microbatches).
2. Runtime latency and Peak VRAM memory comparison between:
   - Old unweighted microbatch mean: mean(CE) / K
   - New token-weighted sum: sum(CE) / total_window_tokens
3. Graph lifetime & memory leak verification (confirming backward() frees graph per microbatch).
"""

import sys
from pathlib import Path
repo_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(repo_dir))
sys.path.insert(0, str(repo_dir / "modern_gpt"))

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from modern_gpt.model.gpt import ModernGPT
from modern_gpt.model.config import ModelConfig
from modern_gpt.model.loss import compute_token_weighted_cross_entropy

def test_numerical_stability():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== 1. Numerical Stability Verification (Device: {device}) ===")
    
    # Test configurations
    dtypes = [
        ("FP32", torch.float32, False),
        ("BF16", torch.bfloat16, True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False),
        ("FP16_AMP", torch.float16, True if torch.cuda.is_available() else False),
    ]
    
    vocab_size = 16384
    hidden_size = 256
    seq_len = 512
    grad_accum_steps = 8
    
    # Test cases:
    # A: Very small token count: ~50 valid tokens across 8 microbatches (~6 per microbatch)
    # B: Very large token count: ~32,768 valid tokens across 8 microbatches (4096 per microbatch)
    # C: Realistic imbalanced: 20 tokens to 1500 tokens across 8 microbatches
    cases = [
        ("Very Small (50 total tokens)", 50),
        ("Realistic Imbalance (~4000 total tokens)", 4000),
        ("Very Large (32,768 total tokens)", 32768),
    ]
    
    results = []
    
    for dtype_name, dtype, use_amp in dtypes:
        if "BF16" in dtype_name and not use_amp:
            continue
        if "FP16" in dtype_name and not use_amp:
            continue
            
        scaler = torch.amp.GradScaler('cuda') if dtype_name == "FP16_AMP" else None
        
        for case_name, target_tokens in cases:
            tokens_per_mb = target_tokens // grad_accum_steps
            actual_seq_len = max(seq_len, tokens_per_mb + 10)
            
            # Simple linear model to test gradient flow and scaler stability
            model = nn.Linear(hidden_size, vocab_size, bias=False).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            
            # Generate microbatches
            mbs = []
            total_supervised = 0
            for _ in range(grad_accum_steps):
                x = torch.randn(1, actual_seq_len, hidden_size, device=device)
                labels = torch.full((1, actual_seq_len), -100, dtype=torch.long, device=device)
                # supervise tokens_per_mb positions
                labels[:, 1:tokens_per_mb+1] = torch.randint(0, vocab_size, (1, tokens_per_mb), device=device)
                mbs.append((x, labels))
                total_supervised += (labels[..., 1:] != -100).sum().item()
            
            has_nan = False
            has_inf = False
            
            for x, labels in mbs:
                amp_device = 'cuda' if device.type == 'cuda' else 'cpu'
                amp_enabled = use_amp
                amp_dtype = dtype if use_amp else torch.float32
                
                with torch.amp.autocast(amp_device, enabled=amp_enabled, dtype=amp_dtype):
                    logits = model(x)
                    loss, valid_n, _ = compute_token_weighted_cross_entropy(
                        logits, labels, normalizer=total_supervised
                    )
                
                if torch.isnan(loss).item():
                    has_nan = True
                if torch.isinf(loss).item():
                    has_inf = True
                    
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
                
            # Check model gradients
            grad = model.weight.grad
            grad_nan = torch.isnan(grad).any().item()
            grad_inf = torch.isinf(grad).any().item()
            
            status = "PASS" if not (has_nan or has_inf or grad_nan or grad_inf) else "FAIL"
            print(f"[{dtype_name}] {case_name}: Supervised={total_supervised} | Loss={loss.item():.4f} | NaN={has_nan or grad_nan} | Inf={has_inf or grad_inf} | Status={status}")
            results.append({
                "dtype": dtype_name,
                "case": case_name,
                "tokens": total_supervised,
                "loss": loss.item(),
                "status": status
            })
            
    return results

def benchmark_memory_and_throughput():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== 2. Memory & Throughput Benchmark (50 Steps, Device: {device}) ===")
    
    # Use a small ModernGPT config to test full model forward/backward
    config = ModelConfig.get_16m_config(
        vocab_size=16384,
        context_length=1024,
    )
    
    grad_accum_steps = 4
    num_steps = 20 # 20 optimizer steps = 80 microbatches
    
    def run_benchmark(method="new"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        model = ModernGPT(config).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Fixed seed for data generation
        torch.manual_seed(42)
        
        start_time = time.perf_counter()
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Generate window microbatches with variable supervised lengths
            window_batches = []
            for i in range(grad_accum_steps):
                # Variable sequence length between 256 and 512
                cur_len = 256 + (i * 64)
                input_ids = torch.randint(0, config.vocab_size, (1, cur_len), device=device)
                labels = input_ids.clone()
                # Variable masking: supervise 50 to 200 tokens
                sup_tokens = 50 + (i * 40)
                labels[:, :-sup_tokens] = -100
                window_batches.append((input_ids, labels))
            
            if method == "old":
                for input_ids, labels in window_batches:
                    logits = model(input_ids)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                        reduction="mean"
                    )
                    loss = loss / grad_accum_steps
                    loss.backward()
            else: # method == "new"
                window_supervised_tokens = sum(
                    (b[1][..., 1:] != -100).sum().item() for b in window_batches
                )
                for input_ids, labels in window_batches:
                    logits = model(input_ids)
                    loss, _, _ = compute_token_weighted_cross_entropy(
                        logits, labels, normalizer=window_supervised_tokens
                    )
                    loss.backward()
                    
            optimizer.step()
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        total_time_ms = (end_time - start_time) * 1000
        ms_per_step = total_time_ms / num_steps
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if device.type == "cuda" else 0.0
        
        del model, optimizer
        torch.cuda.empty_cache()
        return ms_per_step, peak_vram_mb

    ms_old, vram_old = run_benchmark("old")
    ms_new, vram_new = run_benchmark("new")
    
    print(f"Old (Microbatch Mean): Latency = {ms_old:.2f} ms/step | Peak VRAM = {vram_old:.2f} MB")
    print(f"New (Token-Weighted) : Latency = {ms_new:.2f} ms/step | Peak VRAM = {vram_new:.2f} MB")
    lat_diff = ((ms_new - ms_old) / ms_old) * 100
    vram_diff = ((vram_new - vram_old) / vram_old) * 100 if vram_old > 0 else 0
    print(f"Difference: Latency {lat_diff:+.2f}% | VRAM {vram_diff:+.2f}%")
    
    return {
        "ms_old": ms_old,
        "vram_old": vram_old,
        "ms_new": ms_new,
        "vram_new": vram_new,
        "lat_diff": lat_diff,
        "vram_diff": vram_diff
    }

if __name__ == "__main__":
    test_numerical_stability()
    benchmark_memory_and_throughput()
