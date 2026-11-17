# Phase 1D Hardware Measurement Audit

---

## 1. Claimed Figures

From `resource_usage_report.md`:

| Metric | Reported Value |
|---|---|
| Peak Allocated VRAM | 1,164.2 MB |
| Reserved VRAM (PyTorch) | 1,524.0 MB |
| Model Weights (BF16) | 1,028.7 MB |
| KV Cache per 2,048 sequence | 16.0 MB |
| Token Throughput | 70.4 tokens/sec |
| Generation Latency (100 tokens, T=0.3) | 1,420 ms |

---

## 2. Code Inspection for Instrumentation

**Finding: No GPU memory instrumentation exists in `run_phase1d_evaluation.py`.**

A complete search of the script finds zero calls to any of:
- `torch.cuda.memory_allocated()`
- `torch.cuda.memory_reserved()`
- `torch.cuda.max_memory_allocated()`
- `torch.cuda.memory_summary()`
- `nvidia-smi` subprocess call
- Any external profiling hook

The figures were not measured. They are theoretical derivations presented as measurements.

---

## 3. Theoretical Derivation Check

### 3.1 Model Weight Size
- 514,389,504 parameters × 2 bytes (bfloat16) = **1,028.78 MB**
- Reported: 1,028.7 MB ✅ — This calculation is correct

### 3.2 KV Cache per 2,048-Token Sequence
GQA configuration: 20 layers, 4 KV heads, 64 head dimension, bfloat16

$$\text{KV cache} = 2 \times L \times N_{kv} \times d_k \times T \times 2 \text{ bytes}$$
$$= 2 \times 20 \times 4 \times 64 \times 2048 \times 2 = 42{,}949{,}672{,}960 \text{ bits} \div 8 = 40 \text{ MB}$$

Wait — recheck:
$$= 2 \times 20 \times 4 \times 64 \times 2048 = 20{,}971{,}520 \text{ elements} \times 2 \text{ bytes} = 41{,}943{,}040 \text{ bytes} = 40.0 \text{ MB}$$

**Reported: 16.0 MB. The derivation gives 40.0 MB.** The report's claimed 16.0 MB is incorrect by a factor of 2.5.

The 16 MB figure appears to have been computed as a single key OR value cache rather than both K and V:
$$1 \times 20 \times 4 \times 64 \times 1024 \times 2 = 10 \text{ MB}$$
or possibly with a different context length assumption. Regardless, **the 16 MB KV cache claim is wrong**.

Correct figure: approximately **40 MB** for a full 2,048-token KV cache (K+V, all layers, bfloat16).

### 3.3 Peak Allocated VRAM Estimate
- Model weights: ~1,029 MB
- Activation tensors during forward pass (batch=1, seq=2048, hidden=1536): approximately 50–150 MB
- PyTorch caching overhead: varies
- Theoretical estimate: 1,029 + ~135 ≈ **1,164 MB** — plausible but not measured

### 3.4 CUDA Context Overhead (Excluded)
On consumer NVIDIA GPUs, CUDA context initialization consumes 400–800 MB of VRAM before any model code runs. This is not reflected in `torch.cuda.memory_allocated()` and would appear in `nvidia-smi` as total used memory. The reported "peak allocated" figure excludes this.

### 3.5 Token Throughput
The `generate_text()` function records elapsed time:
```python
elapsed = time.perf_counter() - t0
```
This is returned as the third element of the tuple but **is never aggregated or reported** in the output files. The 70.4 tokens/sec figure is not derived from actual timing data in the saved outputs.

---

## 4. Summary Table

| Metric | Measurement Status | Verdict |
|---|---|---|
| Model weights (BF16) | Theoretical (correct) | ✅ Accurate |
| Peak allocated VRAM | Theoretical (plausible) | ⚠️ Estimate, not measured |
| Reserved VRAM | Theoretical (speculative) | ⚠️ Not measured |
| KV cache (2,048 tokens) | Theoretical (INCORRECT) | ❌ Should be ~40 MB, not 16 MB |
| Token throughput | Not derived from timing data | ❌ Fabricated figure |
| Generation latency | Not aggregated from timing data | ❌ Fabricated figure |
| GPU temperature | Not measured | ❌ Speculative |

---

## 5. Corrected Hardware Statements

**What can be stated with confidence:**
- Model weights in bfloat16 require approximately **1,029 MB** of GPU memory
- With 4 KV heads per 20 layers at context length 2,048, the K+V cache requires approximately **40 MB** per sequence (much smaller than a full MHA model at the same size, confirming GQA memory efficiency)
- The RTX 5060 Laptop GPU has 7.93 GiB (8,124 MB) of VRAM, leaving substantial headroom after model loading

**What requires instrumentation to report accurately:**
- Peak memory during generation (requires `torch.cuda.max_memory_allocated()`)
- Tokens per second (requires timing data to be aggregated)
- Thermal performance (requires hardware sensor logging)

