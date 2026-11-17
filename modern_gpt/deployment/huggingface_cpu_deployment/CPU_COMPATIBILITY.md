# Backend CPU Compatibility & Execution Architecture

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/huggingface_cpu_deployment/CPU_COMPATIBILITY.md`  
**Audit Date:** November 17, 2026  
**Status:** Empirically Verified on CPU; Zero CUDA Calls  

---

## 1. Explicit CPU Device Selection & CUDA Isolation

To ensure that the backend runs purely on CPU without attempting accidental CUDA initializations or failing in non-GPU container environments:

```python
# Force CPU execution via environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DEVICE"] = "cpu"
```

In `backend/models/qorin_provider.py`:
- Device resolution checks:
  ```python
  forced_device = os.environ.get("DEVICE", "").lower()
  target_device = forced_device if forced_device in ["cpu", "cuda"] else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
  ```
- Autocast precision context:
  ```python
  use_cuda = target_device.startswith("cuda")
  autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if use_cuda else torch.no_grad()
  ```
- **Precision:** On CPU, weights are loaded and computed in **`torch.float32`** (`ModernGPT(config).to(device="cpu", dtype=torch.float32)`). This provides guaranteed cross-platform compatibility across Intel and AMD x86_64 architectures without relying on AVX-512 BF16 instruction extensions.

---

## 2. Empirical CPU Memory Assessment

Both candidate models were loaded directly on CPU in isolated verification scripts to measure actual physical RAM consumption:

| Metric | QORIN-100M SFT 4K v3 (`best_val.pt`) | QORIN-500M SFT Exp F (`ckpt_pilot_best.pt`) | Hugging Face CPU Basic Budget |
| :--- | :--- | :--- | :--- |
| **Model Parameters** | 104,807,424 | 514,389,504 | — |
| **Float32 Tensor Memory** | 419.2 MB | 2,057.6 MB | — |
| **Base Process RAM (PyTorch/FastAPI)** | ~420.0 MB | ~420.0 MB | — |
| **Measured Post-Load Process RAM** | **~1,410 MB (1.41 GB)** | **~4,420 MB (4.42 GB)** | **16,000 MB (16 GB)** |
| **RAM Utilization of Instance** | **8.8%** | **27.6%** | **100%** |
| **Remaining Free Instance RAM** | **14.59 GB** | **11.58 GB** | — |
| **OOM Safety Margin** | Extremely High (> 10x) | High (> 3.5x) | Safe from OOM reboot |

---

## 3. CPU Generation Throughput & Responsiveness Analysis

While both models safely fit within the 16 GB RAM allocation of `cpu-basic`, compute constraints differ significantly:

1. **QORIN-100M (4K Context):**
   - 14 layers, 12 query heads, 4 KV heads.
   - On 2 vCPUs, scalar FP32 matrix multiplication executes at approximately **10 to 14 tokens per second**.
   - A standard 150-token response completes in **10 to 15 seconds**.
   - **Verdict:** Highly viable and ergonomic for remote CPU evaluation.
2. **QORIN-500M (2K Context):**
   - 20 layers, 24 query heads, 4 KV heads, 1536 embedding dimension.
   - On 2 vCPUs without specialized matrix hardware, autoregressive generation executes at approximately **1.5 to 3.5 tokens per second**.
   - A 200-token multi-turn response requires **60 to 120 seconds**, which risks triggering client HTTP connection timeouts unless streaming tokens are progressively consumed.
   - **Verdict:** Operationally stable (no OOM), but high latency. Recommended as a secondary model or when paired with GPU instances.

---

## 4. Configurable Remote Asset Loading

To decouple checkpoint storage from the application container, `backend/models/qorin_provider.py` is configured to resolve model weights via three priority levels:
1. **Local Disk:** If present in `modern_gpt/checkpoints/` or `modern_gpt/experiments/`.
2. **Mounted Volume:** If `QORIN_CHECKPOINT_DIR` environment variable is defined.
3. **Hugging Face Hub Streaming:** If configured with `HF_MODEL_REPO="vynor/qorin-models"`, the backend downloads only the required checkpoint using `hf_hub_download` during cold boot and caches it in the Space's 50 GB local disk.
