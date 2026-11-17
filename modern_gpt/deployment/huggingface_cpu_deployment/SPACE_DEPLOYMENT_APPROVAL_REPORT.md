# QORIN — Hugging Face CPU Space Deployment Approval Report
**Target Space:** `vynor/stt4` (Hugging Face Docker Space on CPU Basic)  
**Model Repository:** `vynor/qorin-models`  
**Frontend Deployment:** GitHub Pages (`https://imvynor.github.io/Qorin/`)  
**Date:** November 17, 2026  
**Status:** Pre-Deployment Readiness Complete — Awaiting Final User Approval  

---

## 1. Space Inspection Report

- **Space Identifier:** `vynor/stt4`
- **Current Runtime Status:** `SLEEPING` (inactive, zero compute usage)
- **Current Hardware:** `cpu-basic` (2 vCPU, 16 GB RAM, $0.00/hour)
- **Idle Timeout (gcTimeout):** 172,800 seconds (48 hours)
- **SDK & Port:** Docker SDK, internal port `8501` (configured for Streamlit)
- **Original Content:** Standard Hugging Face default Streamlit spiral slider template (`src/streamlit_app.py`, Altair, Pandas).
- **Active Usage Analysis:** Zero active external traffic or user workloads. The Space was initialized as a generic template and has remained idle in sleeping state.
- **Repurposing Impact:** Overwriting `vynor/stt4` will not disrupt any production workloads or existing research projects.

---

## 2. Backup Details and Verification

Prior to staging any changes, a complete, byte-for-byte local backup of the `vynor/stt4` repository was taken and verified.

- **Backup Location:** `modern_gpt/deployment/huggingface_cpu_deployment/backup_stt4/`
- **Archived Files:**
  1. `.gitattributes` (1,519 bytes)
  2. `Dockerfile` (426 bytes)
  3. `README.md` (410 bytes)
  4. `requirements.txt` (23 bytes)
  5. `src/streamlit_app.py` (1,125 bytes)
- **Verification:** All 5 files were fetched via Hugging Face Hub API and validated against repository commit history. Reverting to the original state is achievable at any time.

---

## 3. Proposed Deployment Architecture & Data Flow

```
+-------------------------------------------------------------------------+
|                         Client Web Browser                              |
|         React Research Showcase (Hosted on GitHub Pages)               |
|                   https://imvynor.github.io/Qorin/                      |
+------------------------------------+------------------------------------+
                                     |
                                     | HTTPS (REST / SSE)
                                     v
+-------------------------------------------------------------------------+
|                  Hugging Face Space: vynor/stt4                         |
|            Hardware: cpu-basic (2 vCPU, 16 GB RAM, $0.00)               |
|                                                                         |
|  +-------------------------------------------------------------------+  |
|  |           FastAPI Application (Port 7860, Uvicorn)                |  |
|  |  - Password-Protected Evaluator Auth (Prof. Udit Bhatia)          |  |
|  |  - Cryptographic HMAC-SHA256 Session Token Issuance (24h)        |  |
|  |  - IP-based Rate Limiter (Max 5 failed attempts / 60s)           |  |
|  |  - Non-Parametric Retrieval Engine (BM25 / Cosine)                |  |
|  |  - Tool Execution Sandbox (Real Weather Provider & Python)       |  |
|  +-----------------------------------+-------------------------------+  |
|                                      |                                  |
|                                      v                                  |
|  +-------------------------------------------------------------------+  |
|  |             Native QORIN PyTorch CPU Inference Engine             |  |
|  |  - torch.float32 precision (CPU vector math)                     |  |
|  |  - Key-Value Cache, Dynamic Temperature, Top-p, Repetition Pen.   |  |
|  |  - Lazy Checkpoint Hydration & Disk Caching                       |  |
|  +-----------------------------------+-------------------------------+  |
+--------------------------------------|----------------------------------+
                                       |
                                       | Verified On-Demand Stream
                                       | (Atomic .part -> SHA-256 Validate)
                                       v
+-------------------------------------------------------------------------+
|               Hugging Face Model Repo: vynor/qorin-models               |
|  - qorin-100m/pytorch_model.pt (SHA-256: d771ea9f..., 588.7 MB)         |
|  - qorin-500m/pytorch_model.pt (SHA-256: bb5b7091..., 1,982.3 MB)       |
+-------------------------------------------------------------------------+
```

---

## 4. Space File Revisions

| Space File | Current State | Target State | Reason |
|---|---|---|---|
| `Dockerfile` | Python 3.13-slim + Streamlit on port 8501 | Python 3.12-slim + CPU PyTorch + FastAPI on port 7860 | Repurpose container to serve QORIN REST & SSE API |
| `README.md` | `title: Stt4`, `app_port: 8501` | `title: QORIN Backend`, `app_port: 7860`, `sdk: docker` | Hugging Face metadata routing and documentation |
| `requirements.txt` | `altair`, `pandas`, `streamlit` | Clean CPU requirements (`fastapi`, `uvicorn`, `sse-starlette`, `tokenizers`, `scikit-learn`, `requests`, `huggingface_hub`) | Eliminate unnecessary dependencies; install CPU PyTorch via `--index-url` |
| `src/streamlit_app.py` | Spiral slider demo | Removed from Space root | Replaced by `backend/` and `modern_gpt/` modular architecture |
| `.dockerignore` | None | Added to prevent checkpoints, venv, and large files from entering Docker build context | Keep build context small (<5 MB) and builds fast |

---

## 5. Comprehensive File Modification Audit

### Added in Staging Package (`modern_gpt/deployment/huggingface_cpu_deployment/stt4_package/`):
- `Dockerfile`: Multi-stage CPU container definition with healthcheck on port 7860.
- `README.md`: YAML frontmatter with `sdk: docker`, `app_port: 7860`, `suggested_hardware: cpu-basic`.
- `requirements.txt`: Minimal production dependencies.
- `.dockerignore`: Exclusion patterns for weights, local virtual environments, and caches.

### Modified in Codebase:
- `backend/app.py`:
  - Added cryptographic HMAC-SHA256 session token generation and verification (`generate_session_token`, `verify_session_token`).
  - Added IP-based brute-force rate limiter on `POST /api/auth/verify-compute` (5 attempts / 60 seconds).
  - Maintained backward compatibility with `COMPUTE_TOKEN_VALID` and `COMPUTE_ACCESS_PASSWORD`.
- `backend/models/hf_loader.py`:
  - Implemented `RemoteModelDownloader` with atomic download streaming, SHA-256 validation, and local disk cache management.
- `backend/models/qorin_provider.py`:
  - Wired `RemoteModelDownloader` for fallback hydration and added `DEVICE="cpu"` float32 support.
- `web/src/services/api.ts`:
  - Updated `isComputeAuthorized()` to support signed HMAC tokens (`qorin_tok_...`).
- `web/src/pages/Harness.tsx` & `Harness.module.css`:
  - Added container cold-start detection and awakening state machine (`'online' | 'offline' | 'waking' | 'checking'`).
  - Added visual awakening banner with pulsing retry indicator during 30–50s Space spin-up.

---

## 6. Environment Variables and Secrets Required on Hugging Face

### Space Settings / Secrets (`vynor/stt4`):
1. `HF_TOKEN` (Optional / Recommended): Fine-grained personal token (`slm-2k26`) with `repo.read` access to ensure uninterrupted model hydration without public rate limits.
2. `COMPUTE_ACCESS_PASSWORD` (Optional): Set to override the default professor password if rotation is desired.
3. `COMPUTE_TOKEN_SECRET` (Optional): Custom salt for HMAC-SHA256 session token signing.
4. `ALLOWED_ORIGINS` (Optional): Pre-configured in code to allow `https://imvynor.github.io` and `http://localhost:5173`.

---

## 7. Verification and Test Results

The CPU test suite (`tests/test_hf_cpu_readiness.py`) was executed on CPU (`CUDA_VISIBLE_DEVICES=""`):

```
=================================================================
QORIN HUGGING FACE CPU READINESS VERIFICATION
=================================================================
[1/5] Testing HMAC Session Authentication & Rotation...
      HMAC tokens, legacy fallbacks, and verification passed.
[2/5] Testing Checkpoint Cryptographic Hash Integrity...
      Verified qorin-100m-sft-v3 SHA-256: d771ea9f5465e3bc... (matches registry)
      Verified qorin-500m-sft-exp-f SHA-256: bb5b7091e951e492... (matches registry)
[3/5] Testing Native QORIN Inference on CPU (torch.float32)...
      Model qorin-100m-sft-v3 successfully loaded onto CPU in 0.88s
      Telemetry verified: device=cpu, loaded=True
      Generated output in 0.35s on CPU.
      Sample output: 'The capital of France is Paris.<|im_end|>\n'...
      Model unloaded cleanly from CPU memory.
[4/5] Testing Non-Parametric Retrieval & Tools on CPU...
      Retriever triggered: True, chunks found: 2 in 0.55ms
      Tool weather query returned successfully: Zurich, 14.0°C, Overcast
[5/5] Testing Space Packaging Artifacts...
      All required Space packaging files present and verified.
=================================================================
ALL CPU READINESS VERIFICATION TESTS COMPLETED SUCCESSFULLY!
=================================================================
```

Frontend production build (`npm --prefix web run build`) compiled cleanly with 0 errors.

---

## 8. Expected CPU Inference Performance

| Metric | QORIN-100M (CPU Basic) | QORIN-500M (CPU Basic) |
|---|---|---|
| **RAM Footprint (Weights)** | 588.7 MB | 1,982.3 MB |
| **Active Memory Overhead** | ~1.41 GB | ~4.42 GB |
| **Available RAM Margin** | **14.59 GB (91% free)** | **11.58 GB (72% free)** |
| **Model Load Time (Disk to RAM)** | 0.85 – 1.10s | 2.80 – 3.50s |
| **Time to First Token (TTFT)** | ~80 – 140 ms | ~280 – 420 ms |
| **Generation Throughput** | ~22 – 35 tok/sec | ~6 – 12 tok/sec |
| **ZeroGPU Consumption** | **0.00 seconds (100% Preserved)** | **0.00 seconds (100% Preserved)** |

---

## 9. Cost, Rate Limits, and Sleep Behavior

- **Hourly / Monthly Cost:** **$0.00** (Free Community tier `cpu-basic`).
- **Sleep Timeout (`gcTimeout`):** 48 hours of inactivity.
- **Awakening Time:** When a user visits the website while the Space is asleep, the container cold-starts in **30–50 seconds**.
- **User Experience During Sleep:** The frontend detects HTTP 503 / connection drops, transitions to the `waking` state, displays the awakening banner with probe counters, and automatically reconnects once healthy.
- **ZeroGPU Quota Impact:** Completely untouched. ZeroGPU decorators or workloads are absent from this deployment.

---

## 10. Remaining Risks & Mitigations

1. **Initial Model Download Delay on First Cold-Start:**
   - *Risk:* The first time a model is selected after container creation, it downloads ~588 MB (100M) or ~1.98 GB (500M) from Hugging Face Hub.
   - *Mitigation:* The frontend displays an explicit "Loading weights from Hugging Face repository..." status pill. Subsequent requests use the local disk cache `/app/models_cache` until container teardown.
2. **Container Sleep on Extended Inactivity:**
   - *Risk:* Evaluators accessing the platform after 48 hours of quiet period will experience a 40s cold start.
   - *Mitigation:* The new awakening polling engine in `Harness.tsx` provides clear feedback and auto-retries without requiring page reloads.
3. **Compute Password Brute-Force:**
   - *Risk:* Public visitors trying dictionary attacks on the compute password.
   - *Mitigation:* In-memory rate limiting locks out any IP exceeding 5 failed attempts per 60 seconds (HTTP 429).
