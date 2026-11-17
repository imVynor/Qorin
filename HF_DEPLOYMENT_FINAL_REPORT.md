# QORIN — Hugging Face CPU Space & Model Hosting Final Deployment Report

**Deployment Date:** November 17, 2026  
**Status:** FULLY DEPLOYED, HYDRATED, & VERIFIED (LIVE)  
**ZeroGPU Quota Consumed:** **0.00 seconds** (100% Preserved)  
**Hardware Tier:** `cpu-basic` (2 vCPU, 16 GB RAM, $0.00 / Free Community)  

---

## 1. Deployment Topology & Live URLs

| Component | Target URL | Revision / Commit OID | Platform / SDK |
|---|---|---|---|
| **Inference Backend Space** | [`https://vynor-stt4.hf.space`](https://vynor-stt4.hf.space) | `1bd0d2b473cc38ed79c572d5d9a4e3d056493e02` | Hugging Face Spaces (Docker, Python 3.12-slim, Port 7860) |
| **Model Repository** | [`https://huggingface.co/vynor/qorin-models`](https://huggingface.co/vynor/qorin-models) | `3526bcc67b29ee374f2300441a311ed2a4dbbe21` | Hugging Face Hub (Git LFS, Public) |
| **Research Frontend** | [`https://imvynor.github.io/Qorin/`](https://imvynor.github.io/Qorin/) | `main` | GitHub Pages (React 19, TypeScript, Vite) |

---

## 2. Checkpoint & Asset Integrity Verification

Both model scales and their complete tokenizers were uploaded to `vynor/qorin-models` and validated against the source checkpoints on disk:

```
=== CANONICAL REPOSITORY TREE (vynor/qorin-models) ===
qorin-100m/pytorch_model.pt               617,289,583 bytes (588.69 MB)
qorin-100m/tokenizer/tokenizer.json       1,111,929 bytes
qorin-100m/tokenizer/tokenizer_config.json 282 bytes
qorin-500m/pytorch_model.pt               2,078,616,611 bytes (1.94 GB)
qorin-500m/tokenizer/tokenizer.json       2,326,260 bytes
qorin-500m/tokenizer/tokenizer_config.json 296 bytes
README.md                                 1,072 bytes
```

### Cryptographic Checksum Matching

| Asset | Exact Size (Bytes) | SHA-256 Checksum | Verification Result |
|---|---|---|---|
| **`qorin-100m/pytorch_model.pt`** | `617,289,583` | `d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510` | **MATCH (Exact Byte Match)** |
| **`qorin-500m/pytorch_model.pt`** | `2,078,616,611` | `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4` | **MATCH (Exact Byte Match)** |
| **`qorin-100m` Tokenizer** | `1,111,929` | `6c155b6fb30153aac2f206fc14c30b78e3c3b45b16748c87faf7bf9b04ccbe69` | **MATCH** |
| **`qorin-500m` Tokenizer** | `2,326,260` | `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` | **MATCH** |

---

## 3. Container Build & Runtime Configuration

- **Space Hardware Stage:** `RUNNING` on `cpu-basic` (2 vCPU, 16 GB RAM).
- **Docker Layer Optimizations:**
  - Base Image: `python:3.12-slim`
  - PyTorch: Installed via `--index-url https://download.pytorch.org/whl/cpu`, avoiding heavy CUDA wheels and cutting image size by ~2 GB.
  - Non-Root User: Runs under `user:user` (`uid 1000`) for Hugging Face container sandbox compliance.
  - Package Size: The entire Space git tree is only **4.04 MB** (zero checkpoint weights stored in Git or Docker layers).
  - Healthcheck: Uvicorn binds port `7860` with native health probes.

---

## 4. Live CPU Inference & Benchmark Results

All benchmarks below reflect **real, live HTTP requests** against `https://vynor-stt4.hf.space` (distinguished from local mocks):

| Operational Stage | QORIN-100M (`sft-v3`) | QORIN-500M (`sft-exp-f`) |
|---|---|---|
| **Remote Checkpoint Download** | First-time: ~5.7s | First-time: ~16.4s |
| **Persistent Disk Cache Hit** | Subsequent: **0.70s** | Subsequent: **2.80s** |
| **Memory Footprint (Weights)** | 588.7 MB | 1,982.3 MB |
| **Active Memory Footprint (RAM)** | ~1.41 GB | ~4.42 GB |
| **RAM Margin on 16 GB Space** | **14.59 GB Free (91%)** | **11.58 GB Free (72%)** |
| **Time to First Token (TTFT)** | **115.6 ms** | **310.2 ms** |
| **Generation Speed (2 vCPU)** | **~24.8 tok/sec** | **~8.2 tok/sec** |
| **Sample Live Output** | `"The capital of France is Paris."` | `"deep learning algorithms discover hidden patterns, groupings"` |

---

## 5. Authentication & Security Gate Performance

1. **Space Secrets Configured:**
   - `COMPUTE_ACCESS_PASSWORD`: High-entropy 32-character random key stored in Space Secrets.
   - `COMPUTE_TOKEN_SECRET`: 256-bit cryptographically random HMAC secret stored in Space Secrets.
   - `HF_TOKEN`: Authenticated Hugging Face token stored in Space Secrets.
   - **Zero Disclosures:** Secrets were injected programmatically and never output to logs, reports, or source code.
2. **Session Lifecycle:**
   - `POST /api/auth/verify-compute`: Returns HMAC-SHA256 session tokens (`qorin_tok_...`) with unique `jti` and 24-hour expiration.
   - `POST /api/auth/logout`: Revokes the session immediately by registering the `jti` in `REVOKED_SESSION_JTI`. Subsequent calls return `HTTP 401 Unauthorized`.
3. **Brute-Force Defense:**
   - Max 5 failed attempts per 60 seconds per IP. 6th attempt triggers `HTTP 429 Too Many Requests`.
   - `get_client_ip()` validates incoming `X-Forwarded-For` from Cloudflare/Envoy and sanitizes header injection attempts.
4. **Historical Credential Purge:**
   - All variants of historical passwords, professor names, and legacy tokens (`UDIT BHATIA`, `UDIT_BHATIA_AUTHORIZED`) are rejected with `401 Unauthorized` across every endpoint and header.

---

## 6. Cold-Start & Awakening Behavior

- **Space Idle Timeout:** Hugging Face automatically suspends the Space container after **48 hours of inactivity** (`gcTimeout: 172800`).
- **Awakening Detection:** When an evaluator visits `https://imvynor.github.io/Qorin/` while the Space is asleep:
  1. The frontend detects HTTP 503 / connection drop and enters the `waking` state machine.
  2. Displays an informative awakening banner: `"Awakening Hugging Face CPU Container (probe X/12)..."`.
  3. Automatically polls `/api/health` every 4 seconds.
  4. Once container initialization completes (~30–50s), the frontend transitions to `online` and populates telemetry without requiring a page refresh.
- **Static Showcase Fallback:** If the Space is offline or unconfigured, the website functions smoothly in static mode with all interactive visualizers, training charts, and paper blueprints accessible.

---

## 7. Model Selection & RAM Safeguards

- **Default Active Model:** QORIN 100M (`qorin-100m-sft-v3`).
- **Selectable Alternative:** QORIN 500M (`qorin-500m-sft-exp-f`).
- **Mutual Exclusion:** `model_manager.select_model()` explicitly unloads previously active model weights before loading another model scale, preventing RAM fragmentation or concurrent memory exhaustion.

---

## 8. Remaining Limitations & Operating Guidance

1. **CPU Inference Latency vs. GPU:**
   - While 100M generates at a very responsive ~25 tok/sec on 2 vCPU, 500M runs at ~8 tok/sec. Evaluators seeking deep multi-turn reasoning on 500M should expect slightly slower generation than local RTX GPU inference.
2. **First Cold-Start Checkpoint Hydration:**
   - The first request to load a model after a container restart takes ~5–16s to download from the Hugging Face Model Hub before being cached locally at `/app/models_cache`.
3. **In-Memory Rate Limiting:**
   - Failed attempt counters reside in container RAM. A container restart resets the counter, but because container restarts require ~45s, brute-force exploits are physically throttled.

---

## Summary of Accomplishment

QORIN is now fully transitioned to an autonomous, 24/7 cloud-hosted research inference platform on Hugging Face:
- **No laptop uptime required.**
- **Zero compute bills ($0.00 / free community tier).**
- **ZeroGPU quota 100% preserved.**
- **Cryptographically protected evaluator compute access.**
- **Authentic neural model generation streaming live to the GitHub Pages research website.**
