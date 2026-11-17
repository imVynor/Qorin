# QORIN — Final Deployment Security Gate Report

**Target Space:** `vynor/stt4` (Hugging Face Docker Space on CPU Basic)  
**Model Repository:** `vynor/qorin-models`  
**Frontend Deployment:** GitHub Pages (`https://imvynor.github.io/Qorin/`)  
**Audit Timestamp:** November 17, 2026  
**Status:** ALL 10 SECURITY GATES VERIFIED — AWAITING EXPLICIT FINAL APPROVAL  

---

## 1. Credential Rotation Confirmation

All previous evaluator credentials and hardcoded access strings have been completely removed from the entire codebase.

- **New Password Generation:** A high-entropy, 32-character random access key was generated using the Python cryptographic `secrets` module (`secrets.token_urlsafe(32)`).
- **HMAC Signing Secret Generation:** An independent, 256-bit cryptographically random HMAC secret key (`secrets.token_hex(32)`) was generated for signing session tokens.
- **Secret Isolation:** The new credentials are stored strictly in `backend/.secrets.json` (local execution) and mapped to environment variables `COMPUTE_ACCESS_PASSWORD` and `COMPUTE_TOKEN_SECRET` in Hugging Face Space secrets.
- **Zero Disclosures:** Neither secret has been printed, logged, or recorded in terminal output, test logs, reports, or source code.
- **Non-Disclosure Compliance:** All code, frontend UI elements, and API responses use generic terms ("Evaluator Authorization Key", "Evaluator Compute Gate"). All professor names, hints, visual blanks (`____ ______`), and pattern markers (`XXXX XXXXXX`) have been excised.

---

## 2. Old Credential Rejection Results

All historical credentials and legacy string tokens were tested across every authentication path:

| Tested Input | Auth Path Tested | Result | Security Behavior |
|---|---|---|---|
| Historical Password String | `POST /api/auth/verify-compute` | **401 Unauthorized** | Rejected immediately (`"Invalid evaluator authorization key."`) |
| Historical Password (lowercase) | `POST /api/auth/verify-compute` | **401 Unauthorized** | Rejected immediately |
| Legacy Static Token (`UDIT_...`) | `POST /api/auth/verify-compute` | **401 Unauthorized** | Rejected immediately |
| Historical Password String | `X-Compute-Token` Header | **401 Unauthorized** | Direct token bypass rejected |
| Historical Password String | `Authorization: Bearer` Header | **401 Unauthorized** | Direct token bypass rejected |
| Legacy Static Token | `X-Compute-Token` Header | **401 Unauthorized** | Legacy static tokens completely invalidated |
| Raw New Password as Token | `X-Compute-Token` Header | **401 Unauthorized** | Raw passwords cannot be used as tokens; must obtain signed session token |

---

## 3. Authentication Test Results (10 Gates)

The full security verification suite (`tests/test_auth_security_gate.py`) was executed against the FastAPI application:

```
======================================================================
QORIN AUTHENTICATION SECURITY GATE VERIFICATION
======================================================================
[Gate 1/10] Testing Correct New Evaluator Credential...
           PASS: Valid session token issued with zero credential disclosure.
[Gate 2/10] Testing Incorrect Password Rejection...
           PASS: Incorrect password rejected with generic 401 error.
[Gate 3/10] Testing Old Disclosed Password & Legacy String Rejection...
           PASS: All old disclosed credentials and legacy strings rejected across all auth paths.
[Gate 4/10] Testing Invalid Cryptographic Signature...
           PASS: Invalid signature rejected cryptographically.
[Gate 5/10] Testing Expired Session Token...
           PASS: Expired token rejected automatically.
[Gate 6/10] Testing Logged-Out / Revoked Session...
           PASS: Session invalidation / logout successfully revokes token.
[Gate 7/10] Testing Session Payload Tampering...
           PASS: Payload tampering detected; signature mismatch enforced.
[Gate 8/10] Testing Brute-Force Rate Limiting (5 Attempts / 60s)...
           PASS: 6th brute-force attempt triggered HTTP 429 rate limit.
[Gate 9/10] Testing Session Independence...
           PASS: Sessions are cryptographically isolated with independent jti values.
[Gate 10/10] Testing Direct Unauthenticated API Access Rejection...
           PASS: All protected inference endpoints reject unauthenticated access.
[Bonus] Testing Reverse-Proxy IP Header Sanitization...
           PASS: Reverse-proxy edge client resolution and injection mitigation verified.
======================================================================
ALL 10 SECURITY GATE TESTS PASSED WITH 100% SUCCESS!
======================================================================
```

---

## 4. Rate-Limiter & Reverse-Proxy IP Behavior

### Rate Limiting Policy
- **Threshold:** Maximum 5 failed authentication attempts per IP address within a sliding 60-second window.
- **Penalty:** Any request exceeding 5 failed attempts is immediately rejected with `HTTP 429 Too Many Requests`.
- **Reset on Success:** A successful login clears the failed attempt count for that IP.

### Reverse-Proxy Edge IP Resolution
When deployed on Hugging Face Spaces behind Cloudflare / Envoy edge proxies:
- The backend parses the incoming `X-Forwarded-For` header.
- The **first entry** in the comma-separated proxy chain represents the client IP as seen by the outermost edge gateway.
- `get_client_ip()` strictly validates candidate IP strings (length <= 45, character set restricted to valid IPv4/IPv6 notation `0-9, a-f, A-F, ., :`).
- Maliciously injected SQL/command strings or malformed headers are rejected and fall back safely to `request.client.host`.

### Application Restart Persistence Analysis
- **Storage Mechanism:** In-memory dictionary (`FAILED_AUTH_ATTEMPTS: Dict[str, List[float]]`).
- **Restart Survival:** **Does NOT survive application restarts or container recreation.**
- **Impact & Assessment:** In a single-instance container on Hugging Face CPU Basic, restarting the container resets the tracking dictionary. This is standard for single-node web services. Because the brute-force window is brief (60 seconds), an attacker cannot exploit restarts to bypass the rate limiter, as container restarts require ~45 seconds. For multi-replica distributed scaling in the future, a shared Redis store would be appropriate.

---

## 5. Model Integrity Manifest

All checkpoint and tokenizer files have been cryptographically verified against the original weights stored on disk:

### 1. QORIN-100M Supervised Fine-Tuned Model (`qorin-100m-sft-v3`)
* **Local Source:** `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt`
* **Target Repo Path:** `qorin-100m/pytorch_model.pt` in `vynor/qorin-models`
* **Exact Size:** `617,289,583` bytes (~588.69 MB)
* **SHA-256:** `d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510`
* **Tokenizer:** `tokenizer_16k_v2`
  - `tokenizer.json`: `1,111,929` bytes | SHA-256: `6c155b6fb30153aac2f206fc14c30b78e3c3b45b16748c87faf7bf9b04ccbe69`
  - `tokenizer_config.json`: `282` bytes | SHA-256: `7e728ff718209cac30d736ff877c20d66845b1859eaec0ac6efa6a22124446c2`
* **Architecture Configuration:**
  - Vocabulary Size: `16,000`
  - Context Window: `4,096` tokens
  - Hidden Dimension: `768`
  - Query Heads: `12`
  - Key-Value Heads: `4` (Grouped-Query Attention ratio 3:1)
  - Transformer Layers: `14`
  - FFN Hidden Dimension: `2,048` (SwiGLU)
  - RoPE Theta: `10,000.0` | RoPE Scale: `8.0` (Linear context extension)

### 2. QORIN-500M Supervised Fine-Tuned Model (`qorin-500m-sft-exp-f`)
* **Local Source:** `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt`
* **Target Repo Path:** `qorin-500m/pytorch_model.pt` in `vynor/qorin-models`
* **Exact Size:** `2,078,616,611` bytes (~1.94 GB)
* **SHA-256:** `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4`
* **Tokenizer:** `tokenizer_32k`
  - `tokenizer.json`: `2,326,260` bytes | SHA-256: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`
  - `tokenizer_config.json`: `296` bytes | SHA-256: `b803c75253c44fd9a0745e83df72e9830335b9535c4d3fefc74f3381cb5e45ac`
* **Architecture Configuration:**
  - Vocabulary Size: `32,768`
  - Context Window: `2,048` tokens
  - Hidden Dimension: `1,536`
  - Query Heads: `24`
  - Key-Value Heads: `4` (Grouped-Query Attention ratio 6:1)
  - Transformer Layers: `20`
  - FFN Hidden Dimension: `3,840` (SwiGLU)
  - RoPE Theta: `10,000.0` | RoPE Scale: `1.0` (Default)

### 3. Exclusions
- **Training Data:** Zero JSONL files, tokenized shards, or raw datasets will be uploaded.
- **Optimizer States:** Excluded; only inference-ready model weights are staged.
- **Unrelated Research Artifacts:** Excluded; pilot checkpoints and scratch runs remain local.

---

## 6. Repository Visibility and Access Design

* **Model Repository:** `vynor/qorin-models`
* **Confirmed Visibility:** **Public (`Private: False`)**
* **Repository HEAD Revision:** `3db679138c68b6df8e774e998fc1a2cf8a006304`
* **Access Consequences:**
  1. **Zero-Token Hydration:** Because the repository is public, the Space application can download checkpoints via standard HTTPS without requiring an embedded or leaked `HF_TOKEN`.
  2. **Auditability:** Evaluators, course reviewers, and the scientific community can download and verify model weights independently.
  3. **Privacy Tradeoff:** While the weights are public, inference execution remains gated behind the backend evaluator key.

---

## 7. Final Deployment Package Verification

The packaging files staged at `modern_gpt/deployment/huggingface_cpu_deployment/stt4_package/` have been verified:

1. **Docker Build Context:**
   - Multi-stage CPU container based on `python:3.12-slim`.
   - PyTorch installed exclusively via CPU wheel (`--index-url https://download.pytorch.org/whl/cpu`), reducing image size by ~2 GB and removing all CUDA libraries.
2. **`.dockerignore` Validation:**
   - Explicitly excludes `backend/.secrets.json`, `*.pt`, `checkpoints/`, `node_modules/`, `web/`, and `tests/`.
   - Secrets are NEVER baked into Docker image layers.
3. **Execution Security:**
   - Non-root container execution enforced via `useradd -m -u 1000 user` and `USER user`.
4. **Port Binding:**
   - Uvicorn configured to bind `--host 0.0.0.0 --port 7860`, matching Hugging Face Space Docker requirements.
5. **Cold-Start & Awakening UX:**
   - Space sleeps after 48h of inactivity (`gcTimeout: 172800`).
   - GitHub Pages frontend automatically detects sleeping containers, displays a non-blocking awakening banner with probe counters, and auto-connects when healthy.

---

## 8. Remaining Operational Risks & Mitigation

1. **First-Load Hydration Latency:**
   - *Risk:* The very first time a model scale is selected after container initialization, it streams ~588 MB (100M) or ~1.94 GB (500M) from Hugging Face Hub.
   - *Mitigation:* The frontend displays an explicit "Downloading verified checkpoint..." status pill with progress indication. Weights are cached persistently in `/app/models_cache` for subsequent queries.
2. **Container Sleep on Inactivity:**
   - *Risk:* Evaluators accessing the showcase after 48 hours of quiet period will experience a ~40s cold start.
   - *Mitigation:* The frontend awakening polling engine retries automatically without requiring page refreshes or user intervention.
3. **ZeroGPU Quota Protection:**
   - *Confirmation:* ZeroGPU runtime allowance remains **100% untouched (0.00 seconds consumed)**.

---

## 9. Current Deployment Status

In strict adherence to project restrictions:
- [x] No changes pushed to `vynor/stt4`.
- [x] No model weights uploaded to `vynor/qorin-models`.
- [x] No paid hardware activated.
- [x] Zero GPU quota consumed.
- [x] Canonical checkpoints unmodified on disk.
- [x] No uncommitted code pushed to GitHub.

**Standing by for explicit user deployment approval.**
