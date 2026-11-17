# Authentication & Security Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/AUTHENTICATION_SECURITY_AUDIT.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Authentication Architecture Overview

To protect the author's local workstation compute (NVIDIA RTX 5060 Laptop GPU, 8GB VRAM) and prevent unauthorized script execution in the Python sandbox while permitting seamless course grading for the instructor, an evaluation access gate was established across the frontend and backend.

- **Designated Password:** `UDIT BHATIA` (normalized via `.strip().upper()`).
- **Target User:** Course Professor / Evaluator.
- **Visual Blanks Hint:** `____ ______` (`XXXX XXXXXX` — 4 letters, space, 6 letters).

---

## 2. Granular Implementation Audit

### 1. Submission Flow
The user submits credentials through the portal-mounted `ComputeAuthModal` (`web/src/components/auth/ComputeAuthModal.tsx`).
- The input field auto-uppercases input in real time.
- The form dispatches an HTTP `POST` to `/api/auth/verify-compute` with payload `{"password": "<INPUT>"}`.

### 2. Verification Location
Verification takes place in `backend/app.py`:
```python
COMPUTE_ACCESS_PASSWORD = "UDIT BHATIA"
COMPUTE_TOKEN_VALID = "UDIT_BHATIA_AUTHORIZED"

@app.post("/api/auth/verify-compute")
def verify_compute_auth_endpoint(req: ComputeAuthRequest):
    if req.password.strip().upper() == COMPUTE_ACCESS_PASSWORD:
        return {
            "success": True,
            "token": COMPUTE_TOKEN_VALID,
            "message": "Compute access authorized: Welcome, Prof. Udit Bhatia!",
        }
    raise HTTPException(
        status_code=401,
        detail="Invalid compute access password. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)",
    )
```

### 3. Backend Enforcement (Independent from Frontend)
- **Status:** **FULLY INDEPENDENT AND STRICTLY ENFORCED**.
- Access control is NOT merely a frontend UI toggle. The backend independently inspects every execution request using `require_compute_auth(request, body_token)`.
- Protected endpoints include:
  - `POST /api/chat/stream`: SSE token generation.
  - `POST /api/models/select`: Checkpoint loading into VRAM.
  - `POST /api/models/unload`: Checkpoint unloading.
  - `POST /api/tools/execute`: Direct Python sandbox and web research execution.
  - `POST /api/retrieval/compare`: Model inference comparison.
- Any request lacking valid authorization receives HTTP `401 Unauthorized` with detail containing the professor name blanks hint.

### 4. Session-Based vs Request-Based
- **Mechanism:** **Request-Based Authorization**.
- The server does not maintain server-side session cookies or stateful session tables.
- Instead, each HTTP request must present the authorization credential via:
  1. Header `X-Compute-Token: UDIT_BHATIA_AUTHORIZED` (or `UDIT BHATIA`).
  2. Header `Authorization: Bearer UDIT_BHATIA_AUTHORIZED` (or `Bearer UDIT BHATIA`).
  3. Body parameter `compute_token: "UDIT_BHATIA_AUTHORIZED"`.

### 5. Client-Side Credential Storage
- Upon receiving a successful response from `/api/auth/verify-compute`, the frontend persists the token string (`"UDIT_BHATIA_AUTHORIZED"`) in the browser's `localStorage` under the key:
  `qorin_compute_token`.
- The raw password (`"UDIT BHATIA"`) is cleared from memory after submission and is NOT stored in `localStorage`.
- Revoking access via the UI calls `localStorage.removeItem('qorin_compute_token')` and dispatches a window event `'qorin-compute-auth-changed'` to update UI badges.

### 6. Credential Visibility in Logs & API Responses
- **API Responses:** The token `"UDIT_BHATIA_AUTHORIZED"` is returned in the response body of `POST /api/auth/verify-compute`.
- **Server Logs:** In standard Uvicorn access logging, request headers (`X-Compute-Token`) and request bodies are NOT printed to stdout. Only HTTP method, path, and response status code are logged.
- **Experiment Traces:** In `backend/data/experiment_traces.jsonl`, generation parameters and conversation turns are logged, but the compute token header is NOT recorded.

### 7. Expiration & Revocation Behavior
- **Expiration:** The token does NOT currently expire automatically. It remains valid until the user clicks "Lock / Revoke Access" or clears browser cache.
- **Revocation:** Handled client-side by clearing `localStorage`.

### 8. Concurrent User Independence
- Because authorization is request-based via HTTP headers, concurrent users have completely independent authorization states:
  - Client A (with header) -> `200 OK`, generates tokens.
  - Client B (without header) -> `401 Unauthorized`, blocked from compute.
- However, note that once Client A triggers `model_manager.select_model("qorin-500m-sft-exp-f")`, that model remains loaded in shared VRAM on the server. If Client B queries `/api/health`, it will see that the 500M model is loaded, but Client B cannot invoke inference on it without credentials.

### 9. Direct API Access Without Activation
- Direct curl or Postman calls without the token fail immediately:
  ```bash
  $ curl -X POST http://127.0.0.1:8000/api/chat/stream -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hi"}]}'
  # Response: HTTP/1.1 401 Unauthorized
  # {"detail":"Personal compute access protected. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)"}
  ```

---

## 3. Security Vulnerabilities & Recommended Hardening

While the current implementation meets the academic requirement of gating access to the course professor, a production public deployment should address the following findings:

| Finding | Severity | Description | Recommended Correction (For Future Implementation) |
| :--- | :--- | :--- | :--- |
| **Static Bearer Token** | Medium | The token `"UDIT_BHATIA_AUTHORIZED"` is a static string hardcoded in `backend/app.py`. Anyone who obtains this string can reuse it indefinitely. | Replace static string with cryptographically signed JSON Web Tokens (JWT) signed by a server-side secret (`COMPUTE_SECRET_KEY`) with a 24-hour expiration (`exp`). |
| **No Expiration (TTL)** | Low | The client credential in `localStorage` has no time-to-live expiration. | Include an expiration timestamp in the token payload and reject tokens older than $N$ hours on the backend. |
| **Password Hardcoded in Source** | Medium | `COMPUTE_ACCESS_PASSWORD = "UDIT BHATIA"` is defined in plaintext in `backend/app.py`. | Store the authorized password or its argon2/bcrypt hash in an environment variable (`COMPUTE_PASSWORD_HASH`) instead of code. |
| **Rate Limiting Absent** | Low | `/api/auth/verify-compute` does not rate-limit repeated password attempts. | Add slow-down middleware (e.g. `slowapi` or 5 attempts per minute per IP) to prevent brute-forcing. |
