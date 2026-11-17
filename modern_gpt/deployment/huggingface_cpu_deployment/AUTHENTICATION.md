# Evaluator Authentication & Compute Gate Architecture

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/huggingface_cpu_deployment/AUTHENTICATION.md`  
**Audit Date:** November 17, 2026  
**Status:** Multi-Layered Protection Verified; Strict Backend Enforcement  

---

## 1. Authentication Design & Workflow

To maintain open access for the public to review research monographs, architecture visualizers, training charts, and experiment archives, while strictly restricting interactive model inference and Python execution to the course evaluator:

```
[Evaluator on Website]
       │
       ▼
1. User Clicks "Compute Access" in Navbar / Mobile Drawer / Harness
       │
       ▼
2. Portal Modal Presents Blanks Hint:
   "Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)"
       │
       ▼
3. User Submits: "UDIT BHATIA"
       ├──► Backend POST /api/auth/verify-compute
       │      Validates password against server secret.
       │      Returns: {"success": true, "token": "UDIT_BHATIA_AUTHORIZED"}
       ▼
4. Frontend Saves Token to localStorage['qorin_compute_token']
       │
       ▼
5. All Subsequent Requests Attach Header:
   X-Compute-Token: UDIT_BHATIA_AUTHORIZED
       │
       ▼
6. Live Inference & Closed-Loop Tool Execution Unlocked
```

---

## 2. Independent Backend Authorization Enforcement

The backend does not rely on client-side state. Every compute-intensive or sandbox execution endpoint independently enforces authorization:

```python
def require_compute_auth(request: Request, body_token: Optional[str] = None):
    token = request.headers.get("X-Compute-Token") or request.headers.get("Authorization") or body_token
    if check_compute_auth(token):
        return True
    raise HTTPException(
        status_code=401,
        detail="Personal compute access protected. Hint: Professor's name in ALL CAPS (____ ______ / XXXX XXXXXX)",
        headers={"WWW-Authenticate": "Bearer"},
    )
```

### Protected Endpoints:
- `POST /api/chat/stream`: Live token streaming.
- `POST /api/models/select`: Checkpoint loading into RAM/VRAM.
- `POST /api/models/unload`: Checkpoint unloading.
- `POST /api/retrieval/compare`: Empirical A/B inference comparison.
- `POST /api/tools/execute`: Direct Python sandbox and web search execution.

---

## 3. Security & Privacy Guarantees

1. **No Credentials in Browser Bundles:**
   - The React frontend contains zero hardcoded passwords or server secrets.
   - The Vite build output on GitHub Pages contains no Hugging Face tokens or evaluator passwords.
2. **No Credentials in Server Access Logs:**
   - Standard Uvicorn logs record only method, path, and response code. Headers (`X-Compute-Token`, `Authorization`) are omitted.
3. **Server-Side Model Access:**
   - If model checkpoints are stored in a private Hugging Face repository, the backend downloads them using its internal container environment variable `HF_TOKEN`.
   - The frontend and visitors never receive or see the `HF_TOKEN`.
4. **Session Termination & Logout:**
   - The `ComputeAuthModal` provides a "Lock / Revoke Access" button that immediately deletes the token from browser `localStorage` and resets the UI state to `Compute: Locked`.
