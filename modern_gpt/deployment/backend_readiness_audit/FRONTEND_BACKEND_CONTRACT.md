# Frontend-Backend Contract & Integration Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/FRONTEND_BACKEND_CONTRACT.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. API Base URL Resolution & Static Showcase Guarantee

The API configuration in `web/src/services/api.ts` enforces a strict production contract:

```typescript
const configuredEnvUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;

export const isBackendConfigured: boolean = Boolean(
  configuredEnvUrl || (import.meta.env.DEV && !import.meta.env.VITE_STATIC_MODE)
);

export const API_BASE: string = configuredEnvUrl
  ? configuredEnvUrl.replace(/\/$/, '')
  : (import.meta.env.DEV && !import.meta.env.VITE_STATIC_MODE ? '/api' : '');
```

### Verified Behaviors:
1. **Local Development (`import.meta.env.DEV == true`):**
   - Defaults to `/api`.
   - Vite proxy in `web/vite.config.ts` transparently forwards requests to `http://127.0.0.1:8000`.
2. **Production on GitHub Pages (`https://imvynor.github.io/Qorin/`):**
   - If `VITE_API_URL` is NOT set at build time:
     - `isBackendConfigured` evaluates strictly to `false`.
     - `API_BASE` evaluates to `""`.
     - **Guarantee:** The frontend **NEVER** makes broken relative calls to `https://imvynor.github.io/api/...`.
     - The application gracefully switches to **Static Showcase Mode**.
   - If `VITE_API_URL` IS set (e.g. `https://qorin-api.yourdomain.com` or Cloudflare tunnel URL):
     - `isBackendConfigured` evaluates to `true`.
     - Cross-origin requests are dispatched directly to the configured remote endpoint.
     - Backend `CORSMiddleware` permits `https://imvynor.github.io`.

---

## 2. API Contract Schema Alignment

| Contract Field / Type | Frontend (`web/src/types/`) | Backend (`backend/app.py`) | Verified Match |
| :--- | :--- | :--- | :--- |
| **Model Info** | `ModelInfo` (`id`, `name`, `parameters`, `context_length`, `architecture`, `status`, `is_native`, `vram_estimate_mb`) | `ModelMetadata` dict returned by `/api/models` | **Identical** |
| **Telemetry** | `TelemetryInfo` (`active_model_id`, `qorin.cuda_available`, `qorin.vram_allocated_mb`, `qorin.device_name`) | `model_manager.get_telemetry()` returned by `/api/health` | **Identical** |
| **Chat Request** | `ChatRequest` (`model_id`, `messages`, `temperature`, `top_p`, `top_k`, `max_new_tokens`, `system_prompt`, `retrieval_mode`, `web_research_enabled`, `python_sandbox_enabled`) | `ChatRequest(BaseModel)` at `/api/chat/stream` | **Identical** |
| **Compute Auth** | `verifyComputeAccess(password)` -> `POST /api/auth/verify-compute` (`{"password": "..."}`) | `ComputeAuthRequest(BaseModel)` returning `{"success": true, "token": "..."}` | **Identical** |
| **Provider Usage** | `UsageSummaryResponse` (`providers`, `status`, `account_name`, `plan_tier`, `inference_credits`, `other_resources`) | `UsageSummaryResponse` Pydantic model at `/api/usage` | **Identical** |
| **Comparison** | `compareRetrieval(modelId, prompt)` -> `POST /api/retrieval/compare` | `CompareRetrievalRequest` returning `{without_retrieval, with_retrieval, latency_overhead_ms}` | **Identical** |

---

## 3. Streaming Protocol & Event Handling

The frontend `streamChat()` function (`web/src/services/api.ts:167`) consumes the SSE wire protocol:
- Reads the response body chunk-by-chunk using `TextDecoder` and split buffer lines.
- Intercepts event types:
  - `event: retrieval`: Calls `callbacks.onRetrieval(data)`.
  - `event: tool_start`: Calls `callbacks.onToolStart(data)`.
  - `event: tool_result`: Calls `callbacks.onToolResult(data)`.
  - `event: token`: Calls `callbacks.onToken(data)` (updates incremental live token text and TTFT).
  - `event: done`: Calls `callbacks.onDone(data)` (finalizes assistant message, records latency).
  - `event: error`: Calls `callbacks.onError(data)`.
- If client unmounts or user clicks Stop, `controller.abort()` sends an abort signal, gracefully disconnecting the stream.

---

## 4. Evaluator Inference Activation Workflow

```
[User on Website]
       │
       ├──► 1. Opens Options Popup (or Mobile Drawer / Hardware Telemetry Pill)
       │       Displays status: "Compute Access: Protected (Restricted)"
       ▼
[Clicks "Compute Access"]
       │
       ▼
2. Portal-Mounted ComputeAuthModal Opens
       │  - Displays "Workstation GPU Gate Active"
       │  - Displays Password Hint: Professor's name in ALL CAPS
       │  - Displays Visual Blanks: [ ____ ] [ ______ ] (XXXX XXXXXX)
       ▼
3. User Enters Password
       ├── IF INCORRECT:
       │     API returns 401 -> Modal renders red alert with blanks hint.
       └── IF "UDIT BHATIA":
             API returns 200 with token "UDIT_BHATIA_AUTHORIZED".
             Modal renders: "Welcome, Prof. Udit Bhatia! Personal GPU inference unlocked."
             Frontend saves token to localStorage['qorin_compute_token'].
             Dispatches 'qorin-compute-auth-changed' event.
       ▼
4. Subsequent Requests
       │  All API calls automatically attach:
       │  Header: X-Compute-Token: UDIT_BHATIA_AUTHORIZED
       │  Body: compute_token: UDIT_BHATIA_AUTHORIZED
       ▼
5. Instantaneous Live GPU Inference
       Harness top bar displays: "Compute: Authorized (Prof. Udit Bhatia)"
       User can click "Lock / Revoke Access" at any time to re-gate compute.
```

---

## 5. Static Showcase Mode Graceful Degradation

If the inference backend is offline, asleep, or running in an unconfigured static deployment (such as standard GitHub Pages):
1. **Zero Console Errors:** Health checks catch connection errors cleanly and set `isBackendAvailable(false)` without throwing unhandled exceptions.
2. **Offline Telemetry Badge:** Hardware strip displays `Static Showcase Mode`.
3. **Fallback Model Specs:** The interface populates the model selector with `FALLBACK_MODELS` (16M, 50M, 100M, 500M) allowing inspection of parameters, context lengths, and architectures.
4. **Interactive Chat Informational Notice:** Submitting a chat query while backend is offline renders a polite guidance message explaining that the live PyTorch runtime is offline and pointing the evaluator to local startup instructions.
5. **Research Monograph & Observability:** The Research Journey, Architecture Lineage Visualizer, Learning Lab, and Experiments Archive operate 100% client-side with complete historical data.
