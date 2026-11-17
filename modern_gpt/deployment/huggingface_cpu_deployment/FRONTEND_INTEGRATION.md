# Frontend-to-Backend Remote Integration

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/huggingface_cpu_deployment/FRONTEND_INTEGRATION.md`  
**Audit Date:** November 17, 2026  
**Status:** Architecture Configured; Static Fallback Maintained  

---

## 1. Remote Endpoint Configuration Contract

The frontend React application (`web/src/services/api.ts`) is designed to communicate with any remotely deployed backend via Vite environment variables:

```typescript
const configuredEnvUrl = import.meta.env.VITE_API_URL || import.meta.env.VITE_API_BASE_URL;

export const isBackendConfigured: boolean = Boolean(
  configuredEnvUrl || (import.meta.env.DEV && !import.meta.env.VITE_STATIC_MODE)
);

export const API_BASE: string = configuredEnvUrl
  ? configuredEnvUrl.replace(/\/$/, '')
  : (import.meta.env.DEV && !import.meta.env.VITE_STATIC_MODE ? '/api' : '');
```

### Production Integration Target:
When connecting to a remote Hugging Face backend (e.g. `https://vynor-stt4.hf.space` or custom domain):
```bash
VITE_API_URL="https://vynor-stt4.hf.space" npm --prefix web run build
```
- The build produces static assets that target `https://vynor-stt4.hf.space` directly.
- Cross-origin resource sharing (CORS) is authorized on the backend via `ALLOWED_ORIGINS="https://imvynor.github.io"`.

---

## 2. Preserved UI & Interactive Components

| Component | Integration Behavior | Offline / Sleeping Fallback |
| :--- | :--- | :--- |
| **Navbar Options Popup** | Displays live compute status (`Protected` vs `Authorized: Prof. Udit Bhatia`). | Retains full navigation and options. |
| **Usage & Credits Modal** | Queries `GET /api/usage` for live Hugging Face account and quota data. | Falls back to `FALLBACK_USAGE_REPORT` (Static Mode notice). |
| **Compute Gate Modal** | Submits evaluator password to `POST /api/auth/verify-compute`. | In static mode, evaluates locally for demonstration. |
| **Harness Chat Interface** | Streams tokens via `POST /api/chat/stream`. Attaches `X-Compute-Token`. | Shows informative notice explaining how to connect live backend. |
| **Tool Execution Visualizers** | Renders expandable badges for Web sources, Python execution stdout, and weather conditions. | Fully rendered from historical traces. |
| **Training Observability** | Queries `/api/training/runs` and downsampled loss metrics. | Populates complete historical training runs. |
| **Research Monograph & Lab** | 100% client-side React interactive modules. | Operates with zero backend dependencies. |

---

## 3. Cold-Start & Awakening UX Handling

Hugging Face Spaces on the free tier automatically enter a `SLEEPING` state after 48 hours of inactivity (`gcTimeout: 172800` seconds).

### Handling Wake-Up Latency:
1. **Initial Connection Delay:** When an evaluator first visits the website and the Space is sleeping, the initial HTTP request to `/api/health` initiates container awakening.
2. **Boot Duration:** Standard Docker container boot and PyTorch initialization on `cpu-basic` takes **35 to 65 seconds**.
3. **User Guidance Notice:**
   - In `web/src/pages/Harness.tsx`, when the backend returns a 503 Service Unavailable (Hugging Face waking state) or times out during initial sync:
   - The UI displays:
     `"Backend container is currently waking from sleep. Please wait approximately 45 seconds and click Sync in the top bar."`
   - Prevents confusing container wake-up with an application failure.
