# Runtime Dependencies & Environment Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/RUNTIME_DEPENDENCIES.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Runtime Environment & Package Stack

The current development environment operates on Windows 11 with the following verified runtime specifications:

| Component | Audited Version / Value | Compatibility Notes |
| :--- | :--- | :--- |
| **Python** | `3.12.13` (64-bit AMD64) | Compatible with Python 3.10, 3.11, and 3.12. |
| **PyTorch** | `2.16.0.dev20261022+cu134` | Requires PyTorch 2.0+ for `torch.amp.autocast` and SDPA attention kernels. |
| **CUDA Driver & Toolkit** | CUDA 13.4 / Compute Capability 12.0 | Compatible with CUDA 11.8, 12.1, 12.4, and 12.6. |
| **Web Framework** | `fastapi==0.162.5`, `starlette==1.10.0` | Asynchronous ASGI request handling. |
| **ASGI Server** | `uvicorn==0.54.0` | Production HTTP / SSE server. |
| **Streaming** | `sse-starlette==3.7.0` | Server-Sent Events protocol implementation. |
| **Data Validation** | `pydantic==2.14.1` | V2 schema validation and serialization. |
| **Tokenizer Engine** | `tokenizers==0.23.3` | Hugging Face Rust-backed fast BPE tokenization. |
| **Model Utilities** | `transformers==5.22.2` | Used for base configuration schemas and benchmark comparisons. |
| **HTML / Scraping** | `beautifulsoup4==4.15.0` | Parses DuckDuckGo and Wikipedia search HTML. |
| **Math / ML Kernels** | `numpy==2.5.4`, `scipy==1.18.1`, `scikit-learn==1.9.1` | TF-IDF vectorization and cosine similarity. |

---

## 2. Dependency Manifest Findings

- **Missing Root `requirements.txt`:** The repository root currently does NOT contain a frozen `requirements.txt` or `pyproject.toml` (only a subfolder `modern-gpt-instruction-data/requirements.txt` exists).
- **Deployment Requirement:** Before containerization (Docker) or remote cloud deployment, a consolidated, production-ready `requirements.txt` containing only runtime packages (omitting Jupyter, notebook extensions, and debuggers) must be generated.

### Recommended Production `requirements.txt`:
```text
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.1.0
pydantic>=2.8.0
torch>=2.2.0
tokenizers>=0.19.0
beautifulsoup4>=4.12.0
numpy>=1.26.0
scikit-learn>=1.4.0
requests>=2.31.0
```

---

## 3. Environment Variable Inventory

| Variable | Required | Purpose | Secret? | Default / Fallback |
| :--- | :--- | :--- | :--- | :--- |
| `HF_TOKEN` | Optional | Authenticates to Hugging Face Hub for live account and inference quota monitoring (`/api/usage`). | **Yes** | `None` (Falls back to unauthenticated public API or Static Mode). |
| `ALLOWED_ORIGINS` | Optional | Comma-separated list of allowed CORS origins for cross-domain API calls. | No | `http://localhost:5173,http://127.0.0.1:5173,https://imvynor.github.io` |
| `COMPUTE_ACCESS_PASSWORD` | Optional | Overrides default evaluator password for personal compute gating. | **Yes** | `"UDIT BHATIA"` |
| `VITE_API_URL` / `VITE_API_BASE_URL` | Optional | Frontend environment variable directing the React client to the backend API base. | No | In dev: `"/api"`. In prod: `""` (Static Mode unless explicitly set). |
| `PORT` | Optional | Standard port configuration for cloud host runtime (e.g. Hugging Face Spaces, Render). | No | `8000` |
| `HOST` | Optional | Bind host for Uvicorn server. | No | `127.0.0.1` (local) / `0.0.0.0` (production). |

---

## 4. Platform Assumptions & Operating System Dependencies

The current code contains several platform-specific assumptions that must be addressed during cloud containerization:

1. **Windows-Specific Filesystem Defaults in Python Sandbox:**
   - In `backend/tools/python_sandbox.py:84`:
     ```python
     clean_env = {
         "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"),
         "PATH": os.environ.get("PATH", ""),
         "PYTHONPATH": "",
         "TEMP": str(run_dir),
         "TMP": str(run_dir),
     }
     ```
   - *Impact on Linux Deployment:* `C:\Windows` does not exist on Linux containers (Docker / Cloud Run / HF Spaces). On Linux, `clean_env` must provide `PATH="/usr/local/bin:/usr/bin:/bin"`.
2. **Subprocess Window Creation Flag:**
   - In `backend/tools/python_sandbox.py:109`:
     ```python
     creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
     ```
   - *Status:* Already guarded by `if os.name == "nt" else 0`. Safe for Linux.
3. **Repository Path Resolution:**
   - In `backend/app.py:12` and model providers:
     ```python
     REPO_ROOT = Path(__file__).resolve().parent.parent
     ```
   - *Status:* Resolves reliably across all POSIX and Windows filesystems.
