# Backend Container Packaging for Cloud Deployment

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/huggingface_cpu_deployment/BACKEND_PACKAGING.md`  
**Audit Date:** November 17, 2026  
**Status:** Container Manifests & Production Dependencies Prepared  

---

## 1. Containerization Strategy

To preserve the complete FastAPI REST and Server-Sent Events (SSE) contract without introducing bloated wrappers or Gradio interface overhead, the backend is packaged as a standard **Docker container** targeting Hugging Face Spaces port `7860`.

### Production Dockerfile (`Dockerfile.huggingface`):
```dockerfile
FROM python:3.12-slim

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal OS dependencies for compilation and networking
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Establish application directory
WORKDIR /app

# Copy production requirements and install
COPY modern_gpt/deployment/huggingface_cpu_deployment/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy backend source tree, models, tools, and tokenizers
COPY backend /app/backend
COPY modern_gpt/model /app/modern_gpt/model
COPY modern_gpt/data /app/modern_gpt/data
COPY modern_gpt/artifacts /app/modern_gpt/artifacts
COPY modern_gpt/canonical_model_manifest.json /app/modern_gpt/canonical_model_manifest.json

# Ensure data and cache directories exist with appropriate permissions
RUN mkdir -p /app/backend/data/sandbox /app/backend/data/cache && chmod -R 777 /app/backend/data

# Default runtime configuration
ENV PORT=7860
ENV HOST=0.0.0.0
ENV DEVICE=cpu
ENV DEFAULT_MODEL_ID=qorin-100m-sft-v3
ENV ALLOWED_ORIGINS=https://imvynor.github.io,http://localhost:5173,http://localhost:4173

# Expose standard Hugging Face Spaces web port
EXPOSE 7860

# Launch Uvicorn ASGI server
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
```

---

## 2. Production Dependencies (`requirements-prod.txt`)

Derived by stripping non-runtime development tools (Jupyter, debuggers, Matplotlib, Pytest) from the active virtualenv:

```text
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
sse-starlette>=2.1.0
pydantic>=2.8.0
torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
tokenizers>=0.19.0
beautifulsoup4>=4.12.0
numpy>=1.26.0
scikit-learn>=1.4.0
requests>=2.31.0
huggingface_hub>=0.23.0
```

*(Note: Specifying `--index-url https://download.pytorch.org/whl/cpu` reduces the Docker container image from >4 GB to <800 MB by omitting massive CUDA binaries that are unused on CPU-basic).*

---

## 3. Environment Variable Injection Map

| Variable | Recommended Production Value | Description |
| :--- | :--- | :--- |
| `PORT` | `7860` | Hugging Face Spaces incoming web port. |
| `HOST` | `0.0.0.0` | Binds server to all container network interfaces. |
| `DEVICE` | `cpu` | Enforces pure CPU inference without probing CUDA. |
| `DEFAULT_MODEL_ID` | `qorin-100m-sft-v3` | Responsive CPU default model. |
| `ALLOWED_ORIGINS` | `https://imvynor.github.io` | Permits cross-origin browser fetches from GitHub Pages. |
| `COMPUTE_ACCESS_PASSWORD` | `UDIT BHATIA` | Evaluator password required to unlock inference. |
| `HF_TOKEN` | *(Configured in HF Space Secrets)* | Secret token for querying Hugging Face usage APIs. |

---

## 4. Preservation of FastAPI & SSE Contracts

The packaged container:
1. Retains all 17 REST endpoints unchanged.
2. Emits raw Server-Sent Events (`text/event-stream`) directly through port 7860.
3. Automatically maps to `https://<space-name>.hf.space/api/...`.
4. Executes isolated Python sandboxing in `/tmp` using Linux standard paths.
