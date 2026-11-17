# QORIN Research Platform & Inference Backend
# Target Platform: Hugging Face CPU Basic (2 vCPU, 16 GB RAM)
# Exposed Port: 7860 (Hugging Face Docker standard)

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860 \
    DEVICE=cpu \
    QORIN_REMOTE_MODELS_ENABLED=true \
    QORIN_CACHE_DIR=/app/models_cache

WORKDIR /app

# Install minimal system packages for networking and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch build first to eliminate heavy CUDA dependencies (~2GB savings)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create models cache directory
RUN mkdir -p /app/models_cache

# Copy application source directories
COPY backend /app/backend
COPY modern_gpt /app/modern_gpt

# Set permissions for Hugging Face non-root container execution (uid 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
