---
title: QORIN Research Platform & Inference Backend
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: cpu-basic
---

# QORIN — Research Platform & Inference Backend

Unified backend for the **QORIN** (*The Quiet Origin of Reasoning and Intelligence Networks*) open-weights SLM research project.

## Architecture

- **Runtime:** CPU Basic (2 vCPU, 16 GB RAM) on Hugging Face Spaces.
- **Inference Engine:** Native PyTorch float32 execution with KV-caching.
- **Model Distribution:** Automated, verified on-demand checkpoint hydration from [`vynor/qorin-models`](https://huggingface.co/vynor/qorin-models) with cryptographic SHA-256 verification.
- **Frontend Client:** React research showcase deployed on GitHub Pages at [`https://imvynor.github.io/Qorin/`](https://imvynor.github.io/Qorin/).
- **Evaluation Security:** Password-protected evaluator compute activation with cryptographic HMAC-SHA256 session tokens.

## Primary Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Container healthcheck and telemetry status |
| `GET` | `/api/models` | Available model scales and configurations |
| `POST` | `/api/auth/verify-compute` | Evaluator authentication and session issuance |
| `POST` | `/api/models/select` | Load verified checkpoint into CPU memory |
| `POST` | `/api/chat/stream` | Server-Sent Events (SSE) token generation stream |
| `POST` | `/api/retrieval/query` | Non-parametric lexical/embedding retriever |
| `POST` | `/api/tools/execute` | Isolated mathematical and meteorological sandbox |

## Deployment Notes

- Checkpoints are loaded lazily upon model selection to ensure immediate cold-start container responsiveness.
- Space sleeps after 48 hours of inactivity. The GitHub Pages frontend includes automatic wakeup polling and container spin-up detection.
