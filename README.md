# QORIN: Tool-Augmented Small Language Models

**The Quiet Origin of Reasoning and Intelligence Networks**  
*An empirical investigation into training, tool augmentation, and grounded reasoning across 16M to 500M parameter language models.*

- **Research Website**: [https://imvynor.github.io/Qorin/](https://imvynor.github.io/Qorin/)
- **Model Checkpoints**: [https://huggingface.co/vynor/qorin-models](https://huggingface.co/vynor/qorin-models)
- **Inference Space**: [https://huggingface.co/spaces/vynor/stt4](https://huggingface.co/spaces/vynor/stt4)

---

## 1. Project Introduction

QORIN is an academic machine learning research project exploring how lightweight autoregressive transformers can acquire structured tool invocation, evidence-grounded synthesis, and reliable reasoning without the computational overhead of large-scale foundation models.

While frontier models rely on hundreds of billions of parameters to execute external actions, QORIN investigates the boundaries of sub-billion parameter architectures. Through custom byte-pair encoding (BPE-16k and BPE-32k), curated multi-domain pretraining corpora, multi-stage supervised fine-tuning (SFT), and dual-mode tool augmentation (deterministic Python sandboxing and real-time Web retrieval), the project evaluates when small models succeed, where parametric capacity breaks down, and how tool grounding mitigates factual hallucination.

The research investigates four architectural tiers:
- **16M Scale** (*ModernGPT-S 16M*): 6-layer foundational baseline used for early tokenizer validation and initial loss dynamics.
- **50M Scale** (*ModernGPT-S 50M*): Architectural ablations comparing Grouped-Query Attention (GQA 2:1) against deep-and-thin layer configurations.
- **100M Scale** (*ModernGPT-S 100M*): Pretrained foundational model (1.5B tokens, 512 context), extended to 4,096 tokens via RoPE linear frequency scaling ($\theta=10000, s=8.0$), and aligned across three successive SFT stages.
- **500M Scale** (*QORIN-500M*): Flagship 20-layer, 514M-parameter architecture with Grouped-Query Attention (GQA 6:1, 24 query heads, 4 KV heads) and 32k vocabulary, evaluated across active pretraining and targeted tool-augmented SFT branches.

---

## 2. Repository Architecture & Directory Structure

This repository contains the complete implementation across modeling, backend inference, tool execution, and the frontend interactive showcase:

```
Qorin/
├── backend/                  # FastAPI inference & tool orchestration server
│   ├── app.py                # Core REST & Server-Sent Events (SSE) API
│   ├── models/               # Model loading, memory caching & inference runner
│   ├── tools/                # Python sandbox executor & DuckDuckGo web researcher
│   ├── retrieval/            # Vector indexing, lexical BM25 & semantic reranking
│   ├── training/             # Live training dashboard telemetry & loss tracking
│   └── usage/                # Hugging Face & compute resource quota accounting
├── modern_gpt/               # Native QORIN PyTorch model architecture & training
│   ├── model/                # Modern Transformer: RoPE, GQA, SwiGLU, RMSNorm
│   ├── configs/              # Model configurations (16M, 50M, 100M, 500M)
│   ├── scripts/              # Training, SFT alignment, benchmark & export scripts
│   ├── evaluations/          # Evaluation harnesses & automated validation metrics
│   ├── diagnostics/          # Memory profiling, attention inspection & sanity checks
│   └── tests/                # Unit test suites & architecture assertions
├── gpt2/                     # Baseline standard GPT-2 reference implementation
├── web/                      # React + TypeScript research website & Learning Lab
│   ├── src/pages/            # Overview, Journey, Research, Experiments, Model, Harness, Lab
│   ├── src/components/       # Interactive attention visualizers, telemetry & modals
│   └── src/data/             # Pre-compiled experiment registries & benchmark data
├── dataset_registry.json     # Pretraining & SFT dataset composition records
├── experiment_registry.json  # Comprehensive log of 100M and 500M training runs
├── model_registry.json       # Canonical architecture hyperparameters & specs
├── references_registry.json  # Annotated bibliography and literature citations
├── start_harness.ps1         # One-click startup script (PowerShell)
└── start_harness.bat         # One-click startup script (Windows Command Prompt)
```

---

## 3. Quick Start: Running Locally

### Prerequisites
- Python 3.10+ (PyTorch, FastAPI, Uvicorn)
- Node.js 18+ and npm

### One-Click Launch (Windows)
```powershell
.\start_harness.ps1
```
This launches the FastAPI backend on `http://127.0.0.1:8000` and the Vite frontend on `http://127.0.0.1:5173/harness`.

### Manual Frontend Launch
```bash
cd web
npm install
npm run dev
```
Open `http://localhost:5173` in your browser. The frontend operates in **Static Showcase Mode** with full interactivity even without a local backend.

### Running Backend Tests
```bash
pytest backend/test_backend.py
```

### Running Frontend Verification Audit
```bash
npm test
```

---

## 4. Model Checkpoints & Cloud Inference

Due to GitHub file size limits, large binary model checkpoints (`.pt`, `.bin`) and raw multi-gigabyte corpora are hosted separately on the Hugging Face Hub:

- **Model Repository**: [vynor/qorin-models](https://huggingface.co/vynor/qorin-models)
  - `qorin-100m/`: Pretrained & SFT v3 checkpoints with 16k tokenizer assets.
  - `qorin-500m/`: 514M parameter tool-augmented checkpoints with 32k tokenizer assets.
- **Remote CPU Inference Space**: [vynor/stt4](https://huggingface.co/spaces/vynor/stt4)
  - Live Docker container running FastAPI backend on CPU Basic.
  - Automatically loads weights from `vynor/qorin-models` for real-time SSE chat streaming and tool execution.

---

## 5. Security & Compute Authorization

To protect personal compute resources from unauthenticated automated abuse while remaining accessible for academic review:
- Interactive model execution on the live backend is protected via password authorization.
- The password hint is provided in the interface: Professor's name in ALL CAPS (`____ ______` / `XXXX XXXXXX`).
- Valid authorization generates a cryptographically signed HMAC-SHA256 session token valid for 24 hours.

---

## 6. License

This repository is published for academic evaluation, reproducibility review, and educational research purposes.
