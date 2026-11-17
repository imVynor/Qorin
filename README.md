# QORIN: Tool-Augmented Small Language Models

**The Quiet Origin of Reasoning and Intelligence Networks**  
*An empirical investigation into training, tool augmentation, and grounded reasoning across 16M to 500M parameter language models.*

Research Website: *(Deployment in progress — public URL will be published upon GitHub Pages release)*

---

## 1. Project Introduction

QORIN is an academic machine learning research project exploring how lightweight autoregressive transformers can acquire structured tool invocation, evidence-grounded synthesis, and reliable reasoning without the computational overhead of large-scale foundation models.

While modern frontier models rely on hundreds of billions of parameters to execute external actions, QORIN investigates the boundaries of sub-billion parameter architectures. Through custom byte-pair encoding (BPE-16k and BPE-32k), curated multi-domain pretraining corpora, multi-stage supervised fine-tuning (SFT), and dual-mode tool augmentation (deterministic Python sandboxing and real-time Web retrieval), the project evaluates when small models succeed, where parametric capacity breaks down, and how tool grounding mitigates factual hallucination.

The research investigates four architectural tiers:
- **16M Scale** (*ModernGPT-S 16M*): 6-layer foundational baseline used for early tokenizer validation and initial loss dynamics.
- **50M Scale** (*ModernGPT-S 50M*): Architectural ablations comparing Grouped-Query Attention (GQA 2:1) against deep-and-thin layer configurations.
- **100M Scale** (*ModernGPT-S 100M*): Pretrained foundational model (1.5B tokens, 512 context), extended to 4,096 tokens via RoPE linear frequency scaling ($\theta=10000, s=8.0$), and aligned across three successive SFT stages.
- **500M Scale** (*QORIN-500M*): Flagship 20-layer, 514M-parameter architecture with Grouped-Query Attention (GQA 6:1, 24 query heads, 4 KV heads) and 32k vocabulary, evaluated across active pretraining and targeted tool-augmented SFT branches.

The complete research narrative, interactive exploration labs, dataset registries, architectural specifications, and empirical benchmark visualizations are presented through the project website:  
*(The public research website is currently being prepared for deployment. The live URL will be published here once GitHub Pages deployment is complete.)*

---

## 2. Repository Scope (Frontend-Only Release)

This repository contains the source code for the public **QORIN Research Showcase and Interactive Exploration Platform**.

The repository is organized **frontend-first** to allow researchers, educators, and reviewers to inspect the research interface, mathematical derivations, experiment registries, and interactive modules locally without requiring GPU compute or multi-gigabyte checkpoint downloads.

### Included in this Release
- **Research Presentation & Visualizations**: Structured technical presentation of pretraining curricula, tokenization strategies, fine-tuning protocols, loss curves, and ablation figures.
- **Interactive Architecture & Attention Labs**: Nine real-time, client-side visualizers implemented in the Learning Lab:
  1. *Byte-Pair Encoding (BPE)*: Live token boundary segmentation and vocabulary rank viewer.
  2. *Scaled Dot-Product Attention*: Interactive Query-Key-Value projection and causal masking visualizer.
  3. *Rotary Position Embeddings (RoPE)*: Dynamic 2D frequency rotation and complex vector visualizer.
  4. *Grouped-Query Attention (GQA)*: Multi-Head vs. Multi-Query vs. Grouped-Query KV-cache comparison.
  5. *Transformer Decoder Block*: End-to-end dataflow through RMSNorm, SwiGLU FFN, and residual streams.
  6. *Pretraining Dynamics*: Empirical scaling laws, loss trajectories, and learning-rate schedules.
  7. *Supervised Fine-Tuning (SFT)*: Loss masking and ChatML special-token formatting.
  8. *Retrieval-Augmented Generation (RAG)*: Non-parametric vector search and context injection.
  9. *Model Evaluation*: Cross-entropy perplexity, benchmark metrics, and win-rate analysis.
- **Pre-Compiled Research Registries**: Verified JSON records detailing pretraining datasets (`dataset_registry.json`), experiment configurations (`experiment_registry.json`), model specifications (`model_registry.json`), and bibliography (`references_registry.json`).
- **Research Harness Interface**: Client-side workbench featuring parameter controls, prompt inspection, and a resilient **Static Showcase Mode** that functions out-of-the-box without a live backend.

### Decoupled / Not Included in this Release
- **Model Checkpoints**: PyTorch weight files (checkpoints up to 3.0–3.5 GB for 500M models) are hosted separately due to repository size limits.
- **Raw Training Corpora**: Multi-gigabyte JSONL datasets and tokenized binary shards are excluded.
- **Inference Backend**: The FastAPI server and GPU runtime operate as a separate service and are not included in this repository.

---

## 3. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/imVynor/Qorin.git
cd Qorin
```

---

## 4. Running Locally

### Prerequisites
- [Node.js](https://nodejs.org/) (version 18 or higher recommended)
- `npm` (bundled with Node.js)

### Quick Start
Install dependencies and launch the local development server:

```bash
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

By default, the application runs in **Static Showcase Mode**, rendering all interactive labs, architecture visualizers, and experiment archives client-side with zero manual configuration. 

If a local QORIN inference backend is running on `http://127.0.0.1:8000`, the Vite development server automatically proxies `/api` requests to it out-of-the-box. To configure a different or remote backend URL, set `VITE_API_URL` in `web/.env.local`.

### Verification Tests
To run the automated reproducibility audit verifying all JSON registries, static assets, and client routes:

```bash
npm test
```

---

## 5. Live Showcase & GitHub Pages Deployment

The QORIN frontend showcase is optimized for static hosting on **GitHub Pages** with complete client-side routing, interactive modules, and offline-resilient fallbacks.

- **Live URL**: [https://imvynor.github.io/Qorin/](https://imvynor.github.io/Qorin/)

### Automated Deployment (GitHub Actions)
The repository includes a GitHub Actions workflow (`.github/workflows/deploy-pages.yml`) that automatically builds and publishes the website on every push to `main`.

To enable native GitHub Actions deployment on your repository:
1. Navigate to **Settings** > **Pages** in your GitHub repository.
2. Under **Build and deployment** > **Source**, select **GitHub Actions** (or select **Deploy from a branch** and choose the `gh-pages` branch).
3. Push to `main` or trigger the workflow manually from the **Actions** tab.

### Manual / One-Command Deployment (gh-pages)
You can also build and deploy directly to the `gh-pages` branch from your local terminal:
```bash
npm run deploy
```
This runs `npm run build:pages` (bundling with the `/Qorin/` base path and `.nojekyll` / `404.html` SPA routing support) and pushes the distribution to the `gh-pages` branch.

---

## 6. Model Weights and Checkpoints

Model checkpoint weights and raw pretraining corpora are **not hosted within this Git repository** due to standard repository size limits.

Checkpoint weights, model cards, and curated evaluation splits will be published and linked separately in forthcoming model releases.

---

## 7. Backend Status and Roadmap

The QORIN inference engine and tool-dispatch harness (FastAPI, PyTorch CUDA inference provider, isolated Python subprocess sandbox, and DuckDuckGo web retrieval grounding) operate as a decoupled backend service.

Detailed instructions for self-hosting the inference backend, setting up the PyTorch GPU execution environment, and configuring local model providers will be released in an upcoming update alongside model checkpoint availability.

---

## 8. License

This project is currently released for academic review and educational research purposes. Formal license terms will be designated upon repository finalization.
