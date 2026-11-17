# QORIN: Tool-Augmented Small Language Models

**The Quiet Origin of Reasoning and Intelligence Networks**  
*An empirical investigation into parameter scaling, autonomous tool augmentation, and grounded reasoning across 16M to 500M decoder-only Transformers.*

- **Live Research Website**: [https://imvynor.github.io/Qorin/](https://imvynor.github.io/Qorin/)
- **Hugging Face Model Repository**: [https://huggingface.co/vynor/qorin-models](https://huggingface.co/vynor/qorin-models)
- **Hugging Face CPU Inference Space**: [https://huggingface.co/spaces/vynor/stt4](https://huggingface.co/spaces/vynor/stt4)
- **Primary Research Question**: *"How does increasing the capacity of a small decoder-only Transformer affect conversational LM performance, and can external retrieval compensate for a smaller model?"*

---

## 1. Project Overview

QORIN is an academic machine learning research investigation exploring the interaction between parametric model capacity and external deterministic tools. While contemporary state-of-the-art systems rely on hundreds of billions of parameters to execute external actions, QORIN isolates and evaluates sub-billion parameter decoder-only Transformers.

The central empirical question is:
> **Do external tools actually improve the performance of our language model when the underlying model checkpoint is held constant?**

Across controlled experiments holding the identical checkpoint constant, tool-augmented execution improved end-to-end task success from **43.0% (Parametric Only)** to **64.0% (Autonomous Tools Enabled)** on 100 held-out tasks—a **+21.0% absolute improvement (+48.8% relative)** confirmed statistically significant via McNemar's paired test ($\chi^2 = 12.9032, p = 0.000328$, odds ratio $5.2$).

---

## 2. Model Family & Availability Matrix

All four advertised model configurations have been physically verified on disk, validated for CPU inference compatibility, and synced to the Hugging Face Model Hub:

| Model ID (API) | Display Name | Parameters (Runtime) | Context | Architecture | Checkpoint Hash (SHA-256) | Availability Status |
|---|---|---:|---:|---|---|---|
| `qorin-100m-sft-v3` | **QORIN-100M SFT v3** | 100,401,408 | 4,096 | 14L-12Q-4KV (GQA 3:1), SwiGLU, RoPE $s=8.0$ | `d771ea9f...` | **VERIFIED** (Production SFT) |
| `qorin-100m-base` | **QORIN-100M Baseline** | 100,401,408 | 512 | 14L-12Q-4KV (GQA 3:1), SwiGLU, RoPE $s=1.0$ | `bded509c...` | **VERIFIED** (Base Pretrained) |
| `qorin-500m-sft-exp-f` | **QORIN-500M SFT Exp F** | 514,389,504 | 2,048 | 20L-24Q-4KV (GQA 6:1), SwiGLU, RoPE $s=1.0$ | `bb5b7091...` | **VERIFIED** (100-Step SFT Pilot) |
| `qorin-500m-pretrain-pilot`| **QORIN-500M Pretrain Pilot** | 514,389,504 | 2,048 | 20L-24Q-4KV (GQA 6:1), SwiGLU, RoPE $s=1.0$ | `28ac47b0...` | **VERIFIED** (Unconverged Pretrain) |

*Full parameter accounting, serialized vs tied counts, and VRAM footprints are detailed in [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md) and [`docs/models/MODEL_REGISTRY.md`](docs/models/MODEL_REGISTRY.md).*

---

## 3. Implementation Source-Code Map

Every research component links directly to authoritative source files inside this repository:

| Research Component | Verified Source File | Architectural Purpose |
|---|---|---|
| **Transformer Backbone** | [`modern_gpt/model/gpt.py`](modern_gpt/model/gpt.py) | Full causal autoregressive decoder implementation |
| **Attention & GQA** | [`modern_gpt/model/attention.py`](modern_gpt/model/attention.py) | Causal Grouped-Query Attention with KV-caching |
| **Rotary Embeddings (RoPE)** | [`modern_gpt/model/rope.py`](modern_gpt/model/rope.py) | Complex rotary position embedding with linear frequency scaling |
| **RMSNorm** | [`modern_gpt/model/norm.py`](modern_gpt/model/norm.py) | Pre-normalization root mean square layer norm |
| **SwiGLU MLP** | [`modern_gpt/model/mlp.py`](modern_gpt/model/mlp.py) | Gated Swish feed-forward expansion layers |
| **Model Configuration** | [`modern_gpt/model/config.py`](modern_gpt/model/config.py) | Strict dataclass hyperparameter schemas |
| **Pretraining Pipeline** | [`modern_gpt/train.py`](modern_gpt/train.py) | Multi-epoch causal training loop with learning-rate scheduling |
| **100M SFT Pipeline** | [`modern_gpt/train_sft_4k.py`](modern_gpt/train_sft_4k.py) | Supervised fine-tuning with 4K context extension |
| **500M Multi-Turn SFT Pilot** | [`modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/train_pilot_stage14.py`](modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/train_pilot_stage14.py) | Controlled 100-step multi-turn tool chaining pilot |
| **BPE Tokenizers** | [`modern_gpt/artifacts/tokenizer_16k_v2/tokenizer.json`](modern_gpt/artifacts/tokenizer_16k_v2/tokenizer.json) & [`32k`](modern_gpt/artifacts/tokenizer_32k/tokenizer.json) | Production BPE-16k and BPE-32k vocabulary files |
| **Controlled Ablation Test** | [`modern_gpt/tests/test_final_tool_ablation.py`](modern_gpt/tests/test_final_tool_ablation.py) | Rigorous paired statistical evaluation harness |
| **Blind Capability Eval** | [`modern_gpt/experiments/phase2_experiment_f/stage_15_blind_evaluation/run_stage_15_evaluation.py`](modern_gpt/experiments/phase2_experiment_f/stage_15_blind_evaluation/run_stage_15_evaluation.py) | Greedy held-out capability grader across Corpora A, B, and C |
| **Tool Orchestrator** | [`backend/tools/orchestrator.py`](backend/tools/orchestrator.py) | Deterministic state machine coordinating tool calls & execution |
| **Python Sandbox Contract** | [`backend/tools/python_contract.py`](backend/tools/python_contract.py) | Sandboxed AST parsing and timeout-protected code execution |
| **Web Research Engine** | [`backend/tools/web_researcher.py`](backend/tools/web_researcher.py) | Live DuckDuckGo search integration with snippet extraction |
| **Grounding Validator** | [`backend/tools/grounding_validator.py`](backend/tools/grounding_validator.py) | Anti-hallucination verification against tool observation buffers |
| **CPU Inference Provider** | [`backend/models/qorin_provider.py`](backend/models/qorin_provider.py) | Zero-GPU lazy loading and streaming causal token generation |

---

## 4. Research Documentation & Audit Index

- [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md): Authoritative model inventory, parameter counts, tensor dimensions, and verified SHA-256 hashes.
- [`docs/models/500M_SFT_EXPERIMENT_F.md`](docs/models/500M_SFT_EXPERIMENT_F.md): Lineage, 100-update pilot nature, Stage 13 dataset composition, and evaluation of the 500M SFT model.
- [`docs/models/MODEL_REGISTRY.md`](docs/models/MODEL_REGISTRY.md): Backend inference specification, lazy loading, memory safety, and CPU execution profile.
- [`docs/data/DATASET_REGISTRY.md`](docs/data/DATASET_REGISTRY.md): Canonical dataset catalog and resolution of the 100 vs 90 benchmark count discrepancy.
- [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md): Traceability matrix mapping every website claim to its exact raw experiment artifact.
- [`docs/REFERENCES.md`](docs/REFERENCES.md): Academic citations for architectures, algorithms, datasets, and statistical tests.
- [`docs/TEAM_CONTRIBUTIONS.md`](docs/TEAM_CONTRIBUTIONS.md): Team contribution documentation detailing technical division of responsibilities (Anish Raj: Model, Training, Evaluation, Fine-Tuning; Jnandeep: Harness, Retrieval, Dataset Preparation, Tool Adaptation).
- [`docs/FINAL_REPRODUCIBILITY_AND_RELEASE_REPORT.md`](docs/FINAL_REPRODUCIBILITY_AND_RELEASE_REPORT.md): Comprehensive final audit sign-off.

---

## 5. Local Setup & Reproduction Instructions

### 5.1 Prerequisites
- Python 3.11 or 3.12
- Node.js 18+ and npm
- Git

### 5.2 Frontend Installation & Local Server
```bash
# Clone the repository
git clone https://github.com/imVynor/Qorin.git
cd Qorin

# Install web dependencies
cd web
npm install

# Start Vite development server
npm run dev
# The website will be available at http://localhost:5173
```

### 5.3 Backend Setup & Offline Inference
```bash
# From repository root, create and activate a Python virtual environment
python -m venv .venv

# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install backend dependencies
pip install -r backend/requirements.txt

# Run the backend server
python backend/app.py
# Backend API will be available at http://localhost:8000
```

### 5.4 Offline Model Weight Access
By default, the backend looks for model checkpoints in the local directory structure. If running offline, download the verified weights from [https://huggingface.co/vynor/qorin-models](https://huggingface.co/vynor/qorin-models):
```bash
# Automatic download utility
python modern_gpt/scripts/download_checkpoints.py --model qorin-100m-sft-v3
# Or download all models:
python modern_gpt/scripts/download_checkpoints.py --model all
```

Expected local file paths:
- **QORIN-100M SFT v3**: `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt`
- **QORIN-100M Base**: `modern_gpt/checkpoints/modern_gpts_100m/best_val.pt`
- **QORIN-500M SFT Exp F**: `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt`
- **QORIN-500M Pretrain Pilot**: `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt`

If local files are present, the backend runs in **100% air-gapped offline mode** with zero external network requests.

### 5.5 Running Automated Tests
```bash
# Execute PyTorch model and ablation tests
pytest modern_gpt/tests/test_final_tool_ablation.py -v

# Execute backend API tests
pytest backend/test_backend.py -v

# Execute frontend build and linter
cd web
npm run build:pages
npm run lint
```

---

## 6. Research Integrity & Testing Scope Disclosures

### 6.1 What Frontend & Unit Tests Establish
To maintain strict research integrity, we explicitly demarcate the boundaries of different testing layers:
- **Frontend Regression Tests (`npm test`, Playwright)**: Verify browser UI rendering, tab navigation, responsive layout, and client-side state transitions. *They do NOT evaluate model accuracy, tool selection quality, or reasoning capability.*
- **Backend API Tests (`backend/test_backend.py`)**: Verify HTTP route contracts, session token validation, and error payload structures. *They do NOT prove model capability.*
- **Model Loading Tests**: Verify weight deserialization, tensor dimension matching, and memory reclamation. *They do NOT prove conversational quality.*
- **Harness Unit Tests**: Verify sandbox isolation and tool execution timeouts. *They do NOT prove that generated code is semantically correct.*
- **Benchmark Evaluation (`Stage 15`, `test_final_tool_ablation.py`)**: Executes actual model inferences against held-out prompts with deterministic scoring against reference ground truths.
- **Human-Audited Semantic Evaluation**: Manual cross-turn inspection of raw generated traces for factual hallucination and reasoning coherence.

### 6.2 Data Access Disclosures
- **Public Pretraining Data**: Sourced from `HuggingFaceFW/fineweb-edu` (ODC-By 1.0).
- **Public Instruction Mixtures**: Sourced from SmolTalk, Alpaca, and OpenCodeInstruct.
- **Internal Benchmark Sets**: Stage 13 multi-turn benchmark files (`train.jsonl`, `validation.jsonl`, `test.jsonl`) are included in `modern_gpt/experiments/phase2_experiment_f/stage_13_dataset/`.
- **Large Binary Shards**: Multi-gigabyte pretraining shards (`shards_250m`, `data_500m`) and large checkpoint tensors are excluded from Git to prevent repository bloat and are hosted on the Hugging Face Hub.

---

## 7. AI-Use Disclosure

In accordance with transparent research principles:
- **AI Assistance**: LLM coding assistants (Google Antigravity) were utilized for rapid code refactoring, frontend React component styling, script scaffolding, automated test authoring, LaTeX formatting, and documentation. But the initial model architecture code and experiments were completely written without ai assistance.
- **Model-Generated Content**: Synthetic dialogue seeds in one of the supervised fine tuning dataset were initially generated via automated tooling and subsequently filtered through deterministic Python sandboxes and human quality audits before training.
- **Result Verification**: All numerical results, hashes, parameter counts, and test passes cited in this repository were independently executed and verified directly on the host hardware.

---

## 8. License & Academic Attribution

This project is released for academic research, review, and educational purposes. When citing QORIN in academic research, please refer to [`docs/REFERENCES.md`](docs/REFERENCES.md).
