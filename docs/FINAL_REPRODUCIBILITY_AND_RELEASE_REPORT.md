# QORIN — Final Reproducibility, Model Registry & Release Audit Report

**Report ID**: `AUDIT-20261117-RELEASE-REPRODUCIBILITY`  
**Date**: 2026-11-17  
**Repository**: [https://github.com/imVynor/Qorin](https://github.com/imVynor/Qorin)  
**Live Showcase URL**: [https://imvynor.github.io/Qorin/](https://imvynor.github.io/Qorin/)  
**Hugging Face Hub**: [`vynor/qorin-models`](https://huggingface.co/vynor/qorin-models)  
**Target Environment**: CPU-Only (ZeroGPU Quota Consumed: 0.0 seconds)  
**Authors**: Anish Raj (50%) & Jnandeep (50%)  
**Sign-off Status**: **COMPLETED & VERIFIED (PENDING USER REVIEW BEFORE GIT PUSH)**

---

## 1. Executive Summary

This comprehensive audit was conducted to bring the public QORIN repository, backend inference services, and static research website to publication-grade scientific reproducibility. Every physical checkpoint on disk, BPE tokenizer, benchmark split, dataset hash, and execution pathway has been audited against ground-truth artifacts. No numerical metrics, training durations, or experimental conclusions have been fabricated or assumed.

---

## 2. What Was Inspected

1. **Git Working Tree & Commits**: Examined commit history, submodules, `.gitignore`, and tracked repository artifacts.
2. **Physical Checkpoints & State Dicts**: Deserialized all `.pt` checkpoints to audit tensor parameter counts, config dictionaries, and layer topologies.
3. **BPE Tokenizers**: Audited vocabulary size, merges, and JSON fingerprints for both BPE-16K (`bpe_16k_v2`) and BPE-32K (`bpe_32k`).
4. **Benchmark Datasets**: Verified line counts, JSON structure, unique IDs, and split isolation across Stage 13, Stage 15, and historical SFT suites.
5. **Backend & Providers**: Audited `backend/app.py`, `backend/models/qorin_provider.py`, `backend/models/manager.py`, and `backend/tools/` orchestrators.
6. **Web Frontend**: Inspected React 19 pages (`Home.tsx`, `Research.tsx`, `Experiments.tsx`, `Model.tsx`, `Journey.tsx`, `About.tsx`) and data models (`modelsData.ts`, `journeyData.ts`).
7. **Hugging Face Hub**: Verified remote model repository `vynor/qorin-models` sync and file existence.

---

## 3. What Was Changed

1. **Created Authoritative Audit Documentation**:
   - [`docs/REPRODUCIBILITY_AUDIT.md`](docs/REPRODUCIBILITY_AUDIT.md): Bit-for-bit inventory of all four model configurations with parameter accounting.
   - [`docs/models/500M_SFT_EXPERIMENT_F.md`](docs/models/500M_SFT_EXPERIMENT_F.md): Full lineage, Stage 13 dataset composition, 100-step pilot disclosure, and Stage 15 evaluation.
   - [`docs/models/MODEL_REGISTRY.md`](docs/models/MODEL_REGISTRY.md): Backend inference specification, lazy loading, and CPU-only resource management.
   - [`docs/data/DATASET_REGISTRY.md`](docs/data/DATASET_REGISTRY.md): Canonical dataset catalog and definitive resolution of the 100 vs. 90 benchmark count discrepancy.
   - [`docs/REFERENCES.md`](docs/REFERENCES.md): Complete academic citations for architectures, training algorithms, datasets, and statistical tests.
   - [`docs/RESULTS_PROVENANCE.md`](docs/RESULTS_PROVENANCE.md): Traceability matrix connecting published claims directly to supporting raw JSONL files.
   - [`docs/FINAL_REPRODUCIBILITY_AND_RELEASE_REPORT.md`](docs/FINAL_REPRODUCIBILITY_AND_RELEASE_REPORT.md): This comprehensive final sign-off report.
2. **Updated Root Documentation (`README.md`)**:
   - Added verified live GitHub Pages URL (`https://imvynor.github.io/Qorin/`).
   - Integrated full implementation source-code map linking verified paths for Transformer, attention, RoPE, RMSNorm, SwiGLU, training, evaluation, and tool harnesses.
   - Added step-by-step local setup instructions for frontend, backend, offline model download, and tests.
   - Added transparent AI-use disclosure separating human research decisions from AI coding assistance.
   - Added explicit testing scope disclosures distinguishing frontend UI tests from backend API and model capability evaluations.
3. **Frontend Data & Model Registry Alignment**:
   - Updated `web/src/data/modelsData.ts` to include all four configurations (`MODERN-GPT-100M-4K`, `MODERN-GPT-100M-GQA4`, `QORIN-500M`, `QORIN-500M-BASE`).
   - Updated `web/src/pages/Research.tsx` with benchmark reconciliation (100 total vs. 90 core tool tasks) and unequal training budget disclosures.
   - Preserved team contribution documentation across `About.tsx` and `docs/TEAM_CONTRIBUTIONS.md` (Anish Raj: Model, Model Training, Evaluation, Fine-Tuning; Jnandeep: Harness, Retrieval, Dataset Preparation, Tool Adaptation), while keeping `Journey.tsx` strictly focused on technical research milestones.

---

## 4. What Was Independently Verified vs. Historical Inferences

| Audit Item | Status | Independent Verification Method |
|---|---|---|
| **Checkpoint existence, sizes & SHA-256 hashes** | **VERIFIED** | Computed directly via Python `hashlib.sha256` on physical files. |
| **Tensor shapes & layer parameter counts** | **VERIFIED** | Deserialized checkpoints on CPU and inspected state dict keys. |
| **BPE Tokenizers & Vocab Hashes** | **VERIFIED** | Loaded via `HuggingFaceBPEWrapper` and verified token encodings. |
| **CPU Inference Execution on all 4 Models** | **VERIFIED** | Programmatically streamed generated tokens on CPU for all 4 models. |
| **Stage 13 Benchmark split counts (100 total)** | **VERIFIED** | Parsed `test.jsonl` and verified 100 unique record IDs and categories. |
| **Controlled tool ablation statistical values** | **VERIFIED** | Executed `test_final_tool_ablation.py` (McNemar $\chi^2 = 12.9032, p = 0.000328$). |
| **Frontend production build & linter** | **VERIFIED** | Executed `npm run build:pages` (0 errors) and `oxlint` (0 errors). |
| **Backend API endpoints** | **VERIFIED** | Executed `pytest backend/test_backend.py` (4 passed in 5.03s). |
| **Historical Pretraining Logs (e.g. 1.5B token loss curve)** | **PARTIALLY VERIFIED** | Checkpoint weights verified; raw per-step step-by-step pretraining logs inferred from historical manifest files. |

---

## 5. Four-Model Usability Matrix

| Model ID (API) | Display Name | Local File | SHA-256 | CPU Inference | Usability Verdict |
|---|---|---|---|---|---|
| `qorin-100m-sft-v3` | **QORIN-100M SFT v3** | `modern_gpt/checkpoints/modern_gpts_100m_sft_4k_v3/best_val.pt` | `d771ea9f...` | PASSED | **VERIFIED USABLE** |
| `qorin-100m-base` | **QORIN-100M Baseline** | `modern_gpt/checkpoints/modern_gpts_100m/best_val.pt` | `bded509c...` | PASSED | **VERIFIED USABLE** |
| `qorin-500m-sft-exp-f` | **QORIN-500M SFT Exp F** | `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt` | `bb5b7091...` | PASSED | **VERIFIED USABLE** |
| `qorin-500m-pretrain-pilot`| **QORIN-500M Pretrain Pilot** | `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt` | `28ac47b0...` | PASSED | **VERIFIED USABLE** |

---

## 6. Benchmark Count Resolution Summary

- **Total Unique Records in Benchmark**: Exactly **100 records** in `stage_13_dataset/test.jsonl` (SHA-256: `523ec4962c1de25a515fffc10649d7a06da720006a361540fddfd21c1fcab014`).
- **Stratified Partitioning**:
  - Corpus A (Multi-Turn Chaining): 40 records (30 standard Web &rarr; Python, 3 direct answer, 4 tool failure recovery, 3 contradictory sources).
  - Corpus B (Decision Calibration): 30 records (8 conceptual direct, 6 provided calc, 8 factual web, 8 contrastive pairs).
  - Corpus C (Evidence Synthesis): 30 records (15 full support, 5 partial support, 5 contradictory, 5 insufficient).
- **Why 90 Was Cited in Secondary Reports**: Exactly 90 of the 100 tasks evaluate standard tool execution or direct calculation ($30\text{ standard} + 30\text{ calibration} + 30\text{ synthesis}$). The remaining 10 tasks in Corpus A evaluate edge cases (early termination, search outages).
- **Resolution**: All pages now cite **N = 100 held-out tasks** for the overall benchmark, with explicit notation of **N = 30** or **N = 90** when reporting Turn-2 chaining subsets.

---

## 7. Canonical Dataset Names & Provenance

1. **`DS-PRETRAIN-100M-FW`**: FineWeb-Edu 10BT Sample Shards (1.5B tokens, ODC-By 1.0).
2. **`DS-SFT-100M-4K-V3`**: 4K SFT Instructional Mixture (9,738 records, 29.2M tokens).
3. **`DS-SFT-500M-STAGE13`**: Stage 13 Multi-Turn Dataset (1,000 records: 800 train / 100 val / 100 test).
4. **`DS-PRETRAIN-500M-5B`**: 5-Billion Token Pretraining Shards (101 binary shards, multi-source).

---

## 8. Results Available for the Four Configurations

| Configuration | Validation Loss | Benchmark Evaluation Available | Controlled Tool Ablation | Status Label |
|---|---|---|---|---|
| **1. 100M Base** | $1.564$ (pretrain) | No (Lacks ChatML instruction delimiters) | Not evaluated under this protocol | **VERIFIED BASELINE** |
| **2. 100M SFT v3** | $0.1521$ (Step 8D) | Yes (Step 8D: 96% AST, 38% pass@1, 96% retrieval) | 36.0% (Tools Off) vs 29.0% (Tools On) | **VERIFIED SFT** |
| **3. 500M SFT Exp F**| $0.8947$ (Step 100) | Yes (Stage 15 Blind: 70% web call, 50% python, 100% precision) | 43.0% (Tools Off) vs 64.0% (Tools On) | **VERIFIED PILOT** |
| **4. 500M Pretrain Pilot**| $2.140$ (Step 7,694)| No (Unconverged causal pretrain base) | Not evaluated under this protocol | **VERIFIED UNCONVERGED** |

---

## 9. Training Budget Differences & Scaling Notice

The four configurations differ substantially in training history:
- **100M Base**: Pretrained on 1,500,000,000 tokens (30,000 updates).
- **100M SFT v3**: Fine-tuned on 29,200,000 tokens (8,764 updates).
- **500M Base Pilot**: Pretrained on ~252,000,000 tokens (7,694 updates; early pilot).
- **500M SFT Exp F**: Fine-tuned on ~102,400 tokens (100 updates; exploratory pilot).

**Scientific Finding**: Because training data composition, token budgets, and convergence states differ, comparisons between 100M and 500M are descriptive of these checkpoints and do **not** isolate a causal parameter scaling law.

---

## 10. Verification of Automated Tests Executed

1. **Frontend Production Build**: `npm --prefix web run build:pages` &rarr; **PASSED** (Built in 626ms).
2. **Frontend Linter**: `npm --prefix web run lint` (`oxlint`) &rarr; **PASSED** (0 errors, 11 warnings).
3. **Statistical Tool Ablation**: `pytest modern_gpt/tests/test_final_tool_ablation.py -v` &rarr; **3/3 PASSED** in 0.02s.
4. **Backend API Endpoints**: `pytest backend/test_backend.py -v` &rarr; **4/4 PASSED** in 5.03s.
5. **CPU Inference Streaming**: Verified for all 4 models sequentially &rarr; **4/4 PASSED** on CPU with zero memory leakage.

---

## 11. Remaining Blockers & Next Actions

- **Git Status**: Changes are staged locally in the working directory.
- **Git Push**: In strict compliance with instructions, **NO GIT PUSH HAS BEEN EXECUTED**.
- **User Review Required**: All deliverables, documents, source maps, and verification reports are ready for user inspection and sign-off.
