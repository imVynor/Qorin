# QORIN — Stage 6.5: Experiment E Design Review and Feasibility Gate Report

**Document**: `design_review_report.md`  
**Review Status**: Formal Engineering & Scientific Feasibility Review  
**Subject Baseline**: Frozen Experiment D Step 425 (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`)  
**Target Architecture**: QORIN-500M Dual-Stage Model + Deterministic Sandbox Orchestrator  
**Review Directory**: `modern_gpt/experiments/phase2_experiment_e/design_review/`

---

## 1. Executive Summary & Feasibility Overview

Prior to committing compute and fine-tuning resources to **Experiment E**, a focused design-correction and engineering-feasibility review was conducted across the QORIN repository. This review scrutinized multi-step tool orchestration, evidence-grounding evaluation protocols, dataset accounting, factorial branch exposure, statistical power, and evidence representation architectures.

### Review Highlights:
1. **Multi-Step Tool Orchestration**: Feasibility is **confirmed**. The failure of all 15 Web $\to$ Python tasks in Stage 5 was caused by the evaluation script terminating after a single tool turn and the model lacking multi-turn training tokens. A minimal, deterministic Directed Acyclic Graph (DAG) state machine (`max_tool_rounds = 2`, global timeout 15.0s, AST pre-flight gate) has been specified in [`multistep_orchestration_spec.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/multistep_orchestration_spec.md).
2. **Claim-Level Grounding Protocol**: Superficial 3-word n-gram overlap has been completely **deprecated and replaced** by an Atomic Claim Decomposition Framework with a 4-tier grounding taxonomy (`SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`) and a 50-task human-verified Gold Sub-Suite ([`evidence_grounding_protocol.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/evidence_grounding_protocol.md)).
3. **Rigorous Dataset Accounting**: Ambiguity in the 1,000-record dataset has been **fully resolved**. Corpus 1 (500 records) and Corpus 2 (500 records) are 100% disjoint. The 1,000 records are strictly partitioned into 800 Train / 100 Validation / 100 Test, with complete entity topic quarantine and assistant-target loss masking ([`dataset_allocation.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/dataset_allocation.md)).
4. **Controlled Branch Exposure**: The 4-branch factorial design has been fully audited ([`branch_exposure_matrix.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/branch_exposure_matrix.json)). Branch D is controlled for epoch exposure (2.0 epochs across all branches) and audited against an explicit update-matched ablation (200 optimizer steps).
5. **Statistical Power**: The primary outcome (overall semantic pass rate on $N = 150$) has sufficient statistical power ($\beta \ge 0.95$ for $p < 0.005$ if target $50\%$ is reached). However, category subsets (e.g. multi-step with $N = 15$) have wide confidence intervals ($\pm 22\%$), mandating an expanded 250-task benchmark for subsequent stages ([`statistical_power_review.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/statistical_power_review.md)).
6. **Architecture Recommendation**: We officially adopt a **Hybrid Structured Evidence Record (SER) + Claim-Evidence Mapping (CEM)** architecture, establishing a strict separation between deterministic harness code and learned model capabilities ([`architecture_decision.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/architecture_decision.md)).

---

## 2. Review Findings by Area

### 2.1 Multi-Step Orchestration
- **Root Cause**: In `run_audit_evaluation.py`, Stage 2 was hardcoded as the terminal generation. Even if the model emitted `<|tool_call|>` for Python, the evaluator treated it as the final assistant response.
- **Solution**: The proposed state machine implements an explicit 2-turn ceiling ($Web \to Python$). Self-loops are forbidden by DAG topology. The AST gate blocks malicious imports (`os`, `sys`, `subprocess`) before spawning the Python sandbox.
- **Implementation Requirement**: Requires a coordinated harness extension (adding the state machine loop to both the API backend and evaluation runner) and a prompt adjustment (showing the model that Python tool calls can follow web observations).

### 2.2 Evidence Grounding
- **Deficiency in Stage 5**: A 3-word overlap metric allowed models to achieve "98.1% evidence utilization" while hallucinating all numbers and dates.
- **New Framework**: Six formal metrics are pre-registered: Claim Precision ($CP$), Claim Coverage ($CC$), Numerical Extraction Accuracy ($NEA$), Entity-Value Association Accuracy ($EVAA$), Citation Correctness ($CR$), and Unsupported Claim Rate ($UFCR$). A 50-task Gold Sub-Suite provides deterministic factual baselines.

### 2.3 Dataset Accounting
- **Disjoint Partitioning**: Corpus 1 (Calibration) and Corpus 2 (Synthesis) share zero dialogue records.
- **Combined Assembly**: Branch D trains on the joint pool ($400 + 400 = 800\text{ records}$) with stratified 1:1 alternating batches and balanced multi-task loss ($\alpha = 0.5$).
- **Test Set Protection**: The 100-record Test split is completely isolated from training and early stopping.

### 2.4 Branch Exposure Matrix
- All branches start from frozen `ckpt_experiment_d_best.pt` (SHA-256: `c6b388da...`).
- Branch B (Calibration) and Branch C (Synthesis) receive 200 optimizer updates (2.0 epochs of 400 records).
- Branch D receives 400 optimizer updates (2.0 epochs of 800 records). An ablation condition (`branch_d_matched_updates`) evaluates 200 updates (1.0 epoch) to rule out update-volume artifacts.

### 2.5 Statistical Power & Significance
- **Full Benchmark ($N = 150$)**: Baseline = 42/150 (28.0%). Target = 75/150 (50.0%). Paired McNemar $\chi^2 = 21.79$, $p = 3.05 \times 10^{-6}$. Highly significant.
- **Category Subsets**: Direct-task FPR ($N = 36$) CI width is $\pm 9.5\%$; Multi-step ($N = 15$) CI width is $\pm 22.2\%$. Claims of $p < 0.005$ are restricted to the benchmark-wide primary outcome.

### 2.6 Architecture Decision
- Raw snippets are rejected due to attention dispersion in 500M models.
- Structured Evidence Records (SER) clean the prompt deterministically in harness code.
- Claim-Evidence Mapping (CEM) enforces causal self-attention in model generation tokens.

---

## 3. Deliverables Manifest & Checksums

All 7 review artifacts and the manifest have been written to `modern_gpt/experiments/phase2_experiment_e/design_review/`:

| Artifact | Purpose | SHA-256 |
| :--- | :--- | :--- |
| [`multistep_orchestration_spec.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/multistep_orchestration_spec.md) | DAG state machine, loop prevention, AST gate | `[Calculated in Manifest]` |
| [`evidence_grounding_protocol.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/evidence_grounding_protocol.md) | Claim-level extraction and 4-tier grounding taxonomy | `[Calculated in Manifest]` |
| [`dataset_allocation.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/dataset_allocation.md) | Disjoint 1,000-record pool, splits, masking | `[Calculated in Manifest]` |
| [`branch_exposure_matrix.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/branch_exposure_matrix.json) | 4-branch token, epoch, and update accounting | `[Calculated in Manifest]` |
| [`statistical_power_review.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/statistical_power_review.md) | Wilson CIs, McNemar power, 250-task suite | `[Calculated in Manifest]` |
| [`architecture_decision.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/architecture_decision.md) | Formal ADR for Hybrid SER + CEM | `[Calculated in Manifest]` |
| [`design_review_report.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/design_review_report.md) | This synthesis and authorization gate | `[Calculated in Manifest]` |
| [`SHA256SUMS`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/design_review/SHA256SUMS) | Cryptographic verification manifest | Verified |

---

## 4. Final Authorization Gate Decision

### **DECISION: READY WITH CONDITIONS**

The research design, engineering specifications, statistical protections, and dataset architectures for Experiment E are sound, rigorous, and feasible. However, before dedicating active GPU compute to dataset generation or fine-tuning, two operational conditions must be fulfilled:

### Condition 1: Non-Destructive Harness Multi-Step Dry-Run
Prior to training, the multi-step Directed Acyclic Graph (DAG) state machine specified in `multistep_orchestration_spec.md` must be implemented in an isolated test harness (`test_multistep_harness.py`) and verified against the 15 held-out multi-step prompts using mock and live tool responses. This will prove that zero infinite loops, timeouts, or state deadlocks occur in code before training data is prepared.

### Condition 2: Automated Benchmark Decontamination Verification
Before any record in the 1,000-sample pool is finalized, an automated script must run the three decontamination gates (exact deduplication, 10-gram Jaccard shingling $< 0.15$, entity overlap check) against `benchmark.jsonl` and output a verified `decontamination_audit.json` log with zero violations.

### Unresolved Blockers:
- **None**. There are no theoretical, architectural, or hardware blockers preventing progression to dataset construction once the user authorizes fulfillment of the two operational conditions.
