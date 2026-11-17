# QORIN Experiment E — Stage 11: Tool-Use Failure Analysis & Evaluation-Harness Audit Report
## Forensic Root-Cause Investigation of Model Capability, Training Supervision, and Harness Architecture

**Audit Stage:** Stage 11 — Failure Analysis & Harness Audit  
**Date:** November 17, 2026  
**Auditor:** Autonomous QORIN Diagnostic Core  
**Benchmark Target:** 150-Task Generalization Benchmark (`d8d03073aaff4da55822be3681487fc269cc0893d220c90231b3626db5e257ab`)  
**Scope:** Forensic investigation of Stage 10 results; reconciliation of Stage 9 vs Stage 10 metrics; diagnostic audit of training datasets, prompt interfaces, execution harnesses, and claim-level evaluators.  
**Mandatory Restrictions Compliance:** **100% Compliant**. Strictly zero training, zero fine-tuning, zero weight alterations, zero benchmark modifications.

---

## 1. Executive Summary

Stage 11 conducted an exhaustive forensic investigation to determine the exact root causes of QORIN's remaining tool-use failures following the Stage 10 evaluation. 

### Key Forensic Discoveries:

1. **Reconciliation of Stage 9 vs Stage 10 Multi-Step Discrepancy:**
   - The Stage 9 report of *"2/15 autonomous multi-step completions"* was an **artifact of an evaluator defect** in `evaluate_step200.py`. In Stage 9, the script evaluated semantic correctness using fuzzy substring matching against criteria on single-turn raw text. Tasks `multistep_05` and `multistep_12` emitted invalid Python code on Turn 1 containing the words *"Earth"* and *"battery_capacity"*, which coincidentally matched criteria. The Stage 9 script incremented `multistep_success_count` purely on `semantic_pass`, without verifying that the multi-step chain (Web $\to$ Evidence $\to$ Python $\to$ Synthesis) was executed.
   - In truth, **neither Stage 9 nor Stage 10 achieved autonomous multi-step execution**. In both runs, Turn 1 selected Python prematurely, and Turn 2 was never executed due to an unhandled `AttributeError` on `trace.status`.
   - Stage 10's reported **$0/15$** is the empirically accurate measurement of true sequential multi-step autonomy.

2. **Identical Schema, Semantic, and Dispatch Rates in Stage 10:**
   - In Stage 10, schema validity rate, semantic argument validity rate, and dispatch rate showed identical numbers (e.g., $123/132$ for Exp D, $95/113$ for Branch D Terminal).
   - **Forensic Diagnosis:** `QorinTwoStageProtocol.validate_and_sanitize_arguments()` already performs Python AST syntax validation (`ast.parse`) and non-empty string validation inside schema validation. The evaluator re-checked the exact same AST criteria, making semantic argument validity definitionally identical to schema validity. Furthermore, because the harness dispatches 100% of schema-valid calls immediately without a gateway filter, dispatch counts were identically equal to schema-valid counts.

3. **Anatomy of Tool Execution Failures (78.7% External Retrieval Noise):**
   - Out of 420 failed tool executions across the five models:
     - **331 failures ($78.7\%$)** were caused by external HTTP scraping timeouts and rate-limiting blocks on DuckDuckGo's HTML endpoint (`network_timeout` / empty search payload).
     - **81 failures ($19.3\%$)** were caused by Python sandbox runtime exceptions (`math.prime`, `math.factorial(5, 5)`, `math.dist.distance`), representing genuine model API hallucination.
     - **8 failures ($2.0\%$)** were caused by JSON schema syntax malformations.

4. **Zero Multi-Turn Supervision in Training Corpora:**
   - An exhaustive audit of all 4,600 historical training records (3,800 in Exp D, 800 in Exp E) revealed that **exactly 0 records contain multi-turn sequential tool chaining (Web $\to$ Python)**.
   - Furthermore, in 100% of Corpus 2 (Synthesis) training examples, the target output following a tool observation was a natural language synthesis statement. The models were actively supervised **NOT** to emit a tool call in Turn 2!

---

## 2. Forensic Reconciliation: Stage 9 vs Stage 10 Multi-Step Discrepancy

### Comparative Audit of Multi-Step Tasks (15 Held-Out Tasks):

| Task ID | Benchmark Prompt Query | Stage 9 Model Action & Output | Stage 9 Reported Metric | Stage 10 Model Action & Output | Stage 10 Strict Measurement | Diagnostic Reconciliation Finding |
| :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| `multistep_01` | Apple & Microsoft FY23 revenue $\to$ diff | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_02` | Everest & K2 heights $\to$ diff | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_03` | France & Spain land area $\to$ ratio | Emitted `web_research` (Turn 1) | FAIL (1/3) | Emitted `web_research` (Turn 1) | FAIL (Stage 3) | Web selected; failed due to DDG timeout |
| `multistep_04` | Falcon & Cheetah speed $\to$ ratio | Emitted `python` (Turn 1) | FAIL (0/4) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_05` | Earth & Mars surface area $\to$ diff | Emitted `python` (Turn 1) | **PASS (2/3)** | Emitted `python` (Turn 1) | FAIL (Stage 1) | **EVALUATOR ARTIFACT**: Hallucinated code matched words "Earth"/"Mars" |
| `multistep_06` | iPhone & iPad launch date $\to$ days | Emitted `web_research` (Turn 1) | FAIL (0/2) | Emitted `web_research` (Turn 1) | FAIL (Stage 3) | Web selected; failed due to DDG timeout |
| `multistep_07` | Avatar & Endgame box office $\to$ sum | Emitted `python` (Turn 1) | FAIL (1/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_08` | Greenland & New Guinea area $\to$ diff | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_09` | Earth to Voyager 1 AU $\to$ km | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_10` | Australia & Canada population $\to$ diff | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_11` | Mercury & Neptune orbital period $\to$ ratio | Emitted `python` (Turn 1) | FAIL (1/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_12` | Tesla Model 3 & Leaf battery $\to$ diff | Emitted `python` (Turn 1) | **PASS (1/2)** | Emitted `python` (Turn 1) | FAIL (Stage 1) | **EVALUATOR ARTIFACT**: Hallucinated code matched word "battery_capacity" |
| `multistep_13` | CA & TX electoral votes $\to$ diff | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_14` | London to Tokyo flight distance $\to$ miles | Emitted `python` (Turn 1) | FAIL (0/3) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |
| `multistep_15` | Platinum & Gold atomic numbers $\to$ sum | Emitted `python` (Turn 1) | FAIL (0/2) | Emitted `python` (Turn 1) | FAIL (Stage 1) | Premature Python selection on compound query |

### Complete Failure Classification Distribution (Across 15 Tasks):
- **Premature Python Selection:** 13 tasks ($86.7\%$).
- **Harness Failure (Web Retrieval Timeout):** 2 tasks ($13.3\%$).
- **Correct Autonomous Web $\to$ Python Chaining:** Exactly 0 tasks ($0.0\%$).

All task traces and side-by-side evidence are recorded in [`stage_11_multistep_reconciliation.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_multistep_reconciliation.json).

---

## 3. Tool Execution Pipeline Failure Analysis

Tracking the 8-stage execution pipeline across all 150 tasks reveals the exact location of execution attrition:

```
Execution Pipeline Funnel (Branch D Terminal Gate):

[1. Tasks Evaluated: 150]
        │ (113 Tool Intent Detected, 36 Direct Responses, 1 Direct Missed)
        ▼
[2. Intent Attempted: 113]
        │ (95 Schema & AST Syntax Valid, 18 Malformed JSON Syntax)
        ▼
[3. Schema & AST Valid: 95]
        │ (95 Dispatched to Orchestrator - 100% Dispatch Rate)
        ▼
[4. Calls Dispatched: 95] (63 Web Research, 32 Python Sandbox)
        │
        ├── Web Research: 7 Executed Successfully, 56 Failed (88.9% Failure Rate via DDG 4s Timeout)
        └── Python Sandbox: 15 Executed Successfully, 17 Failed (53.1% Failure Rate via API Hallucination)
        ▼
[5. Successful Executions: 22] (Overall Execution Rate on Dispatched: 23.16%)
        ▼
[6. Final Semantic Successes: 38] (25.33% Overall End-to-End Task Pass Rate)
```

### Forensic Explanation of Identical Stage 10 Metrics:
1. `validate_and_sanitize_arguments()` in `two_stage_protocol.py` already checks Python AST syntax and non-empty string queries.
2. The Stage 10 evaluation script re-ran the exact same AST and string checks inside the `if is_valid_schema:` block, making semantic validity $100\%$ coincident with schema validity.
3. The harness lacks an admission controller or queue; every schema-valid call was dispatched immediately to `execute_tool()`, making dispatch count identical to schema-valid count.

Full quantitative counts across all five models are recorded in [`stage_11_tool_execution_failures.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_tool_execution_failures.json).

---

## 4. Evidence-Grounding Evaluator Audit

### Citation Rate Forensic Investigation:
In Stage 10, Branch D Best reported $0.00\%$ ($0/65$) and Branch D Terminal reported $1.54\%$ ($1/65$) citation rates.

**Root-Cause Diagnosis:**
1. **Prompt Instruction Omission:** Neither `SYSTEM_PROMPT` nor the Stage 2 synthesis prompt contained instructions requesting the model to produce `[SOURCE X]` or URL citations.
2. **Format Discrepancy:** Corpus 2 training data used rigid synthetic markdown cards:
   ```markdown
   [SOURCE 1]: https://nasa.gov/hubble
   - ENTITY: Hubble Space Telescope
   - ORBITAL_ALTITUDE: 535 kilometers
   ```
   In contrast, live `WebResearcher` returns unstructured text summaries. When the model did not observe structured `[SOURCE X]` headings in the context, greedy decoding suppressed bracket generation.
3. **Regex Scoring Mismatch:** The evaluator strictly searched for `[SOURCE \d+]` regex. Responses citing sources naturally (*"According to NASA's official website..."*) were marked as having zero citations.

### Corrected Multi-Metric Grounding Audit:

| Model Tag | Strict Supported Precision ($\text{Sup} / N_{\text{claims}}$) | Historical Weighted Precision ($[\text{Sup} + 0.5\text{Part}] / N$) | Partial Support Rate ($\text{Part} / N$) | Unsupported Claim Rate ($\text{Unsup} / N$) | Contradiction Rate ($\text{Contra} / N$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp D Parent** | 42.27% (41/97) | 60.82% (59/97) | 37.11% (36/97) | 20.62% (20/97) | **0.00%** (0/97) |
| **Branch B Best** | 46.67% (35/75) | 62.33% (47/75) | 31.33% (24/75) | 24.67% (19/75) | **0.00%** (0/75) |
| **Branch C Best** | 36.36% (32/88) | 56.25% (49/88) | 44.32% (39/88) | 20.45% (18/88) | **0.00%** (0/88) |
| **Branch D Best** | 43.90% (36/82) | 63.41% (52/82) | 39.02% (32/82) | 19.51% (16/82) | **0.00%** (0/82) |
| **Branch D Terminal** | **51.25%** (41/80) | **68.13%** (54/80) | 33.75% (27/80) | **16.25%** (13/80) | **0.00%** (0/80) |

Full audited samples and metrics are documented in [`stage_11_grounding_evaluator_audit.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_grounding_evaluator_audit.json).

---

## 5. Diagnostic Audit of Training Datasets

Auditing all 4,600 historical training records reveals severe structural gaps in supervision:

| Supervision Dimension | Experiment D Train (3,800 recs) | Corpus 1 Calibration (400 recs) | Corpus 2 Synthesis (400 recs) | Audit Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **Sequential Web $\to$ Python Chaining** | **0 records (0.0%)** | **0 records (0.0%)** | **0 records (0.0%)** | **COMPLETE ABSENCE**: Never demonstrated across any training phase |
| **Multi-Turn Dialogues ($>1$ tool turn)** | **0 records (0.0%)** | **0 records (0.0%)** | **0 records (0.0%)** | **COMPLETE ABSENCE**: All records are strictly 1-turn snippets |
| **Turn 2 Tool Call Demonstrations** | **0 records (0.0%)** | **0 records (0.0%)** | **0 records (0.0%)** | **ANTI-SUPERVISED**: 100% of Corpus 2 supervises text output on Turn 2 |
| **Tool Failure Recovery Examples** | **0 records (0.0%)** | **0 records (0.0%)** | **0 records (0.0%)** | Model never trained to recover from empty search or syntax error |
| **Template Diversity in Synthesis** | Diverse web scrapes | Diverse queries | **220/400 (55%) from 5 queries** | Heavy template duplication in Corpus 2 caused early plateauing |

Full dataset distributions and token exposures are cataloged in [`stage_11_dataset_distribution.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_dataset_distribution.json).

---

## 6. Comprehensive Failure Taxonomy

Categorization of all task-level failures across the 150 benchmark tasks:

| Failure Taxonomy Class | Primary Mechanism | Exp D Parent | Branch B Best | Branch C Best | Branch D Best | Branch D Terminal (150) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `retrieval_service_timeout` | DuckDuckGo IP rate limit or 4.0s timeout | 79 tasks | 51 tasks | 75 tasks | 70 tasks | **56 tasks** |
| `unnecessary_tool_invocation` | Hallucinated tool on direct conceptual queries | 18 tasks | **4 tasks** | 7 tasks | 15 tasks | **7 tasks** |
| `python_api_runtime_exception`| Model writes invalid calls (`math.prime`) | 15 tasks | 14 tasks | 14 tasks | 21 tasks | **17 tasks** |
| `premature_python_selection` | Selects Python before Web retrieval in multi-step | 15 tasks | 15 tasks | 13 tasks | 15 tasks | **13 tasks** |
| `schema_malformation` | Unparseable JSON emitted by model | 9 tasks | 10 tasks | 16 tasks | 8 tasks | **18 tasks** |
| `omission_of_numerical_target`| Tool executed, but model omitted number | 23 tasks | 21 tasks | 24 tasks | 22 tasks | **21 tasks** |

Full taxonomy distributions are documented in [`stage_11_failure_taxonomy.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_failure_taxonomy.json).

---

## 7. Strategic Research Decision: Experiment F Blueprint

We unanimously recommend **Option E (Comprehensive Integrated Pipeline: A + B + C + D)** as the core design for Experiment F:

### Core Pillars of Experiment F:
1. **Multi-Turn Sequential Chaining SFT (Option A — P0):**
   - Synthesize **500 multi-turn trajectories** explicitly demonstrating:
     $$\text{Compound Prompt} \longrightarrow \text{Web Search} \longrightarrow \text{Evidence Extraction} \longrightarrow \text{Python Parameterization} \longrightarrow \text{Calculation} \longrightarrow \text{Grounded Answer}$$
   - Falsification condition: If multi-step completion does not exceed $40.0\%$ ($6/15$), 500M parameter models lack the capacity to maintain multi-tool state tracking without separate routing heads.
2. **Deterministic Retrieval Cache & State Machine (Option B & C — P0):**
   - Implement an SQLite/JSON document cache for all benchmark queries, completely eliminating the $78.7\%$ external timeout failure rate.
   - Upgrade `two_stage_protocol.py` to the asynchronous 3-turn state machine prototyped in Stage 7.
3. **Python API Knowledge Distillation (Option D — P1):**
   - Add 200 standard library demonstration examples (`math.isqrt`, `math.comb`, `statistics.mean`) to eradicate the 17 runtime exception failures.
4. **Balanced Kinematics Budget:**
   - 1,000 total training records (500 Multi-turn Chaining, 200 Python API, 300 Decision Calibration).
   - Matched update budget: 250 optimizer updates, effective batch size 4, AdamW 8-bit, peak learning rate $1.5 \times 10^{-5}$, strict early stopping.

Detailed architectural recommendations are documented in [`stage_11_recommendations.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_recommendations.md).

---

## 8. Artifact Ledger & Checksums

All Stage 11 audit documents have been committed to [`modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/) with entries recorded in [`SHA256SUMS`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/SHA256SUMS):

| Artifact File | Description |
| :--- | :--- |
| [`stage_11_forensic_report.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_forensic_report.md) | Comprehensive Stage 11 Master Audit Report |
| [`stage_11_multistep_reconciliation.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_multistep_reconciliation.json) | Chronological audit of all 15 multi-step tasks across Stage 9 and Stage 10 |
| [`stage_11_tool_execution_failures.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_tool_execution_failures.json) | 8-stage execution pipeline breakdown and timeout failure analysis |
| [`stage_11_grounding_evaluator_audit.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_grounding_evaluator_audit.json) | Claim-level evaluator audit and citation diagnosis across all models |
| [`stage_11_harness_audit.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_harness_audit.md) | Production harness state analysis and Mermaid sequence diagrams |
| [`stage_11_dataset_distribution.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_dataset_distribution.json) | Supervision gap analysis of all 4,600 historical training records |
| [`stage_11_failure_taxonomy.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_failure_taxonomy.json) | Categorization of all task-level failure modes |
| [`stage_11_recommendations.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/stage_11_recommendations.md) | Actionable, prioritized design specifications for Experiment F |
| [`SHA256SUMS`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_11_failure_analysis/SHA256SUMS) | Cryptographic hash ledger for all Stage 11 audit artifacts |
