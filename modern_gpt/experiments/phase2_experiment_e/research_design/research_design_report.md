# QORIN — Stage 6: Experiment E Research Design & Forensic Audit Report

**Report Title**: QORIN-500M Experiment E Research Design & Forensic Audit  
**Author**: Antigravity Autonomous Research Harness  
**Date**: November 17, 2026  
**Target Architecture**: QORIN-500M (20 Transformer layers, $d_{model}=1536$, 24 query heads, 4 KV heads, context 2048, vocabulary 32768, 514,389,504 unique parameters)  
**Parent Checkpoint**: Frozen Experiment D Best Checkpoint (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`)  
**Baseline Checkpoint**: Frozen Experiment C Final Checkpoint (`ckpt_experiment_c_final.pt`, SHA-256: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`)  
**Primary Directory**: `modern_gpt/experiments/phase2_experiment_e/research_design/`

---

## 1. Executive Summary

Following the completion of the 150-task Stage 5 Generalization Audit, a comprehensive, line-by-line forensic investigation was executed on all raw evaluation logs (`experiment_c_results.jsonl`, `experiment_d_results.jsonl`, and `tool_traces.jsonl`). 

This audit conclusively establishes that while Experiment D succeeded in activating tool-use capabilities in QORIN-500M (elevating Web tool selection recall from **0.00%** to **80.26%** and achieving 100% intent recall), it introduced two severe architectural vulnerabilities:
1. **Decision Calibration Collapse**: Experiment D developed an aggressive bias toward external search, triggering unnecessary `web_research` calls on **50.00% (18 / 36)** of direct conceptual questions where no tool was required.
2. **Synthesis-Evidence Decoupling**: When search snippets were retrieved and executed cleanly (106 unique tasks), the model translated that evidence into a semantically correct final response in only **24.53% (26 / 106)** of cases. In over 75% of executions, the model either extracted incorrect entity values from dense, noisy snippets or overrode the retrieved facts with parametric hallucinations.

Because gains from new tool capabilities (+19 tasks) were partially offset by regressions on conceptual knowledge (-11 tasks), the overall net improvement in semantic correctness (+5.33%, moving from 22.67% in Exp C to 28.00% in Exp D) is **not statistically significant** under McNemar's test ($\chi^2 = 1.6333$, $p = 0.2012$).

To overcome these vulnerabilities, **Experiment E** is formally designed as a dual-intervention, 4-branch factorial research experiment:
- **Intervention 1 (Tool-Decision Calibration)** introduces balanced contrastive dyads and negative-control loss masking to restore direct parametric answers for conceptual queries ($\le 10\%$ false-positive rate).
- **Intervention 2 (Evidence-Grounded Synthesis)** replaces unstructured search snippets with a **Structured Evidence Record (SER)** schema and causal Claim-Evidence Mapping, increasing factual extraction fidelity to $\ge 60\%$.
- A multi-turn orchestrator handoff is designed to unlock true **Web $\to$ Python** chaining.

All artifacts, schemas, protocols, and pre-registered success criteria have been codified without modifying frozen model weights or training datasets.

---

## 2. Part A: Forensic Recomputation & Anomaly Resolution

All metrics were recomputed directly from raw task-level records. The investigation resolved seven specific apparent inconsistencies identified between `generalization_report.md` and the raw data:

### Core Metric Reconciliation Matrix

| Metric Dimension | Recomputed Exp C | Recomputed Exp D | Stage 5 Report Status | Forensic Reconciliation |
| :--- | :---: | :---: | :---: | :--- |
| **Total Benchmark Tasks** | 150 | 150 | Verified | Exact match across all splits |
| **Intent True Positives ($TP$)** | 61 | 114 | Verified | Exact match |
| **Intent False Positives ($FP$)** | 3 | 18 | Verified | Exact match (+15 FP degradation in D) |
| **Intent False Negatives ($FN$)** | 53 | 0 | Verified | Exp D eliminated all false negatives |
| **Intent Precision** | 95.31% | 86.36% | Verified | Exact match |
| **Intent Recall** | 53.51% | 100.00% | Verified | Exact match (+46.49% activation in D) |
| **Intent F1-Score** | 0.6854 | 0.9268 | Verified | Exact match |
| **Web Tool Selection Recall** | 0.00% (0/76) | 80.26% (61/76) | Verified | Exp D correctly selected Web on 61 of 76 tasks |
| **Python Tool Selection Recall** | 94.74% (36/38) | 92.11% (35/38) | Verified | Python capability maintained (-2.63%) |
| **Tool Dispatch Rate** | 38.67% (58/150) | 82.00% (123/150) | Verified | Valid tool calls routed to sandboxes |
| **Clean Tool Executions** | 41 calls | 106 calls | Verified | 106 unique tasks executed cleanly |
| **End-to-End Semantic Pass** | 22.67% (34/150) | 28.00% (42/150) | Verified | Net gain of +8 tasks (+5.33%) |

---

### Detailed Resolution of Seven Investigated Anomalies:

1. **Category 100% vs Overall 80.26% Web Selection**:
   In `colloquial_web_research` (30/30), `current_information` (20/20), `tool_discrimination` (6/6), and `tool_failure_recovery` (5/5), Exp D achieved 100% Web selection. However, in `multistep_web_python` (15 tasks), the benchmark expected an initial `web_research` call to gather empirical data; Exp D emitted `python` immediately on **0/15 (0.00%)** tasks. Total Web recall is exactly $(30 + 20 + 6 + 5 + 0) / 76 = 61 / 76 = \mathbf{80.26\%}$.
2. **50% Unnecessary Tool-Call Rate**:
   Across 36 direct-expected benchmark tasks (25 conceptual, 7 discrimination, 4 recovery), Exp D emitted `<|tool_call|>` on **18 tasks (50.00%)**, all 18 invoking `web_research`. Exp C had only 3 false positives (8.33%). The denominator is verified as exactly 36.
3. **McNemar's Statistical Significance**:
   Paired contingency analysis: $n_{11} = 23$, $n_{10} = 19$ (Exp D won), $n_{01} = 11$ (Exp C won), $n_{00} = 97$. Edwards continuity-corrected $\chi^2 = (|19 - 11| - 1)^2 / 30 = 49 / 30 = \mathbf{1.6333}$, two-tailed $p = \mathbf{0.2012}$. Without continuity correction: $\chi^2 = 2.1333$, $p = 0.1441$. In both cases, $p > 0.05$; the improvement is not statistically significant.
4. **106 Tool Executions**:
   `tool_traces.jsonl` contains 123 traces for Exp D across 123 unique tasks (106 completed, 17 failed). The 106 completed executions represent **106 unique tasks** (no multi-turn or duplicate executions).
5. **Evidence Utilization Metric Defect**:
   In `compute_audit_metrics.py`, 18 direct-response tasks were erroneously added to the numerator, yielding $\frac{86 + 18}{106} = \frac{104}{106} = 98.11\%$ (and $217.07\%$ for Exp C). Recomputed on executed tasks alone, the 3-word token overlap rate was **81.13% (86 / 106)**. Furthermore, this 3-word overlap is a superficial token presence metric; only **26 of those 106 executed tasks (24.53%)** met semantic correctness.
6. **Evaluator Model Blindness**:
   `check_semantic_correctness()` in `run_audit_evaluation.py` takes only `(task, final_answer, tool_trace)` and contains zero conditionals or references to model identity. Evaluation was 100% blind and programmatic.
7. **Task-Level Consistency**:
   All aggregate percentages in `generalization_report.md` were confirmed against raw task lines, with the exception of the documented Evidence Utilization numerator defect.

---

## 3. Part B: Failure Mechanism Taxonomy

Across all 150 benchmark tasks, Experiment D produced **108 semantic failures**. Each failure was traced to its earliest point of origin and cataloged into `failure_taxonomy.json`:

```
                       [108 Total Semantic Failures in Exp D]
                                        │
        ┌───────────────────┬───────────┴───────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
      Model            Evidence Rep         Harness / Orch       Retrieval Engine
   46 (42.6%)           40 (37.0%)            13 (12.0%)             9 (8.3%)
```

### Breakdown by Failure Class:
1. **Evidence Extraction Failure (40 tasks, 37.0%)** — *Evidence Representation Origin*:
   Raw search snippets contained excessive prose, noisy disclaimers, or competing figures. The 500M model failed to bind the correct number/date to the requested entity.
2. **Incorrect Computational Logic (16 tasks, 14.8%)** — *Model Origin*:
   Python code executed cleanly in the sandbox, but the mathematical formula or variable relationship was incorrect (e.g. inverted ratio, wrong trigonometric formulation).
3. **Unnecessary Tool Invocation (15 tasks, 13.9%)** — *Model Origin*:
   Model called `web_research` on core conceptual definitions (e.g. gradient descent, bias-variance tradeoff), receiving noisy search results and failing to answer clearly.
4. **Multi-Step Orchestration Absence (13 tasks, 12.0%)** — *Harness Origin*:
   Tasks requiring empirical web lookup followed by computation were forced into single-turn execution. The model guessed numbers and jumped straight to Python.
5. **Schema Syntax Malformation (12 tasks, 11.1%)** — *Model Origin*:
   Model generated invalid JSON, unescaped quotes, or malformed argument keys in Stage 1.
6. **Tool Execution Errors (7 tasks, 6.5%)** — *Retrieval Engine Origin*:
   DuckDuckGo search rate limits, timeouts, or empty snippet returns.
7. **Retrieval Service Failures & Other (5 tasks, 4.6%)**:
   Network disconnects, 403 HTTP errors, or runtime exceptions.

---

## 4. Part C: Experiment E Research Hypotheses

Experiment E formulates three testable scientific hypotheses:

### Hypothesis 1: Contrastive Decision-Boundary Calibration (H1)
*Fine-tuning on paired contrastive dyads with assistant-target loss masking will reduce false-positive tool calls on direct questions from $50.0\%$ to $\le 15.0\%$ without reducing Web selection recall below $80.0\%$ or Python selection recall below $90.0\%$.*
- **Mechanism**: Sharpens the linear separation in early Transformer layers between definitional/conceptual queries and empirical/temporal queries.

### Hypothesis 2: Structured Evidence Schema vs Raw Snippets (H2)
*Conditioning Stage 2 on a Structured Evidence Record (SER) schema rather than raw web snippets will increase factual evidence extraction accuracy on executed calls from $24.53\%$ to $\ge 55.0\%$.*
- **Mechanism**: Eliminates noisy non-factual tokens, reducing attention dispersion and allowing the 24 query heads to directly align target attributes with extracted key-values.

### Hypothesis 3: Factorial Independence & Synergy (H3)
*Stage 1 decision calibration and Stage 2 evidence synthesis operate on disjoint parameter subspaces and prompt prefixes. A joint curriculum (Branch D) will achieve additive performance gains without destructive interference.*

---

## 5. Part D: Dataset Design & Decontamination

Experiment E specifies a curated 1,000-record multi-turn dialogue corpus (`dataset_design.md`):

```
                        [Total Pool: 1,000 Records]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        Corpus 1: Calibration                   Corpus 2: Synthesis
         (500 Multi-Turn Records)                (500 Multi-Turn Records)
                 │                                       │
     ├── 150 Direct Conceptual               ├── 120 Single-Fact Extraction
     ├── 60 Direct Parametric                ├── 80 Multi-Entity Comparative
     ├── 40 Direct Negative Controls         ├── 100 Conflicting Source Handling
     ├── 125 Web Retrieval Calls             ├── 100 Insufficient Evidence Refusal
     ├── 75 Python Computation Calls         ├── 50 Grounded Python Math
     └── 50 Multi-Step Web -> Python         └── 50 Strict Inline Citations
```

### Partitioning & Leakage Elimination:
- **Splits**: 800 Train (80%) / 100 Validation (10%) / 100 Held-Out Evaluation (10%).
- **Strict Topic Isolation**: All records for a given entity or concept are quarantined within a single split.
- **Decontamination Gates**: 
  - 0 exact prompt/answer matches against the 150-task Stage 5 Benchmark.
  - Jaccard similarity on 10-gram shingles $< 0.15$.
  - Automated entity overlap filters.
- **Curation Proportions**: 40% manually authored gold records (400 records), 60% frontier-model synthetic records with human verification (600 records).

---

## 6. Part E: Evidence Representation & System Architecture

Detailed in `architecture_options.md`, we reject raw text concatenation and adopt a **Hybrid Structured Evidence Record (SER) + Claim-Evidence Mapping (CEM)** architecture:

```
[User Query]
     │
     ▼
[Stage 1: QORIN-500M] ──> Emits: {"name": "web_research", "arguments": {"query": "..."}}
     │
     ▼
[Search Sandbox] ──> Returns raw web snippets
     │
     ▼
[Harness Normalizer] ──> Parses into Structured Evidence Record (SER):
                          [SOURCE 1]: https://...
                          - ENTITY: ...
                          - METRIC: ...
                          - DATE: ...
     │
     ▼
[Stage 2: QORIN-500M] ──> Generates Claim-Evidence Mapping + Grounded Synthesis
```

### Multi-Step Orchestration Upgrade:
The harness will support an iterative 2-turn execution loop:
1. Turn 1 executes `web_research` and injects the SER.
2. The model may then either output the final answer OR emit a `python` tool call using the retrieved metrics as explicit numeric literals.
3. Turn 2 executes `python` and feeds the result back for final conversational synthesis.

---

## 7. Part F: Experimental Protocol & 4-Branch Ablation

Experiment E establishes a strict 4-branch factorial matrix (`experimental_protocol.md`):

| Branch | Identity | Training Curriculum | Target Parameters | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **Branch A** | Frozen Control | 0 steps (Exp D Step 425 Checkpoint) | Frozen | Reference baseline on 150 tasks. |
| **Branch B** | Calibration Only | Corpus 1 (500 records) | Stage 1 Decision Logits | Lower FPR on direct tasks to $\le 12\%$. |
| **Branch C** | Synthesis Only | Corpus 2 (500 records) | Stage 2 In-Context Heads | Elevate evidence extraction to $\ge 55\%$. |
| **Branch D** | Combined E | Joint Curricula (1,000 records) | Joint Stage 1 & 2 | Calibrated tool use + grounded synthesis. |

### Training Kinematics & Safety Safeguards:
- **Optimizer**: `AdamW8bit` (bitsandbytes), peak $\eta = 1.5 \times 10^{-5}$, cosine decay to $1.5 \times 10^{-6}$, 10% warmup, weight decay 0.05.
- **Batch Size**: 1 sequence per device, gradient accumulation = 4 (effective batch size 4).
- **Early Stopping**: Check validation loss every 25 steps; terminate if no improvement for 3 consecutive points (75 steps) or if validation loss diverges by $> 10\%$.
- **Hardware Duty-Cycle**: Max 105 minutes continuous GPU compute, followed by mandatory 15-minute idle cooldown.

---

## 8. Part G: Measurable Success Criteria

Pre-registered quantitative thresholds from `success_criteria.json`:

| Performance Metric | Exp C Baseline | Exp D Baseline | Experiment E Target | Minimum Acceptable Floor |
| :--- | :---: | :---: | :---: | :---: |
| **Unnecessary Tool-Call Rate (FPR)** | 8.33% | 50.00% | **$\le 10.0\%$** | $\le 20.0\%$ |
| **Web Selection Recall** | 0.00% | 80.26% | **$\ge 90.0\%$** | $\ge 80.0\%$ |
| **Python Selection Recall** | 94.74% | 92.11% | **$\ge 95.0\%$** | $\ge 88.0\%$ |
| **Evidence Extraction Accuracy** | 7.32% | 24.53% | **$\ge 65.0\%$** | $\ge 50.0\%$ |
| **Multi-Step Web $\to$ Python Success** | 0.00% | 0.00% | **$\ge 60.0\%$** | $\ge 33.3\%$ |
| **End-to-End Semantic Pass Rate** | 22.67% | 28.00% | **$\ge 50.0\%$** | $\ge 40.0\%$ |
| **McNemar $p$-value vs Exp D** | — | $p = 0.2012$ | **$p < 0.005$** | $p < 0.05$ |

---

## 9. Part H: Explicit Categorization of Knowledge State

In strict adherence to scientific integrity standards, we explicitly demarcate what is empirically proven from what is hypothesized or unresolved:

### Category 1: Measured Empirical Findings (Ground Truth)
- Exp D achieves 100% intent recall on tool-bearing queries, eliminating the 53 false negatives of Exp C.
- Exp D suffers a 50.00% false-positive rate on direct tasks, with all 18 false positives calling `web_research`.
- Exp D's net semantic pass gain (+8 tasks out of 150) is not statistically significant ($p = 0.2012$).
- The previously reported 98.11% evidence utilization rate was corrupted by including direct tasks in the numerator; the actual 3-word overlap among executed calls was 81.13%.
- Only 24.53% of executed tool calls in Exp D resulted in semantically correct final responses.
- Exp D failed 100% (0/15) of multi-step Web $\to$ Python tasks due to single-turn harness limitations.

### Category 2: Proposed Scientific Hypotheses
- Contrastive training pairs will restore decision boundary calibration without degrading tool recall (H1).
- Structured Evidence Records will reduce attention dispersion in a 500M parameter model and double evidence extraction accuracy (H2).
- Decision calibration and evidence synthesis can be fine-tuned simultaneously without destructive gradient interference (H3).

### Category 3: Design Decisions Established
- Experiment E will branch directly from frozen `ckpt_experiment_d_best.pt` without altering parent weights.
- A 4-branch factorial matrix (A: Baseline, B: Calibration, C: Synthesis, D: Combined) will be executed under identical greedy decoding settings.
- Evidence will be ingested via Structured Evidence Records (SER) and generated via Claim-Evidence Mappings (CEM).
- The harness will be extended to support a 2-turn execution loop for multi-step tasks.

### Category 4: Unresolved Uncertainties
- **Capacity Limits of 500M Architecture**: It remains an open empirical question whether a 514M parameter model can simultaneously master multi-hop reasoning, complex arithmetic, and strict attribution without exhibiting capacity saturation.
- **Search Engine Robustness**: DuckDuckGo's public search endpoint occasionally experiences rate-limiting, 403 blocks, or low-quality scraper sites. While SER parsing mitigates snippet noise, it cannot recover information if the search engine returns zero valid hits.
- **In-Context Length Tradeoffs**: Adding structured keys and claim-evidence chains increases Stage 2 context tokens by ~100 tokens, which may slightly increase generation latency on constrained GPU hardware.

---

## 10. Deliverable Verification & Next Steps

All 8 Stage 6 research artifacts have been generated in `modern_gpt/experiments/phase2_experiment_e/research_design/`:
1. `stage5_forensic_audit.md` (Detailed forensic reconciliation of all Stage 5 metrics)
2. `failure_taxonomy.json` (Structured JSON catalog of all 108 Exp D failures)
3. `experiment_e_hypotheses.md` (Formal hypotheses, mechanisms, and falsification criteria)
4. `dataset_design.md` (1,000-sample dataset architecture, splits, and leakage controls)
5. `architecture_options.md` (Evaluation of 5 evidence representations and SER recommendation)
6. `experimental_protocol.md` (4-branch factorial ablation kinematics, LR schedule, duty cycle)
7. `success_criteria.json` (Pre-registered quantitative performance targets)
8. `research_design_report.md` (This comprehensive executive synthesis)
9. `SHA256SUMS` (Cryptographic checksum manifest for all artifacts)

No weights were altered, no training was conducted, and the live model registry remains unchanged. The project is fully primed for formal execution of Experiment E upon user authorization.
