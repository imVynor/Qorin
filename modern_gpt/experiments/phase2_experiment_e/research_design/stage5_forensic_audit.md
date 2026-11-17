# QORIN Stage 5: Independent Forensic Audit & Reconciliation Report

**Audit Date**: November 17, 2026  
**Auditor**: Autonomous Forensic Evaluation Harness  
**Subject Baseline Checkpoint**: QORIN-500M Experiment C Final (`ckpt_experiment_c_final.pt`, SHA-256: `6cb51258...`)  
**Subject Target Checkpoint**: QORIN-500M Experiment D Step 425 Best (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388da...`)  
**Evaluation Benchmark**: 150-task held-out generalization suite (`benchmark.jsonl`)  
**Raw Task Records**: `experiment_c_results.jsonl`, `experiment_d_results.jsonl`, `tool_traces.jsonl`

---

## 1. Executive Summary

An independent, line-by-line programmatic audit of the raw evaluation logs and execution traces from the Stage 5 Generalization Audit was conducted. This audit validates all primary findings, reconciles aggregate statistics with task-level ground truth, and resolves seven specific apparent anomalies identified between the summary report (`generalization_report.md`) and raw JSONL records.

### Key Audit Conclusions:
1. **Experiment D achieves dramatic tool-use activation**: Tool intent recall increased from **53.51%** (Exp C) to **100.00%** (Exp D), and valid Web retrieval dispatch increased from **0/76 (0.00%)** to **61/76 (80.26%)**.
2. **Experiment D suffers from severe calibration collapse on direct questions**: Exp D invoked tools unnecessarily on **18 out of 36 (50.00%)** direct-answer tasks, compared to **3 out of 36 (8.33%)** for Exp C. All 18 false positives were unnecessary `web_research` calls.
3. **Evidence-grounded synthesis remains the primary failure bottleneck**: Despite 106 tool calls executing successfully, only **26 of those 106 tasks (24.53%)** achieved semantic correctness. The model frequently extracted incorrect numbers from dense snippets or overrode retrieved facts with parametric hallucination.
4. **End-to-end semantic improvement is not statistically significant at $\alpha = 0.05$**: Across all 150 tasks, Exp D achieved **42 semantic passes (28.00%)** vs Exp C's **34 semantic passes (22.67%)**. McNemar's test with Edwards continuity correction yields $\chi^2 = 1.6333$, two-tailed $p = 0.2012$.
5. **A reporting defect was identified in the Evidence Utilization metric**: The previously reported 98.11% evidence utilization rate in Exp D was distorted by adding 18 direct-response tasks into the numerator (`104 / 106 = 0.9811`). Among actually executed tool calls, the 3-word n-gram overlap rate was **81.13% (86 / 106)**. Furthermore, this n-gram overlap metric is superficial and does not reflect factual correctness.

---

## 2. Recomputed Confusion Matrices & Core Metrics

All metrics were recomputed directly from the 150 raw records in `experiment_c_results.jsonl` and `experiment_d_results.jsonl`, joined with `benchmark.jsonl`. 95% Confidence Intervals are calculated using the Wilson score method.

### 2.1 Tool Intent Recognition Confusion Matrix

| Metric | Experiment C (Baseline) | Experiment D (Step 425) | Delta |
| :--- | :---: | :---: | :---: |
| **True Positives ($TP$)** | 61 | **114** | +53 |
| **False Positives ($FP$)** | **3** | 18 | +15 (Degradation) |
| **False Negatives ($FN$)** | 53 | **0** | -53 (Eliminated) |
| **True Negatives ($TN$)** | **33** | 18 | -15 |
| **Intent Precision** | **95.31%** [87.10%, 98.39%] | 86.36% [79.47%, 91.20%] | -8.95% |
| **Intent Recall** | 53.51% [44.39%, 62.40%] | **100.00%** [96.74%, 100.00%] | **+46.49%** |
| **Intent F1-Score** | 0.6854 | **0.9268** | **+0.2414** |
| **Overall Accuracy** | 62.67% [54.70%, 70.00%] | **88.00%** [81.83%, 92.27%] | **+25.33%** |

*Note: Total tasks = 150. Positive intent expected = 114 tasks; Direct response expected = 36 tasks.*

---

### 2.2 Tool Selection Accuracy & Recall

| Dimension | Experiment C | Experiment D | Delta |
| :--- | :---: | :---: | :---: |
| **Overall Tool Selection Accuracy** | 31.58% [23.77%, 40.59%] (36/114) | **84.21%** [76.42%, 89.77%] (96/114) | **+52.63%** |
| **Web Retrieval Recall** | 0.00% [0.00%, 4.81%] (0/76) | **80.26%** [69.96%, 87.66%] (61/76) | **+80.26%** |
| **Python Execution Recall** | **94.74%** [82.71%, 98.54%] (36/38) | 92.11% [79.20%, 97.28%] (35/38) | -2.63% |

---

### 2.3 Execution Funnel & Behavioral Progression

| Funnel Stage | Experiment C | Experiment D | Analysis |
| :--- | :---: | :---: | :--- |
| **1. Tool Calls Attempted** | 64 | 132 | Exp D generated tool syntax on 132 tasks (+68 calls) |
| **2. Schema Validity Rate** | 90.62% (58/64) | **93.18%** (123/132) | Exp D improved JSON argument compliance (+2.56%) |
| **3. Dispatch Rate** (of total 150) | 38.67% (58/150) | **82.00%** (123/150) | Valid calls routed to tool sandboxes (+43.33%) |
| **4. Execution Success Rate** | 70.69% (41/58) | **86.18%** (106/123) | Sandbox execution without crash/timeout (+15.49%) |
| **5. Executed Tasks Count** | 41 unique tasks | **106 unique tasks** | Real external tool outputs returned (+65 tasks) |
| **6. Raw Evidence Word Overlap ($\ge 3$)** | 7.32% (3/41) | **81.13%** (86/106) | Response contained $\ge 3$ words from tool output |
| **7. End-to-End Semantic Pass** | 22.67% (34/150) | **28.00%** (42/150) | Passed all factual & uncertainty criteria (+5.33%) |

---

## 3. Investigation of Seven Apparent Inconsistencies

### Inconsistency 1: Why does the report show 100% Web selection in colloquial and current-information categories, but only 80.26% overall Web recall?
**Resolution & Ground Truth**:
Across the entire 150-task benchmark, exactly **76 tasks** expected `web_research`. These 76 tasks were distributed across five distinct benchmark categories:
1. `colloquial_web_research` (30 tasks): Exp D selected `web_research` on **30/30 (100.00%)**.
2. `current_information` (20 tasks): Exp D selected `web_research` on **20/20 (100.00%)**.
3. `tool_discrimination` (6 web-expected tasks): Exp D selected `web_research` on **6/6 (100.00%)**.
4. `tool_failure_recovery` (5 web-expected tasks): Exp D selected `web_research` on **5/5 (100.00%)**.
5. `multistep_web_python` (15 web-first expected tasks): Exp D selected `web_research` on **0/15 (0.00%)**.

In every one of the 15 `multistep_web_python` tasks (e.g. `multistep_01`: "Find the annual fiscal year 2023 revenue of Apple and Microsoft, then calculate the difference"), the prompt required first retrieving current empirical numbers via Web and then computing differences via Python. However, because QORIN's single-turn dispatch harness did not provide multi-step tool-chaining, Exp D immediately emitted a `<|tool_call|>` for `python` with fabricated/estimated values (e.g. `apple = 383; msft = 211; diff = apple - msft`), completely bypassing the Web search phase.
- Recomputed Total Web Recall: $(30 + 20 + 6 + 5 + 0) / 76 = 61 / 76 = \mathbf{80.26\%}$.
- Conclusion: The report's category-level 100% numbers were mathematically correct for those isolated categories, but the overall recall dropped to 80.26% solely due to the 0/15 failure in multi-step tool chaining.

---

### Inconsistency 2: Verify the 50% unnecessary tool-call rate and its exact denominator.
**Resolution & Ground Truth**:
In `benchmark.jsonl`, exactly **36 tasks** expected a direct response (`expected_intent: false` and `expected_tool: null`):
- `direct_conceptual`: 25 tasks (e.g. "What is backpropagation?", "Explain bias-variance tradeoff").
- `tool_discrimination`: 7 tasks (contrast-pair negative controls designed to test parametric knowledge).
- `tool_failure_recovery`: 4 tasks (unfulfillable, speculative, or conversational prompts requiring direct uncertainty acknowledgment).

Task-level audit of model decisions on these 36 tasks:
- **Experiment D invoked a tool on 18 out of 36 direct tasks ($\mathbf{50.00\%}$ False Positive Rate)**:
  - `direct_conceptual`: **10 / 25 (40.00%)** triggered `web_research`.
  - `tool_discrimination`: **4 / 7 (57.14%)** triggered `web_research`.
  - `tool_failure_recovery`: **4 / 4 (100.00%)** triggered `web_research`.
- **Tool Type**: In all 18 cases, the invoked tool was `web_research`. Zero direct tasks invoked `python`.
- **Experiment C Baseline**: Exp C invoked a tool on only **3 out of 36 direct tasks (8.33% FPR)**.
- Conclusion: The 50.00% unnecessary tool-call rate is verified. The denominator is exactly 36 direct tasks. Experiment D suffered a significant loss of calibration on direct queries, developing an aggressive prior that favors Web retrieval over parametric recall.

---

### Inconsistency 3: Recompute McNemar's test from the actual paired task outcomes.
**Resolution & Ground Truth**:
We cross-tabulated the paired semantic correctness of both models for all 150 tasks:

```
                      Experiment D Success    Experiment D Failure    Total
Experiment C Success         n11 = 23                n01 = 11           34
Experiment C Failure         n10 = 19                n00 = 97          116
Total                        42                     108                150
```

- Discordant pairs:
  - $b = n_{10} = 19$ (Tasks where Exp D succeeded and Exp C failed)
  - $c = n_{01} = 11$ (Tasks where Exp C succeeded and Exp D failed)
- McNemar's test with Edwards continuity correction:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|19 - 11| - 1)^2}{19 + 11} = \frac{(8 - 1)^2}{30} = \frac{49}{30} \approx \mathbf{1.6333}$$
  - $p\text{-value} = P(\chi^2_1 \ge 1.6333) = \mathbf{0.2012}$ (two-tailed).
- Standard McNemar's test (without continuity correction):
  $$\chi^2 = \frac{(19 - 11)^2}{30} = \frac{64}{30} \approx 2.1333 \implies p = 0.1441$$
- Conclusion: Under both formulations, $p > 0.05$. The overall net improvement (+8 tasks out of 150, or +5.33%) is **not statistically significant** at the standard 5% significance level. While Exp D gained 19 tasks through successful tool retrieval, it simultaneously lost 11 tasks that Exp C answered correctly from parametric memory, primarily due to unnecessary Web calls degrading into incorrect snippets.

---

### Inconsistency 4: Check whether the 106 tool executions represent unique tasks or multiple calls per task.
**Resolution & Ground Truth**:
Inspection of `tool_traces.jsonl` reveals:
- Total trace records: 181 (123 for `exp_d`, 58 for `exp_c`).
- Unique tasks among `exp_d` traces: **123 unique tasks**.
- Trace execution statuses for Exp D: **106 completed**, **17 failed**.
- Conclusion: The 106 completed tool executions correspond to **106 unique tasks**. There were zero duplicate calls or multi-turn loops. Each task had at most one tool call executed during Stage 1.

---

### Inconsistency 5: Verify that 98.11% evidence utilization actually means relevant evidence was used, rather than merely that retrieved text appeared in the prompt.
**Resolution & Ground Truth**:
Two distinct issues were discovered:
1. **Metric Calculation Defect**:
   In `run_audit_evaluation.py` (lines 271-272), when a task had `detected_intent == False` (direct response), the evaluator assigned `evidence_utilized = True` under the logic that evidence was "not applicable".
   In `compute_audit_metrics.py` (lines 99-100), the formula was:
   $$\text{evidence\_utilization\_rate} = \frac{\sum \text{evidence\_utilized}}{\text{executed\_count}} = \frac{86 + 18}{106} = \frac{104}{106} = \mathbf{98.11\%}$$
   The numerator improperly included 18 direct tasks! For Exp C, this same formula produced $\frac{3 + 86}{41} = \mathbf{217.07\%}$ in `metrics.json`.
   - **Corrected Execution-Conditioned Overlap**: Among the 106 executed tasks, exactly **86 tasks (81.13%)** had $\ge 3$ overlapping words between the tool output and the final response. 20 executed tasks failed to include even 3 words from the tool output.
2. **Superficial Word Overlap vs. Semantic Grounding**:
   The criterion `len(overlap) >= 3` required only three shared alphanumeric words of length $\ge 3$ (e.g. "was", "for", "the", "system").
   - Out of the 106 executed tasks, only **26 tasks (24.53%)** achieved semantic correctness.
   - **80 executed tasks (75.47%)** failed semantic criteria despite passing the 3-word overlap check.
   - Conclusion: The 98.11% figure was an artifact of metric compounding and superficial token overlap. In reality, true evidence-grounded reasoning occurred in fewer than 25% of executed cases.

---

### Inconsistency 6: Determine whether the semantic evaluator was blind to model identity.
**Resolution & Ground Truth**:
Inspection of `run_audit_evaluation.py` (`check_semantic_correctness`, lines 93-130) verifies:
- The function signature is `check_semantic_correctness(task, final_answer, tool_trace)`.
- The evaluation function contains **zero parameters or conditional checks for model name, checkpoint version, or `model_tag`**.
- Scoring is strictly deterministic and programmatic based on task-level ground-truth strings, numerical tolerances, and uncertainty phrase lists.
- Conclusion: The evaluator was **100% blind** to model identity, eliminating any possibility of model-specific evaluation bias.

---

### Inconsistency 7: Check whether task-level results support every aggregate reported in generalization_report.md.
**Resolution & Ground Truth**:
Every single aggregate table and metric reported in `generalization_report.md` was compared against the recomputed raw counts:
- Precision (86.36%), Recall (100.00%), F1 (0.9268): **Exact Match**.
- Web selection accuracy (80.26%), Python selection accuracy (92.11%): **Exact Match**.
- Schema validity (93.18%), Dispatch (82.00%), Execution (86.18%): **Exact Match**.
- Semantic Pass Rate (28.00% vs 22.67%): **Exact Match**.
- The only discrepancy found is the aforementioned `evidence_utilization_rate` numerator defect, which has now been fully documented and isolated.

---

## 4. Comprehensive Audit Reconciliation Table

| Metric Field | Report Summary Value | Recomputed Raw Value | Reconciliation Status | Diagnostic Finding |
| :--- | :---: | :---: | :---: | :--- |
| **Total Benchmark Tasks** | 150 | 150 | Verified | 100% match |
| **Exp D Intent TP / FP / TN / FN** | 114 / 18 / 18 / 0 | 114 / 18 / 18 / 0 | Verified | Exact match |
| **Exp C Intent TP / FP / TN / FN** | 61 / 3 / 33 / 53 | 61 / 3 / 33 / 53 | Verified | Exact match |
| **Exp D Web Recall (All Tasks)** | 80.26% | 80.26% (61/76) | Verified | 15 multi-step tasks missed Web |
| **Exp D Web Recall (Colloquial)** | 100.00% | 100.00% (30/30) | Verified | Category subset |
| **Exp D Web Recall (Current Info)** | 100.00% | 100.00% (20/20) | Verified | Category subset |
| **Exp D Unnecessary Tool Calls** | 50.00% | 50.00% (18/36) | Verified | Denominator = 36 direct tasks |
| **McNemar $\chi^2$ (Edwards corr.)** | 1.6333 | 1.6333 ($p=0.2012$) | Verified | $p > 0.05$ (Not statistically significant) |
| **Exp D Executed Unique Tasks** | 106 | 106 | Verified | 106 distinct tasks executed |
| **Exp D Evidence Overlap Rate** | 98.11% | **81.13%** (86/106) | **Corrected** | Original included 18 direct tasks in numerator |
| **Exp D Semantic Pass on Executed** | Not reported | **24.53%** (26/106) | **Discovered** | 80 of 106 executed calls failed semantic test |
| **Evaluator Model Blindness** | Claimed Blind | **100% Blind** | Verified | Evaluator uses rule-based string matching |

---

## 5. Architectural Implications for Experiment E

1. **Intervention 1 must target Decision Calibration**: Experiment D has become an aggressive tool-caller. Fine-tuning must introduce hard negative controls where calling a tool is penalized, preserving direct answers for conceptual questions.
2. **Intervention 2 must target Evidence-Grounded Synthesis**: Over 75% of successful tool executions failed to translate into a correct answer. Providing raw, unstructured web snippets into Stage 2 overwhelms the 500M model. A structured evidence representation (key-value extraction, claim-evidence pairs) is strictly necessary to bridge the gap between retrieval and reasoning.
