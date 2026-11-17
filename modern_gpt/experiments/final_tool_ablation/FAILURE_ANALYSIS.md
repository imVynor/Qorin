# QORIN Failure Analysis: Controlled Tool-Augmentation Ablation

**Experiment**: Final Controlled Tool-Augmentation Ablation  
**Date**: 2026-11-17  
**Benchmark**: 100 Held-Out Tasks (Stage 13 Stratified Test Split)  
**Evaluated Models**: QORIN-500M (`ckpt_pilot_best.pt`), QORIN-100M (`best_val.pt`)

---

## 1. Executive Failure Taxonomy

While tool augmentation yielded a statistically significant overall performance increase for QORIN-500M (+21.0% points, $p = 0.000328$), fine-grained error analysis reveals critical failure modes and boundaries.

Across the 36 failed tasks in the tools-enabled condition for QORIN-500M and the 71 failed tasks for QORIN-100M, failures cluster into four primary categories:

```
[Tool Augmentation Failures]
 ├── 1. Epistemic Calibration Breakdown (18 tasks / 50.0% of failures)
 │    ├── Insufficient evidence hallucination (5 tasks)
 │    ├── Contradictory evidence collapse (8 tasks)
 │    └── Partial support over-confidence (5 tasks)
 ├── 2. Parametric Capacity Threshold Deficit (100M model breakdown)
 │    ├── Schema formatting errors (Syntax/JSON invalidity)
 │    └── Context distraction from tool observations
 ├── 3. Single-Turn Tool Under-Invocation (8 tasks / 22.2% of failures)
 │    ├── Direct mental math failure on calculation
 │    └── Conservative direct lookup on single-turn web
 └── 4. Execution Latency & Trajectory Overhead (1.92x latency penalty)
```

---

## 2. Failure Mode 1: Epistemic Calibration & Non-Ideal Evidence Handling

The model struggled most severely in three categories where the correct scientific behavior is **hedging or stating uncertainty**:

### 1.1 Insufficient Evidence ($N=5$, Success = 0.0%)
- **Target Behavior**: When the retrieved web passage or database returns zero relevant metrics or explicitly lacks information, the model should state: *"The retrieved evidence does not contain the required valuation/record."*
- **Observed Failure**: In 5 of 5 tasks (`syn_insuff_002`, `syn_insuff_014`, `syn_insuff_028`, `syn_insuff_039`, `syn_insuff_047`), the model instead attempted to synthesize a plausible-sounding factual answer or cited generic background text, exhibiting false certainty.

### 1.2 Contradictory Evidence ($N=8$, Success = 0.0%)
- **Target Behavior**: When two independent sources provide divergent figures (e.g., Agency Alpha reports \$4.2M while Agency Beta reports \$4.8M), the model must synthesize both perspectives: *"Estimates diverge between \$4.2M [1] and \$4.8M [2]."*
- **Observed Failure**: In 8 of 8 contradictory tasks (`syn_contra_021`, `mturn_contra_007`, etc.), the model exhibited **unilateral confirmation bias**, picking the first numeric value encountered in Turn 1 and ignoring the conflicting source in Turn 2.

### 1.3 Partial Support ($N=5$, Success = 0.0%)
- **Target Behavior**: Explicitly stating that the data is an estimate, preliminary, or only covers a subset of the requested parameters.
- **Observed Failure**: The model dropped hedging adverbs and presented partial observations as exhaustive facts.

---

## 3. Failure Mode 2: The Parametric Capacity Floor (Why Tools Hurt 100M)

The most striking empirical finding of this ablation is the negative net tool gain for QORIN-100M:
- **100M Tools Disabled**: 36.0% (36/100)
- **100M Tools Enabled**: 29.0% (29/100)
- **Net Delta**: **-7.0 percentage points (Relative: -19.4%)**

### Root Cause Analysis:
1. **Schema Malformation**: At 120M parameters, the model frequently produced malformed tool call JSON (missing quotes, escaped brackets, or invalid keys), causing the harness parser to fail and trigger fallback re-prompts.
2. **Context Distraction**: When large tool observations (300–500 tokens of raw text) were injected into the 100M model's context window, its attention distribution dispersed, causing it to lose the original user instruction and repeat tool observations verbatim rather than synthesizing a final answer.
3. **Capacity Threshold**: Tool augmentation requires sufficient representational capacity (achieved at 500M, but absent at 100M) to maintain instruction adherence while processing noisy external tokens.

---

## 4. Failure Mode 3: Single-Turn Under-Invocation Bias

In Experiment F, multi-turn sequential data was heavily weighted to overcome earlier deadlocks. An unintended side effect was **under-invocation on simple single-step tasks**:
- On `CAL_DIRECT_PROVIDED_CALC` ($N=6$), the 500M model only invoked Python in 1 of 6 tasks, attempting mental arithmetic on the remaining 5 and getting 4 wrong.
- Similarly, on single-turn web lookup queries, the model exhibited a conservative bias toward direct answers.

---

## 5. Failure Mode 4: Latency & Efficiency Trade-offs

| Condition | Mean Latency | Model Generation | Tool Execution | Tokens Out |
|---|---:|---:|---:|---:|
| Tools Disabled | **1.291s** | 1.291s | 0.000s | 52.4 |
| Tools Enabled | **2.483s** | 2.470s | 0.013s | 80.5 |
| **Overhead** | **+1.192s (+92.3%)** | +1.179s | +0.013s | +28.1 |

- External tool calls require multi-turn conversational loops.
- Average response time nearly doubled ($1.92\times$), representing a clear performance-versus-latency trade-off that must be considered in resource-constrained environments.
