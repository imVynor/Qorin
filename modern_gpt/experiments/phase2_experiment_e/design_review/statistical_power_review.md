# QORIN Experiment E: Statistical Power Review, Uncertainty Bounds & Pre-Registration

**Document**: `statistical_power_review.md`  
**Status**: Formal Statistical Review & Pre-Registration  
**Baseline Dataset**: 150-Task Held-Out Stage 5 Generalization Suite (`benchmark.jsonl`)  
**Parent Baseline**: Frozen Experiment D Checkpoint (`ckpt_experiment_d_best.pt`, 42/150 Semantic Passes)

---

## 1. Executive Summary & Statistical Feasibility

A rigorous statistical power analysis was conducted on all proposed target metrics for Experiment E. 

### Key Power Findings:
1. **Primary Outcome (Overall Semantic Pass Rate)** is **adequately powered** on the 150-task benchmark ($N = 150$). Achieving the target semantic pass rate of $\ge 50.0\%$ (75/150) against the Experiment D baseline of $28.0\%$ (42/150) yields a paired McNemar test statistic of $\chi^2 = 21.7872$ ($p = 3.05 \times 10^{-6}$), easily satisfying $p < 0.005$. Even reaching the conservative floor threshold of $40.0\%$ (60/150) achieves $p = 0.0027 < 0.005$.
2. **Category Subsets are Substantially Underpowered**:
   - `multistep_web_python` ($N = 15$ tasks): A 60.0% success rate (9/15) produces a wide Wilson 95% Confidence Interval of **$[35.7\%, 80.2\%]$** (margin of error $\pm 22.2\%$).
   - `direct_conceptual` FPR ($N = 36$ tasks): An 8.3% FPR (3/36) yields a Wilson 95% CI of **$[2.9\%, 21.8\%]$** (margin of error $\pm 9.5\%$).
3. **Statistical Invariance & Claims Mandate**: We do **not** promise $p < 0.005$ on category-level sub-metrics. High statistical significance ($p < 0.005$) is pre-registered strictly for the full-benchmark primary outcome. For robust category-level evaluations, an expanded 250-task benchmark is formally specified.

---

## 2. Metric Specifications, Denominators & Uncertainty Bounds

| Metric Identifier | Exact Numerator ($k$) & Denominator ($N$) | Benchmark Subset ($N$) | Exp D Baseline | Exp E Target | Exp E 95% Wilson CI | CI Margin ($\pm$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Unnecessary Tool-Call Rate (FPR)** | $\frac{N_{\text{tool calls emitted}}}{N_{\text{direct tasks}}}$ | Direct Tasks ($N = 36$) | 50.00% (18/36) | **$\le 10.0\%$** (3/36) | $[2.9\%, 21.8\%]$ | $\pm 9.5\%$ |
| **2. Web Selection Recall** | $\frac{N_{\text{web selected}}}{N_{\text{web expected}}}$ | Web Tasks ($N = 76$) | 80.26% (61/76) | **$\ge 90.0\%$** (68/76) | $[80.6\%, 94.6\%]$ | $\pm 7.0\%$ |
| **3. Python Selection Recall** | $\frac{N_{\text{python selected}}}{N_{\text{python expected}}}$ | Python Tasks ($N = 38$) | 92.11% (35/38) | **$\ge 95.0\%$** (36/38) | $[82.7\%, 98.5\%]$ | $\pm 7.9\%$ |
| **4. Evidence Extraction Accuracy** | $\frac{N_{\text{correct fact extractions}}}{N_{\text{executed web calls}}}$ | Executed Web ($N \approx 70$) | 24.53% (26/106) | **$\ge 65.0\%$** (46/70) | $[53.8\%, 74.7\%]$ | $\pm 10.5\%$ |
| **5. Multi-Step Success Rate** | $\frac{N_{\text{multistep pass}}}{N_{\text{multistep tasks}}}$ | Multi-Step ($N = 15$) | 0.00% (0/15) | **$\ge 60.0\%$** (9/15) | $[35.7\%, 80.2\%]$ | $\mathbf{\pm 22.2\%}$ |
| **6. End-to-End Semantic Pass** *(Primary)* | $\frac{N_{\text{semantic pass}}}{N_{\text{total tasks}}}$ | Full Suite ($N = 150$) | 28.00% (42/150) | **$\ge 50.0\%$** (75/150) | $[42.1\%, 57.9\%]$ | $\mathbf{\pm 7.9\%}$ |

---

## 3. Paired Statistical Hypothesis Testing Framework

All comparisons between Experiment D and Experiment E will be evaluated using paired tests across identical benchmark tasks:

### 3.1 Primary Hypothesis Test: McNemar's Test (Full Benchmark, $N = 150$)
- **Null Hypothesis ($H_0$)**: $P(\text{Exp E pass, Exp D fail}) = P(\text{Exp D pass, Exp E fail})$ (No difference in marginal success proportions).
- **Alternative Hypothesis ($H_1$)**: $P(\text{Exp E pass, Exp D fail}) > P(\text{Exp D pass, Exp E fail})$.
- **Test Statistic**: Edwards continuity-corrected chi-square:
  $$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad \text{where } b = n_{10} \text{ (E win, D loss)}, \ c = n_{01} \text{ (D win, E loss)}$$

#### Power Trajectory on $N = 150$:
- **At Target ($50.0\%$ Semantic Pass, 75/150)**:
  Assuming conservative retention ($c \le 7$ regressions, $b \ge 40$ wins):
  $$\chi^2 = \frac{(|40 - 7| - 1)^2}{40 + 7} = \frac{1024}{47} \approx 21.79 \implies \mathbf{p = 3.05 \times 10^{-6}}$$
- **At Floor ($40.0\%$ Semantic Pass, 60/150)**:
  With $c \le 7$ regressions, $b \ge 25$ wins:
  $$\chi^2 = \frac{(|25 - 7| - 1)^2}{25 + 7} = \frac{289}{32} \approx 9.03 \implies \mathbf{p = 0.0027}$$
- **Conclusion**: The primary outcome has statistical power $(1 - \beta) > 0.95$ to detect a 12-percentage-point gain at $\alpha = 0.01$.

---

## 4. Pre-Registration of Outcomes

### Primary Outcome:
- **Benchmark-Wide Semantic Pass Rate**: Must exceed $\ge 40.0\%$ (floor) with paired McNemar $p < 0.01$ vs Experiment D.

### Secondary Outcomes (Exploratory / Diagnostic):
1. **Unnecessary Tool-Call Rate on Direct Tasks**: Directional reduction from $50.0\%$ to $\le 15.0\%$.
2. **Web Selection Recall**: Maintained at $\ge 85.0\%$.
3. **Python Selection Recall**: Maintained at $\ge 90.0\%$.
4. **Claim-Level Factual Precision ($CP$)**: Measured via the claim-grounding protocol on the 50-task Gold Sub-Suite ($\ge 70.0\%$).
5. **Multi-Step Web $\to$ Python Chaining**: Directional activation from $0/15$ to $\ge 6/15$ ($40.0\%$).

---

## 5. Specification for an Expanded 250-Task Benchmark Suite

To reduce category-level uncertainty in subsequent evaluation stages (Stage 7+), we specify an expanded **250-Task Frozen Benchmark Suite**:

```
                         [Expanded Suite: 250 Tasks]
                                      │
     ┌──────────────────┬─────────────┼─────────────┬──────────────────┐
     ▼                  ▼             ▼             ▼                  ▼
Direct Concept     Current Info   Tool Discr    Multi-Step      Recovery/Refusal
 (50 Tasks)         (50 Tasks)    (50 Tasks)    (50 Tasks)         (50 Tasks)
```

### Impact of Sample Expansion:
- **Multi-Step Tasks**: Expands from $N = 15$ to $N = 50$. At $60\%$ target success (30/50), the 95% CI narrows from $\mathbf{\pm 22.2\%}$ to **$\pm 13.6\%$**, enabling robust statistical testing ($p < 0.001$).
- **Direct Tasks**: Expands from $N = 36$ to $N = 75$, tightening the FPR confidence interval margin from $\mathbf{\pm 9.5\%}$ to **$\pm 5.4\%$**.
