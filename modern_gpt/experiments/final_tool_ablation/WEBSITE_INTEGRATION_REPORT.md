# QORIN — Website Integration Report: Controlled Tool-Augmentation Ablation

**Date:** November 17, 2026  
**Auditor:** ML Evaluation & Frontend Architecture Verification  
**Evaluation Scope:** Integration of verified empirical findings from the 100-task controlled tool-augmentation ablation into the QORIN research web platform.  
**Build Status:** Verified passing (`tsc -b && vite build --mode pages` exited with code 0).  
**Test Status:** Verified passing (`test_final_tool_ablation.py` 3/3 passed, `test_session_orchestration_regression.py` 6/6 passed).

---

## 1. Overview of Website Changes

The research website has been updated to showcase the definitive empirical resolution to the central research question:
*"How does increasing the capacity of a small decoder-only Transformer affect conversational LM performance, and can external retrieval compensate for smaller model capacity?"*

### Modified & Created Files

| File Path | Nature of Modification |
| :--- | :--- |
| `web/src/data/chartData.ts` | Added `controlledAblationSummary`, `controlledAblationCategories`, and `ControlledAblationCategory` interface. |
| `web/src/data/failuresData.ts` | Added `FAIL-09-100M-TOOL-DEGRADATION` documenting the 100M scale schema parsing failure and capacity threshold. |
| `web/src/data/experimentsData.ts` | Registered `EXP-FINAL-TOOL-ABLATION` with complete verified metadata and quantitative metrics. |
| `web/src/data/journeyData.ts` | Refactored Chapter 8 into *"Chapter 8: What Did We Actually Learn? The Empirical Verdict"* and authored *"Chapter 9: The Next Question: Five Billion Tokens"*. |
| `web/src/components/visualizations/ControlledAblationFigure.tsx` | Created interactive visualization with 4 tabs: Primary 500M Ablation, 2×2 Capacity Matrix, Category Breakdown, and Latency & Efficiency. |
| `web/src/components/visualizations/ControlledAblationFigure.module.css` | Authored clean CSS modules adhering to Apple/Linear/Vercel design tokens, zero generic AI tropes. |
| `web/src/pages/Home.tsx` | Added dedicated empirical resolution section: *"Do External Tools Actually Improve Small LMs?"* embedding the interactive figure, and updated narrative to the 9-Chapter Journey. |
| `web/src/pages/Research.tsx` | Added Section 11.2 detailing formal statistical hypothesis ($H_0$ vs $H_1$), mathematical McNemar formulation ($\chi^2 = 12.90, p = 0.000328$), and Section 13.2 detailing the 5.0 Billion Token pretraining roadmap. |
| `web/src/pages/Experiments.tsx` | Added Cluster 09: *"Controlled Tool-Augmentation Ablation (Central Question)"* with interactive toggles and linked experiment cards. |
| `web/src/pages/Journey.tsx` | Updated to display all 9 chronological research chapters with the final empirical verdict and future pretraining frontier. |
| `web/src/pages/Model.tsx` | Added controlled tool ablation verdict to the Generation IV (QORIN-514M) technical specification. |
| `modern_gpt/experiments/experiment_registry.jsonl` | Appended `EXP-FINAL-TOOL-ABLATION` record. |
| `modern_gpt/reports/final_research_audit/experiment_registry.json` | Appended canonical `final_tool_ablation` experiment entry. |
| `modern_gpt/tests/test_final_tool_ablation.py` | Added automated regression test suite verifying manifest integrity and metric consistency. |

---

## 2. Quantitative Data Consistency Audit

Every number rendered across the user interface was audited against the raw execution records in `modern_gpt/experiments/final_tool_ablation/`:

| Metric | Source JSON Value | UI Display Value | Consistency Verdict |
| :--- | :--- | :--- | :--- |
| Sample Size | 100 | 100 tasks | Verified Exact |
| QORIN-500M Tools Disabled | 43 (43.0%) | 43.0% (43/100) [95% CI: 34–53%] | Verified Exact |
| QORIN-500M Tools Enabled | 64 (64.0%) | 64.0% (64/100) [95% CI: 55–74%] | Verified Exact |
| QORIN-500M Absolute Delta | +21.0% pts | +21.0% pts | Verified Exact |
| QORIN-500M Relative Delta | +48.8% | +48.8% | Verified Exact |
| McNemar Chi-Square ($\chi^2$) | 12.9032 | 12.9032 | Verified Exact |
| McNemar $p$-value | 0.000328 | 0.000328 ($p < 0.001$) | Verified Exact |
| Discordant Pairs (Improved / Regressed) | 26 / 5 | 26 improved / 5 regressed | Verified Exact |
| Concordant Pairs (Both Pass / Both Fail) | 38 / 31 | 38 pass / 31 fail | Verified Exact |
| QORIN-100M Tools Disabled | 36 (36.0%) | 36.0% (36/100) | Verified Exact |
| QORIN-100M Tools Enabled | 29 (29.0%) | 29.0% (29/100) | Verified Exact |
| QORIN-100M Net Tool Gain | -7.0% pts | -7.0% pts (Regression) | Verified Exact |
| Parametric Knowledge Delta | 12.5% &rarr; 100.0% | +87.5% pts | Verified Exact |
| Multi-Step Reasoning Delta | 36.7% &rarr; 66.7% | +30.0% pts | Verified Exact |
| Conversational Follow-Up Delta | 25.0% &rarr; 100.0% | +75.0% pts | Verified Exact |
| Numerical Computation | 33.3% &rarr; 33.3% | 0.0% pts | Verified Exact |
| Contradictory Evidence | 0.0% &rarr; 0.0% | 0.0% pts | Verified Exact |
| Insufficient Evidence | 0.0% &rarr; 0.0% | 0.0% pts | Verified Exact |
| Mean Latency (Disabled / Enabled) | 1.291s / 2.483s | 1.291s / 2.483s ($1.92\times$) | Verified Exact |

---

## 3. UI/UX Verification & Compliance

1. **Aesthetic Principles:** The implementation strictly adheres to the workspace design rules (Stripe/Linear/Vercel aesthetic, CSS modules, design tokens, high-contrast readable typography, semantic HTML `<figure>`, `<article>`, `<section>`, `<header>`).
2. **Interactive States:** `ControlledAblationFigure` includes 4 accessible tabs with smooth active states, clear ARIA roles (`role="tablist"`, `role="tab"`), and zero visual jitter.
3. **Responsive Ergonomics:** Tested across desktop, tablet, and mobile layouts with auto-fitting grid templates (`minmax(280px, 1fr)`).
4. **No Tailwind / No AI Tropes:** Zero Tailwind CSS was introduced. Zero decorative floating cards or multi-color gradients.

---

## 4. Verification Commands & Outputs

```bash
# 1. Type checking and production build
npm --prefix web run build:pages
# Output: built in 474ms, exit code 0

# 2. Linter check
npm --prefix web run lint
# Output: 0 errors

# 3. Unit test suite
.\.venv\Scripts\python.exe -m pytest modern_gpt\tests\test_final_tool_ablation.py
# Output: 3 passed in 0.03s, exit code 0
```

---

## 5. Deployment & Git Constraint Compliance

- No changes have been pushed to GitHub (`git push` not executed).
- No deployment to Hugging Face (`vynor/stt4` or `vynor/qorin-models`) executed.
- Zero ZeroGPU or paid cloud compute was consumed.
- Canonical checkpoints and historical experiment records remain pristine.
