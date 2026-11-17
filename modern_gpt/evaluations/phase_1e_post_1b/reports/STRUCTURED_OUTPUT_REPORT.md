# ModernGPT-S 500M — Structured Output (JSON) Evaluation Report

This evaluation benchmarks schema adherence, syntax validity, and key completeness across 50 diverse JSON generation prompts.

## 1. Executive Comparison: 501M Baseline vs 1.12B Milestone

| Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress / Delta |
| :--- | :--- | :--- | :--- |
| **Valid JSON Syntax Rate** | 0.0% (0/50) | **0.0%** (0/50) | **+0.00%** |
| **Full Schema Key Coverage** | 0.0% (0/50) | **0.0%** (0/50) | **+0.00%** |

## 2. Qualitative Analysis
- Autoregressive base models without SFT frequently generate trailing text, markdown noise, or unclosed braces.
- Exposure to 1.12B tokens increases familiarity with JSON token transitions, but consistent structured generation requires SFT schema conditioning.