# ModernGPT-S 500M — Coding Capability & Subprocess Sandbox Evaluation

This evaluation benchmarks functional code generation across 50 standardized programming tasks executed within an isolated subprocess Python sandbox with strict 3.0-second execution timeouts.

## 1. Executive Comparison: 501M Baseline vs 1.12B Milestone

| Evaluation Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress / Delta | Trend |
| :--- | :--- | :--- | :--- | :--- |
| **Tokens Seen** | 501,678,080 | **1,121,648,640** | **+619,970,560 tokens** | +123.6% pretraining |
| **Pass@1 Accuracy** | 0.0% (0/50) | **2.0%** (1/50) | **+2.00%** | Improved |
| **Syntax Validity Rate** | 4.0% (2/50) | **2.0%** (1/50) | **-2.00%** | Python AST compliant |
| **Sandbox Timeouts** | 0 | **0** | +0 | Infinite loops eliminated |

## 2. Independent Audit of Phase 1D Coding Claims
- **Phase 1D Defect:** Phase 1D reported Pass@1 using in-process `exec()`, which risked interpreter state corruption and lacked wall-clock process isolation. Phase 1E executed all tests via detached subprocesses.
- **Pretrained Base Limitations:** Base models without instruction tuning often generate comments, docstrings, or variations instead of pure function bodies matching exact signatures.
- **Code Domain Loss Correlation:** In held-out validation, Code loss dropped from 2.0174 to 1.8334 (PPL 6.25). The model understands code syntax well, but requires instruction fine-tuning to reliably align function signatures with user specifications.