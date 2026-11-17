# ModernGPT-S 500M — Tool-Use Readiness Evaluation Report

This evaluation benchmarks zero-shot tool intent recognition across 20 scenarios requiring python execution, weather lookup, web search, or pure text reasoning (no tool).

## 1. Executive Comparison: 501M Baseline vs 1.12B Milestone

| Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress / Delta |
| :--- | :--- | :--- | :--- |
| **Tool Intent Accuracy** | 15.0% (3/20) | **0.0%** (0/20) | **-15.00%** |

## 2. Assessment of Pretrained Base vs SFT Requirement
- The base model lacks explicit tool-calling tokens `<call:...>` or role markers until instruction fine-tuning.
- Pretraining exposes the model to code semantics and tool concepts, but tool-call invocation protocols must be taught during SFT.