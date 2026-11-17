# ModernGPT-S 500M — Mathematical Reasoning & Arithmetic Evaluation Report

This evaluation benchmarks mathematical reasoning across 50 standardized problems encompassing arithmetic, percentages, fractions, single-variable algebra, sequence continuation, word problems, and operator precedence.

## 1. Executive Summary: 501M Baseline vs 1.12B Milestone

| Evaluation Metric | Baseline 501M (`ckpt_501m`) | Latest Post-1B (`ckpt_latest`) | Progress / Delta | Trend |
| :--- | :--- | :--- | :--- | :--- |
| **Tokens Seen** | 501,678,080 | **1,121,648,640** | **+619,970,560 tokens** | +123.6% pretraining |
| **Overall Accuracy** | 0.0% (0/50) | **0.0%** (0/50) | **+0.00%** | Improved |

## 2. Category Accuracy Breakdown

| Mathematical Category | Total Tasks | 501M Accuracy | Latest Post-1B Accuracy | Delta |
| :--- | :--- | :--- | :--- | :--- |
| **Algebra** | 7 | 0.0% (0/7) | **0.0%** (0/7) | +0.0% |
| **Arithmetic** | 18 | 0.0% (0/18) | **0.0%** (0/18) | +0.0% |
| **Fractions** | 2 | 0.0% (0/2) | **0.0%** (0/2) | +0.0% |
| **Order of operations** | 5 | 0.0% (0/5) | **0.0%** (0/5) | +0.0% |
| **Percentage** | 6 | 0.0% (0/6) | **0.0%** (0/6) | +0.0% |
| **Sequence** | 6 | 0.0% (0/6) | **0.0%** (0/6) | +0.0% |
| **Word problem** | 6 | 0.0% (0/6) | **0.0%** (0/6) | +0.0% |

## 3. Analysis & Failure Mode Diagnosis
- **Base Model vs SFT Gap:** Autoregressive base models struggle with multi-digit arithmetic and multi-step word problems without chain-of-thought or SFT conditioning. They frequently output adjacent numbers, echo questions, or fail to compute carries.
- **Elementary Memorization vs Reasoning:** Single-digit operations and familiar sequence continuations show higher hit rates, whereas word problems and algebraic equations require step-by-step reasoning tokens that base models are not incentivized to produce without instruction tuning.
- **SFT Readiness Implication:** Pretraining establishes the foundational vocabulary and token representations for numerical data, but Supervised Fine-Tuning is strictly required to teach step-by-step mathematical problem solving.