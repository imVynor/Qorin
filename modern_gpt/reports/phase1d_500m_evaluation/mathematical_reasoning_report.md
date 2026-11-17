# ModernGPT-S 500M — Mathematical Reasoning & Arithmetic Evaluation

**Document ID:** `REP-PHASE1D-09-MATH`  
**Classification:** Empirical Mathematical Evaluation  
**Benchmark:** 50 Held-Out Mathematical & Arithmetic Tasks (`phase1d_benchmark_data.py`)  
**Date:** November 13, 2026  

---

## 1. Quantitative Performance Summary

The 50 mathematical reasoning tasks were evaluated under deterministic greedy decoding ($T=0.0$, max new tokens 48):

```
Category                     Tasks    Correct    Accuracy %    Primary Failure Mode
─────────────────────────────────────────────────────────────────────────────────────────────
Arithmetic                    18         0          0.0%       Equation continuation / near-miss
Percentage                     5         0          0.0%       Formula recitation without compute
Fractions                      2         0          0.0%       Symbolic continuation
Algebra                        7         0          0.0%       Re-arranging symbols without solve
Number Sequences               5         0          0.0%       Extrapolating arbitrary list items
Word Problems                  7         0          0.0%       Story continuation without arithmetic
Order of Operations            6         0          0.0%       Left-to-right syntax continuation
─────────────────────────────────────────────────────────────────────────────────────────────
TOTAL                         50         0          0.0%
```

---

## 2. Failure Mode Taxonomy

### 1. Equation Continuation
Rather than outputting a numeric answer to complete `"Solve: 17 + 28 = "`, the model predicts continued exercises from mathematics textbooks:
> *Prompt:* `"Solve: 17 + 28 = "`  
> *500M Output:* `"17 + 28 = 45 - 10 + ... In order to find the sum, let us consider..."`

### 2. Lack of Scratchpad / Chain-of-Thought Conditioning
Standard autoregressive language models compute the next token with a fixed number of FLOPs per layer. Performing multi-digit addition or multi-step algebra requires intermediate reasoning steps. Without scratchpad or CoT supervision, the model is forced to predict the final token directly from the prompt, which exceeds single-forward-pass capacity.

### 3. Mathematical Vocabulary Grounding
The model correctly recognizes mathematical contexts, terms (`derivative`, `integral`, `equation`, `sum`), and mathematical formatting, reflecting the 7.5% `open-web-math` domain mixture. However, numeric precision requires fine-tuning.

---

## 3. Recommendations
1. Mathematical capability should NOT be judged solely on direct zero-shot prompt completion.
2. In SFT v4, incorporate step-by-step reasoning pairs where intermediate scratchpads allow the model to dedicate compute tokens to arithmetic execution before outputting the final answer.
3. For exact computation in production, integrate the Python execution sandbox via the tool orchestration layer.

