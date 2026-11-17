# ModernGPT-S 500M — Phase 1D Validation Loss & Perplexity Report

**Document ID:** `REP-PHASE1D-06-VAL-LOSS`  
**Classification:** Empirical Language Modelling Quality Report  
**Target Checkpoint:** `ckpt_501m_tokens_501678080.pt` (SHA-256: `a5755655...`)  
**Validation Corpus:** `modern_gpt/data_500m/final_5b/val/` (20,000,000 isolated tokens)  
**Evaluated Token Budget:** 358,400 tokens across 175 contiguous sequences of 2,048 tokens  
**Date:** November 13, 2026  

---

## 1. Domain-Specific Loss & Perplexity Breakdown

```
Domain                 Loss (CE)    Perplexity    Target %    Evaluated Tokens    Relative Rank
───────────────────────────────────────────────────────────────────────────────────────────
Source Code (code)       2.0272         7.59       15.0%          51,200           1 (Best)
Curated Dialogue (dial)  2.2169         9.18        5.0%          51,200           2
Educational (edu)        2.6204        13.74       15.0%          51,200           3
Mathematics (math)       2.8965        18.11        7.5%          51,200           4
Books & Prose (books)    3.4118        30.32       10.0%          51,200           5
Wikipedia (wiki)         3.5398        34.46        7.5%          51,200           6
Quality Web (web)        3.5868        36.12       40.0%          51,200           7 (Highest)
───────────────────────────────────────────────────────────────────────────────────────────
OVERALL WEIGHTED         3.0666        21.47      100.0%         358,400
```

---

## 2. Analysis of Domain Dynamics

### 1. Source Code Dominance (`Loss: 2.0272 | PPL: 7.59`)
Source code exhibits the lowest cross-entropy loss across all domains. This aligns with empirical scaling observations: code contains highly structured keywords, repetitive syntax templates, and deterministic token sequences (`def`, `import`, `return`, `class`), allowing the 500M model to achieve low entropy very early in training.

### 2. Conversational Dialogue Alignment (`Loss: 2.2169 | PPL: 9.18`)
The curated dialogue domain (`ultrachat_200k`) achieves the second-lowest perplexity. The presence of natural turn indicators and colloquial phrasing is effectively captured by the production 32K tokenizer, showing strong token efficiency.

### 3. Web & Wikipedia Entropy (`Loss: 3.54 – 3.59 | PPL: 34.4 – 36.1`)
Web and Wikipedia domains exhibit the highest cross-entropy loss. These corpora contain massive lexical diversity, diverse scientific nomenclature, varied stylistic forms, and foreign entity names. At 501M tokens, the model has only seen ~200M web tokens (10% of the 2.0B web budget), meaning vocabulary and factual representations are still actively learning.

---

## 3. Positional Attention & Context Window Dynamics

We measured token-level cross-entropy loss as a function of sequence position within the 2,048-token context window:

- **First 512 Tokens (Prefix Window):** Mean CE Loss = **`3.0662`**
- **Last 512 Tokens (Deep Context Window):** Mean CE Loss = **`2.8642`**
- **Delta:** **`-0.2020`** (Perplexity improves by ~18.3% as context expands).

### Significance
In poorly configured attention mechanisms or unstable positional encodings, loss explodes at deeper positions (as observed in earlier 4K extrapolation experiments). The fact that loss consistently decreases by `0.20` across the 2,048 window proves that:
1. Rotary Position Embeddings (`RoPE` with $\theta = 10,000.0$) are operating correctly.
2. Grouped-Query Attention (6:1 ratio) effectively aggregates context across earlier tokens to improve next-token predictability.

