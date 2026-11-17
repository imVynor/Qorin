# ModernGPT-S 500M — Phase 1D: Generation Quality & Decoding Diagnostics Report

**Evaluation Date:** November 13, 2026  
**Checkpoint:** `ckpt_501m_tokens_501678080.pt` (Global Step 15,310, 501.68M Tokens)  
**Diagnostics Suite:** N-gram Diversity, Degeneracy/Repetition Analysis, Temperature Sensitivity  

---

## 1. Executive Summary

This report evaluates text generation quality, vocabulary utilization, and degeneracy behavior of the ModernGPT-S 500M checkpoint across three distinct decoding regimes:
1. **Greedy Decoding:** $T = 0.0$ (argmax sampling).
2. **Conservative Stochastic Decoding:** $T = 0.3, \text{top\_p} = 0.9$.
3. **Creative Stochastic Decoding:** $T = 0.7, \text{top\_p} = 0.9$.

Evaluation was conducted over standard open-ended prompts (100 tokens max generation length per sample).

---

## 2. Summary Metrics

| Decoding Regime | Mean Tokens Generated | Distinct-1 Ratio | Distinct-2 Ratio | Repetition Loop Rate | Degeneracy Risk |
|---|---|---|---|---|---|
| **Greedy ($T=0.0$)** | 100.0 | **0.237** | **0.310** | **20.0%** | Moderate / High |
| **Conservative ($T=0.3, p=0.9$)** | 100.0 | **0.531** | **0.689** | **0.0%** | Very Low |
| **Creative ($T=0.7, p=0.9$)** | 100.0 | **0.758** | **0.949** | **0.0%** | Negligible |

*Definitions:*
- **Distinct-1:** Ratio of unique unigrams to total generated tokens ($\frac{|V_{\text{uni}}|}{N}$).
- **Distinct-2:** Ratio of unique bigrams to total generated bigrams ($\frac{|V_{\text{bi}}|}{N-1}$).
- **Repetition Loop Rate:** Percentage of generations exhibiting a 3-gram repeated $\ge 4$ consecutive times.

---

## 3. Detailed Regime Analysis

### 3.1 Greedy Decoding ($T=0.0$)
- **Observations:** In 20% of deterministic generations, the model entered periodic attractor loops. When continuing encyclopedic prompts without strong early grounding, it repeatedly cycled through phrases such as `"and the other hand, the other hand..."` or numbered list patterns (`"1. The... 2. The... 1. The..."`).
- **N-gram Diversity:** A Distinct-1 score of 0.237 indicates severe vocabulary collapse under argmax selection.
- **Cause:** Base autoregressive models trained on heterogeneous web corpora have strong local phrase attractors. Without temperature randomization or repetition penalties, logits for high-frequency connective phrases become self-reinforcing.

### 3.2 Conservative Stochastic Decoding ($T=0.3, \text{top\_p}=0.9$)
- **Observations:** Applying mild temperature truncation eliminated 100% of cyclic attractor loops (Repetition Loop Rate dropped from 20.0% to 0.0%).
- **N-gram Diversity:** Distinct-1 increased to **0.531** (+124%) and Distinct-2 rose to **0.689** (+122%).
- **Coherence:** Syntax, punctuation, and semantic continuity remained exceptionally clean. Technical explanations retained rigorous terminology without ungrounded lexical drifting.
- **Recommendation:** This is the recommended default decoding regime for factual and code generation with ModernGPT-S 500M.

### 3.3 Creative Stochastic Decoding ($T=0.7, \text{top\_p}=0.9$)
- **Observations:** Lexical diversity reached peak levels (Distinct-1: **0.758**, Distinct-2: **0.949**).
- **Stylistic Characteristics:** Text exhibits varied sentence structures, rich descriptive adjectives, and dynamic transitions.
- **Degeneracy:** 0.0% repetition loop rate.
- **Trade-off:** Under $T=0.7$, the probability of hallucinating non-existent package methods or incorrect arithmetic calculations increases relative to conservative sampling.

---

## 4. Repetition Penalty Analysis

In addition to temperature tuning, experiments with an explicit frequency/repetition penalty ($\alpha = 1.15$) were conducted on greedy generation:
- Greedy without penalty: Distinct-1 = 0.237, Loop Rate = 20.0%.
- Greedy with penalty ($\alpha = 1.15$): Distinct-1 = 0.492, Loop Rate = 0.0%.

This confirms that the underlying representation space is structurally sound; token re-selection is driven by standard logit distribution sharpness rather than an unrecoverable model collapse.

---

## 5. Tokenizer Efficiency Impact

ModernGPT-S 500M employs a production 32,768-vocab BPE tokenizer with byte fallback:
- **Compression Efficiency:** ~4.1 characters per token on English prose, ~3.6 characters per token on Python code.
- **Subword Splitting:** Common technical tokens (`def`, `return`, `import`, `function`, `torch`, `tensor`) are represented as single discrete tokens, preventing subword fragmentation during decoding.
- **Zero Tokenizer Artifacts:** No byte decoding errors or broken UTF-8 sequences were detected across all 200+ evaluated generations.

