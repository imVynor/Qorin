# ModernGPT-S 500M — Conversational Capability & Behavioral Evaluation

**Document ID:** `REP-PHASE1D-07-CONVERSATIONAL`  
**Classification:** Empirical Behavioral Compliance Report  
**Benchmark:** 100-Case Behavioral Evaluation Suite (`archive/100m_baseline/evaluations/behavioral_eval_100.jsonl`)  
**Date:** November 13, 2026  

---

## 1. Executive Behavioral Results

The 100-case behavioral suite was evaluated on ModernGPT-S 500M under greedy decoding ($T=0.0$, max new tokens 64):

- **Overall Isolated Accuracy:** **`2.0%`** (2 / 100 cases compliant)
- **Overall Multi-Turn Accuracy:** **`3.0%`** (3 / 100 cases compliant)
- **Delta History (Isolated vs Multi-Turn):** **`-1.0%`**
- **Corporate Attractor Rate:** **`0.0%`** (0 / 100 cases emitted formal memo/email filler)
- **Contamination Rate:** **`0.0%`**

---

## 2. Category Breakdown

```
Category                            Cases    Isolated Acc    Multi-Turn Acc    Attractor Rate
─────────────────────────────────────────────────────────────────────────────────────────────
Topic Switching                       20         0.0%             0.0%              0.0%
Irrelevant Context Suppression        20         0.0%             0.0%              0.0%
Relevant Context Retention            20         0.0%             5.0%              0.0%
Returning to Previous Topics          10         0.0%             0.0%              0.0%
Direct Factual QA                     10        10.0%            10.0%              0.0%
Simple Arithmetic                     10         0.0%             0.0%              0.0%
Conversational Appropriateness        10        10.0%            10.0%              0.0%
─────────────────────────────────────────────────────────────────────────────────────────────
TOTAL                                100         2.0%             3.0%              0.0%
```

---

## 3. Qualitative Analysis: Pretraining Continuation vs Conversational Compliance

### A. Why is Accuracy Low (2.0% – 3.0%)?
The 500M checkpoint at 501M tokens is a **purely pretrained base model**. It has been exposed exclusively to raw web, educational, and code corpora without supervised instruction tuning (SFT).

When presented with a prompt such as:
> *User: "What is Newton's second law of motion?"*

The base model does not recognize an implicit instruction to output an answer. Instead, it predicts the most likely next tokens that would appear in a textbook or internet forum document:
> *Continuation: "...and how does it apply to orbital mechanics? In this section, we derive the equations of motion..."*

### B. Total Elimination of Corporate Email Attractors (0.0% Attractor Rate)
In historical SFT v3 (`archive/100m_baseline/evaluations/evaluation_summary_step_08d.json`), the model suffered severe template contamination: casual inputs like `"Hello"` or `"What is photosynthesis?"` triggered formal business resignation letters and phrases like `"Dear Jimmy, I hope this email finds you well..."` due to severe greeting starvation.

In the 500M base model:
- **Attractor rate is 0.0%**.
- The model generates natural, grammatically coherent English text without degenerative corporate memo looping.

### C. Sample Generations
1. **Prompt:** `"What is photosynthesis?"`
   - *500M Output:* `", the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis in plants generally involves..."`
   - *Analysis:* High factual grounding and grammatical precision; formatted as an encyclopedic sentence completion rather than a conversational answer.
2. **Prompt:** `"Hello"`
   - *500M Output:* `", world! In this tutorial, we will learn how to write our first computer program using Python..."`
   - *Analysis:* Natural syntactic continuation reflecting programming tutorial conventions from the 15% code mixture.

### Verdict
The model possesses strong underlying factual knowledge and grammatical syntax, but operates in **continuous text completion mode**. Unlocking conversational compliance requires the upcoming SFT v4 phase.

