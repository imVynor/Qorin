# ModernGPT-S Phase 2B — Before vs After Comparative Evaluation

**Comparison Checkpoints:**
- **Pretrained Base Checkpoint:** [`ckpt_latest.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt)
- **Experiment A Pilot Checkpoint:** [`ckpt_pilot_final.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt)

---

## 1. Quantitative Benchmark Comparison Table

| Metric | Pretrained Base Checkpoint | Experiment A Pilot Checkpoint | Delta / Improvement |
|---|---:|---:|---|
| **SFT Held-out Validation Loss** | 2.1127 | **1.8799** | **-11.02% (Improved)** |
| **SFT Held-out Perplexity** | 8.27 | **6.55** | **-20.80% (Improved)** |
| **Pretraining Weighted Val Loss (7 Domains)** | 2.8621 | **2.8487** | **-0.0134 (Preserved/Improved)** |
| **Pretraining Weighted Perplexity** | 17.50 | **17.26** | **-1.37% (Preserved)** |
| **Distinct-1 (Vocabulary Diversity)** | 0.1287 | **0.1441** | **+11.96% (Improved)** |
| **Distinct-2 (Bigram Diversity)** | 0.2241 | **0.2346** | **+4.68% (Improved)** |
| **Distinct-3 (Trigram Diversity)** | 0.2557 | **0.2706** | **+5.82% (Improved)** |
| **4-gram Repetition Rate** | 0.7251 | **0.7026** | **-3.10% (Reduced loops)** |
| **ChatML Turn Delimiter Adherence** | 0.0% (Echoed prompt) | **100.0%** | **Qualitative Leap** |
| **Markdown Code Block Generation** | 0.0% | **100.0%** | **Qualitative Leap** |

---

## 2. Capability Analysis

### A. Capabilities Gained
1. **Assistant Persona & Turn Taking:** The model transitions cleanly from raw completion to assistant turn generation upon encountering `<|im_start|>assistant\n`.
2. **Markdown Code Structuring:** Correctly emits ````python` blocks with function signatures and return statements.
3. **Conversational Directness:** Immediately provides definitions (e.g. quantum entanglement, French revolution summary) rather than hallucinating fictitious user replies.

### B. Capabilities Retained
1. **Pretraining Generalization:** Loss across all 7 pretraining validation domains showed zero degradation (loss slightly dropped from 2.8621 to 2.8487).
2. **Code Syntax Awareness:** Python token associations and AST structure remained completely intact.

### C. Residual Weaknesses & Limitations
1. **Repetition on Long Contexts:** In greedy or low-temperature generation, repetition loops still manifest after ~100 tokens. Repetition penalty (1.20) or nucleus sampling (top-p=0.9) is essential.
2. **Arithmetic Reasoning:** The model does not calculate arithmetic reliably in-weights; multi-step arithmetic requires tool execution (APIGen / Python sandbox) as planned in Experiment B.
