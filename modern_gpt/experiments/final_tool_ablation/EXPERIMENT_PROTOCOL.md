# QORIN Research Protocol: Controlled Tool-Augmentation Ablation

**Document Version**: 1.0.0 — Authoritative Pre-Execution Protocol  
**Date**: 2026-11-17  
**Principal Question**: *Does augmenting small decoder-only language models with external tools actually improve task performance when model capacity and checkpoints are strictly held constant?*

---

## 1. Experimental Objectives

This controlled experiment tests the foundational premise of tool-augmented language modeling:
1. When the exact same model weights are held constant, does granting access to external tools (Web search and a sandboxed Python execution engine) increase semantic task success?
2. In which task categories do external tools provide clear utility, and in which categories do they introduce failure modes, latency penalties, or regressed behavior?
3. Does external tool augmentation compensate for smaller model capacity (comparing a 120M parameter model against a 514M parameter model)?

---

## 2. Experimental Conditions

The evaluation evaluates a $2 \times 2$ factorial matrix:
- **Factor 1: Tool Availability**
  - **Condition A (Tools Disabled)**: All tool execution and retrieval capabilities are turned off. The model is presented with the user prompt and must answer directly using its internal parametric representations. If a tool call is generated, tool dispatch returns a standardized notice that external tools are unavailable, requiring direct response.
  - **Condition B (Tools Enabled)**: The complete QORIN multi-turn tool harness is active. The model autonomously decides whether to invoke `web_search` or `python_interpreter`. Real execution environments provide live observations, and the model synthesizes the multi-step trajectory.
- **Factor 2: Model Capacity**
  - **Primary Model**: QORIN-500M SFT Experiment F Pilot Best (`ckpt_pilot_best.pt`, 514M parameters, 32k vocabulary, SHA-256 `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4`).
  - **Secondary Model**: QORIN-100M SFT v3 (`best_val.pt`, 120M parameters, 16k vocabulary, SHA-256 `d771ea9f5465e3bcf2b619a32f7797742e500a01704a4d5f582eb5db4adc7510`).

---

## 3. Strict Experimental Controls

To guarantee internal validity:
1. **Identical Weights**: Within each model tier, Condition A and Condition B run against the exact same bitwise checkpoint file.
2. **Identical Tokenizer**: Token vocabulary, special tokens, and decoding logic are unchanged.
3. **Identical Prompts**: Both conditions receive identical system prompts and task instructions.
4. **Deterministic Decoding**: Greedy decoding ($T=0.0$, $\text{top\_p}=1.0$) is enforced to eliminate stochastic generation variance.
5. **Standardized Context Window**: Identical maximum context window limits and turn generation limits (160 tokens per assistant turn).
6. **Isolated Timing**: Model generation latency is metered strictly separately from external tool execution latency.
7. **Paired Statistical Testing**: Every evaluation metric is computed across identical task instances, enabling paired McNemar's tests and paired non-parametric bootstrap resampling ($B=2000$).

---

## 4. Evaluation Benchmark & Categories

The evaluation runs over the 100 independently held-out tasks from the verified Stage 13 test split (`test.jsonl`), categorized into 10 capability domains:
1. **Parametric knowledge** ($N=8$): Concepts and foundational definitions answerable purely from internal parameters.
2. **Current information** ($N=8$): Factual lookup tasks requiring exact external data.
3. **Numerical computation** ($N=6$): Quantitative calculations requiring precise arithmetic or formula evaluation.
4. **Multi-step reasoning** ($N=30$): Sequential tasks requiring Web information gathering followed by Python code computation and evidence synthesis.
5. **Evidence synthesis** ($N=15$): Synthesizing coherent answers from provided evidence passages with numeric citation tags.
6. **Partial evidence** ($N=5$): Recognizing when retrieved information only partially addresses the prompt and explicitly hedging.
7. **Contradictory evidence** ($N=8$): Synthesizing divergence across conflicting sources and avoiding arbitrary adherence to one.
8. **Insufficient evidence** ($N=5$): Correctly acknowledging that evidence is missing rather than hallucinating plausible facts.
9. **Tool discrimination** ($N=11$): Correctly deciding when NOT to invoke tools on conceptual tasks vs. invoking tools when needed.
10. **Conversational follow-ups** ($N=4$): Preserving multi-turn context and recovering from empty or failing queries.

---

## 5. Metric Formulations

### Primary Outcome: Semantic Task Success Rate
$$\text{Success Rate} = \frac{\sum_{i=1}^N \mathbb{I}(\text{Answer}_i \text{ is semantically correct})}{N}$$

### Paired McNemar's Contingency Matrix
For binary outcomes on identical tasks:
$$b = \text{Success in A, Failure in B}$$
$$c = \text{Failure in A, Success in B}$$
$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}, \quad p = P(\chi_1^2 \ge \chi^2)$$

### Evidence Grounding & Citation Precision
$$\text{Citation Precision} = \frac{\text{Supported Claims with Grounded Citations}}{\text{Total Claims Emitted}}$$
$$\text{Hallucination Rate} = \frac{\text{Unsupported Factual Claims}}{\text{Total Factual Claims}}$$
