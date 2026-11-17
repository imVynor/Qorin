# QORIN — Final Limitations, Negative Findings, and Future Work Report
## ES 335 Machine Learning Research Sprint

**Date:** 2026-11-17  
**Artifact Path:** `modern_gpt/reports/final_8hour_sprint/limitations_and_future_work.md`  
**Model Identity:** **QORIN** (514M parameter decoder-only Transformer)  

---

### 1. Honest Research Accounting & Known Limitations

In compliance with rigorous research reporting principles, we explicitly document the empirical boundaries and limitations of QORIN:

#### A. Parity Heuristic vs. Algorithmic Primality
- **Finding:** When prompted to write an `is_prime(n)` function, QORIN generated:
  ```python
  def is_prime(n):
      if n % 2 == 0:
          return False
      else:
          return True
  ```
- **Empirical Reality:** The model learned a superficial **parity heuristic** (`is_prime := n % 2 != 0`). On an extended 37-case evaluation, it scored 59.46% (22/37), succeeding on even numbers and odd primes, but failing on all odd composites (9, 15, 25, 49), Carmichael pseudoprimes (561, 1105), the number 1, and 2.
- **Scientific Takeaway:** Sub-billion parameter models frequently learn shortcut statistical heuristics that appear competent on small benchmark subsets. We do not claim genuine primality understanding.

#### B. Execution Success vs. Semantic Task Correctness
- **Finding:** A clean exit code 0 from the Python sandbox does **not** imply end-to-end task completion.
- **Empirical Reality:** Across 30 evaluation tasks, 50% of tool invocations executed with exit code 0, but only 6.67% produced algebraically verified ground-truth answers. In multiple cases, the model formulated Python code with semantic bugs (e.g. `math.comb(315, 840)` instead of `math.gcd(315, 840)`).
- **Scientific Takeaway:** Tool-augmented small language models require external semantic verification or multi-step execution reflection to bridge the gap between sandbox execution and mathematical validity.

#### C. Cross-Tool Selection Bias
- **Finding:** When both Python and Web Researcher are advertised simultaneously, QORIN exhibits a strong inductive bias toward Python.
- **Empirical Reality:** Due to a 12.5:1 ratio of Python to Web examples in the Stage 1 training corpus, the model frequently defaults to Python even for factual queries unless the Web tool is explicitly isolated via the capability switch.
- **Scientific Takeaway:** When Web Researcher is isolated via the user-controlled switch, tool accuracy rises from 2% to 88.5%, proving the utility of explicit capability toggles.

#### D. Verbatim Round 2 Interpretation
- **Finding:** In Round 2, QORIN faithfully reproduces the tool's stdout (`"The Python execution yielded: ... Therefore, the requested value is verified."`).
- **Limitation:** The model does not critically audit whether the output from the tool logically answers the original question, treating tool observations as authoritative facts without self-correction.

---

### 2. What Can Safely Be Claimed for the Research Showcase

1. **Autonomous Tool Intent Acquisition:** QORIN (514M) successfully learned when to trigger external tools, increasing tool-intent recall from **0% in Experiment B to 80% in Experiment C** ($F_1 = 0.8889$, Precision = 100%).
2. **Zero Catastrophic Forgetting:** Language modeling retention across 7 distinct pretraining domains remained intact ($2.8651$ validation loss vs $2.8621$ in the frozen base).
3. **High Syntactic Robustness:** Python coding syntax validity reached an all-time high of **96.0%** (up from 2.0% in Base and 52.0% in Exp A).
4. **Working End-to-End System:** The application reliably dispatches Python calculations and live DuckDuckGo web searches through a bounded, secure SSE streaming loop.

---

### 3. What Should NOT Be Claimed

1. **Do NOT claim intrinsic mathematical reasoning:** The model requires external Python execution for arithmetic and relies on heuristics for primality.
2. **Do NOT equate execution success with task success:** 50% tool execution does not mean 50% benchmark accuracy.
3. **Do NOT claim autonomous multi-tool arbitration:** When multiple tools are active, explicit capability toggles or routing prompts are required for reliable tool discrimination.

---

### 4. Future Work & Next Steps

1. **Balanced SFT Mixture:** Train on a 1:1 balanced ratio of authentic Python, Web Researcher, and multi-step trajectories to eliminate cross-tool inductive bias.
2. **Self-Correction & Reflection Supervision:** Fine-tune on trajectories where the model inspects intermediate tool outputs, catches algebraic errors, and re-executes.
3. **Constrained Decoding Engine:** Integrate grammar-based logit masking into the inference engine to eliminate JSON formatting errors entirely.
