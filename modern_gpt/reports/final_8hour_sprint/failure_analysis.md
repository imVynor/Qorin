# QORIN — Sprint Failure Analysis & Error Taxonomy
## Final 8-Hour Sprint: Error Patterns across 4 Operating Modes

**Date:** 2026-11-17  
**Artifact Path:** `modern_gpt/reports/final_8hour_sprint/failure_analysis.md`  

---

### 1. Comparative Error Distribution Across 4 Modes

| Error Category | Mode 1: Standalone | Mode 2: Python Only | Mode 3: Web Only | Mode 4: Two-Tool | Root Cause & System Behavior |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **JSON Truncation / Malformed Call** | 0 | 0 | 0 | 1 | Mitigated by `QorinTwoStageProtocol` AST validation and markdown fallback extraction. |
| **Cross-Tool Selection Error** | 0 | 0 | 0 | 3 | Mitigated when Web Researcher is isolated via capability toggle. In Two-Tool mode, Python inductive prior dominates. |
| **Execution Error (Syntax / Exception)** | 0 | 2 | 0 | 2 | Code contained unimported library (`cmath` / `missing.txt`) or syntax invalidity; caught by AST pre-check or returned via `<|tool_error|>`. |
| **Semantic Calculation Error** | N/A | 4 | N/A | 4 | Code executed cleanly with exit code 0, but formulated mathematically incorrect formulas (e.g. `math.comb` instead of `math.gcd`). |
| **False Positive (Unnecessary Call)** | 0 | 0 | 0 | 0 | Model correctly avoided calling tools on 100% of negative control prompts. |
| **Primality Parity Failure** | 1 | 1 | 1 | 1 | Model relies on parity heuristic (`n % 2 != 0`), failing on composite odd integers like 91 ($7 \times 13$). |

---

### 2. Representative Error Case Studies

#### Case 1: Semantic Divergence Despite Clean Execution
- **Task ID:** `task_01_arithmetic`
- **Prompt:** `Calculate (24 * 35) + (1024 / 32) - (15 ** 2) using Python.`
- **Ground Truth:** $(840) + (32) - (225) = \mathbf{647.0}$
- **Generated Code:**
  ```python
  import math
  r = 24 * 35
  h = 1024 / 32
  ans = math.comb(24, 35)
  print('Result:', ans)
  ```
- **Execution Output:** `Result: 0` (Clean exit code 0!)
- **Model Synthesis:** `"The Python execution yielded: Result: 0. Therefore, the requested value is verified."`
- **Failure Classification:** **Semantic Calculation Error.**
- **Diagnostic:** The model possesses syntactic competency (code runs without throwing exceptions) and execution ingestion competency, but suffers from symbolic formula hallucination in the generated code body.

#### Case 2: Inductive Tool Selection Prior
- **Task ID:** `task_06_transformer_paper`
- **Prompt:** `Search the web for the authors and year of the Transformer paper Attention Is All You Need.`
- **Behavior in Mode 4 (Both Tools ON):**
  - Model emitted: `<|tool_call|>{"name": "python", "arguments": {"code": "import math..."}}`
  - *Outcome:* Wrong tool selected due to 12.5:1 Python training bias.
- **Behavior in Mode 3 (Web Researcher Isolated):**
  - Model emitted: `<|tool_call|>{"name": "web_research", "arguments": {"query": "The authors of the Attention Is All You Need paper"}}`
  - *Outcome:* Live search executed, retrieved arXiv citations, synthesized author names ("Vaswani et al.") and year ("2017").
  - *Key Finding:* Capability controls are essential to prevent cross-tool competition in sub-billion parameter models.

#### Case 3: Primality Parity Shortcut
- **Task ID:** `task_20_primality_91`
- **Prompt:** `Is the number 91 a prime number? Compute its smallest divisor greater than 1 using Python.`
- **Ground Truth:** Composite ($91 = 7 \times 13$, smallest divisor = 7).
- **Generated Code:**
  ```python
  def is_prime(n):
      return n % 2 != 0
  print(is_prime(91))
  ```
- **Output:** `True` (Wrong!)
- **Failure Classification:** **Heuristic Shortcut.**
- **Diagnostic:** Because 91 is an odd number, the parity heuristic classifies it as prime, failing to test $d=7$.
