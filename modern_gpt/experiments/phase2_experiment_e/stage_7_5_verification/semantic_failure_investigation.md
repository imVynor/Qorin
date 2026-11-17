# QORIN Multi-Step Test Semantic Failure Forensic Investigation

**Document**: `semantic_failure_investigation.md`  
**Execution Timestamp**: November 17, 2026  
**Subject Tasks**: `multistep_02`, `multistep_06`, `multistep_15`  
**Inspection Source**: `multistep_test_results.json` and `benchmark.jsonl`  
**Inspection Directory**: `modern_gpt/experiments/phase2_experiment_e/stage_7_5_verification/`

---

## 1. Executive Summary

In the Phase 1 multi-step harness test, 12 out of 15 tasks achieved `semantic_pass = True` (80.0%). Three tasks (`multistep_02`, `multistep_06`, and `multistep_15`) passed Web execution, SER extraction, Python code generation, and sandbox calculation correctness, but were scored as `semantic_pass = False`.

A line-by-line inspection of the execution traces conclusively establishes:
1. **Evidence Transfer Was 100% Correct**: In all three tasks, the correct factual entities and metrics were extracted into the SER.
2. **Python Sandbox Execution Was 100% Correct**: In all three tasks, the sandbox executed Python code with exit code 0 and printed the mathematically exact answer.
3. **Synthesis Was 100% Factually Accurate**: The final response stated the exact numerical answer.
4. **Failure Origin**: The failures originated entirely from an **Evaluator Criteria Matching Defect** in `_evaluate_semantic_pass()`:
   - **Comma Punctuation Mismatch**: The response formatted numbers with thousands commas (`8,848` and `12,325`), whereas the benchmark criteria list required unpunctuated substrings (`8848` and `12325`).
   - **Conjunctive vs. Disjunctive Criteria Handling**: For tasks where the benchmark specified alternate valid representations (e.g. `['1009', '1,009']` and `['12325', '12,325']`), the naive heuristic evaluated whether $\ge 65\%$ of all listed strings appeared. Because no single sentence contains both forms simultaneously, matching one variant yielded only $1/2 = 50\% < 65\%$.

---

## 2. Case-by-Case Forensic Breakdown

### Case 1: `multistep_02` (Mount Everest vs. K2 Height Difference)
- **Prompt**: "Look up the heights in meters of Mount Everest and Mount K2, then calculate the difference in feet using Python (1 meter = 3.28084 feet)."
- **Retrieved Evidence & SER**:
  - `everest_m`: 8848.86 meters
  - `k2_m`: 8611.0 meters
- **Python Input & Output**:
  ```python
  everest = 8848.86
  k2 = 8611.0
  diff_m = everest - k2
  diff_ft = diff_m * 3.28084
  print(f'Difference: {diff_m:.2f} meters = {diff_ft:.2f} feet')
  ```
  - **Stdout**: `Difference: 237.86 meters = 780.38 feet` (Mathematically exact).
- **Final Synthesized Response**:
  "According to geographic surveys [SOURCE 1], Mount Everest stands at 8,848.86 meters and Mount K2 stands at 8,611 meters. The difference is 237.86 meters, which converts via Python calculation to approximately 780.38 feet."
- **Expected Answer**: "Everest: 8848.86m, K2: 8611m. Difference: 237.86m = ~780.38 feet."
- **Benchmark Semantic Criteria**: `['8848', '8611', '780']`
- **Exact Failure Mechanism**:
  The response wrote `8,848.86` and `8,611` with commas. A substring search for `'8848'` and `'8611'` failed because of the comma character. Only `'780'` matched. Matches = $1/3 = 33.3\% < 65\%$.
- **Attribution**: **Evaluator string-matching defect (punctuation formatting)**.

---

### Case 2: `multistep_06` (iPhone vs. iPad Launch Dates Duration)
- **Prompt**: "Look up the exact launch date of the original iPhone and original iPad, then compute the exact number of days between them using Python datetime."
- **Retrieved Evidence & SER**:
  - `iphone_date`: `2007-06-29`
  - `ipad_date`: `2010-04-03`
- **Python Input & Output**:
  ```python
  from datetime import date
  d1 = date.fromisoformat('2007-06-29')
  d2 = date.fromisoformat('2010-04-03')
  diff = (d2 - d1).days
  print(f'Exact days between launches: {diff} days')
  ```
  - **Stdout**: `Exact days between launches: 1009 days` (Mathematically exact).
- **Final Synthesized Response**:
  "Historical product records confirm the original iPhone was released on June 29, 2007, and the original iPad was released on April 3, 2010. Computing the duration via Python datetime yields exactly 1,009 days between the two launches."
- **Expected Answer**: "iPhone: June 29, 2007; iPad: April 3, 2010. Difference: 1,009 days."
- **Benchmark Semantic Criteria**: `['1009', '1,009']`
- **Exact Failure Mechanism**:
  The benchmark criteria author included both `'1009'` and `'1,009'` to allow either representation. The synthesized response used `'1,009'`. In the harness, `matches = 1` out of 2 total criteria. Score = $1/2 = 50.0\% < 65\%$. It is impossible for a non-repetitive response to pass a $65\%$ threshold when criteria are alternate disjunctive variants of the same number.
- **Attribution**: **Evaluator criteria aggregation defect (conjunctive evaluation of disjunctive alternatives)**.

---

### Case 3: `multistep_15` (Platinum & Gold Atomic Numbers Sum of Squares)
- **Prompt**: "Find the atomic numbers of Platinum (78) and Gold (79), then compute the sum of their squares using Python."
- **Retrieved Evidence & SER**:
  - `pt_num`: 78
  - `au_num`: 79
- **Python Input & Output**:
  ```python
  pt = 78
  au = 79
  sum_sq = (pt ** 2) + (au ** 2)
  print(f'Platinum (78^2=6084) + Gold (79^2=6241) = {sum_sq}')
  ```
  - **Stdout**: `Platinum (78^2=6084) + Gold (79^2=6241) = 12325` (Mathematically exact: $6084 + 6241 = 12325$).
- **Final Synthesized Response**:
  "The atomic number of Platinum is 78 and the atomic number of Gold is 79. Computing the sum of their squares in Python (78^2 + 79^2 = 6,084 + 6,241) yields exactly 12,325."
- **Expected Answer**: "78^2 + 79^2 = 6084 + 6241 = 12,325."
- **Benchmark Semantic Criteria**: `['12325', '12,325']`
- **Exact Failure Mechanism**:
  Identical to `multistep_06`. The criteria listed both unpunctuated `'12325'` and punctuated `'12,325'`. The response used `'12,325'`. Matches = $1/2 = 50.0\% < 65\%$.
- **Attribution**: **Evaluator criteria aggregation defect (conjunctive evaluation of disjunctive alternatives)**.

---

## 3. Systematic Classification Matrix

| Task ID | Evidence Ingestion | Calculation Accuracy | Synthesis Completeness | Punctuation Sensitive | Disjunctive Criteria Conflict | True Factual Correctness | Evaluator Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `multistep_02` | **Valid** | **100% Exact** | **Complete** | Yes (`8,848` vs `8848`) | No | **TRUE PASS** | False |
| `multistep_06` | **Valid** | **100% Exact** | **Complete** | Yes | Yes (`['1009', '1,009']`) | **TRUE PASS** | False |
| `multistep_15` | **Valid** | **100% Exact** | **Complete** | Yes | Yes (`['12325', '12,325']`) | **TRUE PASS** | False |

---

## 4. Evaluator Remediation Specification

To ensure objective scoring in subsequent evaluation phases:
1. **Numerical Punctuation Normalization**: Evaluator scripts must strip commas from both generated text and criteria strings prior to matching (e.g. `re.sub(r",", "", text)`).
2. **Disjunctive Set Grouping**: When criteria contain multiple representations of the same target quantity (e.g. `[{"1009", "1,009"}]`), matching *any* member of the equivalence set must award full credit for that criterion.
3. **Impact**: Under normalized evaluation, **15 out of 15 multi-step tasks (100.0%)** pass semantic evaluation.
