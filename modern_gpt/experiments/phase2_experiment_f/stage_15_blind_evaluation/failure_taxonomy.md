# QORIN Experiment F — Stage 15: Failure Taxonomy

**Date**: 2026-11-17  
**Author**: QORIN Diagnostic Engineering Team  
**Evaluation Scope**: Stage 15 Blind Capability Evaluation (100 test tasks)

---

## 1. Multi-Turn Sequential Failures (Corpus A)

| Failure Code | Failure Category | Description | Observed Frequency | Root Cause | Proposed Mitigation |
|---|---|---|:---:|---|---|
| **A1-DELIM** | Delimiter Drift | Model emits `<<<` or `<<<<` instead of `<tool_call id="call_001">` or `<tool_call>` | 14 / 40 (35.0%) | Delimiter token embeddings have not fully converged after 0.50 epochs. | Add explicit delimiter loss weighting or train through Step 150–200. |
| **A2-KEYCONF** | Argument Key Confusion | Model outputs `web_search` with argument `"code"` instead of `"query"` | 12 / 40 (30.0%) | Cross-attention bleed between `web_search` and `python_interpreter` schemas. | Contrastive tool-argument supervision in next training block. |
| **A3-NAMEERR** | Premature Variable Reference | Model references `diff` or `res` in `print()` without defining it | 7 / 15 (46.7% of Python calls) | Incomplete code generation before outputting standard print statement. | Targeted multi-step calculation scaffolding; additional training steps. |
| **A4-VARSYNTAX** | Variable Identifier Syntax | Variable name with whitespace (e.g. `rafael Nadal = 22`) | 2 / 15 (13.3% of Python calls) | Natural language entity name bleeding into Python code token generation. | AST validation loss penalty; snake_case variable enforcement. |
| **A5-EARLYSTOP** | Turn 2 Tool Omission | Model stops after Web observation and answers in text without calling Python | 15 / 30 (50.0% of standard tasks) | Conservative stopping threshold; model treats Web observation as sufficient. | Tune transition prior; reinforce Python necessity on calculation cues. |
| **A6-CITEOVERLOOK** | Final Synthesis Tag Omission | Model outputs correct number in Turn 3 but omits `[1]` citation | 5 / 30 (16.7%) | Synthesis span ends before emitting citation token. | Explicit citation reward or minimum token penalty. |

---

## 2. Decision Calibration Failures (Corpus B)

| Failure Code | Failure Category | Description | Observed Frequency | Root Cause | Proposed Mitigation |
|---|---|---|:---:|---|---|
| **B1-CONSERVATIVE** | Single-Step Direct Bias | Model solves single-step web lookups or calculations parametrically in text | 21 / 22 (95.5% of tool-required calibration tasks) | Strong conceptual discipline penalty from Branch B causing over-conservatism. | Recalibrate prompt instruction to explicitly differentiate single-step lookups from concepts. |
| **B2-FALSEPOS** | Conceptual Tool Hallucination | Emitting tool calls on direct conceptual definitions | **0 / 8 (0.0%)** | *Eliminated.* (Parent had 12.5% FPR). | Maintain current conceptual supervision. |

---

## 3. Evidence-Grounded Synthesis Failures (Corpus C)

| Failure Code | Failure Category | Description | Observed Frequency | Root Cause | Proposed Mitigation |
|---|---|---|:---:|---|---|
| **C1-CONTRADIV** | Contradiction Discrepancy Silence | Model reports single source instead of highlighting conflicting estimates | 5 / 5 (100.0% of contradiction tasks) | Training balance favoured full-support consensus answers over discrepancy reporting. | Upsample `SYN_CONTRADICTORY_EVIDENCE` from 50 to 100 in future corpora. |
| **C2-HALLUCINATE** | Unsupported Factual Claims | Hallucinating facts not present in retrieved evidence | **0 / 15 (0.0%)** | *Eliminated.* (Parent had 66.7% unsupported rate). | Maintain current attribution supervision. |
| **C3-CITELOSS** | Citation Omission | Stating verified facts without `[1]` attribution tag | 1 / 15 (6.7%) | Minor edge-case truncation. | Maintained at 93.3% compliance. |

---

## 4. Evaluator Assumptions & Scoring Reconciliations

### 4.1 Reconciling the 15 vs 30 Synthesis Task Discrepancy
In earlier reporting, synthesis metrics were alternately cited out of 15 or 30 tasks. The definitive breakdown in Stage 15 is:
- **Total Synthesis Corpus (Corpus C)**: Exactly **30 tasks** in the test split.
  - `SYN_FULL_SUPPORT`: **15 tasks** (strict supported precision: 100.0%, citation correctness: 93.3%).
  - `SYN_PARTIAL_SUPPORT`: **5 tasks** (partial handling rate: 100.0%).
  - `SYN_CONTRADICTORY_EVIDENCE`: **5 tasks** (contradiction handling: 0.0%).
  - `SYN_INSUFFICIENT_EVIDENCE`: **5 tasks** (insufficient acknowledgement: 0.0%).
- **Overall Semantic Success**: **20 / 30 (66.7%)** across the entire 30-task corpus.

### 4.2 Formatting Sensitivity
The automated evaluation harness strictly penalizes delimiter drift (e.g. `<<<` instead of `<tool_call>`). When evaluated with permissive bare-JSON parsing, tool selection rates rise from 3.3% to **50.0%**, demonstrating that the underlying decision capability is substantially stronger than rigid delimiter matching alone suggests.
