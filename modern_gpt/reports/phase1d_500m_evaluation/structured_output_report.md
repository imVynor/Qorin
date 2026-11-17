# ModernGPT-S 500M — Structured Output & JSON Generation Report

**Document ID:** `REP-PHASE1D-11-STRUCTURED-JSON`  
**Classification:** Structured Output Compliance Benchmark  
**Benchmark:** 50 Held-Out JSON Object Synthesis Tasks (`evaluate_sft_benchmarks.py`)  
**Date:** November 13, 2026  

---

## 1. Quantitative Results

The 50 structured JSON synthesis tasks were evaluated under deterministic greedy decoding ($T=0.0$, max new tokens 140):

- **JSON Syntax Validity Rate:** **`2.0%`** (1 / 50 tasks parsed cleanly as valid JSON via `json.loads`)
- **Schema Conformance Rate:** **`0.0%`** (0 / 50 tasks satisfied all required keys and typing constraints)
- **Prose / Markdown Wrapper Rate:** **`76.0%`** (38 / 50 tasks wrapped partial JSON blocks in conversational explanation)
- **Syntax Truncation Rate:** **`22.0%`** (11 / 50 tasks emitted unclosed braces or trailing commas)

---

## 2. Failure Analysis

1. **Unclosed Braces & Continuation:**
   The base model frequently begins emitting key-value pairs (e.g. `{"name": "Alice", "age": 30`), but continues into narrative text or additional sample records without emitting the terminating `}` bracket.
2. **Missing Instruction Constraints:**
   When prompted `"Output a valid JSON object only"`, the base model predicts typical web documents where the prompt is followed by explanatory prose, tutorials, or alternative configurations rather than an isolated raw JSON payload.
3. **Contrast with SFT Models:**
   In historical SFT v2, schema validity reached **`48.0%`** because supervised training explicitly penalized non-JSON tokens between `<|im_start|>assistant` and `<|im_end|>`. Pretrained base models require this delimiter supervision to enforce strict schema adherence.

