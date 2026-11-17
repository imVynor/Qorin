# Phase 1D RAG and Behavioral Audit

---

## PART A: Retrieval-Augmented Generation Audit

> [!CAUTION]
> The factual narrative in `retrieval_evaluation_report.md` is entirely fabricated. The 10 actual RAG queries are project-internal QORIN specifications questions, not general knowledge questions about Shakespeare, Australia, Mars, gold symbols, or the speed of light.

### A1. Query Mismatch — What Was Claimed vs What Was Actually Tested

**The written report claims:**
> *"The test set comprises 10 factual knowledge queries spanning science, geography, biology, and computer history: What is the chemical symbol for gold? Who wrote Romeo and Juliet? What is the capital of Australia? ..."*

**The actual queries in `phase1d_benchmark_data.py`:**
```python
RETRIEVAL_QUERIES = [
    {"id": "rag_01", "query": "What is QORIN's GQA ratio across 16M, 50M, and 100M?", ...},
    {"id": "rag_02", "query": "What was the passkey retrieval accuracy on the 4K context pilot?", ...},
    {"id": "rag_03", "query": "What is the token-weighted loss correction implemented in Phase 1A?", ...},
    {"id": "rag_04", "query": "What is the greeting density percentage in SFT v2 dataset?", ...},
    {"id": "rag_05", "query": "What is QORIN-500M's target pretraining corpus size?", ...},
    {"id": "rag_06", "query": "What is the attention head configuration for QORIN-500M?", ...},
    {"id": "rag_07", "query": "What activation function is used in ModernGPT-S 500M?", ...},
    {"id": "rag_08", "query": "What is the context window length for ModernGPT-S 500M?", ...},
    {"id": "rag_09", "query": "What is the thermal operating schedule for 500M pretraining?", ...},
    {"id": "rag_10", "query": "What is the production tokenizer vocabulary size for ModernGPT-S 500M?", ...},
]
```

The queries test retrieval of internal QORIN project-specific information. These queries are not part of either model's training data in any meaningful form. The comparison to general factual knowledge retrieval (the standard for RAG benchmarking) does not apply here.

The per-query comparison table in the original report (showing Shakespeare, gold, Mars, etc.) was not generated from the evaluation code or data. It was fabricated narrative.

### A2. Scoring Artifact Analysis

The scoring function:
```python
score = any(t.lower() in out.lower() for t in terms)
```
This returns True if **any** key term is a substring of the model's output.

**Identified prompt-echo artifacts:**

| RAG ID | Query | Key Term | Artifact |
|---|---|---|---|
| `rag_02` | "...passkey retrieval accuracy on the 4K context pilot?" | `"passkey"` | Term appears in the question itself; model echoing the question scores True |
| `rag_03` | "...token-weighted loss correction implemented in Phase 1A?" | `"token-weighted"` | Same: term in question |
| `rag_04` | "...greeting density percentage in SFT v2 dataset?" | `"greeting"` | Same: term in question |

**Identified substring collision artifacts:**

| RAG ID | Key Term | Collision String | RAG Record Output Excerpt |
|---|---|---|---|
| `rag_02` | `"4%"` or `"4"` | Model echoed the question verbatim | `"passkey retrieval accuracy on the 4K context"` → "4" matches |
| `rag_06` | `"24"` | Parameter string `"23,305,344"` | `"contains 16,768,128 parameters (23,305,344"` → "24" matches as substring... wait: `"23,305,34"` contains `"4"` and near `"24"` but the actual match was `"24"` from `"24 attention heads"` text |

**Revised per-query truth table based on JSONL inspection:**

| RAG ID | A (100M Off) | B (100M On) | C (500M Off) | D (500M On) | Artifact Type |
|---|---|---|---|---|---|
| rag_01 | ❌ | ❌ | ❌ | ❌ | None; all wrong |
| rag_02 | ✅* | ✅* | ✅* | ❌ | Prompt echo: "passkey" in echo |
| rag_03 | ✅* | ✅* | ✅* | ✅* | Prompt echo: "token-weighted" in echo |
| rag_04 | ✅* | ✅* | ❌ | ❌ | Prompt echo: "greeting" in echo |
| rag_05 | ❌ | ❌ | ❌ | ❌ | None |
| rag_06 | ❌ | ✅* | ❌ | ✅* | Substring: "24" in parameter count |
| rag_07 | ❌ | ❌ | ❌ | ❌ | None |
| rag_08 | ❌ | ❌ | ❌ | ❌ | None |
| rag_09 | ❌ | ❌ | ❌ | ❌ | None |
| rag_10 | ❌ | ❌ | ❌ | ❌ | None |
| **Total (reported)** | **3** | **4** | **2** | **2** | — |
| **Genuine correct** | **0** | **0** | **0** | **0** | — |

\* = Artifact-driven score, not genuine retrieval

### A3. What the RAG Results Actually Tell Us

The RAG matrix scores (A=3, B=4, C=2, D=2) are entirely explained by prompt-echo scoring artifacts. No model correctly retrieved and extracted a factual answer from project-internal documentation.

The correct conclusion is:
> *"Neither model demonstrates reliable retrieval and answer extraction from internal QORIN project documentation at zero-shot. The evaluation test set is not an appropriate benchmark for general RAG capability; it tests recall of internal proprietary facts absent from both models' training corpora. All apparent positive scores are artifacts of substring matching against echoed prompt terms."*

### A4. Retrieval Engine Note

The `RetrievalEngine.retrieve()` call injects source blocks formatted as:
```text
[Source 1: QORIN Model Architecture Specifications - QORIN-16M Proof of Concept]
QORIN-16M contains 16,768,128 parameters...
```
In the `rag_06` Condition D output, the model literally echoes this `[Source N: ...]` block format, suggesting it treats the retrieved context as a template to continue in document-continuation mode, not as evidence to extract an answer from.

---

## PART B: Behavioral Evaluation Audit

### B1. Attractor Rate — VERIFIED

The attractor pattern list from `phase1d_benchmark_data.py`:
```python
ATTRACTORS = [
    "how can i help you today?",
    "dear ",
    "i hope this email finds you well",
    "i hope this message finds you well",
    "as an ai language model",
    "i am an ai language model",
    "jimmy",
]
```

Inspecting all 100 behavioral records in `raw_generations.jsonl`: **0 records** contain any of these strings in `isolated_output` or `multiturn_output`. The 0.0% attractor rate is **verified correct**.

### B2. Accuracy Scores — Verified

Isolated accuracy: 2/100 = 2.0% ✅  
Multi-turn accuracy: 3/100 = 3.0% ✅  
Contamination rate: 23/100 = 23.0% — this is a negative trigger appearing in multi-turn outputs, not a prompt injection.

### B3. Scoring Methodology Assessment

The scoring uses exact substring matching of `expected_answer`. For base pretrained models operating in document-continuation mode, this severely under-counts cases where the model produces the correct information embedded in a continuation paragraph.

**Example:** If the expected answer is `"canberra"` and the model generates:

> *"Australia's capital is located in the Australian Capital Territory, situated between Sydney and Melbourne, in a region chosen..."*

This scores as INCORRECT because the string "canberra" is not present, even though the model may have produced correct associative context. This is a known limitation of substring-only behavioral scoring for base pretrained models.

The behavioral results are correct as measured. The 2.0% and 3.0% figures accurately reflect exact string match accuracy. They should not be interpreted as an upper bound on what the model knows.

### B4. Multi-Turn Format Assessment

The multi-turn evaluation format:
```python
history = ""
for m in item["multi_turn_messages"]:
    history += f"{m['role'].capitalize()}: {m['content']}\n"
history += "Assistant: "
```

This raw `Role: Content` format was not seen during pretraining. The behavioral improvement from isolated (2.0%) to multi-turn (3.0%) is small and within noise for 100 cases, though the relevant_context_retention category shows 10% → 15%, suggesting the model does partially exploit multi-turn context even in this unstructured format.

### B5. Math Behavioral Cases

Simple arithmetic tasks within behavioral evaluation (10 cases): 0.0% accuracy. All cases verified in JSONL. The model generates discussion text around arithmetic rather than computing values. Consistent with the dedicated math benchmark (0.0% across 50 tasks).

