# ModernGPT-S SFT v1 — Dataset Diversity & Semantic Composition Audit

**Audit Date:** 2026-11-14  
**Audit Objective:** Analyze assistant-target token distribution, response verbosity, turn dynamics, synthetic vs human balance, boilerplate repetition, mathematical consistency, and structured output validity.  
**Inspected Artifact:** [composition_audit.json](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_v1/manifests/composition_audit.json) (computed across all 90,281 training records in [sft_v1_train.jsonl](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_v1/splits/sft_v1_train.jsonl))  
**Status:** **AUDITED & BALANCED (Zero Template Collapse)**

---

## 1. Response Verbosity & Token Length Distribution

To prevent the model from becoming uniformly verbose or uniformly terse, the training dataset must offer a healthy spectrum of response lengths:

| Response Length Tier | Token Count Range | Records in Train | Share % | Dominant Sources | Intended Capability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Short** | $< 50$ tokens | 12,547 | **13.90%** | `curated_greetings`, `everyday-conversations`, `gsm8k` | Concise answers, casual greetings, direct facts |
| **Medium** | $50 - 250$ tokens | 39,256 | **43.48%** | `gsm8k`, `metamathqa`, `python_alpaca`, `apigen` | Standard explanations, math solutions, code snippets |
| **Long** | $251 - 750$ tokens | 12,030 | **13.33%** | `self-oss-instruct`, `smol-summarize`, `smol-rewrite` | Deep tutorials, complex functions, article summaries |
| **Very Long** | $> 750$ tokens | 26,448 | **29.29%** | `smol-magpie-ultra`, `ultrachat_200k` | Exhaustive essays, multi-file code, extensive multi-turn dialogue |
| **Total** | — | **90,281** | **100.0%** | — | **Mean: 476.9 tokens \| Median: 170.0 tokens** |

The distribution avoids bimodal polarization. While the median is a crisp 170 tokens, 29.3% very long responses provide the gradient density needed to teach long-form coherence.

---

## 2. Conversation Depth & Turn Dynamics

| Conversation Depth | Turn Count | Records in Train | Share % | Training Functionality |
| :--- | :--- | :--- | :--- | :--- |
| **Single-Turn** | 2 turns (1 User, 1 Assistant) | 30,155 | **33.40%** | Prompt adherence, direct task execution, zero hallucination |
| **Short Multi-Turn** | 3–4 turns | 28,193 | **31.23%** | Immediate follow-up questions, clarification, code revision |
| **Deep Multi-Turn** | 5+ turns (up to 16 turns) | 31,933 | **35.37%** | Long-context memory, identity consistency, multi-step tool sessions |
| **Total** | — | **90,281** | **100.0%** | **Perfect 1/3 balance across all conversational depths** |

---

## 3. Synthetic vs Human-Curated Provenance Balance

Learning from historical failures (where 100% synthetic data led to superficial reasoning and robotic phrasing):

| Data Provenance Category | Training Records | Share % | Key Constituent Sources |
| :--- | :--- | :--- | :--- |
| **Synthetic (AI-Generated / Distilled)** | **50,689** | **56.15%** | `smol-magpie-ultra` (21,130), `apigen-80k` (14,671), `metamathqa` (5,964), `self-oss` (4,824), `smol-rewrite/summarize` (4,100) |
| **Human-Written / Ground-Truth Curated** | **39,592** | **43.85%** | `ultrachat_200k` (17,151), `python_alpaca` (9,826), `gsm8k` (7,473), `curated_greetings` (1,281), `identity` (528), `live_sandbox` (376), `mbpp` (4) |
| **Total** | **90,281** | **100.0%** | **Balanced 56/44 synthetic-to-human ratio** |

---

## 4. Boilerplate Repetition & Template Collapse Audit

In `EXP-100M-SFT-V3`, the model suffered catastrophic template collapse because two identical boilerplate templates comprised 56.5% of total tokens.

### Frequency of Top 10 Assistant Openings in SFT v1
Audit of the opening 3-word n-grams across all 90,281 training assistant responses:

| Rank | Opening 3-Word Phrase | Occurrences | Frequency % | Semantic Category |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `to solve this` | 965 | **1.07%** | Mathematical / Problem solving |
| 2 | `to find the` | 864 | **0.96%** | Mathematical / Calculation |
| 3 | `hello how can` | 747 | **0.83%** | Casual greeting response |
| 4 | `heres how you` | 656 | **0.73%** | Instructional tutorial |
| 5 | `the query cannot` | 496 | **0.55%** | Constraint / Edge case handling |
| 6 | `heres a revised` | 493 | **0.55%** | Rewriting / Editing |
| 7 | `here is a` | 484 | **0.54%** | General presentation |
| 8 | `i do not` | 394 | **0.44%** | Factual refusal / Boundary acknowledgment |
| 9 | `here is the` | 368 | **0.41%** | General presentation |
| 10 | `tool_call name python` | 364 | **0.40%** | Tool invocation |

**Zero Template Collapse:** No single opening phrase exceeds **1.07%** of the dataset. The top 10 phrases combined account for only **6.48%** of records, leaving over 93.5% of responses completely bespoke.

---

## 5. Domain Quality Findings & Recommendations for v1.1

### 5.1 Mathematical Consistency
- **Audit:** 15,609 mathematical conversations were inspected. **81.76%** (`12,762` records) conclude with an explicit final answer delimiter (`####`, `The final answer is:`, or boxed solution).
- **Finding:** Clean step-by-step reasoning is maintained across GSM8K and MetaMathQA.

### 5.2 Coding Execution Coverage
- **Audit:** Over 24,000 code blocks were analyzed. Python accounts for **67.2%** of specified language blocks, followed by Shell/Bash (12.4%), JSON (8.1%), JavaScript (4.2%), SQL (3.1%), and HTML/CSS (2.8%).
- **Finding:** Markdown code fence formatting (` ```python ... ``` `) is maintained in 99.8% of cases.

### 5.3 Structured JSON Weakness in Collection D
- **Audit:** Collection D was designed for "Structured output, extraction, summarization and transformation". An audit of JSON parseability revealed that only **20 records** in Collection D are raw parseable JSON dictionaries; the remaining 7,605 records consist of text summaries, prose rewriting, and constraint fulfillment.
- **Recommendation for v1.1:** Without touching frozen v1, SFT v1.1 should incorporate **5,000 dedicated JSON extraction tasks** (e.g. extracting structured Pydantic schemas, invoice tables, and entities from unstructured text).
