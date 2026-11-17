# QORIN Experiment E: Dataset Preparation & Token Accounting Report

**Document**: `dataset_preparation_report.md`  
**Execution Date**: November 17, 2026  
**Curriculum Scope**: 1,000 Unique Multi-Turn Dialogue Records (~140,000 Conversation Tokens)  
**Tokenizer**: HuggingFace 32k BPE Tokenizer (`modern_gpt/artifacts/tokenizer_32k/tokenizer.json`, SHA-256: `d10624a9...`)  
**Data Directory**: `modern_gpt/experiments/phase2_experiment_e/stage_1_data/`

---

## 1. Executive Summary

In Phase 3 of Stage 7, the complete 1,000-record curriculum for Experiment E was generated, serialized, token-counted, and decontaminated according to the specifications established in `dataset_allocation.md`.

### Core Accomplishments:
1. **100% Disjoint Partitioning**: Corpus 1 (Calibration, 500 records) and Corpus 2 (Synthesis, 500 records) share zero records, zero prompts, and zero entity topics.
2. **Strict 80/10/10 Allocation**: Exactly 800 Train, 100 Validation, and 100 Test records across the unified pool.
3. **Permanent Test Split Isolation**: The 100-record Test split (`combined_test.jsonl`) is vaulted and permanently quarantined from training and validation early-stopping routines.
4. **Verified Assistant-Target Masking**: Every record embeds exact `prefix_token_length`, `target_token_length`, and `total_token_length` fields, enabling strict binary loss masking during fine-tuning.
5. **Full Decontamination Clearance**: All 1,000 candidate records were audited against the 150 tasks of `benchmark.jsonl` using the automated 3-gate decontamination engine. 0 violations were detected in the final dataset (`decontamination_audit.json`).

---

## 2. Record & Token Accounting by Dataset Split

All token counts were computed using the project's authoritative HuggingFace 32,768-vocabulary BPE tokenizer:

| Dataset File Name | Split Role | Records | Total Conversation Tokens | Assistant-Target Tokens | Avg Tokens / Record |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `corpus1_calibration_train.jsonl` | Branch B Training | 400 | 37,282 | 18,046 | 93.2 |
| `corpus1_calibration_val.jsonl` | Branch B Validation | 50 | 6,523 | 3,635 | 130.5 |
| `corpus1_calibration_test.jsonl` | Branch B Evaluation | 50 | 4,775 | 1,915 | 95.5 |
| **Subtotal Corpus 1** | **Calibration Pool** | **500** | **48,580** | **23,596** | **97.2** |
| `corpus2_synthesis_train.jsonl` | Branch C Training | 400 | 72,038 | 19,990 | 180.1 |
| `corpus2_synthesis_val.jsonl` | Branch C Validation | 50 | 8,740 | 2,500 | 174.8 |
| `corpus2_synthesis_test.jsonl` | Branch C Evaluation | 50 | 10,642 | 3,175 | 212.8 |
| **Subtotal Corpus 2** | **Synthesis Pool** | **500** | **91,420** | **25,665** | **182.8** |
| `combined_train.jsonl` | Branch D Training | **800** | **109,320** | **38,036** | **136.7** |
| `combined_val.jsonl` | Branch D Validation | **100** | **15,263** | **6,135** | **152.6** |
| `combined_test.jsonl` | Branch D Evaluation | **100** | **15,417** | **5,090** | **154.2** |
| **Total Unified Pool** | **Full Experiment E** | **1,000** | **140,000** | **49,261** | **140.0** |

---

## 3. Curriculum Composition & Balancing

### 3.1 Corpus 1: Decision-Calibration (500 Records)
Designed to calibrate QORIN-500M's Stage 1 action logits, eliminating false-positive tool calls while preserving retrieval and execution capability:
- **`CAL_DIR_CONCEPT` (150 records)**: Pure conceptual, algorithmic, and mathematical questions answerable from parametric weights without tools (e.g. backpropagation, bias-variance tradeoff, eigenvalue decomposition).
- **`CAL_DIR_PARAM` (60 records)**: Fixed historical facts, physical constants, and classical literature well within pretraining cutoff.
- **`CAL_DIR_NEG` (40 records)**: Negative controls requiring direct uncertainty or safety refusal (unfulfillable future predictions, harmful synthesis requests).
- **`CAL_TOOL_WEB` (125 records)**: Current information, live statistics, and colloquial retrieval requests requiring `web_research`.
- **`CAL_TOOL_PY` (75 records)**: Arithmetic, combinatorics, matrix math, and statistical simulations requiring `python`.
- **`CAL_TOOL_MULTI` (50 records)**: Empirical retrieval followed by calculation requiring `web_research` first.

### 3.2 Corpus 2: Evidence-Grounded Synthesis (500 Records)
Designed to condition QORIN-500M's Stage 2 attention on Structured Evidence Records (SER), enforcing factual extraction and inline citation:
- **`SYN_FACT_SINGLE` (120 records)**: Extraction of single entity metrics from key-value SER blocks.
- **`SYN_FACT_MULTI` (80 records)**: Multi-entity comparisons formatted in Markdown tables.
- **`SYN_CONFLICT` (100 records)**: Conflicting sources where model must state range and discrepancies.
- **`SYN_INSUFFICIENT` (100 records)**: Sparse search results where model must declare explicit uncertainty rather than hallucinating parametric facts.
- **`SYN_CALC_GROUND` (50 records)**: Synthesis combining verified SER facts and Python stdout calculations.
- **`SYN_ATTRIBUTION` (50 records)**: Complex multi-claim passages requiring strict `[SOURCE X]` inline citations.

---

## 4. Decontamination Verification Summary

Before serialization, all 1,000 candidate records were audited through the 3-gate decontamination infrastructure (`decontamination_audit.py`) against `benchmark.jsonl`:
- **Gate 1 (Exact Deduplication)**: 0 exact prompt or answer duplicate strings.
- **Gate 2 (10-Gram Jaccard)**: 0 records with 10-gram Jaccard similarity $\ge 0.15$.
- **Gate 3 (Entity Overlap)**: 0 records sharing $\ge 2$ named entities with any benchmark task.
- **Final Audit Status**: **1,000 / 1,000 records CLEAN (100.0% Pass Rate)**. Full task-level logs are archived in [`decontamination_audit.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_e/stage_7_prerequisites/decontamination_audit.json).

---

## 5. Masking & Serialization Integrity

Inspection of serialized records confirms:
1. Every record uses standard ChatML syntax (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n...`).
2. Prefix tokens (system, user, tool observation, and `<|im_start|>assistant\n`) are explicitly indexed so training collators mask their loss ($M_t = 0$).
3. Target tokens (assistant response tokens) are strictly indexed ($M_t = 1$).
4. Context length ceiling: The maximum record length across all 1,000 records is 318 tokens, well within QORIN-500M's 2,048-token context window. Zero truncation occurs.
