# Stage 0 Data Audit Report: Forensic Analysis of Experiment C SFT Dataset

**Experiment ID**: `QORIN-EXP-D-STAGE-0`  
**Audit Date**: 2026-11-17T02:55:00+05:30  
**Target Audited**: `modern_gpt/experiments/phase2_experiment_c/stage_1_data/`  
**Checkpoint Audited**: `ckpt_experiment_c_final.pt` (`6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`)  
**Investigator**: Antigravity Autonomous Systems Engineering Team  

---

## 1. Executive Summary & Verification of Imbalance

Our quantitative audit of the serialized Experiment C dataset (`train.jsonl` and `val.jsonl`) confirms the exact structural defects identified during the live web retrieval incident investigation:

1. **Extreme Python-to-Web Discrepancy**:
   - Training Set (`train.jsonl`): **3,962 total examples**
     - Python Execution Calls: **2,183 (55.10%)**
     - Direct Conversational Answers: **1,703 (42.98%)**
     - Web Research Calls: **76 (1.92%)**
     - **Exact Serialized Ratio**: **28.72 to 1** in favor of Python.
   - Validation Set (`val.jsonl`): **208 total examples**
     - Python Execution Calls: **117 (56.25%)**
     - Direct Answers: **87 (41.83%)**
     - Web Research Calls: **4 (1.92%)**

2. **Template Rigidity & Prefix Monoculture**:
   - **100% (76/76)** of the web research training examples began with the literal formulaic string `Search the web for: <query>`.
   - The queries were drawn from only **5 fixed topics** (Nobel Prize in Physics 2024, Nobel Prize in Chemistry 2024, Attention Is All You Need date, James Webb Space Telescope deep field date, and Python 3.12 release date).
   - Zero colloquial variations (`"grab info on"`, `"who is"`, `"tell me about"`, `"find out about"`) were present in the training corpus.

3. **Train-Validation Overlap**:
   - **59 out of 208 validation prompts (28.37%)** were exact duplicate copies of training set prompts, artificially inflating validation metrics in Experiment C.

4. **Training-Inference Delimiter Mismatch**:
   - The dataset serializer ([`modern_gpt/data/conversation_serializer.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data/conversation_serializer.py)) mapped `role="tool"` to `role="tool_observation"`:
     ```
     <|im_start|>tool_observation\n<|tool_result|>{...}<|im_end|>\n
     ```
   - In live inference ([`backend/models/qorin_provider.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/models/qorin_provider.py)), the prompt generator emitted:
     ```
     <|im_start|>tool\n<|tool_result|>{...}<|im_end|>\n
     ```
   - This role token mismatch degraded the model's closed-loop conditioning when interpreting observation payloads.

---

## 2. Quantified Breakdown of Experiment C Dataset

| Dataset Split | Total Records | Python Calls | Web Calls | Direct Answers | Multi-Tool | Python:Web Ratio |
|---|---|---|---|---|---|---|
| `train.jsonl` | 3,962 | 2,183 | 76 | 1,703 | 755 | **28.72 : 1** |
| `val.jsonl` | 208 | 117 | 4 | 87 | 45 | **29.25 : 1** |
| **Combined** | **4,170** | **2,300** | **80** | **1,790** | **800** | **28.75 : 1** |

### Web Prompt Prefix Distribution in Experiment C Training Set:
```
"Search the web for: Nobel Prize in Physics 2024":     31 records
"Search the web for: Attention Is All You Need":       15 records
"Search the web for: James Webb Space Telescope":      15 records
"Search the web for: Python 3.12 release date":        15 records
-----------------------------------------------------------------
Total Unique Web Phrasing Templates:                    4 templates
Colloquial / Question / Verification Formulations:       0 records
```

---

## 3. Serialization and Delimiter Audit

### Canonical Training Token Formats in Experiment C:
1. **Tool Call Invocation**:
   `<|tool_call|>{"name": "<tool_name>", "arguments": {<args>}}`
   - Loss Masking: Supervised (labels = `token_ids`).
2. **Tool Observation**:
   `<|im_start|>tool_observation\n<|tool_result|>{<json_payload>}<|im_end|>`
   or
   `<|im_start|>tool_observation\n<|tool_error|>{<json_payload>}<|im_end|>`
   - Loss Masking: Ignored (labels = `-100`).
3. **Assistant Synthesis**:
   `<|im_start|>assistant\nBased on verified web research:\n...\n\nSource: [<title>](<url>)<|im_end|>`
   - Loss Masking: Supervised (labels = `token_ids`).

---

## 4. Required Experiment D Architectural & Curriculum Correctives

1. **Rebalance Multi-Tool Distribution**:
   - Expand Web Research representations from 1.9% to over **40%** of the SFT dataset (~1,500+ web-involved interactions).
   - Invert the ratio from **28.72:1** down to approximately **0.61:1**, eliminating the model's severe Python selection bias.
2. **Linguistic Phrasing Diversity**:
   - Introduce 4 distinct semantic intent classes: Colloquial (`"grab info on"`, `"look into"`), Question-based (`"Who is"`, `"What is"`), Explicit Search (`"Search the web for"`), and Fact Verification (`"Verify whether"`).
3. **Zero Contamination**:
   - Guarantee 0% overlap with all 50 protected evaluation benchmark prompts and 0% train/val split overlap.
4. **Role Normalization**:
   - Ensure inference and training role representations are aligned strictly to `tool_observation`.
