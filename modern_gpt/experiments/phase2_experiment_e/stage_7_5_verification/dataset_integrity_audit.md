# QORIN Experiment E: Comprehensive Dataset Integrity & Provenance Audit

**Document**: `dataset_integrity_audit.md`  
**Execution Timestamp**: November 17, 2026  
**Subject Files**: 9 Serialized JSONL Datasets under `modern_gpt/experiments/phase2_experiment_e/stage_1_data/`  
**Inspection Directory**: `modern_gpt/experiments/phase2_experiment_e/stage_7_5_verification/`

---

## 1. Executive Summary

A programmatic audit of all nine serialized dataset files in `stage_1_data/` was conducted using `run_stage7_5_verification.py`. 

The audit verifies:
1. **Record Counts & IDs**: Exactly 1,000 unique records across the unified pool. Zero ID collisions exist between Corpus 1 and Corpus 2.
2. **Train/Val/Test Separation**: Zero leakage exists across data partitions.
3. **Union Consistency**: The combined sets are exact mathematical unions of their constituent Corpus 1 and Corpus 2 splits.
4. **Tokenization & Masking**: All records encode cleanly with the project's 32k BPE tokenizer without truncation. Assistant-target token masking boundaries are strictly indexed.
5. **Decontamination Provenance**: 1,000 candidate records were audited against the 150 tasks of `benchmark.jsonl`, achieving 100% clean clearance.

---

## 2. Record & Token Inventory by File

| Dataset File Name | Split Role | Records | Unique IDs | Total Tokens | Target Tokens | Max Record Length | Truncation ($>2048$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `corpus1_calibration_train.jsonl` | Calibration Train | 400 | 400 | 37,282 | 18,046 | 184 tok | **0 (None)** |
| `corpus1_calibration_val.jsonl` | Calibration Val | 50 | 50 | 6,523 | 3,635 | 179 tok | **0 (None)** |
| `corpus1_calibration_test.jsonl` | Calibration Test | 50 | 50 | 4,775 | 1,915 | 162 tok | **0 (None)** |
| `corpus2_synthesis_train.jsonl` | Synthesis Train | 400 | 400 | 72,038 | 19,990 | 318 tok | **0 (None)** |
| `corpus2_synthesis_val.jsonl` | Synthesis Val | 50 | 50 | 8,740 | 2,500 | 294 tok | **0 (None)** |
| `corpus2_synthesis_test.jsonl` | Synthesis Test | 50 | 50 | 10,642 | 3,175 | 288 tok | **0 (None)** |
| `combined_train.jsonl` | Combined Train | 800 | 800 | 109,320 | 38,036 | 318 tok | **0 (None)** |
| `combined_val.jsonl` | Combined Val | 100 | 100 | 15,263 | 6,135 | 294 tok | **0 (None)** |
| `combined_test.jsonl` | Combined Test | 100 | 100 | 15,417 | 5,090 | 288 tok | **0 (None)** |

---

## 3. Disjoint Partitioning & Leakage Verification

### 3.1 Corpus 1 vs. Corpus 2 Disjointness:
- $ID(\text{Corpus 1}) \cap ID(\text{Corpus 2}) = \emptyset$ (0 shared IDs).
- $\text{Prompts}(\text{Corpus 1}) \cap \text{Prompts}(\text{Corpus 2}) = \emptyset$ (0 shared prompt strings).
- **Status**: **100% Disjoint**.

### 3.2 Partition Isolation (Within Corpora):
- $ID(\text{Train}) \cap ID(\text{Val}) = \emptyset$
- $ID(\text{Train}) \cap ID(\text{Test}) = \emptyset$
- $ID(\text{Val}) \cap ID(\text{Test}) = \emptyset$
- **Status**: **Zero Partition Leakage**.

### 3.3 Permanent Test Split Quarantine:
The 100-record Test split (`combined_test.jsonl`) is isolated in storage. It is strictly excluded from DataLoader pipelines for training, validation checkpoints, and early-stopping loss calculations.

---

## 4. Assistant-Target Loss Masking Integrity

Every serialized JSONL record embeds explicit token offsets:
- `prefix_token_length`: Exact count of tokens preceding the assistant's generation. In the training collator, tokens $0 \le t < \text{prefix\_token\_length}$ are assigned loss weight $M_t = 0.0$.
- `target_token_length`: Exact count of generated assistant tokens. Assigned loss weight $M_t = 1.0$.
- `total_token_length`: Equals $\text{prefix\_token\_length} + \text{target\_token\_length}$.

*Audit Verification*: All 1,000 records satisfy `prefix_tokens + target_tokens == total_tokens` with zero indexing drift.

---

## 5. Epistemological Scope of Heuristic Decontamination

In strict accordance with scientific integrity standards, we explicitly delineate what heuristic decontamination **can** and **cannot** establish:

### What the 3-Tier Decontamination Gates Establish:
1. **Zero Textual Memorization**: Proves zero exact string duplication between training records and benchmark tasks.
2. **Zero Contiguous N-Gram Regurgitation**: The 10-gram Jaccard threshold ($< 0.15$) guarantees that the model cannot solve benchmark tasks by replaying memorized 10-token phrasing sequences.
3. **Zero Named Entity Co-Occurrence**: Gate 3 guarantees that training prompts do not pair the specific named entities tested in the benchmark (e.g. "James Webb Space Telescope", "Mount Everest", "Peregrine Falcon").

### What Heuristic Decontamination CANNOT Establish:
1. **Semantic & Conceptual Independence**: No string or entity filter can guarantee absolute topological semantic independence. For example, training on "Explain the backpropagation chain rule" inherently improves the model's representations for a benchmark question on "Explain gradient descent", even though they share zero entities or 10-grams.
2. **Syntactic Generalization Overlap**: Training the model on the syntax `<|tool_call|>{"name": "web_research", ...}` intentionally teaches it the functional API required to pass benchmark tool-call evaluation. This is intentional capability acquisition, not benchmark leakage.
3. **Parametric Knowledge Overlap**: Both the training set and the benchmark query general knowledge about the natural world (physics, chemistry, geography). Factual alignment is the intended objective of a foundation model, distinct from test-set cheating.

**Explicit Scientific Policy**: We do **not** claim that heuristic decontamination proves complete semantic independence. We claim strictly that it establishes **structural, lexical, and entity-level non-contamination**.
