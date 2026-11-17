# QORIN Experiment E: Definitive Dataset Allocation & Token Accounting Specification

**Document**: `dataset_allocation.md`  
**Status**: Pre-Implementation Specification (Zero Training Files Created)  
**Total Curated Pool**: Exactly 1,000 Multi-Turn Dialogue Records (~350,000 tokens)  
**Integrity Invariance**: Zero record overlap between Calibration and Synthesis corpora; 100% strict isolation of the Test split.

---

## 1. Disambiguation of Corpus Structure & Disjoint Partitioning

To eliminate ambiguity from earlier research proposals:
1. **Corpus 1 (Calibration)** and **Corpus 2 (Synthesis)** are **100% disjoint**. They share zero dialogue records, zero prompt strings, and zero synthetic seeds.
2. The **Total Curated Pool** consists of exactly **1,000 unique records**:
   - **500 records** dedicated to Tool-Decision Calibration (Corpus 1).
   - **500 records** dedicated to Evidence-Grounded Synthesis (Corpus 2).
3. The **Combined Branch (Branch D)** draws its training data directly from the unified pool of Train splits from Corpus 1 and Corpus 2 ($400 + 400 = 800\text{ training records}$). No new or secret records are created.

---

## 2. Global Allocation Matrix (Train / Validation / Test)

The 1,000 curated records are partitioned using a strict 80 / 10 / 10 split architecture:

```
                            [Total Curated Pool: 1,000 Records]
                                            │
        ┌───────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                       ▼
Corpus 1: Calibration (500 Records)                     Corpus 2: Synthesis (500 Records)
  ├── Train:      400 Records (80.0%)                     ├── Train:      400 Records (80.0%)
  ├── Validation:  50 Records (10.0%)                     ├── Validation:  50 Records (10.0%)
  └── Test:        50 Records (10.0%)                     └── Test:        50 Records (10.0%)
```

### Aggregate Split Totals Across Entire Pool:
- **Total Training Split ($D_{train}$)**: Exactly **800 records** (Used exclusively for gradient updates).
- **Total Validation Split ($D_{val}$)**: Exactly **100 records** (Used exclusively for evaluation checkpoints and early stopping).
- **Total Held-Out Test Split ($D_{test}$)**: Exactly **100 records** (Vaulted; strictly barred from training, hyperparameter tuning, and early stopping).

---

## 3. Record-Level Allocation by Category

### 3.1 Corpus 1: Tool-Decision Calibration (500 Records)

| Category ID | Semantic Focus | Target Action | Train | Val | Test | Total Records |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `CAL_DIR_CONCEPT` | Algorithmic, mathematical, and scientific definitions. | `DIRECT` | 120 | 15 | 15 | **150** |
| `CAL_DIR_PARAM` | Historical facts and literature within pretraining cutoff. | `DIRECT` | 48 | 6 | 6 | **60** |
| `CAL_DIR_NEG` | Unfulfillable, harmful, or subjective prompts (Refusal). | `DIRECT` | 32 | 4 | 4 | **40** |
| `CAL_TOOL_WEB` | Post-cutoff events, live stats, colloquial retrieval. | `WEB` | 100 | 12 | 13 | **125** |
| `CAL_TOOL_PY` | Arithmetic, linear algebra, simulations. | `PYTHON` | 60 | 8 | 7 | **75** |
| `CAL_TOOL_MULTI` | Contrastive empirical lookup + computation. | `WEB -> PY` | 40 | 5 | 5 | **50** |
| **Subtotal Corpus 1** | — | — | **400** | **50** | **50** | **500** |

*Note: Exactly 50 paired contrastive dyads (100 records) are embedded within Corpus 1. Both members of each dyad are assigned to the same split to avoid cross-split contamination.*

---

### 3.2 Corpus 2: Evidence-Grounded Synthesis (500 Records)

| Category ID | Evidence Conditioning Schema | Factual Target | Train | Val | Test | Total Records |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `SYN_FACT_SINGLE` | Structured Evidence Record (SER) | Single Entity Metric | 96 | 12 | 12 | **120** |
| `SYN_FACT_MULTI` | Structured Markdown Table | Multi-Entity Comparison | 64 | 8 | 8 | **80** |
| `SYN_CONFLICT` | Conflicting SER Records | Range / Discrepancy Statement | 80 | 10 | 10 | **100** |
| `SYN_INSUFFICIENT` | Tangential / Sparse SER | Uncertainty / Refusal | 80 | 10 | 10 | **100** |
| `SYN_CALC_GROUND` | Web SER + Python Execution Output | Verified Math Result | 40 | 5 | 5 | **50** |
| `SYN_ATTRIBUTION` | Multi-Source Document SER | Strict `[Source X]` Citations | 40 | 5 | 5 | **50** |
| **Subtotal Corpus 2** | — | — | **400** | **50** | **50** | **500** |

---

## 4. Multi-Turn Dialogue Serialization & Loss Masking

All records are serialized using standard QORIN ChatML syntax. Assistant-target loss masking is enforced at the token level using binary cross-entropy masks ($M_t \in \{0, 1\}$).

### 4.1 Calibration Record Serialization (Corpus 1)

#### Case A: Direct Response (`DIRECT`)
```xml
<|im_start|>system
You are QORIN, a calibrated AI assistant. Answer direct conceptual questions directly. Use tools only when external retrieval or calculation is strictly necessary.<|im_end|>
<|im_start|>user
Explain the bias-variance tradeoff in machine learning.<|im_end|>
<|im_start|>assistant
The bias-variance tradeoff describes the tension between two sources of error in predictive models...<|im_end|>
```
* **Masking**: Tokens from `<|im_start|>system` through `<|im_start|>assistant\n` have $M_t = 0$. All assistant answer tokens have $M_t = 1$.

#### Case B: Tool Call (`WEB` or `PYTHON`)
```xml
<|im_start|>system
...<|im_end|>
<|im_start|>user
Find the current market price of Brent Crude oil.<|im_end|>
<|im_start|>assistant
<|tool_call|>{"name": "web_research", "arguments": {"query": "Brent crude oil price current"}}<|end_tool_call|><|im_end|>
```
* **Masking**: Tokens prior to `<|tool_call|>` have $M_t = 0$. The tool call syntax, name, and JSON arguments have $M_t = 1$. The trailing `<|im_end|>` has $M_t = 1$.

---

### 4.2 Synthesis Record Serialization (Corpus 2)

```xml
<|im_start|>system
You are QORIN, an evidence-grounded AI assistant. Base your response strictly on the verified facts in the tool observation. Cite sources using [SOURCE X].<|im_end|>
<|im_start|>user
What is the primary mirror diameter and launch date of the James Webb Space Telescope?<|im_end|>
<|im_start|>assistant
<|tool_call|>{"name": "web_research", "arguments": {"query": "James Webb Space Telescope mirror diameter launch date"}}<|end_tool_call|><|im_end|>
<|im_start|>tool
<|tool_result|>
[SOURCE 1]: https://science.nasa.gov/webb
- ENTITY: James Webb Space Telescope
- LAUNCH_DATE: December 25, 2021
- PRIMARY_MIRROR: 6.5 meters diameter
<|end_tool_result|><|im_end|>
<|im_start|>assistant
The James Webb Space Telescope features a primary mirror diameter of 6.5 meters and was launched on December 25, 2021 [SOURCE 1].<|im_end|>
```
* **Masking**: All tokens from `<|im_start|>system` up through `<|im_start|>assistant\n` following the tool observation have $M_t = 0$. Only the final assistant synthesis tokens have $M_t = 1$.

---

## 5. Combined-Branch (Branch D) Curriculum Assembly & Weighting

During training of the Combined Branch (Branch D):
1. **Curriculum Mixing**: Records from $D_{train}^{\text{Corpus 1}}$ (400 records) and $D_{train}^{\text{Corpus 2}}$ (400 records) are merged into a single training manifest ($N = 800$ records).
2. **Stratified Alternating Mini-Batches**:
   Batches are sampled with a 1:1 ratio between Calibration records and Synthesis records to prevent gradient thrashing between Stage 1 logits and Stage 2 projection weights.
3. **Loss Weighting Factor**:
   $$\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{cal}} + (1 - \alpha) \mathcal{L}_{\text{syn}}, \quad \text{with } \alpha = 0.5$$
   Because both corpora use the identical assistant-target masking scheme, $\alpha = 0.5$ provides balanced, stable optimization without gradient dominance.
