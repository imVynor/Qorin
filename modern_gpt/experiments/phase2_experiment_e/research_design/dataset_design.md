# QORIN Experiment E: Dataset Architecture & Curation Specification

**Document**: `dataset_design.md`  
**Status**: Pre-Implementation Specification (No training files created)  
**Target Corpus Size**: 1,000 total multi-turn dialogue records  
**Split Architecture**: 800 Train / 100 Validation / 100 Held-Out Evaluation  
**Decontamination Mandate**: Zero token, entity, or semantic overlap with the 150-task Stage 5 Benchmark.

---

## 1. Dataset Design Principles & Rationale

In Experiment D, the model received an overabundance of tool-positive examples without sufficient counterbalancing negative controls, skewing its decision boundary toward indiscriminate retrieval. In Stage 2, it was trained on noisy, loosely formatted snippets that did not enforce factual grounding.

Experiment E requires a surgically balanced, high-fidelity curriculum that addresses both failure modes while remaining lightweight enough to prevent catastrophic forgetting of general English and Python syntax.

### Core Architecture:
1. **Calibration Component (Corpus 1)**: 500 multi-turn records targeting Stage 1 tool dispatch.
2. **Synthesis Component (Corpus 2)**: 500 multi-turn records targeting Stage 2 evidence-grounded generation.
3. **Total Unified Corpus**: 1,000 records (~350,000 tokens).

---

## 2. Corpus 1: Tool-Decision Calibration (500 Records)

Corpus 1 balances the prior probability between direct conceptual generation and tool invocation at exactly 1:1 (50% Direct, 50% Tool).

### 2.1 Category Breakdown & Sample Allocations

| Category ID | Description | Expected Action | Sample Count | Difficulty Distribution (Easy / Med / Hard) |
| :--- | :--- | :---: | :---: | :---: |
| `CAL_DIR_CONCEPT` | Core scientific, technical, philosophical, and algorithmic concepts. | `DIRECT` | 150 | 50 / 60 / 40 |
| `CAL_DIR_PARAM` | Historical facts, dates, constants, and literature clearly within pretraining cutoff. | `DIRECT` | 60 | 20 / 25 / 15 |
| `CAL_DIR_NEG` | Unfulfillable, speculative, subjective, or potentially harmful tool requests. | `DIRECT` (Refusal/Uncertainty) | 40 | 10 / 15 / 15 |
| `CAL_TOOL_WEB` | Current events, post-cutoff metrics, live stats, and colloquial search requests. | `WEB` | 125 | 35 / 55 / 35 |
| `CAL_TOOL_PY` | Arithmetic, algebraic solving, statistical simulations, and combinatorics. | `PYTHON` | 75 | 20 / 35 / 20 |
| `CAL_TOOL_MULTI` | Contrastive queries requiring empirical fact retrieval followed by computation. | `WEB -> PYTHON` | 50 | 10 / 25 / 15 |
| **Total Corpus 1** | — | — | **500** | **145 / 215 / 140** |

### 2.2 Contrastive Dyad Architecture
To explicitly train the decision boundary, 100 records in Corpus 1 will be constructed as **50 paired contrastive dyads**:

```json
// Example Contrastive Dyad:
{
  "dyad_id": "dyad_phys_01",
  "pair_A": {
    "prompt": "What is the physical principle behind gravitational lensing?",
    "expected_action": "DIRECT",
    "rationale": "General relativity theory; self-contained conceptual explanation."
  },
  "pair_B": {
    "prompt": "What was the date of the first observation of gravitational lensing around galaxy cluster MACS J1149?",
    "expected_action": "WEB",
    "rationale": "Specific astronomical observation event requiring empirical citation."
  }
}
```

### 2.3 Assistant-Target Masking Strategy (Stage 1)
- For `DIRECT` examples: Loss is calculated on all assistant response tokens (`weight = 1.0`), training the model to begin directly answering without prepending tool tags.
- For `TOOL` examples: Loss is calculated strictly on the `<|tool_call|>` tag, tool name, and JSON arguments up to `<|end_tool_call|>` (`weight = 1.0`). Prompt tokens and subsequent observation tokens have `weight = 0.0`.

---

## 3. Corpus 2: Evidence-Grounded Synthesis (500 Records)

Corpus 2 trains the model to act as a faithful information-extraction and synthesis engine when provided with retrieved tool observations.

### 3.1 Category Breakdown & Sample Allocations

| Category ID | Description | Evidence Condition | Sample Count | Failure Mode Mitigated |
| :--- | :--- | :--- | :---: | :--- |
| `SYN_FACT_SINGLE` | Single entity, single metric lookup (e.g. height, population, founding date). | Structured Evidence Schema | 120 | Extraction failure, number scrambling |
| `SYN_FACT_MULTI` | Multi-entity comparative tables (e.g. comparing GDPs of 3 nations). | Structured Table Schema | 80 | Entity-metric confusion |
| `SYN_CONFLICT` | Sources report differing dates or figures; model must state the range. | Conflicting Source Records | 100 | Arbitrary selection, false certainty |
| `SYN_INSUFFICIENT` | Retrieved snippet contains topic-adjacent text but lacks the requested fact. | Tangential Search Snippet | 100 | Parametric hallucination override |
| `SYN_CALC_GROUND` | Python calculation executed on numbers extracted from web research. | Web Fact + Python Result | 50 | Ungrounded math, fabricated constants |
| `SYN_ATTRIBUTION` | Complex multi-claim answer requiring strict inline `[Source X]` citations. | Multi-Source Document | 50 | Unverifiable claims, citation drift |
| **Total Corpus 2** | — | — | **500** | — |

### 3.2 Target Output Syntax: Structured Evidence Record Format
All web tool observations in Corpus 2 will use the **Structured Evidence Record (SER)** syntax:

```xml
<|tool_result|>
[SOURCE 1]: https://www.esa.int/Science_Exploration/Space_Science/Juice
- TITLE: ESA - Juice space mission overview
- ENTITY: Jupiter Icy Moons Explorer (Juice)
- LAUNCH_DATE: 14 April 2023
- DESTINATION: Jupiter (Ganymede, Callisto, Europa)
- ARRIVAL_ESTIMATE: July 2031
<|end_tool_result|>
```

Target Assistant Generation Format:
```xml
<|assistant|>
The Jupiter Icy Moons Explorer (Juice) was launched on 14 April 2023 by the European Space Agency [SOURCE 1]. It is scheduled to arrive at the Jovian system in July 2031 to study Ganymede, Callisto, and Europa [SOURCE 1].
<|end_assistant|>
```

### 3.3 Assistant-Target Masking Strategy (Stage 2)
In Stage 2 records:
- Input: `<|system|>...<|user|>...<|assistant|><|tool_call|>...<|end_tool_call|><|tool_result|>...<|end_tool_result|>`
- Masking: All input tokens up to `<|end_tool_result|>` are masked with `weight = 0.0`.
- Target: Only the final `<|assistant|>` synthesis tokens are trained with `weight = 1.0`.

---

## 4. Train, Validation, and Test Split Isolation

To prevent subtle data contamination, the 1,000 records will be partitioned into three strictly isolated splits:

```
Total Curated Pool: 1,000 Records
├── Train Split:      800 Records (80.0%) - Used for optimizer gradient steps
├── Validation Split: 100 Records (10.0%) - Used for checkpoint selection & early stopping
└── Test Split:       100 Records (10.0%) - Used for pre-release benchmark verification
```

### 4.1 Strict Splitting Constraints:
1. **Cluster Isolation**: Entire semantic topics (e.g. all James Webb Space Telescope records) must reside entirely within either Train or Validation/Test. No cross-split topic leakage.
2. **Contrast-Dyad Integrity**: Both pairs of a contrastive dyad must stay in the same split. Splitting a pair between train and val would contaminate validation loss.

---

## 5. Decontamination & Leakage Prevention Protocol

Before any record is approved for the Experiment E corpus, it must pass a three-tier automated audit against:
1. The **150-Task Stage 5 Benchmark** (`benchmark.jsonl`).
2. The **Experiment C Training Dataset**.
3. The **Experiment D Training Dataset**.

### Audit Gates:
- **Gate 1: Exact String Deduplication**: Zero identical prompt strings or answer target substrings (case-insensitive normalized).
- **Gate 2: 10-Gram Token Shingling**: Jaccard similarity of 10-gram token sets between any proposed candidate and any benchmark task must be $< 0.15$.
- **Gate 3: Named Entity Overlap Filter**: If a candidate prompt shares 2 or more named entities with a benchmark prompt, it is automatically flagged for manual review and rewritten.

---

## 6. Synthetic vs. Manually Curated Proportions

To ensure both diversity of linguistic expression and uncompromising factual precision:
- **40% Manually Curated (400 records)**:
  - 150 hard conceptual & negative control prompts written directly by domain experts.
  - 100 conflicting evidence and insufficient evidence cases verified against live source pages.
  - 150 gold multi-step Web $\to$ Python math chains with exact numerical ground truths.
- **60% Synthetically Generated with Human Verification (600 records)**:
  - Generated using frontier models (Claude 3.5 Sonnet / Gemini 1.5 Pro) with strict JSON schema validators.
  - Every synthetic record must pass automated syntax validation, DuckDuckGo retrieval verification, and human editorial approval before insertion into the training manifest.
