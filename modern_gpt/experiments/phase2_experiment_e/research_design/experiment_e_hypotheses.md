# QORIN Experiment E: Scientific Hypotheses & Intervention Formulations

**Document**: `experiment_e_hypotheses.md`  
**Status**: Pre-Experimental Theoretical Specification  
**Model Scale**: QORIN-500M (20 layers, $d_{model}=1536$, 24 query / 4 KV heads, 514M params)  
**Parent Checkpoint**: Frozen Experiment D Best Checkpoint (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388da...`)

---

## 1. Executive Problem Statement

Stage 5 demonstrated that fine-tuning QORIN-500M on tool-use activation successfully unlocked Web retrieval capability (moving Web recall from 0% to 80.26%). However, this activation created two severe behavioral pathologies:

1. **Pathology 1 (Calibration Collapse)**: The model developed an indiscriminate bias toward external retrieval, invoking `web_research` on 50.00% of direct conceptual queries that it previously answered correctly from parametric memory.
2. **Pathology 2 (Synthesis Decoupling)**: When web evidence was retrieved, the model achieved end-to-end semantic success on only 24.53% of executed calls (26/106). The model treated the retrieved raw text blob either as an ignored context background (falling back to parametric hallucinations) or failed to extract the precise entity/numerical values embedded in dense text.

Experiment E is formulated around two distinct, independently testable interventions designed to resolve these two pathologies.

---

## 2. Formal Hypotheses

### Hypothesis 1: Contrastive Decision-Boundary Calibration (H1)
*Providing paired contrastive training examples (where a conceptual question is paired with a structurally similar temporal/empirical question) combined with explicit negative-control loss masking will reduce false-positive tool invocations on direct questions to $\le 15\%$ without degrading Web recall below $80\%$ or Python recall below $90\%$.*

#### Theoretical Mechanism:
In Experiment D, the loss function rewarded tool token generation (`<|tool_call|>`) whenever colloquial triggers ("find info on", "search for") appeared. Because the training corpus had high tool-use density (over 70% tool-bearing records), the prior probability $P(\text{tool} \mid \text{user\_query})$ shifted drastically upward, overpowering the internal attention heads responsible for recognizing self-contained conceptual queries. 

By introducing contrastive pairs (e.g. "Explain backpropagation" vs "What was the date of Rumelhart's 1986 backpropagation paper?"), the model's early multi-layer perceptron (MLP) layers learn a sharp decision hyperplane separating parametric definitions from external empirical needs.

#### Falsification Criteria for H1:
H1 is falsified if:
- Unnecessary tool calls remain above 25% on held-out direct conceptual tasks; OR
- Web recall drops below 75%; OR
- Python recall drops below 85%.

---

### Hypothesis 2: Structured Evidence Schema vs. Raw Snippet Extraction (H2)
*Transforming raw search engine snippets into a standardized, semi-structured Evidence Schema (comprising Source ID, Claim Summary, Fact Entities, and Extracted Numerical Quantities) prior to Stage 2 conditioning will increase semantic evidence extraction accuracy from $24.53\%$ to $\ge 50.0\%$ in a 500M parameter model, compared to feeding raw snippet text.*

#### Theoretical Mechanism:
A 500M parameter transformer with 20 layers has limited cross-attention capacity over long, noisy text windows. When DuckDuckGo returns three raw paragraphs (300–600 tokens) filled with navigational boilerplate, disclaimers, and conflicting numbers, the self-attention weights in the lower 10 layers disperse across irrelevant tokens. 

A structured evidence representation concentrates the relevant factual tokens into an explicit key-value syntax:
```xml
<|tool_result|>
[SOURCE 1]: https://...
- ENTITY: James Webb Space Telescope
- METRIC: Primary Mirror Diameter = 6.5 meters
- DATE: Launched December 25, 2021
<|end_tool_result|>
```
This reduces the attention routing distance from $O(L_{snippet})$ to $O(L_{fact})$, allowing the 24 query heads to directly align generation tokens with verified entity values.

#### Falsification Criteria for H2:
H2 is falsified if:
- End-to-end semantic correctness on executed web tasks does not improve by at least +15 percentage points over the Experiment D baseline (i.e. fails to exceed 40.0%); OR
- Hallucination rate on numerical extraction remains above 35%.

---

### Hypothesis 3: Orthogonality & Factorial Independence (H3)
*Tool-decision calibration (Intervention 1) and evidence-grounded synthesis (Intervention 2) operate on disjoint functional pathways within the model and orchestrator. Consequently, a combined model (Branch D) will achieve additive gains in overall benchmark semantic success without negative cross-task interference.*

#### Theoretical Mechanism:
- **Decision Calibration** acts purely during **Stage 1 generation** (outputting `<|tool_call|>` vs `<|direct_response|>`), governed by early token logit biases and prompt classification attention.
- **Evidence Synthesis** acts purely during **Stage 2 generation** (conditioning on `<|tool_result|>` to synthesize `<|assistant|>`), governed by in-context cross-referencing and factual fidelity.
Because these two stages occur under disjoint prompt prefixes and target token masks, gradient updates during fine-tuning will update complementary parameter subspaces if multi-task loss is properly balanced.

#### Falsification Criteria for H3:
H3 is falsified if:
- The combined branch (Branch D) exhibits destructive interference (e.g. achieves lower semantic success than either Branch B or Branch C alone).

---

## 3. Intervention 1: Tool-Decision Calibration Design

### 3.1 Four-Way Action Space
The model must reliably classify queries into one of four distinct action trajectories:
1. `DIRECT`: Query is conceptual, definitional, mathematical derivation, or answerable from parametric knowledge.
2. `WEB`: Query requires current factual data, live statistics, recent post-cutoff events, or unknown entities.
3. `PYTHON`: Query requires exact arithmetic, algebraic solving, string manipulation, or deterministic simulation.
4. `WEB -> PYTHON`: Query requires empirical retrieval followed by mathematical computation or statistical analysis.

### 3.2 Contrast-Pair Construction Methodology
Every training cluster will contain explicit contrastive dyads and triads:

```
[CLUSTER: Physics & Space]
- Direct:   "Explain why objects in orbit experience apparent weightlessness." -> DIRECT
- Web:      "What is the current altitude of the International Space Station right now?" -> WEB
- Python:   "Calculate the orbital period of a satellite at an altitude of 400 km around Earth." -> PYTHON
- Multistep:"Look up the dry mass of Falcon 9 and Starship, then compute their ratio." -> WEB -> PYTHON
```

### 3.3 Loss Masking Strategy for Calibration
During Stage 1 training:
- For `DIRECT` records: Assistant tokens are masked normally; loss is computed on the introductory reasoning tokens to reinforce immediate, direct answers without tool tokens.
- For `TOOL` records: Loss is computed strictly on `<|tool_call|>` and the exact JSON argument dictionary.
- Ratio of Direct-to-Tool records in the calibration corpus: **1.0 : 1.0** (balanced 50% direct, 50% tool).

---

## 4. Intervention 2: Evidence-Grounded Synthesis Design

### 4.1 Comparison of Evidence Representations

| Representation Style | Description | Cognitive Burden on 500M Model | Feasibility with Existing Harness | Recommendation |
| :--- | :--- | :---: | :---: | :--- |
| **Option 1: Raw Snippets** | Concatenate top-3 search snippets verbatim inside `<|tool_result|>`. | **High**: Noise, ads, and irrelevant numbers drown out facts. | Trivial | **Rejected** (Root cause of Stage 5 failures). |
| **Option 2: Structured Evidence Schema** | Normalizer extracts Title, Source URL, Key Entities, and Numeric Values into standard key-value blocks. | **Low**: High signal-to-noise ratio; easily parsed by 24 attention heads. | Straightforward regex/LLM normalizer in orchestrator. | **Recommended Primary**. |
| **Option 3: Evidence Tables** | Tabular markdown representation of metrics across entities. | **Medium**: Excellent for multi-entity comparison, but rigid for prose topics. | Moderate | **Recommended for comparison tasks**. |
| **Option 4: Claim-Evidence Mappings** | Intermediate model reasoning step outputting `Claim: [Fact] (Source: [ID])` before final prose. | **Medium**: Teaches explicit attribution, highly effective for chain-of-thought. | High (Requires Stage 2 CoT prompting). | **Recommended for Stage 2 target format**. |

### 4.2 Two-Stage Prompting Contract for Stage 2
When tool execution returns, the orchestrator constructs the Stage 2 input prompt:

```xml
<|system|>
You are QORIN, an evidence-grounded AI assistant. Base your answer strictly on the verified facts in the tool observation. If the retrieved evidence does not contain the answer, state that the information is unavailable. Cite sources using [Source X].
<|user|>
What is the population and land area of Iceland?
<|assistant|>
<|tool_call|>
{"name": "web_research", "arguments": {"query": "Iceland population and land area"}}
<|end_tool_call|>
<|tool_result|>
[SOURCE 1]: https://statice.is
- ENTITY: Iceland
- POPULATION: 393,600 (as of early 2024)
- LAND AREA: 103,000 square kilometers
<|end_tool_result|>
<|assistant|>
According to Statistics Iceland [SOURCE 1], Iceland has an estimated population of approximately 393,600 people and covers a total land area of 103,000 square kilometers.
```

### 4.3 Handling Conflicting & Insufficient Evidence
The training dataset for Intervention 2 must explicitly feature:
1. **Conflicting Sources**: Source A claims \$10.5B, Source B claims \$11.2B. The model must acknowledge the discrepancy ("Estimates vary between \$10.5B [SOURCE 1] and \$11.2B [SOURCE 2]").
2. **Insufficient Evidence**: Search returns tangential information without the requested answer. The model must declare uncertainty rather than hallucinating parametric approximations ("The search results do not specify the exact figure for 2024").
3. **Evidence Extraction vs. Calculation**: When numbers are retrieved, calculations must be performed strictly using the retrieved figures, not recalled figures.

---

## 5. Summary of Experimental Branches

- **Branch A**: Experiment D Step 425 Checkpoint (Frozen Reference Baseline).
- **Branch B (Calibration Only)**: Fine-tuned strictly on balanced contrastive decision pairs (Stage 1 only).
- **Branch C (Synthesis Only)**: Fine-tuned strictly on structured evidence extraction and attribution (Stage 2 only).
- **Branch D (Combined E)**: Fine-tuned on the joint calibration and structured synthesis curriculum.
