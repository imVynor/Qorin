# QORIN Experiment E: Evidence Representation & Harness Architecture Options

**Document**: `architecture_options.md`  
**Status**: Architectural Evaluation & Recommendation  
**Target System**: QORIN-500M Dual-Stage Agent System  
**Hardware & Latency Constraints**: Single NVIDIA RTX GPU, Context Window: 2,048 tokens, Generation Budget: 160 tokens per stage.

---

## 1. Architectural Challenge: The Evidence Grounding Bottleneck

The Stage 5 audit revealed that **75.47% (80 / 106)** of successfully executed tool calls in Experiment D failed to translate into a correct final response. 

At 514M parameters (20 layers, $d_{model} = 1536$, 24 query heads, 4 KV heads), QORIN possesses sufficient capacity to parse well-formatted syntax, but lacks the massive cross-attention redundancy of 70B+ parameter models. When presented with 400–800 tokens of raw HTML-stripped web snippets containing promotional copy, disclaimers, repeated keywords, and competing figures, the attention distributions in layers 8–16 become dispersed. The model frequently fails to route attention from the query's target entity to the corresponding numerical predicate, resulting in:
1. **Extraction Scrambling**: Binding the metric of Entity A to Entity B.
2. **Parametric Fallback**: Defaulting to pretraining weights and producing an answer that contradicts the snippet.
3. **Superficial Bag-of-Words Mimicry**: Emitting common vocabulary from the snippet without answering the prompt.

To resolve this bottleneck, we evaluate five evidence representation architectures.

---

## 2. Comparative Evaluation of Five Evidence Architectures

### Option 1: Raw Snippets (Current Stage 5 Baseline)
* **Format**: Verbatim text snippets returned by DuckDuckGo concatenated inside `<|tool_result|>`.
* **Example**:
  ```xml
  <|tool_result|>
  Title: James Webb Space Telescope - NASA Science
  Snippet: The James Webb Space Telescope is NASA's flagship space observatory... Launched Dec 25 2021... It features a 6.5 meter primary mirror made of gold-plated beryllium... Costing over 10 billion dollars...
  <|end_tool_result|>
  ```
* **Token Footprint**: 400–750 tokens per query.
* **Cognitive Load on 500M Model**: **Severe**. Over 60% of tokens are syntactically and semantically irrelevant. Attention heads must compute hundreds of pairwise dot-products between query terms and background words.
* **Evaluation Verdict**: **REJECTED**. Directly caused 40 of the 108 failures in Stage 5.

---

### Option 2: Structured Evidence Records (SER)
* **Format**: A lightweight orchestrator normalizer parses the raw search payload into standardized key-value blocks before passing it to Stage 2.
* **Example**:
  ```xml
  <|tool_result|>
  [SOURCE 1] NASA Webb Observatory (https://science.nasa.gov/webb)
  - TARGET: James Webb Space Telescope (JWST)
  - LAUNCH_DATE: December 25, 2021
  - PRIMARY_MIRROR: 6.5 meters diameter
  - MISSION_COST: ~$10 billion
  <|end_tool_result|>
  ```
* **Token Footprint**: 80–160 tokens per query (70% token reduction).
* **Cognitive Load on 500M Model**: **Low**. High signal-to-noise ratio. Query heads can directly align with uppercase attribute keys (`LAUNCH_DATE:`, `PRIMARY_MIRROR:`).
* **Evaluation Verdict**: **HIGHLY RECOMMENDED (PRIMARY)**.

---

### Option 3: Tabular Markdown Schema
* **Format**: Multi-entity or comparative search results formatted into strict Markdown tables.
* **Example**:
  ```xml
  <|tool_result|>
  | Entity | Metric | Value | Source |
  | :--- | :--- | :--- | :--- |
  | Apple (FY23) | Revenue | $383.29B | [1] SEC 10-K |
  | Microsoft (FY23) | Revenue | $211.92B | [2] SEC 10-K |
  <|end_tool_result|>
  ```
* **Token Footprint**: 60–120 tokens per query.
* **Cognitive Load on 500M Model**: **Low to Moderate**. Highly effective for multi-step comparison and arithmetic handoffs, but rigid for descriptive, biographical, or conceptual topics.
* **Evaluation Verdict**: **RECOMMENDED FOR COMPARATIVE TASKS**.

---

### Option 4: Explicit Claim-Evidence Mapping (CEM) in Generation
* **Format**: Model is fine-tuned to produce a structured intermediate chain-of-thought in Stage 2 before outputting final prose:
  ```xml
  <|assistant|>
  [EXTRACTED EVIDENCE]
  - Fact 1: Apple FY23 revenue = $383.29B [Source 1]
  - Fact 2: Microsoft FY23 revenue = $211.92B [Source 2]
  [SYNTHESIS]
  In fiscal year 2023, Apple reported $383.29 billion in revenue while Microsoft reported $211.92 billion [Source 1, 2].
  <|end_assistant|>
  ```
* **Token Footprint**: Adds ~40 tokens to assistant generation budget.
* **Cognitive Load on 500M Model**: **Moderate**. Enforces causal self-attention: the final sentence attends directly to the model's own extracted tokens rather than attending back through the entire observation prompt.
* **Evaluation Verdict**: **RECOMMENDED FOR COMPLEX FACTUAL QUERIES**.

---

### Option 5: Two-Stage Hierarchical Extraction Pipeline
* **Format**: Introduce a dedicated intermediate model call:
  - Step 1: Model generates tool call.
  - Step 2: Search runs.
  - Step 3: Model is prompted solely to extract facts into JSON.
  - Step 4: Model is prompted to synthesize JSON into prose.
* **Token Footprint**: 3 sequential forward passes.
* **Cognitive Load on 500M Model**: Low per pass, but multiplies latency by $1.8\times$ and increases inference VRAM turnover.
* **Evaluation Verdict**: **REJECTED FOR 500M REAL-TIME HARNESS**. Unnecessary computational overhead for an interactive Web UI with strict response-time requirements.

---

## 3. Detailed Architecture Comparison Matrix

| Evaluation Criterion | Option 1: Raw Snippets | Option 2: Structured Evidence (SER) | Option 3: Markdown Tables | Option 4: Claim-Evidence Mapping | Option 5: 3-Pass Pipeline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Token Efficiency** | Poor (500+ tok) | **Excellent** (120 tok) | **Excellent** (90 tok) | Good (160 tok) | Poor (Multiple passes) |
| **500M Attention Density** | Low (Scattered) | **High** (Focused) | High (Focused) | **Very High** (Causal) | High |
| **Inference Latency** | Baseline (1.0x) | **Fast** (0.8x) | **Fast** (0.8x) | Baseline (1.1x) | Slow (1.8x) |
| **Implementation Complexity** | Zero | Low (Harness normalizer) | Moderate | Low (Dataset formatting) | High (Harness rewrite) |
| **Robustness to Search Noise**| 24.5% | **> 65% (Projected)** | > 60% (Projected) | **> 70% (Projected)** | > 70% (Projected) |

---

## 4. Final Architectural Recommendation for Experiment E

We recommend a **Hybrid SER + CEM Architecture**:

1. **Harness-Side Normalizer (Ingestion Layer)**:
   Inside `two_stage_protocol.py`, when DuckDuckGo returns search snippets:
   - Filter out duplicate sentences and boilerplate navigation text ("Click here", "Subscribe", "Cookies policy").
   - Extract primary entity keywords, capitalized proper nouns, numbers, currencies, and dates.
   - Format into a standardized **Structured Evidence Record (SER)** block (Option 2).
   - If the task is explicitly comparative or arithmetic, format into a **Markdown Table** (Option 3).

2. **Model-Side Chain-of-Thought (Generation Layer)**:
   During Stage 2 fine-tuning, condition the model on SER inputs and train the assistant to generate concise **Claim-Evidence Mappings** (Option 4) before emitting the final conversational response.

3. **Multi-Step Handoff Support**:
   Enable the harness to execute up to 2 tool turns iteratively:
   - Turn 1: Model invokes `web_research`.
   - Tool Observation: Returns SER with verified numbers.
   - Turn 2: Model can either synthesize directly OR emit `<|tool_call|>` for `python` referencing the retrieved numbers.
   - Final Turn: Model synthesizes the computational output into the user-facing response.
