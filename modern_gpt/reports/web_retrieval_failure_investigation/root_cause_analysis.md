# Root Cause Analysis: Neural Intent Triggering & Multi-Tool Imbalance in QORIN-500M

**Report ID**: `RCA-20261117-WEB-RETRIEVAL`  
**Target Query**: `grab info on elon musk`  
**Model Investigated**: `QORIN-500M SFT Experiment C`  
**Status**: Resolved / Fully Accounted For  

---

## 1. Core Problem Statement

When presented with colloquial information retrieval queries such as `grab info on elon musk` under active Web Retrieval and Python Sandbox capabilities, QORIN-500M consistently emits direct conversational text rather than a structured `<|tool_call|>` intent. Because the model has a parameter count of 514M and limited pretraining memorization, direct generation produces plausible-sounding but completely fabricated hallucinations (e.g., claiming Elon Musk is a biological statistical model with 400,000 neurons).

This investigation performed a deep forensic audit of the SFT training data, prompt formatting, token generation distributions, and protocol parsing to determine the fundamental root causes.

---

## 2. Root Cause 1: Severe Training Distribution Imbalance (28.7:1 Ratio)

An exhaustive audit of the 3,962 supervised fine-tuning examples in `modern_gpt/experiments/phase2_experiment_c/stage_1_data/train.jsonl` revealed an extreme category imbalance:

```
Total Training Examples: 3,962
├── Python Execution Tool Calls: 2,183 (55.10%)
├── Direct Conversational Answers: 1,703 (42.98%)
└── Web Research Tool Calls:        76 ( 1.92%)
```

- **Ratio of Python to Web Research**: **28.7 to 1**.
- **Impact**: In a 514M parameter model, token prediction probabilities are heavily shaped by training frequency. When both tools are advertised in the system prompt (`two_tool_manifest`), the neural attention weights are overwhelmingly biased toward predicting `"name": "python"` rather than `"name": "web_research"`.

In direct testing of the model with both tools enabled on an explicit search prompt (`Search the web for current, sourced information about Elon Musk.`):
```json
<|tool_call|>{"name": "python", "arguments": {"code": "import math\nans = math.comb(elon Musk, elon Musk)\nprint('Combinations:', ans)"}}
```
The model was so strongly conditioned toward Python that it generated invalid Python code attempting to compute mathematical combinations of string tokens rather than calling Web Research.

---

## 3. Root Cause 2: Extreme Prompt Template Rigidity in SFT Data

Further inspection of the 76 `web_research` training examples revealed that they were synthesized using only **five rigid, formulaic query templates**, repeated cyclically:

1. `Search the web for: Nobel Prize in Physics 2024 winners.`
2. `Search the web for: Nobel Prize in Chemistry 2024 AlphaFold.`
3. `Search the web for: Attention Is All You Need publication date.`
4. `Search the web for: James Webb Space Telescope first deep field image date.`
5. `Search the web for: Python 3.12 release date.`

**Key Finding**:
- **100% (76/76)** of the web research training examples began with the exact literal prefix:
  ```
  Search the web for: <query>
  ```
- **0% (0/76)** of the training examples contained colloquial phrasing, such as:
  - `"grab info on ..."`
  - `"find out about ..."`
  - `"tell me about ..."`
  - `"who is ..."`
  - `"look up ..."`

Consequently, the model's neural weights learned to treat `<|tool_call|>{"name": "web_research"}` as a conditional completion of the literal token sequence `Search the web for: `. When presented with `grab info on elon musk`, the model's conditional probability for `<|tool_call|>` failed to exceed the threshold for generation, falling back to direct autoregressive continuation.

---

## 4. Root Cause 3: SLM Schema Drift and Brittle Parser Fallback

In small language models (~500M parameters), fine-tuned tool calling often suffers from "topic drift," where the model outputs the topic or target entity as the `name` field:
```json
<|tool_call|>{"name": "New York State University", "arguments": {"query": "Institute of Physics..."}}
```
or
```json
<|tool_call|>{"name": "NASA Artemis Program", "arguments": {"query": "Profile: Artemis program"}}
```

Prior to our defensive remediation:
1. `backend/tools/two_stage_protocol.py` evaluated `tool_candidate in ["web_research", "web_search", "search"]`.
2. Because `"New York State University"` or `"NASA Artemis Program"` was not in this hardcoded list, the protocol classified the action as `DIRECT`.
3. It completely ignored the presence of valid search arguments (`{"query": "..."}`).
4. The raw tool call JSON string was streamed to the user as conversational text, creating another apparent retrieval failure.

Similarly, if the model emitted single quotes (`{'name': 'web_research', ...}`) or wrapped its tool call in markdown fences (````json ... ````), `json.loads` threw a decode error and returned `None`, which previously caused the tool call to be dropped to `DIRECT`.

---

## 5. Root Cause 4: Missing Fallback for User Queries on Manual Override

When users in the UI or evaluation harness explicitly set `manual_tool_override="WEB"`, `two_stage_protocol.py` routed to `WEB`, but `validate_and_sanitize_arguments` attempted to extract `"query"` exclusively from the model's first-round generated text.

If the model had generated conversational text (as in `grab info on elon musk`), `query` was missing, and the validator rejected the tool call with:
```
Web research payload missing non-empty 'query' argument.
```
The application lacked an architectural fallback to the user's input prompt (`user_query=last_user_msg`), preventing manual override from rescuing unconditioned queries.

---

## 6. Summary Root Cause Chain

```
User types "grab info on elon musk"
         │
         ▼
Model checks weights: SFT data had 28.7:1 Python vs Web imbalance,
and 100% of Web training prompts used literal prefix "Search the web for: "
         │
         ▼
Model fails to predict <|tool_call|> (Stage A Intent Failure)
         │
         ▼
Protocol classifies response as DIRECT
         │
         ▼
Orchestrator skips WebResearcher dispatch
         │
         ▼
Direct hallucinated tokens streamed to client without citations
```
