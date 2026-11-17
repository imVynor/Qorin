# Forensic Root Cause Diagnosis: Live Inference Discrepancy & Isolation Failures

## Executive Summary

Following Stage 16B, where isolated unit tests passed, live website testing revealed three reproducible failure modes:
1. **Failure A (Wrong entity attribution)**: When asked *"Who is Jensen Huang?"*, the website returned a fabricated claim that he was a Japanese-born student at the University of Tokyo.
2. **Failure B (Cross-turn contamination)**: Asking *"Who is Donald Trump?"* immediately after the Jensen Huang query yielded text about the University of Tokyo.
3. **Failure C (Identity-style generation)**: Querying *"who is donald trump?"* in lowercase yielded *"You are donald trump. I am donald trump."* or an invalid Python tool call `m = donald trump(1, 2)`.

---

## 1. Trace Analysis & Forensic Discoveries

### 1.1 Discrepancy Between Benchmark Tests and Live Website Path
In the Stage 16B benchmark (`test_grounded_synthesis_benchmark.py`), test requests explicitly forced tools via:
`"manual_tool_override": "WEB"` and `"python_sandbox_enabled": False`.

On the live website:
- `manual_tool_override` is **never set** by the user interface.
- Both `web_research_enabled: true` and `python_sandbox_enabled: true` are enabled by default.
- The 500M parameter model (trained for only 100 pilot optimizer updates) does not autonomously emit `web_search` on every factual query. For `Who is Jensen Huang?`, the model emitted direct text without tools (`tools_executed == 0`).
- Because `tools_executed == 0`, `web_traces` was empty.
- In `backend/app.py`, the Grounding Validator was previously guarded by `if web_traces:`. Therefore, **the Grounding Validator was completely bypassed on the live website**, streaming the ungrounded pretraining hallucination straight to the user!

### 1.2 Cross-Turn Contamination Mechanism
In Turn 2 (`Who is Donald Trump?`), the multi-turn conversation payload sent by the frontend included:
```json
[
  {"role": "user", "content": "Who is Jensen Huang?"},
  {"role": "assistant", "content": "Jensen Huang is a Japanese-born student at the University of Tokyo..."},
  {"role": "user", "content": "Who is Donald Trump?"}
]
```
Because the model saw the previous assistant turn in its context window and did not invoke `web_search`, the model's self-attention latched onto the prior turn's topic, continuing to output facts about the University of Tokyo.

### 1.3 Case-Sensitivity and Residual Python Weights
When asked `Who is Donald Trump?` (capitalized), the model emitted `{"name": "web_research"}`.
When asked `who is donald trump?` (lowercase), the model's residual Experiment D Python fine-tuning weights activated, emitting:
`{"name": "python", "arguments": {"code": "m = donald trump(1, 2)\nprint(f'Must be a valid m: {m}')"}}`.
This crashed in the Python sandbox with `SyntaxError`, and the fallback emitted identity repetition (`You are donald trump. I am donald trump.`).

### 1.4 Dangling Pronoun Failure in Conservative Synthesis
In Stage 16B's `synthesize_conservative_grounded_answer()`, sentences were scored purely on query-term overlap. For Donald Trump, `SOURCE_2` (Robert Trump Wikipedia page) contained:
`"He was the younger brother of Donald Trump."`
Because this sentence matched "Donald" and "Trump", it was sorted into the answer after `SOURCE_1`, resulting in:
`"Donald John Trump is the 47th president... [SOURCE_1]. He was the younger brother of Donald Trump... [SOURCE_2]."`
The dangling pronoun `He` falsely attributed the younger brother relationship to Donald Trump himself.

---

## 2. Root Cause Summary Matrix

| Issue | Manifestation | Root Cause | Implemented Resolution |
| :--- | :--- | :--- | :--- |
| **Tool Non-Invocation** | `tools_executed: 0` on factual queries | Model did not emit tool call; validator only ran if `web_traces` existed | Policy routing in `ToolPolicyClassifier` detects `FACTUAL_INQUIRY` and triggers `web_search` |
| **Cross-Turn Contamination** | Donald Trump response mentions University of Tokyo | Model attention overwhelmed by prior turn text | Autonomous retrieval ensures Donald Trump evidence is loaded; validator rejects ungrounded prior-turn text |
| **Errant Python Dispatch** | `m = donald trump(1, 2)` | Residual Python weights activated on lowercase input | Policy layer intercepts Python calls for factual queries and redirects to `web_search` |
| **Dangling Pronouns** | "He was the younger brother of Donald Trump" | Naive snippet sentence concatenation | Pronoun filtering and single-entity primary passage selection |
| **Token-Only Validation** | Passing score on false relationships | Validator only checked global document word presence | Sentence-level co-occurrence alignment and relational predicate verification |
