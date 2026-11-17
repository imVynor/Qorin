# QORIN Stage 16A — Root-Cause Forensic Report: Live Website Inference & Tool-Dispatch Failures

**Date:** 2026-11-17  
**Investigator:** Senior Inference Architect & Systems QA  
**Scope:** Live Website Inference Pipeline, Backend Tool Dispatch, Prompt Construction, RAG Engine, and Frontend Streaming Display  

---

## 1. Executive Summary

A forensic audit of the live QORIN website inference pipeline identified the precise root causes behind the two observed failures:
1. **Test A ("grab info on the weather in ahmedabad"):** Model produced a general overview, emitted a non-dispatched Python call with `math.comb(3, 5)`, and exposed the raw `<code_string>` token to the user.
2. **Test B ("who is donald trump?"):** RAG displayed "Retrieved 0 Knowledge Chunks", model emitted `{"name": "web_research", "arguments": {"query": "Who is donald trump?"}}`, but the call was not dispatched and the raw JSON was exposed in the user chat bubble.

Neither failure is solely caused by the neural model weights. Both are caused by systemic disconnects between prompt engineering, tool manifests, regex-based action detection, error propagation, and unbuffered SSE token streaming.

---

## 2. Failure Case A: Weather in Ahmedabad & `<code_string>` Exposure

### 2.1 Observed Behavior
- **User Query:** `"grab info on the weather in ahmedabad"`
- **Observed Symptoms:**
  - Model failed to retrieve real-time weather.
  - Model emitted an irrelevant Python calculation: `math.comb(3, 5)`.
  - Raw `<code_string>` protocol marker was rendered directly in the user-visible UI.

### 2.2 Root-Cause Trace
1. **Absence of Weather Tool in Manifest:**
   In [`backend/app.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/app.py) (`get_compact_tool_manifest`), only two tools were defined: `python` and `web_research`. No weather tool existed in the registry or manifest.
2. **Harmful Prompt Injection of `<code_string>`:**
   Lines 166 and 174 of `backend/app.py` defined the Python tool format as:
   ```text
   Format: <|tool_call|>{"name": "python", "arguments": {"code": "<code_string>"}}
   ```
   The model literally learned the token `<code_string>` from the system prompt as an expected delimiter rather than a placeholder.
3. **Execution Failure in `math.comb(3, 5)`:**
   The model attempted to calculate combinations where $k = 5 > n = 3$. In standard Python:
   ```python
   math.comb(3, 5) # Raises ValueError: k must be non-negative and <= n
   ```
4. **Action Determination Bypass & Token Leaking:**
   Because `<code_string>` was prepended before the tool call or the closing tags were mismatched, `two_stage_protocol.determine_action` failed to recognize the intent as a valid tool call. It defaulted to `"DIRECT"`, streaming the entire raw string—including `<code_string>` and unexecuted Python code—directly to the browser via SSE `event: token`.

---

## 3. Failure Case B: Donald Trump & Exposed JSON Tool-Call

### 3.1 Observed Behavior
- **User Query:** `"who is donald trump?"`
- **Observed Symptoms:**
  - UI displayed `Retrieved 0 Knowledge Chunks`.
  - Model emitted `{"name": "web_research", "arguments": {"query": "Who is donald trump?"}}`.
  - Tool call was not executed; raw JSON was displayed in the assistant chat bubble.

### 3.2 Root-Cause Trace
1. **False-Positive Keyword Trigger in RAG:**
   In [`backend/retrieval/engine.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/retrieval/engine.py), `FACTUAL_KEYWORDS` contained broad interrogative words including `"who"`, `"what"`, `"when"`, `"where"`.
   The query `"who is donald trump?"` triggered the internal technical documentation indexer. Because Donald Trump is not in the QORIN GPU hardware/architecture documentation, the search returned 0 chunks.
2. **Misleading Frontend RAG Banner:**
   In [`web/src/components/harness/ChatArea.tsx`](file:///c:/Users/Anish/Desktop/Projects/ML-course/web/src/components/harness/ChatArea.tsx), the inline retrieval card checked `if (m.retrieval_trace)`. Even when `retrieved_chunks.length === 0`, it rendered:
   ```text
   Retrieved 0 Knowledge Chunks
   ```
   creating the impression of an aborted retrieval pipeline.
3. **Strict Tag Requirement in `two_stage_protocol.py`:**
   In [`backend/tools/two_stage_protocol.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/two_stage_protocol.py):
   ```python
   is_tool_call = ("<tool_call" in first_round_text or "<|tool_call|>" in first_round_text)
   ```
   When the model emitted bare JSON `{"name": "web_research", "arguments": {"query": "Who is donald trump?"}}` without explicit `<|tool_call|>` tags, `is_tool_call` evaluated to `False`.
4. **Direct SSE Stream Leakage:**
   `determine_action()` returned `("DIRECT", None, None)`. The backend loop assumed this was a conversational answer and streamed all tokens straight to the frontend.

---

## 4. Pipeline Defect Matrix

| Pipeline Stage | Defect Identified | Required Architectural Remediation |
|---|---|---|
| **System Prompt** | Hardcoded `<code_string>` and `<query_string>` placeholder tokens in `get_compact_tool_manifest` taught the model to hallucinate them. | Replace with canonical syntax `<tool_call id="{id}">{"name": "...", "arguments": {...}}</tool_call>` without pseudo-tag placeholders. |
| **Tool Registry** | Missing `weather` tool; inconsistent naming (`web_search` vs `web_research`, `python` vs `python_interpreter`). | Implement canonical `ToolRegistry` with canonical names (`web_search`, `python`, `weather`) and strict alias resolution. |
| **Weather Integration** | No real-time weather provider existed. | Implement `RealWeatherProvider` using Open-Meteo geocoding + current forecast API (temperature, apparent temp, humidity, wind, condition, timestamp). |
| **Tool Parser** | Did not support bare JSON or attribute tags (`<tool_call id="...">`). Regex failed on delimiter variations. | Implement robust multi-format parser with `strict=False` JSON parsing and marker stripping. |
| **Execution State Machine** | Ad-hoc while-loop in `backend/app.py` did not leverage the formal 9-state machine with observation loop. | Standardize inference through the bounded multi-turn state machine. |
| **RAG Fallback** | Over-broad keyword matching ("who") caused 0-chunk technical search and empty banner. | Narrow RAG keywords to technical terms, suppress 0-chunk UI banners, and allow fallback to web search. |
| **Streaming Output** | Direct SSE token emission of unclassified first-turn generations leaked tool calls to UI. | Buffer and classify generation before emitting tokens to client. |
