# Failure Localization and Stage Classification Report

**Incident ID**: `INC-20261117-WEB-RETRIEVAL-FAIL`  
**Query Investigated**: `grab info on elon musk`  
**Evaluator**: Antigravity Autonomous Systems Engineering Team  

---

## 1. Primary Classification

Based on runtime logs, raw model token streams, and protocol execution traces, this incident is categorized as:

> ### **Category A — Model selected DIRECT despite Web being enabled**

### Secondary Compounding Defects
While **Category A** was the direct cause of the ungrounded response to `"grab info on elon musk"`, our forensic trace also revealed three critical secondary vulnerabilities in the harness implementation:
1. **Category B/C Vulnerability (Tool Name Schema Drift)**: When the model *did* attempt web retrieval in other tests (e.g. `"Search the web for: Nobel Prize in Physics 2024 winners."`), it emitted `<|tool_call|>{"name": "New York State University", "arguments": {"query": "..."}}`. The protocol parser rejected it as `DIRECT` because `"New York State University"` was not in the hardcoded tool name list, streaming the raw tool call token string to the user.
2. **Category B Vulnerability (Missing Argument Fallback)**: When `manual_override="WEB"` was selected or when arguments lacked `"query"`, `validate_and_sanitize_arguments` failed to fall back to the user prompt.
3. **Severe Training Bias (Python Selection Bias)**: In the training set (`train.jsonl`), Python examples outnumber Web Research examples 28.7 to 1 (2,183 Python vs 76 Web). When both tools are enabled, the model exhibits a severe bias toward calling `python` even for web research questions.

---

## 2. Stage-by-Stage Forensic Breakdown

### Stage A: Action Determination (`backend/tools/two_stage_protocol.py`)
- **Code Reference**: [two_stage_protocol.py#L30-L38](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/two_stage_protocol.py#L30-L38)
```python
is_tool_call = ("<tool_call>" in first_round_text or "<|tool_call|>" in first_round_text)
if not is_tool_call:
    return "DIRECT", None, None
```
- **Evaluation**: The raw generated text for `"grab info on elon musk"` was:
  ```
  elon musk is a language model research assistant developed by the QORIN research project...
  ```
  Because neither `<|tool_call|>` nor `<tool_call>` appeared anywhere in the text, `is_tool_call` was `False`.
- **Outcome**: The function cleanly and deterministically returned `("DIRECT", None, None)`.

---

### Stage B: Controlled Dispatch Logic (`backend/app.py`)
- **Code Reference**: [app.py#L265-L330](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/app.py#L265-L330)
```python
if action in ["PYTHON", "WEB"] and current_round < max_tool_rounds and tname:
    # Tool execution branch
    ...
else:
    # Direct answer branch
    final_response_text = round_response_text
    for c in round_chunks:
        yield {"event": "token", "data": ...}
    break
```
- **Evaluation**: Because `action == "DIRECT"`, the orchestrator dispatch branch was never entered.
- **Outcome**: Tool execution was skipped. The application did not attempt to call DuckDuckGo or Wikipedia, and did not emit any `tool_start` or `tool_result` SSE events.

---

### Stage C: Frontend Presentation (`web/src/components/harness/ChatArea.tsx`)
- **Code Reference**: [ChatArea.tsx#L225-L260](file:///c:/Users/Anish/Desktop/Projects/ML-course/web/src/components/harness/ChatArea.tsx#L225-L260)
```tsx
{/* Optional Inline Tool Traces */}
{m.tool_traces && m.tool_traces.map((t, tIdx) => (
  <div key={tIdx} className={styles.inlineCard}>
    ...
  </div>
))}
<div className={m.role === 'user' ? styles.userBubble : styles.assistantBubble}>
  {m.content}
</div>
```
- **Evaluation**: The frontend correctly renders whatever SSE events it receives. Because no tool events arrived, `currentTools` remained `[]`.
- **Outcome**: The UI rendered the model's direct hallucinated response in an assistant bubble with no tool trace cards, no knowledge pills, and no citations. The frontend behaved as designed; the failure occurred upstream in Stage A intent recognition.

---

## 3. Localization Matrix of Audited Failure Stages

| Failure Stage Code | Category Description | Verified In Incident? | Evidence / Code Locus |
|---|---|---|---|
| **A** | **Model selected DIRECT despite Web enabled** | **YES (ROOT CAUSE)** | Model emitted direct tokens; `<|tool_call|>` was never predicted by neural weights. |
| **B** | Model selected WEB but generated malformed arguments | **PARTIAL (Secondary)** | Model often emits topic name as `name` or single-quoted JSON. Fixed defensively. |
| **C** | Protocol parser failed to recognize valid action | **PARTIAL (Secondary)** | Parser previously dropped topic-named tool calls to DIRECT. Fixed defensively. |
| **D** | Orchestrator rejected or silently skipped dispatch | **NO** | Dispatch was bypassed because Stage A reported DIRECT. |
| **E** | Web researcher returned zero useful results | **NO** | Component diagnostic proved live DDG/Wikipedia returns 3 verified sources for Elon Musk. |
| **F** | Web results retrieved but not passed to model | **NO** | When tool is executed, payload is injected via `<|tool_result|>`. |
| **G** | Web results reached model but ignored | **NO** | In closed-loop test, model grounded synthesis on retrieved Wikipedia snippets. |
| **H** | Frontend hid tool trace or citations | **NO** | `ChatArea.tsx` renders all received `tool_traces`. No traces were emitted by backend. |
