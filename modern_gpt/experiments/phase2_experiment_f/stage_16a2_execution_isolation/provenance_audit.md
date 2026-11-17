# Backend and Frontend Provenance Audit: Stage 16A.2

## 1. Audit Scope

To permanently resolve cross-request contamination and stale React state rendering, both backend orchestration and frontend consumption pathways were audited from token generation to DOM paint.

---

## 2. Frontend State Machine & Lifecycle Audit

File: `web/src/components/harness/ChatArea.tsx`

### A. Conversation ID Lifecycle
- **Before:** No `conversation_id` existed. When the user sent repeated requests, the backend had no session provenance.
- **After:** Initialized `conversationIdRef = useRef<string>(`conv-${Date.now()}`)`.
  - Persists across consecutive user messages within a chat session.
  - Automatically reset on clicking the "Clear" button (`handleClear()`).
  - Passed in every SSE stream request payload (`req.conversation_id`).

### B. React Key Invalidation & DOM Reconciliation
- **Before:**
  ```tsx
  {m.tool_traces && m.tool_traces.map((t, tIdx) => (
    <div key={tIdx} className={styles.inlineCard}>
  ```
  Using array index `key={tIdx}` allowed React's fiber reconciler to reuse stale DOM nodes when new messages arrived.
- **After:**
  ```tsx
  {m.tool_traces && m.tool_traces.map((t, tIdx) => (
    <div key={t.execution_id || `${m.id}-tool-${tIdx}`} className={styles.inlineCard}>
  ```
  Guarantees 1:1 binding between the rendered tool card and the physical execution instance.

### C. Execution Provenance UI Badge
Added visual verification of the execution ID directly in the tool card header:
```tsx
<span>Tool Invocation: {t.tool_name} ({t.status}) {t.execution_id ? `[${t.execution_id.slice(-6)}]` : ''}</span>
```
Allows researchers to instantly visually confirm whether an execution is fresh or reused.

---

## 3. Backend SSE Stream Event Pipeline

File: `backend/app.py`

### SSE Event Sequencing
1. `retrieval`: Returns RAG knowledge chunk evaluation.
2. `tool_start`: Emitted immediately upon schema validation, exposing `tool_name`, `call_id`, `request_id`, `conversation_id`.
3. `tool_result`: Emitted upon execution completion, exposing `execution_id`, `status`, `stdout`, `stderr`, `output_payload`.
4. `token`: Streams model synthesis tokens (cleaned of any internal protocol markers).
5. `done`: Emits final summary with `trace_id`, `request_id`, `conversation_id`, and `tools_executed`.

Zero fallback to previous execution results is permitted. If an execution fails, `status="failed"` is explicitly recorded with its error traceback.
