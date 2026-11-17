# Code Changes Summary: Stage 16A.2 Python Execution Isolation

## 1. Summary of Changes

Six core modules across the backend and frontend were updated or created to enforce tool policy, result isolation, and unique provenance tracking.

---

## 2. Modified & Created Files

### 1. `backend/tools/policy.py` (New Module)
- Created `ToolPolicyClassifier` and `UserIntent` enum (`CODE_GENERATION`, `SIMPLE_ARITHMETIC`, `EXPLICIT_EXECUTION`, `GENERAL_QUERY`).
- Implemented linguistic regex patterns to distinguish code-writing tasks (including common misspellings like "rite a python function") from execution commands.
- Implemented `_safe_eval_math()` using AST to evaluate elementary arithmetic without invoking subprocesses.
- Implemented `synthesize_code_response()` providing clean markdown code blocks.

### 2. `backend/tools/registry.py`
- Added `request_id` and `conversation_id` to `CanonicalToolCall`.
- Added `execution_id`, `request_id`, `conversation_id`, `stdout`, `stderr` to `CanonicalToolObservation`.
- Updated `execute_tool()` to generate a unique `execution_id = f"exec_{uuid.uuid4().hex[:12]}"` per invocation and populate all fields.

### 3. `backend/tools/output_protocol.py`
- Updated `parse_generation()` to accept `request_id`, `conversation_id`, and a request-scoped `dispatched_calls` set.
- Ensured generated `call_id` includes `request_id` prefix (`call_<req_prefix>_001`) preventing cross-request ID collisions.

### 4. `backend/app.py`
- Imported `ToolPolicyClassifier` and `UserIntent`.
- Added `request_id` and `conversation_id` to `ChatRequest`.
- Updated `get_compact_tool_manifest()` to include strict tool-use policy instructions.
- Enforced tool policy in `chat_stream()`:
  - If `CODE_GENERATION`: Intercepts model tool calls and returns clean markdown code blocks.
  - If `SIMPLE_ARITHMETIC`: Evaluates and returns direct answer (e.g., `3`) without running tools.
  - If `EXPLICIT_EXECUTION`: Executes tool in isolated sandbox with unique `execution_id`.
- Replaced global `output_parser.reset_session()` with a per-request `request_dispatched_calls` set.

### 5. `web/src/types/harness.ts`
- Extended `ToolTrace` interface with `call_id`, `execution_id`, `request_id`, `conversation_id`, `stdout`, `stderr`.

### 6. `web/src/components/harness/ChatArea.tsx`
- Added persistent `conversationIdRef` tracking the conversation session.
- Passed `conversation_id` in chat request payload.
- Updated rendering keys to use `key={t.execution_id || ...}` to prevent React DOM reuse.
- Added visual execution ID badge `[exec_...]` in the tool header.

### 7. `tests/test_execution_isolation.py` (New Test Suite)
- Implemented 10 regression tests verifying execution isolation, sentinel containment, concurrency, and policy enforcement.
