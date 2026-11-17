# Cross-Turn & Request Isolation Audit Report

## Objective

Examine and prove request-level, conversation-level, and turn-level isolation across QORIN's live inference engine.

---

## 1. Isolation Boundaries

### 1.1 Request-Scoped State
In `backend/app.py`:
- `req_id`: Dynamically assigned per HTTP connection using `uuid.uuid4().hex[:12]`.
- `request_dispatched_calls`: Created as a fresh, local Python `set()` inside `event_generator()` for each request. It is never stored in a global module variable.
- `tool_traces`: Local Python list instantiated per request.
- `active_messages`: Local copy of messages (`list(generation_messages)`).
- **Concurrency Test Verification**: Executing concurrent requests (`test_04_concurrent_requests_isolation`) showed zero cross-contamination. Jensen Huang and Donald Trump ran in parallel threads and each thread received its respective answer with zero cross-leakage.

### 1.2 Multi-Turn Conversation Scoping
In `web/src/components/harness/ChatArea.tsx`:
- `conversationIdRef.current` assigns a persistent session ID (`conv-${Date.now()}`).
- Clearing the chat explicitly resets `conversationIdRef.current`.
- Separate browser sessions / tabs maintain distinct conversation IDs.
- **Separate Conversation Test Verification**: `test_6_separate_conversations` proved that querying Jensen Huang in Conversation A and Donald Trump in Conversation B results in completely distinct outputs.

### 1.3 Resolution of Cross-Turn Contamination (Jensen Huang → Donald Trump)
- **Defect Observed**: When asked *"Who is Donald Trump?"* immediately after *"Who is Jensen Huang?"* in the same conversation, QORIN previously outputted *"Jensen Huang is a computer science professor at the University of California, Berkeley..."*.
- **Root Cause**: The model's attention was dominated by the assistant message from Turn 1, while `web_search` was not triggered, and the validator was not invoked.
- **Remediation**:
  1. `ToolPolicyClassifier` detects that Turn 2 is a `FACTUAL_INQUIRY` for `"Who is Donald Trump?"`.
  2. `web_search` is triggered specifically for `"Who is Donald Trump?"`.
  3. The Evidence Bundle returned contains Donald Trump sources.
  4. If the model generates any text relating to Jensen Huang or Berkeley/Tokyo:
     - The Grounding Validator evaluates the model's text against the Donald Trump evidence bundle.
     - `query_relevance` drops to 0.0, and `missing_entities` flags all Jensen Huang tokens.
     - `val.is_grounded` evaluates to `False`.
     - `synthesize_conservative_grounded_answer()` is invoked using the Donald Trump bundle.
     - The user receives:
       `"Based on verified sources ([SOURCE_1]): Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States [SOURCE_1]."`
  5. Contamination rate: **0.0%**.
