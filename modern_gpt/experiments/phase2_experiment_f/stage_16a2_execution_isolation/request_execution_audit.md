# Request and Execution ID Audit: Stage 16A.2

## 1. Audit Overview

This audit verifies that every tool call, subprocess run, and synthesis turn in QORIN operates with end-to-end provenance and unique identity tracking across request, conversation, and execution boundaries.

---

## 2. Identity Hierarchy & Field Specifications

```
Session / Client Browser
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Conversation ID: `conv_<hex12>`                       │
│  (Persists across turns in a conversation session)     │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Request ID: `req_<hex12>`                             │
│  (Generated uniquely per HTTP SSE chat request)        │
└────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Tool Call ID:                 │ │ Tool Call ID:                 │
│ `call_<req_prefix>_001`       │ │ `call_<req_prefix>_002`       │
└───────────────────────────────┘ └───────────────────────────────┘
       │                                 │
       ▼                                 ▼
┌───────────────────────────────┐ ┌───────────────────────────────┐
│ Execution ID:                 │ │ Execution ID:                 │
│ `exec_<hex12>`                │ │ `exec_<hex12>`                │
│ (Assigned upon tool dispatch) │ │ (Assigned upon tool dispatch) │
└───────────────────────────────┘ └───────────────────────────────┘
```

---

## 3. Mandatory Metadata Verification

Every `CanonicalToolObservation` and SSE `tool_result` event now carries the complete 10-point metadata tuple:

1. `request_id`: e.g. `req_28983bceebad`
2. `conversation_id`: e.g. `conv_aea14cb683c1`
3. `call_id`: e.g. `call_req_2898_001`
4. `execution_id`: e.g. `exec_5530ba2e09ce`
5. `tool_name`: `python` | `web_search` | `weather`
6. `status`: `completed` | `failed` | `timeout`
7. `stdout`: Exact raw standard output from the isolated subprocess.
8. `stderr`: Exact raw standard error or traceback.
9. `output_payload`: Structured JSON dictionary containing all execution metrics.
10. `timestamp`: High-precision POSIX timestamp.

---

## 4. Verification under Concurrency

During concurrent multi-threaded execution (5 parallel worker threads executing identical prompts with randomized unique sentinels):
- **Collision Rate:** 0.00% (5 distinct execution IDs, 5 distinct request IDs, 5 distinct conversation IDs).
- **Cross-Contamination:** 0 occurrences of thread $i$ output appearing in thread $j$ stdout/stderr.
- **Deduplication Leakage:** Zero interference between concurrent request tracking sets.
