# Fix and Regression Tests Report: Web Retrieval & Intent Protocol Remediation

**Report ID**: `FIX-20261117-WEB-RETRIEVAL`  
**Target Issue**: Unhandled direct hallucination & brittle tool dispatch on live web queries  
**Scope**: Defensively hardened application protocol, parser, argument validator, and regression test suite.  
**Constraint Compliance**:
- Zero modifications to checkpoint weights (`ckpt_experiment_c_final.pt`).
- No model retraining or fine-tuning.
- No hardcoded keyword routers for "Elon Musk".
- No forced web retrieval on non-web queries (model-selected tool use strictly preserved).
- All existing capability isolation and permission checks intact.

---

## 1. Summary of Targeted Code Fixes

### Fix 1: Robust Tool-Call Parsing (`backend/tools/orchestrator.py`)
- **Locus**: [orchestrator.py#L38-L90](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/orchestrator.py#L38-L90)
- **Problem**: In small language models (SLMs), tool call outputs frequently contain single quotes (`{'name': 'web_research', ...}`), markdown code fences (````json ... ````), or unnested arguments (`{"name": "web_research", "query": "..."}`). Standard `json.loads` threw exceptions on single quotes, causing valid tool calls to be discarded.
- **Remediation**:
  1. Added `import ast` for safe Python literal dictionary parsing fallback.
  2. Stripped markdown code fences before parsing.
  3. Added schema normalization to extract flattened top-level arguments.
  4. Added parameter-based fallback when tool name is omitted but `"query"` or `"code"` is present.

### Fix 2: Schema-Aware Action Routing (`backend/tools/two_stage_protocol.py`)
- **Locus**: [two_stage_protocol.py#L55-L75](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/two_stage_protocol.py#L55-L75)
- **Problem**: The model frequently outputs the entity or topic name as the tool `name` (e.g., `{"name": "NASA Artemis Program", "arguments": {"query": "Profile: Artemis program"}}` or `{"name": "New York State University", "arguments": {"query": "..."}}`). The protocol previously evaluated only hardcoded tool names (`["web_research", "web_search", "search"]`), dropping all topic-named calls to `DIRECT`.
- **Remediation**:
  - Implemented argument schema signature detection: If `name` is unrecognized but `arguments` contains `"query"` or `"search_query"` (and no `"code"`), the protocol routes to `("WEB", "web_research", args)` if `web_enabled` is `True`.
  - Maintained permission enforcement: If `web_enabled` is `False`, the action is strictly blocked and returns `DIRECT`.

### Fix 3: User Query Fallback on Manual Override (`backend/tools/two_stage_protocol.py` & `backend/app.py`)
- **Locus**: [two_stage_protocol.py#L110-L130](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/two_stage_protocol.py#L110-L130) & [app.py#L268](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/app.py#L268)
- **Problem**: When a user or harness explicitly configured `manual_tool_override="WEB"`, `validate_and_sanitize_arguments` failed if the model's first-round output contained only conversational text without a `"query"` parameter.
- **Remediation**:
  - Added `user_query: Optional[str] = None` parameter to `validate_and_sanitize_arguments`.
  - If `"query"` is missing from model arguments and text, it falls back to the sanitized `user_query` (passed from `last_user_msg` in `app.py`).

### Fix 4: Manifest Identity Harmonization (`backend/app.py`)
- **Locus**: [app.py#L160-L190](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/app.py#L160-L190)
- **Remediation**: Replaced historical `ModernGPT-S` references in `get_compact_tool_manifest` with canonical product name `QORIN`.

---

## 2. Comprehensive Regression Test Suite

A new dedicated test module, `tests/test_web_retrieval_regression.py`, was created covering all 8 required scenarios:

| Scenario # | Test Function | Scenario Description | Result |
|---|---|---|---|
| **1a** | `test_scenario_1_ambiguous_request_model_direct` | Ambiguous request (`grab info on elon musk`) where model outputs direct text is strictly honored as `DIRECT` with no phantom tool trace. | **PASS** |
| **1b** | `test_scenario_1_ambiguous_request_manual_override` | Ambiguous request with `manual_override="WEB"` routes to `WEB` and falls back to user prompt for search query. | **PASS** |
| **2a** | `test_scenario_2_canonical_web_call` | Canonical `<|tool_call|>{"name": "web_research", ...}` routes cleanly to `WEB`. | **PASS** |
| **2b** | `test_scenario_2_schema_drift_topic_as_name` | Model emitting topic name (`"name": "Elon Musk"`) with `query` argument is recovered and routed to `WEB`. | **PASS** |
| **3** | `test_scenario_3_web_disabled_blocks_dispatch` | When Web capability is `OFF`, tool calls to `web_research` are blocked and returned as `DIRECT`. | **PASS** |
| **4** | `test_scenario_4_web_zero_sources` | Web researcher gracefully handles zero returned sources with structured error payload without raising unhandled exceptions. | **PASS** |
| **5** | `test_scenario_5_web_multiple_sources` | Live Web researcher retrieves multiple sources with valid titles, HTTP/HTTPS URLs, and snippets. | **PASS** |
| **6** | `test_scenario_6_second_stage_synthesis_error_resilience` | Closed-loop trace packaging handles boundary and empty source payloads without failing. | **PASS** |
| **7** | `test_scenario_7_malformed_tool_actions` | Robust parser recovers single quotes, markdown fences, flattened schemas, and rejects garbage syntax safely. | **PASS** |
| **8** | `test_scenario_8_both_tools_enabled_web_routing` | When both capabilities are `ON`, web research calls route strictly to `web_research` and not `python`. | **PASS** |

### Test Suite Execution Output
```bash
$ .venv\Scripts\python.exe -m pytest tests/test_model_manifest_validation.py tests/test_capability_isolation_and_safety.py tests/test_web_retrieval_regression.py

============================= test session starts =============================
platform win32 -- Python 3.12.13, pytest-9.1.1, pluggy-1.7.0
rootdir: C:\Users\Anish\Desktop\Projects\ML-course
collected 23 items

tests\test_model_manifest_validation.py ....                             [ 17%]
tests\test_capability_isolation_and_safety.py .........                  [ 56%]
tests\test_web_retrieval_regression.py ..........                        [100%]

============================= 23 passed in 5.46s ==============================
```

---

## 3. Live Demonstration Verification

Following deployment to the live backend server:
1. **Query**: `"grab info on elon musk"` with `manual_tool_override="WEB"`.
2. **Outcome**:
   - `event: tool_start` emitted with `tool_name: "web_research"`, `query: "grab info on elon musk"`.
   - `event: tool_result` emitted with 3 verified sources (Wikipedia, Biography.com, Investing.com) and authentic URLs.
   - Second-stage model generation produced a grounded response citing biographical details from retrieved evidence snippets.
   - `ChatArea.tsx` renders inline tool trace card with execution latency (1021.23 ms) and interactive citation hyperlinks.
