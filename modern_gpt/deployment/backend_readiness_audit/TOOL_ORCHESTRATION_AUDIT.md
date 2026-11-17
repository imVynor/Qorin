# Tool Orchestration & Multi-Turn Execution Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/TOOL_ORCHESTRATION_AUDIT.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Web Retrieval Subsystem

The live web research capability is implemented in `backend/tools/web_researcher.py` and `backend/tools/retrieval_providers.py`.

```
User Query ──► SQLite Cache Check ──┬──► Cache Hit: Return Stored Bundle (0.5ms)
                                    └──► Cache Miss:
                                           ├── 1. Primary: DuckDuckGo HTML Lite (4.0s timeout)
                                           └── 2. Fallback: Wikipedia OpenSearch API
                                                   │
                                                   ▼
                                         Parse HTML (BeautifulSoup)
                                         Extract Titles, URLs, Snippets
                                         Format as EvidenceRecord Bundle
                                         Write to retrieval_cache.sqlite
```

### Detailed Component Specifications:
- **Search Providers:**
  - *Primary Provider:* DuckDuckGo HTML Lite (`https://html.duckduckgo.com/html/?q=...`). Employs desktop browser user-agent header to avoid bot gates.
  - *Fallback Provider:* Wikipedia OpenSearch API (`https://en.wikipedia.org/w/api.php?action=opensearch&...`). Automatically engaged if DuckDuckGo returns empty results, rate limits, or times out.
- **Retrieval Timeout:** `TIMEOUT_SECONDS = 4.0` per HTTP request.
- **Retry Behavior:** Up to 2 retries with exponential backoff (`0.2s * 2^attempt`) implemented in `UnifiedWebResearcher`.
- **Persistent Caching:**
  - *Storage Engine:* Embedded SQLite database at `backend/tools/retrieval_cache.sqlite`.
  - *Key Derivation:* SHA-256 hash of `f"{provider_name}:{normalized_query}"` (lowercase, whitespace collapsed).
  - *Hit Counter:* Automatically increments `hit_count` on access.
  - *Performance:* Cached queries resolve in `< 1.0 ms`, eliminating external network latency for repeated benchmark evaluation.
- **Evidence Representation:**
  - Emits an `EvidenceBundle` containing a list of `EvidenceRecord` objects:
    ```python
    EvidenceRecord(
        source_id="SOURCE_1",
        title="Jensen Huang - Wikipedia",
        url="https://en.wikipedia.org/wiki/Jensen_Huang",
        snippet="Jensen Huang is an American businessman and electrical engineer...",
        is_cached=False
    )
    ```
- **Citation Verification:**
  - Handled by `EvidenceGroundingValidator` in `backend/tools/grounding_validator.py`.
  - Extracts bracketed citation markers: `[SOURCE_1]`, `[SOURCE_2]`, `[1]`, `[2]`.
  - Computes entity overlap between claims and cited source snippets, rejecting ungrounded claims or hallucinated citations.

---

## 2. Python Execution Subsystem

The Python execution sandbox is implemented in `backend/tools/python_sandbox.py` and `backend/tools/python_contract.py`.

```
Model Emitted Python Code
       │
       ▼
1. Pre-Execution AST Static Inspection
       │  - Syntax validation
       │  - Disallowed module check (os, sys, subprocess, socket, urllib...)
       ▼
2. Isolated Subprocess Launch
       │  - python -I user_script.py (isolated mode)
       │  - Stripped environment (empty PYTHONPATH)
       │  - Temporary directory: backend/data/sandbox/run_<id>/
       │  - Security header injection: Block sensitive imports in sys.modules
       ▼
3. Execution & Timeout Watchdog
       │  - 5.0s hard timeout
       │  - Output capped at 50,000 bytes
       ▼
4. Cleanup & Result Serialization
       │  - Subprocess termination & kill
       │  - rmtree(run_dir)
       │  - Return PythonSandboxResult (stdout, stderr, exit_code, latency_ms)
```

### Detailed Component Specifications:
- **Execution Mechanism:** Subprocess launch using `subprocess.Popen([sys.executable, "-I", script_path])`.
  - The `-I` flag tells Python to run in isolated mode: ignores `PYTHONPATH`, user site-packages, and environment variables.
- **Sandbox Isolation:**
  - Code executes in a newly generated temporary directory inside `backend/data/sandbox/run_{random_id}/`.
  - Environment variables are stripped: `clean_env` provides only minimal `SYSTEMROOT`, `PATH`, and redirects `TEMP`/`TMP` to the temporary run folder.
  - Pre-execution string filter rejects explicit invocations of `os.system`, `os.popen`, `subprocess`, `rmdir`, `unlink`.
  - Subprocess script prepends `RESTRICTED_HEADER` setting `sys.modules[mod] = None` for `socket`, `urllib`, `http`, `ftplib`, `smtplib`, `webbrowser`, `subprocess`, `shutil`.
- **AST Validation:**
  - `PythonExecutionContract.pre_validate_ast(code)` parses the Python AST before execution.
  - Traverses `ast.Import` and `ast.ImportFrom` nodes. Rejects forbidden root modules (`os`, `sys`, `socket`, `requests`, `pip`).
- **Resource Limits & Output Capture:**
  - *Timeout:* Hard 5.0-second timeout. If exceeded, the process is killed via `proc.kill()` and returns exit code `124`.
  - *Output Truncation:* Maximum output is capped at 50,000 bytes (`MAX_OUTPUT_BYTES = 50_000`). Runaway print loops are truncated.
  - *Post-Run Cleanup:* The temporary folder is deleted in the `finally` block via `shutil.rmtree(run_dir, ignore_errors=True)`.

---

## 3. Multi-Step Orchestration State Machine

Implemented in `backend/tools/state_machine.py` and inline within `backend/app.py:chat_stream`.

### State Lifecycle:
1. `RECEIVE_USER_REQUEST`: Ingests conversation history and last user message.
2. `MODEL_DECISION`: Model generates autoregressive tokens.
3. `VALIDATE_TOOL_CALL`: `OutputProtocolParser` parses XML-style tags `<tool_call id="...">{"name": "...", "arguments": {...}}</tool_call>`. Validates JSON payload against tool schema.
4. `EXECUTE_TOOL`: Dispatches tool call to `PythonSandbox` or `UnifiedWebResearcher`.
5. `APPEND_TOOL_OBSERVATION`: Injects observation turn `<|im_start|>tool name={name} id={id}\n{payload}<|im_end|>\n`.
6. `MODEL_FOLLOWUP_DECISION`: Model inspects tool output and either invokes another tool or generates final synthesis.
7. `FINAL_SYNTHESIS`: Model synthesizes direct grounded response citing sources.
8. `COMPLETE`: SSE stream concludes and logs experiment trace.
9. `FAILED`: Emits structured error if bounds or timeouts are exceeded.

### Bounded Limits & Isolation:
- **Maximum Tool Rounds:** `max_tool_rounds = 4` per user turn. Prevents infinite tool loops.
- **Global Turn Timeout:** Hard 30.0-second timeout across all sequential tool calls.
- **Schema Retries:** `max_schema_retries = 1`. If the model emits malformed JSON syntax, the harness provides one corrective turn (`MalformedToolCallError`). If it fails again, it forces direct synthesis.
- **Request Call Isolation:** Tracks `request_dispatched_calls: Set[str]` to prevent duplicate or looping tool invocations within a single user request.

---

## 4. Model-Generated Decisions vs Harness-Forced Policy Routing

An essential distinction for production deployment:

| Decision Point | Model-Generated Behavior | Harness-Enforced Policy Routing |
| :--- | :--- | :--- |
| **Factual Questions** (e.g. *"Who is Jensen Huang?"*) | In Stage 16B SFT checkpoints, the model autonomously emits `<tool_call id="...">{"name": "web_search", "arguments": {"query": "..."}}</tool_call>`. | **Policy Fallback:** In `app.py:390`, if `user_intent == UserIntent.FACTUAL_INQUIRY` and the model fails to emit a tool call on Round 0, the harness automatically injects a `web_search` call for `last_user_msg`. |
| **Code Generation** (e.g. *"Write a Python script to calculate fibonacci"*) | The model occasionally attempts to execute code when only asked to write code. | **Policy Suppression:** If `user_intent == UserIntent.CODE_GENERATION` and model invokes Python, `app.py:429` intercepts the tool call, suppresses execution, and forces direct markdown synthesis (`'''python ... '''`). |
| **Simple Arithmetic** (e.g. *"What is 1 + 2?"*) | The model may emit a Python tool call for elementary arithmetic. | **Policy Suppression:** `app.py:612` evaluates simple arithmetic directly without wasting GPU turns or sandbox subprocesses. |
| **Manual UI Override** | Ignored by raw model. | When the user explicitly toggles `manual_tool_override="WEB"` or `"PYTHON"`, the harness injects the selected tool call on Round 0 regardless of model parametric output. |

---

## 5. Mock vs Real Execution Paths

- **Mock Providers:**
  - `MockRetrievalProvider` exists in `backend/tools/retrieval_providers.py` strictly for unit testing. It returns canned fixture records.
- **Live Production Paths:**
  - In `backend/app.py`, only live providers are wired: `RealWeatherProvider`, `UnifiedWebResearcher` (DuckDuckGo + Wikipedia), and `PythonSandbox` (real subprocess).
  - No mock providers are active during standard runtime operation.
