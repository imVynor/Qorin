# QORIN Capability Isolation & Safety Audit Report

**Audit Date:** November 17, 2026  
**Audited Modules:** `backend/tools/two_stage_protocol.py`, `backend/tools/orchestrator.py`, `backend/tools/python_sandbox.py`, `backend/tools/web_researcher.py`  
**Test Suite:** [`tests/test_capability_isolation_and_safety.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/tests/test_capability_isolation_and_safety.py)  
**Audit Status:** **VERIFIED & SECURE (Multi-Layer Enforcement)**

---

## 1. Executive Summary

This audit evaluates the reliability, isolation, and safety of QORIN's two external tool integrations: **Web Researcher** and **Python Sandbox**. Both capabilities must remain independently toggleable, default to OFF, and be strictly enforced at the backend level regardless of client or model behavior.

### Primary Audit Verdict:
1. **Four-State Capability Isolation:** The application strictly enforces all four permutations (OFF/OFF, OFF/ON, ON/OFF, ON/ON).
2. **Defect Remediation (DEF-04):** An unauthorized tool re-routing defect in `backend/tools/two_stage_protocol.py` (which previously attempted to convert a disabled tool call into the enabled tool) was identified and fixed. Disabled tool calls are now strictly blocked and treated as direct responses.
3. **Defense-in-Depth Sandbox Safety:** The Python sandbox provides multiple layers of containment: AST syntax pre-validation, system command pattern rejection, import blocking (`socket`, `subprocess`, `urllib`), stripped environment variables, and strict 5.0-second process timeouts.
4. **Automated Verification:** 100% of capability isolation and safety tests passed (9/9).

---

## 2. Four-State Capability Configuration Matrix

| State Name | Web Researcher Toggle | Python Sandbox Toggle | Expected Manifest Injected | Expected Allowed Dispatch | Actual Enforcement Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **OFF_OFF** | `false` (OFF) | `false` (OFF) | `None` (No tools advertised) | **None** (Strict direct parametric response) | **PASSED (Blocked)** |
| **OFF_ON** | `false` (OFF) | `true` (ON) | Python tool schema only | **Python Sandbox only** (Web blocked) | **PASSED (Python only)** |
| **ON_OFF** | `true` (ON) | `false` (OFF) | Web research schema only | **Web Researcher only** (Python blocked) | **PASSED (Web only)** |
| **ON_ON** | `true` (ON) | `true` (ON) | Both tool schemas | **Both tools authorized** | **PASSED (Both tools)** |

### Multi-Layer Enforcement Pipeline:
```mermaid
flowchart LR
    A["Frontend Toggle (HarnessControls.tsx)"] --> B["Compact Manifest Generator (get_compact_tool_manifest)"]
    B --> C["Protocol Action Routing (two_stage_protocol.py)"]
    C --> D["Orchestrator Enforcement (orchestrator.py)"]
    D --> E["Subprocess Execution (python_sandbox.py / web_researcher.py)"]
```

---

## 3. Defect Analysis & Remediation: DEF-04

### 3.1 Defect Description
In `backend/tools/two_stage_protocol.py` (lines 61–68 prior to audit):
```python
# VULNERABLE CODE PRIOR TO FIX:
if web_enabled and not python_enabled:
    # Model wanted tool under web-only manifest: route to WEB
    return "WEB", "web_research", parsed[1] if parsed else None
elif python_enabled and not web_enabled:
    # Model wanted tool under python-only manifest: route to PYTHON
    return "PYTHON", "python", parsed[1] if parsed else None
```

### 3.2 Security & Correctness Impact
If a user enabled Web Research only (Python disabled) and the model emitted a Python call (`{"name": "python", "arguments": {"code": "print(10)"}}`), the protocol re-routed the action to `"WEB"`, attempting to pass Python code as a search query. Conversely, if only Python was enabled, a Web search query was routed to Python, executing natural language as code.

### 3.3 Remediation Applied
The unauthorized re-routing logic was removed. The protocol now strictly inspects tool permissions for the specific candidate tool:
```python
if tool_candidate in ["web_research", "web_search", "search"]:
    if web_enabled:
        return "WEB", "web_research", parsed[1] if parsed else None
    else:
        return "DIRECT", None, None
elif tool_candidate in ["python", "python_interpreter", "run_code"]:
    if python_enabled:
        return "PYTHON", "python", parsed[1] if parsed else None
    else:
        return "DIRECT", None, None
```
If the requested tool is disabled, execution is immediately blocked without side effects.

---

## 4. Subprocess Sandbox Security & Isolation

The Python execution environment (`PythonSandbox` in `backend/tools/python_sandbox.py`) enforces five independent security controls:

1. **AST Syntax Pre-Validation:**
   Before any subprocess is created, `ast.parse(code)` evaluates the code. Malformed syntax is rejected in Python memory without disk I/O or process creation.
2. **Prohibited Primitive Scanning:**
   Code containing dangerous primitives (`os.system`, `os.popen`, `subprocess`, `__import__('subprocess')`, `rmdir`, `unlink`) is intercepted with an immediate `SecurityViolation` error.
3. **Standard Library Import Poisoning:**
   The runner prepends a restricted header setting dangerous network and execution modules (`sys.modules[m] = None` for `socket`, `urllib`, `http`, `ftplib`, `webbrowser`, `subprocess`, `shutil`) to `None`, making network socket creation impossible.
4. **Environment Isolation:**
   Subprocesses run with Python's `-I` (isolated mode) flag, ignoring `PYTHONPATH`, user site-packages, and user environment variables.
5. **Execution Timeout & Process Termination:**
   All execution is wrapped in `subprocess.Popen.communicate(timeout=5.0)`. Runaway loops (e.g. `while True: pass`) are killed with `proc.kill()` and return an explicit `TimeoutError` (exit code 124).

---

## 5. Automated Test Suite Results

Test execution was performed using pytest in the isolated virtual environment:

```text
tests/test_capability_isolation_and_safety.py::test_four_capability_configurations_manifest PASSED [ 11%]
tests/test_capability_isolation_and_safety.py::test_disabled_tool_cannot_be_dispatched_via_protocol PASSED [ 22%]
tests/test_capability_isolation_and_safety.py::test_disabled_tool_blocked_at_orchestrator PASSED [ 33%]
tests/test_capability_isolation_and_safety.py::test_python_argument_validation PASSED [ 44%]
tests/test_capability_isolation_and_safety.py::test_web_argument_validation PASSED [ 55%]
tests/test_capability_isolation_and_safety.py::test_python_sandbox_security_rejections PASSED [ 66%]
tests/test_capability_isolation_and_safety.py::test_python_sandbox_execution_correctness PASSED [ 77%]
tests/test_capability_isolation_and_safety.py::test_python_sandbox_timeout_enforcement PASSED [ 88%]
tests/test_capability_isolation_and_safety.py::test_web_researcher_empty_query PASSED [100%]
============================== 9 passed in 5.60s ==============================
```
