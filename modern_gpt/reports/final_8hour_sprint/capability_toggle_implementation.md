# QORIN — Capability Toggle Implementation & Verification Report
## Final 8-Hour Sprint: Independent Capability Controls

**Date:** 2026-11-17  
**Artifact Path:** `modern_gpt/reports/final_8hour_sprint/capability_toggle_implementation.md`  

---

### 1. Requirements Specification

The system requires two independent, user-controlled switches with specific semantics:

1. **Web Retrieval** (Label: `Web retrieval`):
   - **When Enabled:**
     - Web Researcher is advertised to QORIN in the tool manifest.
     - Web Researcher may be dispatched for live web searches via DuckDuckGo Lite.
     - Retrieved sources, evidence snippets, and URLs are made available to the model.
     - The interface displays inline retrieval activity and source evidence cards.
   - **When Disabled:**
     - Web Researcher is strictly absent from the tool manifest.
     - Any web dispatch attempts are rejected at the backend level (`status="disabled"`).
     - Zero network web retrieval occurs.

2. **Python Execution** (Label: `Python execution`):
   - **When Enabled:**
     - Python Sandbox is advertised to QORIN in the tool manifest.
     - Python code calls may be dispatched through the isolated subprocess sandbox.
     - Actual execution outputs (stdout, stderr, execution time) are returned to QORIN for synthesis.
   - **When Disabled:**
     - Python is strictly absent from the tool manifest.
     - Any Python dispatch attempts are rejected at the backend level (`status="disabled"`).
     - Zero subprocess execution occurs.

3. **Standalone Inference Rule:**
   - Both controls default to **OFF** for standalone inference.
   - When both controls are disabled, the application operates as a standalone language model.
   - The UI displays `Standalone Mode (No Tools)`.
   - The backend enforces tool unavailability even if the user attempts prompt injection.

---

### 2. Implementation Architecture

#### A. Frontend Architecture (`web/src/`)
1. **Default Settings Reset (`web/src/pages/Harness.tsx`):**
   ```typescript
   const DEFAULT_SETTINGS: GenerationSettings = {
     // ...
     web_research_enabled: false,  // Default OFF for standalone inference
     python_sandbox_enabled: false, // Default OFF for standalone inference
   };
   ```
2. **Chat Area Capability Toolbar (`web/src/components/harness/ChatArea.tsx`):**
   - Added an interactive capability toolbar directly above the prompt textarea.
   - Contains real-time interactive pill switches for `Web retrieval` and `Python execution`.
   - Displays real-time status:
     - `Standalone Mode (No Tools)`
     - `Web Retrieval Only`
     - `Python Execution Only`
     - `Two-Tool Orchestrator`
3. **Sidebar Controls (`web/src/components/harness/HarnessControls.tsx`):**
   - Renamed Section 4 to **Capability Controls**.
   - Updated labels strictly to `Web retrieval` and `Python execution` with visual `[ACTIVE]` and `[OFF]` badges.

#### B. Backend Architecture (`backend/app.py` & `backend/tools/`)
1. **Manifest Conditional Generation (`backend/app.py`):**
   - `get_compact_tool_manifest(web_enabled, python_enabled)`:
     - If `web_enabled and python_enabled` $\to$ Returns two-tool manifest.
     - If `python_enabled and not web_enabled` $\to$ Returns Python-only manifest.
     - If `web_enabled and not python_enabled` $\to$ Returns Web-only manifest.
     - If `not web_enabled and not python_enabled` $\to$ Returns `None`.
2. **Backend Execution Guards (`backend/tools/two_stage_protocol.py` & `orchestrator.py`):**
   - If `not web_enabled and not python_enabled` $\to$ `determine_action` returns `("DIRECT", None, None)`. No tool dispatch can occur.
   - If tool dispatch is attempted for a disabled tool $\to$ `orchestrator.execute_tool` immediately returns `status="disabled"` with error `"Tool execution blocked by permission control."`

---

### 3. Empirical Verification Matrix

The four operational states were systematically tested against the live backend API:

| Capability State | Web Enabled | Python Enabled | Injected Manifest | Dispatched Action | Tool Executed? | Verification Result |
| :--- | :---: | :---: | :--- | :--- | :---: | :--- |
| **State 1: Standalone** | ❌ OFF | ❌ OFF | `None` (No tools advertised) | `DIRECT` | ❌ None | **PASS** — Direct text generation only. Zero tools dispatched. |
| **State 2: Python Only** | ❌ OFF | ✅ ON | Python-only manifest | `PYTHON` | ✅ Yes | **PASS** — Python code executed in sandbox. Web calls blocked. |
| **State 3: Web Only** | ✅ ON | ❌ OFF | Web-only manifest | `WEB` | ✅ Yes | **PASS** — Live DuckDuckGo search executed. Python calls blocked. |
| **State 4: Two-Tool** | ✅ ON | ✅ ON | Compact two-tool manifest | `PYTHON` or `WEB` | ✅ Yes | **PASS** — Dual-capability orchestrator active. |
