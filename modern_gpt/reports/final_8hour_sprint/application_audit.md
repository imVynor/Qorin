# QORIN — Baseline Application Audit & Architecture Report
## Final 8-Hour Sprint: Product Reliability and Research Showcase Audit

**Date:** 2026-11-17  
**Artifact Path:** `modern_gpt/reports/final_8hour_sprint/application_audit.md`  
**Target Identity:** **QORIN** (historical development name: ModernGPT-S)  

---

### 1. Executive Summary & Audit Scope

This baseline application audit was executed immediately prior to implementing the independent capability switches and Two-Stage Action Protocol. The audit examined all components across the full stack:
- **Frontend:** React + TypeScript + Vite (`web/src/`)
- **Backend:** FastAPI + Uvicorn + Server-Sent Events (`backend/app.py`)
- **Model Inference:** `ModelManager` + `QorinProvider` (`backend/models/`)
- **Tool Orchestration:** `ToolOrchestrator` (`backend/tools/orchestrator.py`)
- **Tool Sandbox Environments:** `PythonSandbox` (`backend/tools/python_sandbox.py`) and `WebResearcher` (`backend/tools/web_researcher.py`)
- **Observability:** `ExperimentTracker` (`backend/experiments/tracker.py`)

---

### 2. Forensic Baseline Findings

#### A. Model Selection & Architecture Loading
- **Current Behavior:** The backend provides 9 native QORIN model variants (`qorin-16m-sft`, `qorin-50m-sft`, `qorin-100m-base`, `qorin-100m-4k-base`, `qorin-100m-sft-v3`, `qorin-500m-pretrain-pilot`, `qorin-500m-sft-exp-a`, `qorin-500m-sft-exp-b`, and `qorin-500m-sft-exp-c`).
- **Audit Verification:** Following the resolution of GQA vs. MHA head configuration resolution in `qorin_provider.py`, all 9 models load successfully into memory without tensor shape mismatches.
- **VRAM Utilization:** ModernGPT-S/QORIN 514M allocates **1,036.12 MB** in CUDA memory on the NVIDIA GeForce RTX 5060 Laptop GPU.

#### B. Retrieval-Augmented Generation (RAG) Settings
- **Settings:** `retrieval_mode` (`"auto"`, `"always"`, `"off"`), `top_k: 3`, `threshold: 0.08` to `0.15`.
- **Behavior:** The retrieval subsystem routes domain-specific documentation into context when similarity exceeds threshold, returning an event `event: retrieval` on the SSE stream with retrieved chunks, similarity scores, and latency.

#### C. Tool Availability & Manifest Injection
- **Baseline Behavior:** Tool settings were passed as `web_research_enabled: bool` and `python_sandbox_enabled: bool` in `ChatRequest`.
- **Manifest Delivery:** When both flags were enabled, `get_compact_tool_manifest` injected a 4-line compact description of Python and Web Researcher into the system prompt.
- **Baseline Defect:** The frontend `DEFAULT_SETTINGS` defaulted both tools to `true`, violating the requirement that capability controls must default to OFF for standalone inference.

#### D. Standalone Inference Verification
- **Verification Question:** Does standalone mode truly disable tools?
- **Finding:** **Yes, verified at the backend level.** When `web_research_enabled=False` and `python_sandbox_enabled=False`:
  1. `get_compact_tool_manifest` returns `None`.
  2. The system prompt contains no tool documentation or syntax examples.
  3. The backend execution condition `if is_tool_call and (req.web_research_enabled or req.python_sandbox_enabled)` evaluates to `False`.
  4. Even if the user explicitly writes a prompt prompting a tool call (e.g. *"Calculate with Python"*), the backend does not dispatch code to `PythonSandbox`.

#### E. Tool-Call Generation, Parsing, and Dispatch
- **Call Emitted:** `<|tool_call|>{"name": "...", "arguments": {"...": "..."}}`.
- **Parsing:** RegEx search for canonical `<|tool_call|>` or legacy `<tool_call>` tags, followed by `json.loads`.
- **Baseline Flaw (Single-Stage Free-Form Fragility):** The model was tasked with deciding intent, choosing the tool, generating valid JSON, and writing code all within one generation budget. On complex queries, token limits resulted in unclosed braces, failing JSON parsing.

#### F. Tool Result Feedback & SSE Streaming
- **Observation Feeding:** Dispatched tool traces are converted to `<|tool_result|>` or `<|tool_error|>` payloads and appended as `role: "tool"` messages in `active_messages`.
- **Closed-Loop Round 2 Synthesis:** The model is invoked a second time with the tool result in its history, synthesizing a final natural language response.
- **Streaming Contract:** The Server-Sent Events stream yields:
  - `event: retrieval`
  - `event: tool_start`
  - `event: tool_result`
  - `event: token`
  - `event: done`

---

### 3. Identified Deficits & Sprint Remediation Matrix

| Deficit ID | Component | Description | Sprint Remediation |
| :--- | :--- | :--- | :--- |
| **DEF-04** | Frontend / UX | UI lacked direct independent capability switches in the chat action bar; defaulted to ON instead of OFF. | Add explicit `Web retrieval` and `Python execution` toggles to both chat bar and sidebar, defaulting to OFF. |
| **DEF-05** | Backend / Harness | Single-stage free-form tool calling suffered from Python selection bias and token cutoff syntax errors. | Implement `QorinTwoStageProtocol`: separate Action Selection (`DIRECT`, `WEB`, `PYTHON`) from Argument Validation. |
| **DEF-06** | Branding | Historical "ModernGPT-S" branding persisted in chat manifests, UI titles, and documentation. | Standardize all user-facing names to **QORIN**. Preserve exact trained tokens in conditional inference. |
| **DEF-07** | Argument Validation | Malformed Python code strings were dispatched directly to the sandbox, causing preventable execution errors. | Validate Python arguments with `ast.parse` and extract code from markdown blocks before execution. |
