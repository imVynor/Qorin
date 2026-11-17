# QORIN Live Demonstration Script & Defense Protocol

**Audit Date:** November 17, 2026  
**Audited Target:** Live Inference Demonstration via QORIN Harness (`/harness`)  
**Backend Endpoint:** `http://127.0.0.1:8000` (Uvicorn / FastAPI)  
**Active Model:** `qorin-500m-sft-exp-c` (514M ModernGPT-S on NVIDIA RTX 5060 Laptop GPU)  
**Demo Duration:** Exactly 5 Minutes

---

## 1. Pre-Flight Setup Checklist

Before initiating the defense presentation:
1. **Verify Backend Server Status:**
   - Execute in terminal:
     ```powershell
     curl.exe http://127.0.0.1:8000/api/health
     ```
   - Expect: `{"status": "healthy", "telemetry": {"active_model_id": "qorin-500m-sft-exp-c", "cuda_available": true}}`
2. **Open Frontend Web Application:**
   - Navigate to `http://localhost:5173/harness` in browser.
3. **Verify Model Selection:**
   - Check dropdown in left sidebar: **QORIN-500M SFT Experiment C (Autonomous Two-Tool)** must be selected.
   - Status badge must display: **ready** (in green).
4. **Ensure Default Capability State:**
   - Verify both **Web retrieval** and **Python execution** checkboxes are unchecked (OFF).

---

## 2. Live Demonstration Flow Sequence

### Flow 1: Standalone Direct Inference (Both Tools OFF)
- **Objective:** Prove that the model acts as a natural conversational agent when tools are disabled, without false-positive tool calls.
- **Controls Configuration:**
  - Web retrieval: **OFF**
  - Python execution: **OFF**
  - RAG Retrieval: **OFF**
- **Input Prompt:**
  ```text
  Hello QORIN! Can you explain what a transformer attention mechanism is in two sentences?
  ```
- **Observed Behavior:**
  - The model streams an articulate conceptual explanation directly into the chat area.
  - No `<|tool_call|>` tags appear. Zero tool execution cards are rendered.
  - Latency: $\sim 3,200\text{ms}$ total ($\sim 65\text{ tokens/sec}$).

---

### Flow 2: Isolated Python Execution (Python ON, Web OFF)
- **Objective:** Demonstrate that the model recognizes mathematical complexity and autonomously delegates to the Python sandbox.
- **Controls Configuration:**
  - Web retrieval: **OFF**
  - Python execution: **ACTIVE (Checked)**
- **Input Prompt:**
  ```text
  Compute 4321 * 8765 using python.
  ```
- **Observed Behavior:**
  - **Round 1:** Model emits `<|tool_call|>{"name": "python", "arguments": {"code": "print(4321 * 8765)"}}`.
  - **Harness Action:** The two-stage protocol intercepts the call, validates AST syntax, and executes in the isolated sandbox subprocess.
  - **UI Render:** An inline green tool card appears: `Tool Invocation: python (completed) • ~150ms`. Stderr is empty; stdout displays `37873565`.
  - **Round 2:** The model re-prompts with the tool observation and synthesizes the verified answer: `37873565`.

---

### Flow 3: Tool Execution & Second-Stage Synthesis (Closed-Loop)
- **Objective:** Prove that the model reads the tool observation and synthesizes an authoritative final answer incorporating the external calculation.
- **Controls Configuration:**
  - Web retrieval: **OFF**
  - Python execution: **ACTIVE**
- **Input Prompt:**
  ```text
  Calculate the greatest common divisor of 123456 and 7890 using python math.gcd.
  ```
- **Observed Behavior:**
  - Model emits: `import math\nprint(math.gcd(123456, 7890))`.
  - Sandbox output: `6`.
  - Model synthesized output: *"The Python execution yielded: Greatest Common Divisor: 6. Therefore, the requested value is verified."*

---

### Flow 4: Live Web Research with Source Attribution (Web ON, Python OFF)
- **Objective:** Demonstrate live external web retrieval and citation extraction.
- **Controls Configuration:**
  - Web retrieval: **ACTIVE (Checked)**
  - Python execution: **OFF**
- **Input Prompt:**
  ```text
  Search for the latest news on NASA Artemis program.
  ```
- **Observed Behavior:**
  - Model emits: `web_research` call.
  - Harness dispatches query to DuckDuckGo/Wikipedia via `WebResearcher`.
  - Tool card renders retrieved snippets and authentic URLs.
  - Model synthesizes the news summary with explicit source citations.

---

### Flow 5: Subprocess Failure & Honest Error Recovery
- **Objective:** Demonstrate that runtime exceptions in external tools do not crash the server and are ingested by the model for recovery.
- **Controls Configuration:**
  - Python execution: **ACTIVE**
  - Web retrieval: **OFF**
- **Input Prompt:**
  ```text
  Execute python with division by zero: print(1 / 0)
  ```
- **Observed Behavior:**
  - Subprocess catches `ZeroDivisionError: division by zero` (exit code 1).
  - Tool card displays red/amber status: `Tool Invocation: python (failed)`.
  - The observation `<|tool_error|>{"stderr": "...ZeroDivisionError..."}` is fed back into Round 2.
  - Model generates an honest acknowledgment that division by zero is undefined.

---

### Flow 6: Negative-Control Query with Both Tools Active
- **Objective:** Prove that having both tools active does NOT cause spurious or hallucinated tool invocations.
- **Controls Configuration:**
  - Web retrieval: **ACTIVE**
  - Python execution: **ACTIVE**
- **Input Prompt:**
  ```text
  Write a short haiku about spring blossoms falling into a river.
  ```
- **Observed Behavior:**
  - Model recognizes that creative writing requires internal generation, not tools.
  - Generates a 3-line haiku directly into the chat area. Zero tool cards rendered.

---

## 3. Comprehensive Backup Plan

| Failure Scenario | Immediate Remedy | Examiner Explanation |
| :--- | :--- | :--- |
| **Live Web Search Timeout or Rate Limit** | Toggle Web OFF, toggle Python ON, and re-run Flow 2 or Flow 3 with Wikipedia direct fallback. | Explain that live HTML scraping via DuckDuckGo is subject to public IP rate-limiting, and show pre-recorded traces in `/experiments`. |
| **GPU Out-of-Memory (VRAM Leak)** | Click the **"Unload VRAM"** button in `/harness`, then select `qorin-500m-sft-exp-c` again. | Explain single-model VRAM management policy and automatic PyTorch cache flushing. |
| **Backend Process Dies** | Open PowerShell terminal and execute:<br/>`.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000` | Process reloads within 2.5 seconds on localhost. |
| **Network Completely Disconnected** | All Python sandbox, negative-control, and RAG retrieval flows operate 100% locally on localhost without internet. | Emphasize that QORIN is an open-weights local model requiring zero cloud API calls. |
