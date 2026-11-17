# QORIN Multi-Step Tool Orchestration Specification & Evaluation Protocol

**Document**: `multistep_orchestration_spec.md`  
**Status**: Engineering Specification & Protocol Review (Non-destructive, Pre-implementation)  
**Target System**: QORIN-500M Dual-Stage Agent Harness (`backend/tools/two_stage_protocol.py`, `backend/app.py`, and offline audit harnesses)  
**Parent Checkpoint**: Frozen Experiment D Best Checkpoint (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388da...`)

---

## 1. Existing Harness Investigation & Failure Root Cause

In the Stage 5 Generalization Audit, all 15 tasks in the `multistep_web_python` category failed (0.0% semantic success). A technical inspection of the repository reveals a fundamental architectural discrepancy between the API backend and the evaluation harness:

1. **Evaluation Harness Disconnect**:
   In `run_audit_evaluation.py` (lines 248–265), the evaluation loop was implemented as a rigid, hardcoded single-turn dispatch:
   $$\text{Stage 1 Prompt} \xrightarrow{\text{Generate}} \text{Tool Call} \xrightarrow{\text{Execute}} \text{Tool Observation} \xrightarrow{\text{Generate}} \text{Final Response}$$
   The evaluation script terminated immediately after `raw_stage2` was generated. It contained zero logic to inspect whether `raw_stage2` contained a secondary `<|tool_call|>` for `python`.
2. **Model Training Disconnect**:
   In Experiment D, training records only ever contained single-turn `<|tool_call|>` instances (either `web_research` or `python`). The model never saw an assistant turn that emitted `<|tool_call|>` *after* a `<|tool_result|>` observation. Consequently, in Stage 1, when faced with prompts requiring both empirical retrieval and calculation (e.g. "Find the 2023 revenue of Apple and Microsoft, then compute the difference"), the model panicked and jumped immediately to `python` using hallucinated constants, completely bypassing the search step.
3. **API Backend Latent Loop**:
   `backend/app.py` contained a latent `while current_round <= max_tool_rounds` loop (`max_tool_rounds = 2`), but its message serialization for `<|im_start|>tool` and tool transition safety had never been formally tested, calibrated, or coordinated with model training prompts.

To solve this, both a **harness extension** and a **model prompting/training adjustment** are strictly necessary.

---

## 2. Minimal Reliable Extension Architecture

The extension is designed as an explicit, bounded **Deterministic Execution State Machine** that guarantees zero infinite loops, strict execution budgets, and full trace serialization.

### 2.1 Formal Execution State Machine (DAG)

The orchestration operates strictly as a Directed Acyclic Graph (DAG) with a maximum depth of 2 tool executions:

```
                      [START / User Request]
                                │
                                ▼
                      [STATE: STAGE1_DECISION]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
    (No Tool)            (<tool: python>)        (<tool: web>)
        │                       │                       │
        ▼                       ▼                       ▼
 [FINAL_SYNTHESIS]       [PYTHON_EXEC_1]          [WEB_EXEC_1]
        │                       │                       │
        │                       ▼                       ▼
        │               [PYTHON_RETURN_1]       [WEB_RETURN_1]
        │                       │                       │
        │                       ▼                       ▼
        │               [FINAL_SYNTHESIS]    [STAGE2_DECISION]
        │                       │                       │
        │                       │       ┌───────────────┴───────────────┐
        │                       │       ▼                               ▼
        │                       │  (No Further Tool)            (<tool: python>)
        │                       │       │                               │
        │                       │       ▼                               ▼
        │                       │ [FINAL_SYNTHESIS]             [PYTHON_EXEC_2]
        │                       │       │                               │
        │                       │       │                               ▼
        │                       │       │                       [PYTHON_RETURN_2]
        │                       │       │                               │
        │                       │       │                               ▼
        │                       │       │                       [FINAL_SYNTHESIS]
        ▼                       ▼       ▼                               ▼
        └───────────────────────┴───────┴───────────────────────────────┘
                                        │
                                        ▼
                                   [TERMINATED]
```

---

## 3. State Machine Specification & Safety Guarantees

### 3.1 State Definitions & Transitions

| State ID | Description | Allowed Next States | Error / Timeout Fallback |
| :--- | :--- | :--- | :--- |
| `INIT` | Ingests user prompt, applies ChatML templating and system instructions. | `STAGE1_DECISION` | `FAILED` |
| `STAGE1_DECISION` | Model generates initial output (Greedy, max 128 tokens). | `DIRECT_SYNTHESIS`, `WEB_EXEC_1`, `PYTHON_EXEC_1` | `DIRECT_SYNTHESIS` |
| `WEB_EXEC_1` | Orchestrator validates query, calls search engine sandbox. | `WEB_RETURN_1`, `SCHEMA_RETRY_WEB` | `DIRECT_SYNTHESIS` (with error) |
| `WEB_RETURN_1` | Harness normalizes search results into Structured Evidence Record (SER). | `STAGE2_DECISION` | `DIRECT_SYNTHESIS` |
| `STAGE2_DECISION` | Model consumes SER. Can emit final text OR `<tool_call: python>`. | `FINAL_SYNTHESIS`, `PYTHON_EXEC_2` | `FINAL_SYNTHESIS` |
| `PYTHON_EXEC_2` | AST validation; sandbox executes Python code using extracted literals. | `PYTHON_RETURN_2`, `SCHEMA_RETRY_PY` | `FINAL_SYNTHESIS` (with error) |
| `PYTHON_RETURN_2` | Harness captures stdout/stderr, wraps in `<|tool_result|>`. | `FINAL_SYNTHESIS` | `FINAL_SYNTHESIS` |
| `FINAL_SYNTHESIS` | Model generates final response incorporating facts and math. | `TERMINATED` | `TERMINATED` |
| `TERMINATED` | Clean output delivery, metrics finalized, traces committed. | End | End |

### 3.2 Bounded Execution & Loop Prevention Guarantees
1. **Maximum Tool Rounds ($R_{max} = 2$)**:
   The variable `current_tool_round` increments on every dispatch. If `current_tool_round == 2`, the orchestrator **strips any subsequent tool call tags** and forces transition to `FINAL_SYNTHESIS`.
2. **Cycle Prevention Rule**:
   The state graph strictly forbids self-transitions ($Web \to Web$ is illegal; $Python \to Python$ is illegal). The only valid multi-step transition is $Web \to Python$.
3. **Global Execution Timeout ($T_{max} = 15.0\text{ seconds}$)**:
   A monotonic clock starts at `INIT`. If elapsed wall-clock time exceeds 15.0s at any state boundary:
   - Active sub-process is terminated immediately (`SIGKILL`).
   - The harness appends `[TIMEOUT ERROR: Execution exceeded 15.0s]` to context.
   - Transitions directly to `FINAL_SYNTHESIS`.
4. **Retry and Recovery Limits**:
   - Schema syntax errors allow at most **1 retry** (`max_retries = 1`).
   - If the model emits malformed JSON a second time, the error payload `<|tool_error|>` is injected, and the model must proceed directly to answer without tool execution.
5. **Python Execution Sandbox & AST Gate**:
   - Python code must pass pre-flight `ast.parse()`.
   - The AST visitor rejects any import or call of `os`, `sys`, `subprocess`, `socket`, `requests`, `shutil`, `open`, or built-in reflection (`eval`, `exec`).
   - Execution runs in an isolated subprocess with a **3.0-second CPU time ceiling** and a **128 MB RAM limit**.

---

## 4. Structured Intermediate Representation & Trace Schema

Intermediate outputs between turns must not be raw, unformatted text. The harness enforces explicit encapsulation:

### 4.1 Turn 1 Tool Observation (Web Research $\to$ SER):
```xml
<|tool_result|>
[SOURCE 1]: https://investor.apple.com
- ENTITY: Apple Inc.
- METRIC: Fiscal Year 2023 Revenue = $383.29 billion
[SOURCE 2]: https://microsoft.com/investor
- ENTITY: Microsoft Corporation
- METRIC: Fiscal Year 2023 Revenue = $211.92 billion
<|end_tool_result|>
```

### 4.2 Turn 2 Tool Observation (Python Execution $\to$ Structured Result):
```xml
<|tool_result|>
[PYTHON EXECUTION RESULT]
- STDOUT:
383.29 - 211.92 = 171.37
- RETURN_VALUE: 171.37
- STATUS: completed (latency: 142ms)
<|end_tool_result|>
```

### 4.3 Multi-Step Tool Trace Record Schema
Every multi-step interaction serializes a complete trace object into `tool_traces.jsonl`:
```json
{
  "task_id": "multistep_01",
  "total_rounds": 2,
  "elapsed_wall_ms": 1420.5,
  "steps": [
    {
      "round": 1,
      "tool": "web_research",
      "args": {"query": "Apple and Microsoft FY 2023 revenue"},
      "status": "completed",
      "latency_ms": 1180.2,
      "extracted_entities": ["Apple", "Microsoft"],
      "extracted_numbers": [383.29, 211.92]
    },
    {
      "round": 2,
      "tool": "python",
      "args": {"code": "apple = 383.29\nmsft = 211.92\ndiff = apple - msft\nprint(f'{apple} - {msft} = {diff:.2f}')"},
      "status": "completed",
      "latency_ms": 84.1,
      "stdout": "383.29 - 211.92 = 171.37"
    }
  ],
  "final_synthesis_token_count": 94,
  "termination_reason": "natural_completion"
}
```

---

## 5. Evaluation Protocol for Multi-Step Orchestration

Before deploying in the primary benchmark, the multi-step extension must pass an independent **15-task Feasibility Gate**:

### 5.1 Test Battery Composition
The 15 multi-step tasks from Stage 5 (`multistep_01` to `multistep_15`) are executed against the state machine using a mock/live tool provider.

### 5.2 Pass/Fail Criteria for Multi-Step Feasibility
1. **Zero Infinite Loops**: 100% of tasks terminate within $R \le 2$ and $T \le 15.0\text{s}$.
2. **Transition Validity**: The state machine correctly transitions from `WEB_EXEC_1` to `STAGE2_DECISION` and cleanly invokes `PYTHON_EXEC_2` when the model emits code.
3. **Trace Completeness**: Every run emits a valid JSON trace containing both tool steps.
4. **Execution Safety**: AST gate successfully blocks intentional injection payloads (e.g. `import os`).
