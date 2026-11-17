# QORIN — Tool-Calling Reliability and Two-Stage Protocol Report
## Final 8-Hour Sprint: Structured Execution & Error Recovery

**Date:** 2026-11-17  
**Artifact Path:** `modern_gpt/reports/final_8hour_sprint/tool_calling_reliability_report.md`  

---

### 1. The Reliability Problem

In Phase 2D Experiment C, the free-form tool-call generation mechanism exhibited key reliability bottlenecks:
1. **JSON Syntax Truncation:** Free-form generation required QORIN to emit `<|tool_call|>{"name": "...", "arguments": {"...": "..."}}` within a single generation step. If the model generated extensive string prefixes or complex nested quotes, the 120–150 token budget was exhausted before closing quotes and braces, causing JSON parser syntax failures.
2. **Inductive Prior Distortion:** The SFT training corpus contained a 12.5:1 ratio of Python trajectories to Web Research fixtures (625 vs 50). When both tools were advertised, the model had an overwhelming bias toward generating `{"name": "python"}` even on factual lookup tasks.
3. **Semantic Code Errors:** In 50% of dispatched Python calls, the code executed with returncode 0, but computed algebraically incorrect values (e.g. `math.comb` instead of `math.gcd`).
4. **Verbatim Parity Echo:** The model frequently used a parity heuristic (`is_prime(n) := (n % 2 != 0)`) rather than an algorithmic primality test, failing on odd composites.

---

### 2. The Two-Stage Action Protocol

To address free-form fragility without altering the underlying model weights or fine-tuning, we implemented the **Two-Stage Action Protocol** (`backend/tools/two_stage_protocol.py`):

```
                                 [User Query]
                                      │
                                      ▼
                      [Capability Toggle Verification]
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [Both Tools Disabled]                  [At Least One Tool Active]
                   │                                     │
                   ▼                                     ▼
             [Mode: DIRECT]                     [Stage A: Action Decision]
                   │                             - Evaluates conditioned intent
                   ▼                             - Resolves candidate actions
        [Stream Native Output]                   - Enforces permission guards
                                                         │
                                        ┌────────────────┴────────────────┐
                                        ▼                                 ▼
                                 [Action: DIRECT]              [Action: PYTHON or WEB]
                                        │                                 │
                                        ▼                                 ▼
                             [Stream Native Output]         [Stage B: Argument Validation]
                                                            - Extract code/query
                                                            - ast.parse verification
                                                            - Non-empty query check
                                                                          │
                                                         ┌────────────────┴────────────────┐
                                                         ▼                                 ▼
                                                   [Args Valid]                     [Args Malformed]
                                                         │                                 │
                                                         ▼                                 ▼
                                                [Sandbox Dispatch]                [Emit <|tool_error|>]
                                                         │                                 │
                                                         ▼                                 ▼
                                              [Capture Observation]              [Bounded Recovery]
                                                         │
                                                         ▼
                                            [Round 2 Model Synthesis]
```

#### Stage A: Action Determination
- If no tools are enabled $\to$ Action is strictly `DIRECT`.
- If only one tool is enabled $\to$ The model is evaluated under the isolated tool manifest. Under isolated manifests, Web Researcher accuracy jumps from 2% to 88.5%, eliminating cross-tool competition.
- If both tools are enabled $\to$ The model's intent token probability ($P(\text{tool}) = \text{probs}[6]$) is evaluated. If $P(\text{tool}) < 0.20$, the query is classified as `DIRECT`. If $P(\text{tool}) \ge 0.20$, tool selection evaluates conditioned log-probabilities.

#### Stage B: Argument Validation & Sanitization
1. **Python Validation:**
   - Arguments must contain non-empty `code`.
   - If JSON parser truncated outer quotes, extracts code from markdown blocks (` ```python ... ``` `).
   - Code is parsed through Python's Abstract Syntax Tree (`ast.parse(code)`). If a syntax error is detected, execution is halted, preventing sandbox crashes and returning a structured `<|tool_error|>` diagnostic for bounded recovery.
2. **Web Researcher Validation:**
   - Arguments must contain a non-empty `query` string.
   - Cleans leading/trailing escape characters and extraneous punctuation before dispatching to DuckDuckGo Lite.

---

### 3. Execution Verification & Result Interpretation

#### Bounded Closed-Loop Cycle:
1. **Round 1:** QORIN generates the action and validated arguments.
2. **Dispatch:** `ToolOrchestrator` executes the tool with isolated subprocess security limits (5.0s timeout, memory bounds, blocked dangerous modules `[socket, urllib, subprocess]`).
3. **Feedback:** The observation is formatted as `<|tool_result|>` (or `<|tool_error|>`) and appended to the active conversation history.
4. **Round 2:** QORIN ingests the observation and generates a natural language synthesis incorporating the computed output or cited URLs.
5. **Loop Termination:** Bounded to a maximum of 2 rounds per turn, preventing infinite looping or runaway tool recursion.
