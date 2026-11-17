# ModernGPT-S 500M — Tool-Calling & Intent Recognition Report

**Document ID:** `REP-PHASE1D-12-TOOLS`  
**Classification:** Function Calling & Agentic Intent Benchmark  
**Benchmark:** 20 Held-Out Tool Selection & Intent Tasks (`phase1d_benchmark_data.py`)  
**Date:** November 13, 2026  

---

## 1. Executive Performance Results

The 20 tool-calling tasks tested whether the model distinguishes between requests requiring external computation (Python), real-time search (Web), current sensors (Weather), or standard internal knowledge:

- **Tool Intent Recognition Rate:** **`35.0%`** (7 / 20 tasks aligned with correct execution tool)
- **Tool Delimiters Emitted (`<|tool_call|>`):** **`0`** (0 / 20 tasks)
- **Explicit Code Invocation Rate:** **`40.0%`** (8 / 20 tasks emitted inline Python when prompted for calculation)

---

## 2. Analysis: Base Pretraining vs Supervised Tool Delimiters

### A. The Special Token Gap
The production 32K tokenizer defines `<|tool_call|>` (ID `6`) and `<|tool_result|>` (ID `7`). However, during Phase 1C pretraining on the 5B-token corpus, these tokens are never used as supervision targets (they were reserved for future tool-augmented datasets). Consequently, the base model has zero probability of spontaneously emitting `<|tool_call|>` delimiters.

### B. Emergent Intent Recognition
Despite the lack of formal tool tokens, the model displays clear semantic awareness of tool modalities:
- When prompted: *"Run python code to find the 20th Fibonacci number."*
  - The model immediately outputs `def fib(n): ...` and `print(fib(20))`.
- When asked: *"What is the capital of France?"*
  - The model does not attempt to invoke Python or search tools; it completes the answer directly as `"Paris"`.

### C. Recommendation for Tool Integration
1. Base model tool usage should be handled via the harness orchestrator (`backend/tools/orchestrator.py`), which uses the intent classifier and routes requests to sandboxed subprocesses.
2. In SFT v4, incorporate 2,000 synthetic `<|tool_call|>` training pairs to teach the model to natively generate structured JSON arguments between special delimiters.

