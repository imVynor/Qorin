# QORIN Failure Analysis & System Limitations Audit

**Audit Date:** November 17, 2026  
**Audited Target:** QORIN-500M Experiment C & Inference Harness  
**Authoritative Checkpoint:** `ckpt_experiment_c_final.pt` (SHA-256: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`)  
**Audit Status:** **DOCUMENTED & ANALYZED**

---

## 1. Executive Summary

A critical objective of the QORIN research campaign is maintaining complete scientific honesty regarding the capabilities and failure modes of small sub-1B language models. While Experiment C demonstrates successful learned tool intent (F1: 0.889–1.000) and functional tool-assisted execution, it is **not a generalized autonomous agent**. 

This document details the recurring failure patterns, architectural bottlenecks, and model limitations identified during the reliability audit.

---

## 2. Model-Level Failure Modes

### 2.1 Python Tool Selection Bias
- **Observed Behavior:** When both Web Researcher and Python Sandbox toggles are enabled simultaneously, QORIN-500M exhibits an empirical bias toward dispatching the Python tool, even for queries with factual or historical components.
- **Root Cause:** In the Stage 1 training mixture, authentic multi-turn Python trajectories were dense with explicit programming syntax and mathematical prompts, whereas Web Research examples were constrained by Wikipedia and DuckDuckGo response structures.
- **Remediation / Mitigation:** 
  - Independent capability toggles in `HarnessControls.tsx` allow users to isolate Web Researcher exclusively (`web=True, python=False`), which completely restores 100% Web intent recognition.

### 2.2 Semantic Algorithm Deficiencies (Naive Heuristics)
- **Observed Behavior:** When generating code for number-theoretic questions without explicit step-by-step guidance, the model occasionally defaults to a naive parity check rather than a true primality test:
  ```python
  # Observed erroneous model code for primality:
  def is_prime(n):
      return n % 2 != 0  # Erroneously marks 9, 15, 21, 91 as prime!
  ```
- **Significance:** This underscores why **execution success is not semantic success**. The Python sandbox executes the code without errors (exit code 0), but the returned answer is mathematically false.
- **Remediation / Mitigation:**
  - Prompt conditioning in `get_compact_tool_manifest` instructs the model to use standard libraries (e.g. `math.gcd`, `math.comb`).
  - The evaluation harness checks semantic ground truth, not process exit codes.

### 2.3 Prompt Template Sensitivity
- **Observed Behavior:** Tool-intent recall is highly sensitive to the exact phrasing of the capability manifest. If the system prompt header deviates from the fine-tuned pattern (`"You are ModernGPT-S, a helpful and precise AI assistant.\nYou have access to external tools:..."`), tool-intent recall degrades significantly.
- **Root Cause:** 500M parameter models have limited representational capacity and rely heavily on activation patterns established during instruction tuning.
- **Remediation / Mitigation:**
  - `backend/app.py` standardizes system prompt assembly via `get_compact_tool_manifest()`, ensuring deterministic prompt conditioning.

---

## 3. Harness & System-Level Limitations

### 3.1 Single-Token Stop Sequence Handling
- **Observed Behavior:** `backend/models/qorin_provider.py` evaluates stop sequences on individual decoded tokens (`any(s in decoded_text for s in stop_seqs)`).
- **Risk:** If a stop sequence spans across multiple BPE tokens (e.g. a multi-character sequence like ````\n` that tokenizes into separate IDs), the single-token check can fail to trigger until the entire sequence appears in a single token.
- **Current Safeguard:** In `tokenizer_32k`, `<|im_end|>` is an atomic single special token (`token_id = 32761`), so standard ChatML turn termination is robust.

### 3.2 Quadratic Decode Latency ($O(T^2)$ Complexity)
- **Observed Behavior:** `generate_stream` executes a full forward pass on `inp[:, -context_len:]` at every generation step:
  ```python
  logits = model(inp[:, -context_len:])[:, -1, :].clone()
  ```
- **Impact:** While acceptable for short outputs ($< 128$ tokens, achieving $> 65\text{ tokens/sec}$ on RTX 5060), decode latency grows quadratically with sequence length because past key-value states are recomputed rather than cached.
- **Planned Improvement:** Implement a static inference KV-cache in Phase 3.

### 3.3 Context Window Consumption in Multi-Turn Tool Loops
- **Observed Behavior:** In a 2048-token context window, multi-turn interactions (User Query $\to$ Model Tool Call $\to$ Tool Observation $\to$ Model Synthesis) consume tokens rapidly. Web research summaries containing 3 web snippets can consume 400–600 tokens.
- **Impact:** After two tool execution rounds, remaining context for final synthesis may be constrained.
- **Mitigation:**
  - Hard limit of `max_tool_rounds = 2` enforced in `backend/app.py`.
  - Web research snippets are capped at 3 sources with truncated snippets.
  - Automatic prompt history sliding window truncates oldest turns if context length is approached.

---

## 4. Summary Matrix of Limitations & Mitigations

| Failure Mode / Limitation | Severity | Manifestation | Mitigation in Current System |
| :--- | :--- | :--- | :--- |
| **Python Selection Bias** | Moderate | Prefers Python when both tools ON | Independent capability toggle controls |
| **Semantic Code Errors** | High | Generates parity heuristic for primes | Semantic evaluation assertions; AST validation |
| **Quadratic Decoding** | Low (Short sequences) | Decode latency scales with length | Restrict `max_new_tokens` to 256; single-model VRAM lock |
| **Context Exhaustion** | Moderate | Multi-turn tool traces fill 2048 ctx | `max_tool_rounds = 2`; snippet length limits |
| **Unauthorized Re-routing** | Resolved | Morphing disabled tools into enabled tools | Fixed in DEF-04 (`two_stage_protocol.py`) |
