# QORIN Stage 16A — Forensic Failure Case Analysis

**Date:** 2026-11-17  
**Document:** Edge-Case & Failure Mode Analysis  

---

## 1. Catalog of Repaired Failure Modes

### Failure Mode 1: Pseudo-Tag Leakage via Prompt Inversion
- **Manifestation:** Prompt instruction `Format: <|tool_call|>{"name": "python", "arguments": {"code": "<code_string>"}}` caused model to emit literal `<code_string>` as a code fence.
- **Vulnerability:** Downstream token streaming yielded the raw `<code_string>` to the web UI.
- **Remediation:** Removed placeholder tags from system prompt; implemented multi-pass `sanitize_user_visible_text()` to strip protocol markers before token emission.

### Failure Mode 2: Unwrapped / Bare JSON Tool-Calls
- **Manifestation:** Model emitted bare JSON `{"name": "web_research", "arguments": {"query": "..."}}` without surrounding `<|tool_call|>` or `<tool_call>` tags.
- **Vulnerability:** `two_stage_protocol.determine_action` evaluated `is_tool_call` as `False` because it only searched for `<tool_call>` or `<|tool_call|>`.
- **Remediation:** `OutputProtocolParser` supports bare JSON matching `{"name": "...", "arguments": {...}}`, regex-based name/argument recovery, and `strict=False` parsing.

### Failure Mode 3: RAG False-Positive Trigger on Natural Language Interrogatives
- **Manifestation:** Broad interrogative words (`who`, `what`, `where`, `when`) triggered internal engineering RAG for open-domain questions (e.g. "who is donald trump?").
- **Vulnerability:** Produced 0 matching technical chunks; the UI displayed a misleading `Retrieved 0 Knowledge Chunks` banner.
- **Remediation:** Restricted `FACTUAL_KEYWORDS` to technical architectural terminology (`parameters`, `layers`, `gqa`, `swiglu`, `rmsnorm`, `gpu`, etc.); suppressed UI banner when 0 chunks match.

### Failure Mode 4: Non-Existent Weather Tool
- **Manifestation:** When asked for weather in Ahmedabad, model attempted Python combinations `math.comb(3, 5)` or hallucinated outdated weather overviews.
- **Vulnerability:** No weather capability existed in the registry or manifest.
- **Remediation:** Integrated `RealWeatherProvider` using Open-Meteo geocoding and live forecast API; configured canonical `weather` tool with location geocoding and live meteorological observation.

### Failure Mode 5: Sandbox Exception Reporting
- **Manifestation:** Python runtime exceptions (e.g., `NameError: name 'math' is not defined`) were not fed back as error observations for model self-correction.
- **Remediation:** `PythonExecutionContract` captures runtime errors, returns structured `status="FAILED"`, and feeds `SchemaValidationError` / `RuntimeError` back into the context turn so the model never claims a successful calculation.

---

## 2. Residual Edge Cases & Mitigations

1. **High-Entropy Model Delimiter Drift:**
   If a model emits single-quoted dictionary syntax (`{'name': 'web_search'}`), `ast.literal_eval` fallback inside the parser safely recovers the structure without throwing unhandled exceptions.
2. **Offline Weather Network Timeout:**
   If the Open-Meteo external service experiences an outage or timeout, `RealWeatherProvider` returns `status="UNAVAILABLE"` with a structured message. The model is informed that weather cannot be retrieved and does not fabricate fake temperatures.
3. **Runaway Tool Invocation Loops:**
   Enforced hard boundaries: `max_tool_rounds = 4`, `max_schema_retries = 1`, and `global_timeout = 30.0s`.
