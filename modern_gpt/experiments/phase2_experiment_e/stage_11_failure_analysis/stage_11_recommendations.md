# QORIN Experiment E — Stage 11: Prioritized Engineering Recommendations for Experiment F

**Document Version:** 1.0.0  
**Date:** November 17, 2026  
**Auditor:** Autonomous QORIN Research Core  
**Scope:** Actionable, Evidence-Backed Experimental Design for Next-Generation Agent Training (Experiment F)

---

## 1. Executive Summary & Evaluation of Strategic Options

Based on the empirical evidence gathered across Stage 10 and the Stage 11 forensic audit, we evaluated five candidate research directions for Experiment F:

| Candidate Strategy | Addressed Failure Modes | Strategic Priority | Recommendation Status |
| :--- | :--- | :---: | :---: |
| **Option A: Multi-turn Web $\to$ Python SFT** | $0/15$ multi-step completion; Turn 1 premature Python bias; lack of Turn 2 tool calls. | **CRITICAL (P0)** | **APPROVED FOR CORE TASK** |
| **Option B: Production Multi-Turn State Orchestration** | 2-stage hard termination in `two_stage_protocol.py`; absence of tool-call IDs; DDG 4s timeouts. | **CRITICAL (P0)** | **APPROVED FOR CORE HARNESS** |
| **Option C: Unified Evidence Representation & Retrieval Cache** | Format mismatch between training cards and live text; 78.7% retrieval failure rate; missing citations. | **HIGH (P1)** | **APPROVED FOR INTEGRATION** |
| **Option D: Python API Schema Grounding** | 81 Python runtime exceptions from hallucinated APIs (`math.prime`, `math.dist.distance`). | **HIGH (P1)** | **APPROVED FOR DATASET** |
| **Option E: Comprehensive Integrated Pipeline (A + B + C + D)** | Simultaneously resolves model supervision, harness state, retrieval reliability, and API correctness. | **UNANIMOUS** | **FINAL RECOMMENDED PATH** |

---

## 2. Detailed Intervention Specifications

### Intervention 1: Multi-Turn Sequential Tool-Chaining SFT (Option A)

- **Target Failure:** Complete failure of autonomous multi-step reasoning ($0/15$ across all models). In 100% of historical training data (4,600 records across Exp D and E), sequential tool chaining was never demonstrated.
- **Justification:** Language models cannot learn dynamic inter-tool information handoffs without explicit sequential demonstration. The model must learn that compound queries requiring unobserved facts must trigger retrieval first, and that returned observations can be parameterized into a subsequent Python call.
- **Data Specification:**
  - Construct **500 multi-turn synthetic trajectories**:
    - **Turn 1 (Prompt):** Compound question requiring external fact + calculation.
    - **Turn 1 (Target):** Assistant emits `<|tool_call|>{"name": "web_research", "arguments": {"query": "<fact_query>"}}`.
    - **Turn 2 (Observation):** Tool returns authentic factual observation.
    - **Turn 2 (Target):** Assistant extracts retrieved numbers and emits `<|tool_call|>{"name": "python", "arguments": {"code": "res = ...; print(res)"}}`.
    - **Turn 3 (Observation):** Python sandbox returns calculation output.
    - **Turn 3 (Target):** Assistant synthesizes grounded natural language answer citing both the web fact and calculated result.
- **What Remains Frozen:** Base pre-trained transformer architecture, tokenizer vocabulary (32k), and the frozen 150-task generalization benchmark.
- **Target Success Metric:** Fully autonomous multi-step completion on the 15 held-out tasks $\ge 40.0\%$ ($6/15$), up from $0/15$.
- **Falsification Condition:** If models trained on 500 multi-turn trajectories continue to select Python prematurely on $>50\%$ of multi-step tasks, the hypothesis is falsified, proving that token-level attention capacity in a 500M architecture cannot disentangle conflicting lexical priors without separate routing heads.

---

### Intervention 2: Production Multi-Turn Orchestration & Retrieval Cache (Option B & C)

- **Target Failure:** 78.7% of all tool execution failures were caused by live DuckDuckGo rate limiting and 4.0s timeouts; production harness hard-terminates after a single round.
- **Justification:** Evaluating model reasoning through unauthenticated live web scraping introduces severe external noise that obscures actual model competence.
- **Harness Engineering Requirements:**
  1. **Deterministic Retrieval Cache:** Implement a local SQLite/JSON document cache containing verified web snippets for all benchmark queries. Live web search is reserved for open queries, with local cache hit as primary fallback.
  2. **Full State Machine Migration:** Replace the 2-stage stateless `two_stage_protocol.py` with the asynchronous `MultiStepExecutionStateMachine` supporting up to 3 iterative tool turns.
  3. **Tool-Call ID Association:** Adopt standard schema with explicit call IDs (`call_001`) to support robust multi-tool trace tracking.
- **What Remains Frozen:** Benchmark tasks, prompt semantics, and evaluation scoring formulas.
- **Target Success Metric:** Tool execution success rate on valid dispatches increases from $23.16\%$ to $\ge 85.0\%$.
- **Falsification Condition:** If cached retrieval does not increase end-to-end semantic correctness by at least 10 percentage points across factual tasks, retrieval latency was not the primary bottleneck for factual synthesis.

---

### Intervention 3: Python API Knowledge Distillation (Option D)

- **Target Failure:** 81 Python runtime exceptions caused by hallucinated methods (`math.prime`, `math.factorial(x, y)`, `math.dist.distance`).
- **Justification:** Python sandbox errors occur because the 500M parameter model lacks deep memorization of standard library signatures. Providing canonical code snippets in the system prompt or training data eliminates trial-and-error syntax crashes.
- **Data Specification:**
  - Include 200 Python API demonstration records in the training distribution covering standard library signatures (`math.isqrt`, `math.comb`, `math.factorial`, `statistics.mean`, `datetime.date`).
- **Target Success Metric:** Python execution failure rate among dispatched valid calls drops from $34.7\%$ to $<10.0\%$.

---

## 3. Recommended Phased Implementation Plan for Experiment F

```
Phase Roadmap for Experiment F:

[Phase 1: Infrastructure & Cache]
  ├── Implement persistent SQLite retrieval cache for benchmark tasks.
  └── Upgrade backend orchestrator to 3-turn state machine with call IDs.

[Phase 2: Multi-Turn Dataset Construction]
  ├── Synthesize 500 multi-turn Web -> Python chaining trajectories.
  ├── Synthesize 200 Python standard library API demonstration records.
  └── Balance with 300 decision calibration records (Corpus 1) to protect FPR.

[Phase 3: Controlled Training (Experiment F)]
  ├── Match kinematic budget: 1,000 records, effective batch size 4, 250 updates.
  └── Strict early-stopping safeguards with validation loss plateau monitoring.

[Phase 4: Comparative Verification Audit]
  └── Benchmark against Branch D Terminal across all 7 evaluation dimensions.
```

*Status:* All engineering proposals are documented and ready for user authorization. Zero training or code modification has been initiated.
