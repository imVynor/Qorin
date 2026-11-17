# QORIN Stage 12: Empirical Model Capability Evaluation on Stateful Multi-Turn Tasks

## 1. Executive Summary

A core objective of Stage 12 was to definitively determine whether the persistent failure on multi-step tasks (`Web Retrieval -> Python Calculation -> Final Grounded Synthesis`) stems from an **infrastructure deficit** (a harness incapable of mediating multi-turn interactions) or a **model inductive bias deficit** (models lacking multi-turn supervision).

To establish an unassailable empirical baseline:
1. **Harness Capability**: Rigorously verified via automated integration tests (`test_multi_turn_web_to_python_chaining` in `test_stage_12_infrastructure.py`). The harness successfully mediates the complete 9-state sequence, tracks tool call IDs (`call_001`, `call_002`), injects observations into conversational memory, enforces timeouts, and synthesizes answers. **Harness Pass Rate: 100% (14/14 tests passing)**.
2. **Model Capability**: Evaluated across the 15 held-out multi-step benchmark tasks using the frozen checkpoints `branch_d_terminal` (`ckpt_branch_d_step200.pt`) and `exp_d_parent` (`ckpt_experiment_d_best.pt`) operating under greedy decoding with deterministic fixture replay.

### Key Finding
**Neither model checkpoint completed a single autonomous multi-step tool chain (0/15, 0.0%).**
- The harness was ready and fully functional.
- The failure was 100% driven by the **model's prior**: having been trained exclusively on single-turn demonstrations (`User -> Tool -> Observation -> Final`), the models either prematurely dispatched Python on Turn 1 with hallucinated constants or, upon receiving web search results, immediately synthesized an answer without initiating Turn 2 Python calculations.

---

## 2. Comparative Evaluation Results

| Metric | Experiment D Parent (Step 425) | Branch D Terminal (Step 200) | Scripted / Mock Harness Tests |
|---|---:|---:|---:|
| **Checkpoint** | `ckpt_experiment_d_best.pt` | `ckpt_branch_d_step200.pt` | `MockFixtureProvider` |
| **Total Multi-Step Tasks** | 15 | 15 | N/A (Unit/Integration Suite) |
| **Turn 1 Tool Selection: Python** | **15 (100.0%)** | **12 (80.0%)** | Expected tool per test |
| **Turn 1 Tool Selection: Web** | **0 (0.0%)** | **2 (13.3%)** | Expected tool per test |
| **Turn 1 Tool Selection: Direct** | **0 (0.0%)** | **1 (6.7%)** | Expected tool per test |
| **Turn 2 Tool Transitions (Chaining)** | **0 (0.0%)** | **0 (0.0%)** | **100% (Completed Chain)** |
| **Full Chain Completion Rate** | **0.0% (0/15)** | **0.0% (0/15)** | **100.0% (Harness Verified)** |
| **Semantic Pass Rate** | **0.0% (0/15)** | **6.7% (1/15)** | **100.0%** |
| **Evaluation Wall Time** | 92.24s | 417.91s | 1.31s |

---

## 3. Forensic Analysis of Model Decision Dynamics

### A. Turn 1 Premature Python Selection (The "Calculation Priority" Bias)
In 27 out of 30 total model trials (15/15 for Exp D Parent, 12/15 for Branch D Terminal), the models immediately emitted a `python` tool call on Turn 1 despite the prompt requiring factual information that could only be retrieved from the web (e.g., *Task 01: FY2023 revenue of Apple and Microsoft*; *Task 02: Heights of Everest and K2*).

**Why this happens:**
In the training datasets of Phase 1, Phase 2 Exp D, and Exp E:
- Whenever prompts contained mathematical keywords (*"calculate"*, *"compute"*, *"difference"*, *"ratio"*, *"average"*), 100% of the training targets were Python calls.
- The training data contained **zero compound multi-intent examples** where an agent must first retrieve unknown parameters before computing.
- Consequently, the attention heads strongly attend to the computation token span and trigger Python execution with imaginary variable names (e.g., `p = 2020(2023)` or `meters_of_Mount_Everest`).

### B. Turn 2 Termination at Observation 1 (The "Immediate Synthesis" Bias)
In the two instances where `branch_d_terminal` successfully selected `web_research` on Turn 1 (`multistep_03`: France & Spain land area; `multistep_06`: Mercury & Mars orbital radius):
1. The state machine successfully validated the tool schema (`VALIDATE_TOOL_CALL`).
2. Dispatched the query to the provider (`EXECUTE_TOOL`).
3. Appended the structured observation into context (`APPEND_TOOL_OBSERVATION`).
4. Re-prompted the model with the expanded conversation history at `MODEL_FOLLOWUP_DECISION`.

**What the model did:**
Instead of emitting the expected Turn 2 call (`<tool_call>{"name": "python", ...}</tool_call>`), the model emitted natural language text:
> `"The web research yielded: fixture: ..."`

**Root Cause:**
In Stage 8 (Branch C) and Stage 9 (Branch D), every training record consisted of exactly:
`[User Query] -> [Assistant Tool Call] -> [Tool Observation] -> [Assistant Final Answer]`
The conditional probability distribution learned by the model is:
$$P(\text{Final Answer} \mid \text{Observation}) \approx 1.0, \quad P(\text{Tool Call} \mid \text{Observation}) \approx 0.0$$
The model has never been shown that a tool observation can be followed by an intermediate thought or a second tool call!

---

## 4. Disentangling Harness vs. Model: A Methodological Milestone

A critical error in earlier evaluations (Stages 5, 7, and 10) was conflating harness capability with model capability. By implementing an isolated, modular architecture in Stage 12, we establish strict separation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        STAGE 12 HARNESS STATUS                         │
├────────────────────────────────────────────────────────────────────────┤
│  ✓ 9-State Stateful Multi-Turn State Machine                           │
│  ✓ Explicit Tool-Call ID Tracking (call_001 -> observation)            │
│  ✓ SQLite Query Cache (<2ms hit time)                                  │
│  ✓ Resilient Provider Abstraction (8s timeout, backoff, retry)         │
│  ✓ Python AST Pre-Validation & Error Taxonomy                          │
│  ✓ Bounded Looping and Global Timeout Protection                       │
│  VERDICT: FULLY OPERATIONAL (14/14 Integration Tests Passing)          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Evaluates
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FROZEN CHECKPOINT STATUS                        │
├────────────────────────────────────────────────────────────────────────┤
│  ✗ Exp D Parent (Step 425): 0/15 Chains (100% Turn 1 Python)           │
│  ✗ Branch D Terminal (Step 200): 0/15 Chains (80% Python, 13% Web)     │
│  VERDICT: LACKS MULTI-TURN INDUCTIVE BIAS                              │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Blueprint for Stage 13 / Experiment F Training Intervention

The empirical findings from Stage 12 dictate the exact requirements for Experiment F training data:
1. **Multi-Turn Trajectory Supervision**: Include explicit multi-turn training records formatted with matching `call_id` tags:
   - Turn 1: User Request $\to$ Model Web Call $\to$ Observation.
   - Turn 2: Model Python Call $\to$ Observation.
   - Turn 3: Model Final Synthesis.
2. **Intent Deconfliction**: Include contrastive examples where mathematical keywords appear alongside unknown factual entities, penalizing premature Turn 1 Python invocations.
3. **Observation-Conditioned Policy**: Break the $P(\text{Final Answer} \mid \text{Observation}) \approx 1.0$ degenerate shortcut by training the model on both single-turn and multi-turn continuations.
