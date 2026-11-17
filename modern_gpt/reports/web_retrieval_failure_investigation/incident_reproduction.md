# Incident Reproduction Report: Live Web Retrieval Direct-Response Failure

**Incident ID**: `INC-20261117-WEB-RETRIEVAL-FAIL`  
**Timestamp**: 2026-11-17T02:23:45+05:30  
**Target Query**: `grab info on elon musk`  
**Investigator**: Antigravity Autonomous Systems Engineering Team  
**Evaluation Model**: `qorin-500m-sft-exp-c`  
**Checkpoint Path**: `modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt`  
**Checkpoint SHA-256**: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`  

---

## 1. Executive Summary

During testing of the live QORIN web harness, a user submitted the colloquial informational query:
```
grab info on elon musk
```
with both **Web retrieval** and **Python execution** enabled on the frozen `QORIN-500M Experiment C` checkpoint.

The application failed to execute web retrieval or provide citations. Instead, it returned a 221-token direct response containing severe factual hallucinations, asserting that Elon Musk was a "statistical tool" and "language model research assistant developed by the QORIN research project."

This incident was reproduced deterministically against the running live application server (`http://127.0.0.1:8000/api/chat/stream`) and model checkpoint. The complete Server-Sent Events (SSE) stream, raw model token outputs, and dispatch decision traces were captured and preserved.

---

## 2. Active Application and Runtime Configuration

The reproduction was executed under the exact parameters active in the live user demonstration:

| Parameter | Configuration Value | Verification Source |
|---|---|---|
| **Model ID** | `qorin-500m-sft-exp-c` | Model Manifest & Registry |
| **Model Architecture** | QorinForCausalLM (20 L, 1536 D, 24 QH, 4 KVH, 2048 Ctx) | Architecture Audit (514,389,504 params) |
| **Checkpoint SHA-256** | `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b` | Cryptographic SHA-256 verification |
| **Web Retrieval** | `True` (ON) | `req.web_research_enabled` |
| **Python Sandbox** | `True` (ON) | `req.python_sandbox_enabled` |
| **Manual Override** | `None` (Autonomous) | `req.manual_tool_override` |
| **Retrieval Mode** | `auto` (Threshold: 0.35, Top-K: 3) | Non-parametric RAG Engine |
| **Sampling Temperature** | `0.7` | `GenerationConfig` |
| **Top-p / Top-k** | `0.9` / `40` | `GenerationConfig` |
| **Max New Tokens** | `256` | `GenerationConfig` |
| **Repetition Penalty**| `1.15` | `GenerationConfig` |

---

## 3. End-to-End Execution Trace

### Step 1: Local Knowledge Retrieval (RAG)
The non-parametric retrieval engine evaluated the user prompt against the technical QORIN documentation corpus:
- **Query**: `"grab info on elon musk"`
- **Triggered**: `False`
- **Decision Reason**: `"Query did not match technical knowledge base routing threshold."`
- **Latency**: `0.01 ms`
- **Result**: No technical context chunks injected; request proceeded to tool-conditioned generation.

### Step 2: System Prompt Injection
Because both Web Retrieval and Python Sandbox were enabled, the backend injected the compact two-tool manifest:
```
You are QORIN, a helpful and precise AI assistant.
You have access to external tools:
1. `python`: Executes Python code in an isolated sandbox. Use for math, calculations, and data processing. Format: <|tool_call|>{"name": "python", "arguments": {"code": "<code_string>"}}
2. `web_research`: Searches the web for current factual information. Returns evidence with source URLs. Format: <|tool_call|>{"name": "web_research", "arguments": {"query": "<query_string>"}}
When a tool is needed, emit ONLY the tool call. If no tool is needed, respond directly to the user.
```

### Step 3: First-Stage Model Generation
The model (`qorin-500m-sft-exp-c`) began streaming tokens directly without emitting `<|tool_call|>`:
```
elon musk is a language model research assistant developed by the QORIN research project. The language model is a statistical tool that uses linear regression to predict the behavior of individual neurons in the brain. Elon Musk has been working with the QORIN data for over 30 years and has been trained on more than 400,000 neurons in the brain.
The QORIN platform is designed for large-scale analysis, including multi-layer classification, classification, and prediction. This language model is highly sensitive and can accurately predict a wide range of biological systems, including neurological disorders such as Alzheimer's disease, Alzheimer's disease, and multiple sclerosis.
The QORIN platform provides a powerful tool for predicting complex neural networks with high accuracy and efficiency. In this case, the QORIN machine learning model was developed using a combination of the QORIN model and the QORIN dataset. The QORIN model allows the QORIN dataset to be used for statistical analysis of biological systems, while the QORIN dataset provides a better understanding of complex systems with high performance.
```

### Step 4: Protocol Action Determination
`backend/tools/two_stage_protocol.py::determine_action`:
- Checked for `<|tool_call|>` or `<tool_call>` in output text: **Not Found**.
- Evaluated Action: `DIRECT`.
- Tool Candidate: `None`.
- Arguments: `None`.

### Step 5: Orchestrator Dispatch Decision
Because action was `DIRECT`:
- Tool dispatch: **Bypassed / Skipped**.
- `WebResearcher`: **Not Invoked**.
- `PythonSandbox`: **Not Invoked**.

### Step 6: Client Delivery & Frontend Display
- All 221 tokens were streamed directly to the frontend via SSE `event: token`.
- The final `event: done` payload was yielded:
  ```json
  {
    "trace_id": "a973f11b-7551-4742-955f-1623185b3157",
    "total_tokens": 24531,
    "ttft_ms": 116.59,
    "total_latency_ms": 4632.51,
    "tok_per_sec": 47.74,
    "retrieval_triggered": false,
    "tools_executed": 0
  }
  ```
- In `web/src/components/harness/ChatArea.tsx`, because `m.tool_traces` was empty, no inline tool trace cards or citation hyperlinks were displayed, presenting the user with an ungrounded hallucination.

---

## 4. Diagnostic Metrics

| Metric | Measured Value |
|---|---|
| **Total Turn Latency** | 4,632.51 ms |
| **Time to First Token (TTFT)** | 116.59 ms |
| **Generation Throughput** | 47.74 tokens/sec |
| **Tokens Generated** | 221 tokens |
| **Tools Executed** | 0 |
| **Web Retrieval Executed** | No |
