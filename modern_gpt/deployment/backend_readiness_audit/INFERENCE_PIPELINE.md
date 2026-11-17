# Inference Pipeline Technical Audit

**Project:** QORIN (The Quiet Origin of Reasoning and Intelligence Networks)  
**Document:** `modern_gpt/deployment/backend_readiness_audit/INFERENCE_PIPELINE.md`  
**Audit Date:** November 17, 2026  
**Status:** Read-Only Audit Complete  

---

## 1. Step-by-Step Inference Execution Trace

Tracing an incoming request from the HTTP client to the final SSE streamed token:

```
[HTTP Client: Frontend / API]
       │  POST /api/chat/stream
       ▼
1. Request Validation & Compute Auth Check
       │  require_compute_auth() -> verify X-Compute-Token
       │  ChatRequest Pydantic parsing
       ▼
2. User Intent Classification & Retrieval Evaluation
       │  ToolPolicyClassifier.classify_intent()
       │  RetrievalEngine.retrieve() -> TF-IDF cosine ranking
       │  Event "retrieval" emitted to SSE stream
       ▼
3. Prompt Context Injection
       │  Inject retrieved chunks into messages
       │  Append Canonical Tool Manifest to system prompt
       ▼
4. Multi-Turn Orchestration State Machine (Round 0)
       │  Acquire Model Lock: async with qorin_provider._lock
       ▼
5. Tokenizer & ChatML Formatting
       │  HuggingFaceBPEWrapper maps messages to ChatML format:
       │  <|im_start|>role\ncontent<|im_end|>\n
       │  Tokenize to input_ids; truncate if > (context_len - max_new_tokens)
       ▼
6. Autoregressive Generation Loop
       │  Model: ModernGPT on CUDA (bfloat16) or CPU (float32)
       │  Forward pass: logits = model(inp[:, -context_len:])[:, -1, :]
       │  Apply repetition penalty (divide/multiply positive/negative logits)
       │  Sampling: Top-K filtering -> Top-P (nucleus) filtering -> Multinomial sampling
       │  Yield token chunk to EventSourceResponse (await asyncio.sleep(0))
       ▼
7. Generation Output Parsing
       │  OutputProtocolParser.parse_generation()
       │  Classify: DIRECT_RESPONSE vs TOOL_CALL vs MALFORMED_TOOL_CALL
       ▼
8. Tool Dispatch or Final Synthesis
       ├── IF TOOL_CALL:
       │     Emit "tool_start" SSE event
       │     ToolOrchestrator executes tool (Python subprocess or Web search)
       │     Emit "tool_result" SSE event
       │     Append tool observation to active_messages
       │     Increment round; loop back to Step 5
       └── IF DIRECT_RESPONSE:
             Apply EvidenceGroundingValidator
             Emit final token stream & "done" SSE event
             Log trace to backend/data/experiment_traces.jsonl
             Release Model Lock
```

---

## 2. Granular Step Analysis

### Step 1: Request Validation & Auth Check
- Pydantic validates payload against `ChatRequest`.
- `require_compute_auth(request, req.compute_token)` verifies `X-Compute-Token` or `Authorization` header.
- Rejects unauthenticated callers immediately with `401 Unauthorized` before any model evaluation.

### Step 2: Intent Classification & Retrieval
- `ToolPolicyClassifier.classify_intent(last_user_msg)` classifies intent into:
  - `CODE_GENERATION` (suppresses unnecessary execution).
  - `SIMPLE_ARITHMETIC` (answers directly without spawning sandbox).
  - `FACTUAL_INQUIRY` (enforces web retrieval).
  - `GENERAL_CONVERSATION`.
- `RetrievalEngine.retrieve()` computes TF-IDF similarity against `backend/data/knowledge_index.json`.

### Step 3: Prompt Formatting & Delimiters
- ChatML formatting enforces strict boundaries:
  - System: `<|im_start|>system\n{prompt}<|im_end|>\n`
  - User: `<|im_start|>user\n{prompt}<|im_end|>\n`
  - Assistant: `<|im_start|>assistant\n`
  - Tool Observation: `<|im_start|>tool name={name} id={id}\n{content}<|im_end|>\n`
- Special tokens are mapped using `HuggingFaceBPEWrapper` (BPE-16k for 100M, BPE-32k for 500M).

### Step 4: Tokenizer Loading
- Implemented in `modern_gpt/data/bpe_tokenizer.py` wrapping Hugging Face `tokenizers.Tokenizer`.
- Fast Rust-backed tokenization with special token handling (`<|im_start|>`, `<|im_end|>`).
- Loaded once per model selection into `qorin_provider.active_tokenizer`.

### Step 5: Checkpoint Loading & Architecture Initialization
- Checkpoints are loaded via standard `torch.load(ckpt_path, map_location="cpu", weights_only=False)`.
- Architecture parameters (`emb_dim`, `n_layers`, `n_heads`, `num_kv_heads`, `ffn_hidden_dim`, `rope_scale`) are extracted directly from checkpoint metadata or canonical config.
- Instantiates `ModernGPT(config)`.
- Moves to target device (`cuda` or `cpu`) with appropriate dtype (`torch.bfloat16` on CUDA, `torch.float32` on CPU).
- Calls `model.eval()`.

### Step 6: Autoregressive Generation & Sampling
- For each step up to `max_new_tokens`:
  1. Forward pass: `logits = model(inp[:, -context_len:])[:, -1, :].clone()`.
  2. Autocast context: `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)` when on CUDA.
  3. Repetition penalty:
     ```python
     if config.repetition_penalty != 1.0 and generated_tokens:
         for tid in set(generated_tokens):
             if logits[0, tid] > 0:
                 logits[0, tid] /= config.repetition_penalty
             else:
                 logits[0, tid] *= config.repetition_penalty
     ```
  4. Temperature scaling: `logits = logits / config.temperature`.
  5. Top-K truncation: removes logits outside top $K$.
  6. Top-P nucleus sampling: cumulative softmax filtering keeping smallest set exceeding probability $P$.
  7. Sampling: `next_token = torch.multinomial(probs, num_samples=1)`.
  8. Streaming yield: decodes token chunk, emits SSE packet, calls `await asyncio.sleep(0)` to prevent blocking the event loop.
  9. Stopping criteria: `token_id == eos_id` or decoded text matches stop sequences (`<|im_end|>`).

### Step 7: Output Protocol Parsing & Closed-Loop Tool Execution
- `OutputProtocolParser` parses model generation text for:
  ```xml
  <tool_call id="call_1234">{"name": "web_search", "arguments": {"query": "..."}}</tool_call>
  ```
- If a valid tool call is found:
  - Generates `CanonicalToolCall`.
  - Dispatches execution via `tool_orchestrator.execute_tool()`.
  - Python sandbox or web researcher executes and returns `CanonicalToolObservation`.
  - Appends observation to conversation history with role `tool`.
  - Injects Grounded Synthesis instruction:
    `"Answer the user's question directly using ONLY the retrieved sources provided above."`
  - Re-invokes generation for the final synthesis turn.

---

## 3. Supported vs Unsupported Capabilities Matrix

Only mark a capability as supported if the code or tests demonstrate it:

| Capability | Status | Evidence / Implementation Details |
| :--- | :--- | :--- |
| **GPU Inference** | **SUPPORTED** | Implemented via `torch.cuda` with `autocast(dtype=torch.bfloat16)`. Verified on local NVIDIA RTX 5060 Laptop GPU (8GB VRAM). |
| **CPU Inference** | **SUPPORTED** | Implemented as fallback when CUDA unavailable or when `device="cpu"` is passed. Uses `torch.float32`. |
| **BF16 Precision** | **SUPPORTED** | Default CUDA precision. Generation loop runs under `torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)`. |
| **FP16 Precision** | **NOT CONFIGURED** | The code explicitly uses `bfloat16` on CUDA and `float32` on CPU. Standard `float16` is not configured. |
| **FP32 Precision** | **SUPPORTED** | Used automatically on CPU devices (`ModernGPT(config).to(device=target_device, dtype=torch.float32)`). |
| **Weight Quantization** | **NOT SUPPORTED** | Native `QorinModelProvider` loads raw unquantized FP32/BF16 PyTorch state dicts. While `bitsandbytes` is installed in `.venv`, 4-bit / 8-bit NF4 weight quantization is not implemented in `ModernGPT`. (External Ollama provider supports GGUF/quantized models, but native QORIN does not). |
| **Multiple Simultaneous Requests** | **SEQUENTIALIZED** | `QorinModelProvider` contains an explicit `self._lock = asyncio.Lock()`. Multiple concurrent requests are queued and processed sequentially, preventing GPU out-of-memory errors but limiting concurrent throughput. |
| **Multiple Conversations** | **SUPPORTED** | Conversations are stateless on the model side; client passes full `messages` history along with unique `conversation_id`. Active state machine context is request-isolated. |
| **Request Cancellation** | **SUPPORTED** | Frontend abort triggers `AbortSignal` -> closes SSE connection -> Starlette terminates `EventSourceResponse` generator -> generation loop terminates and lock is cleanly released. |
| **Model Unloading & Reloading** | **SUPPORTED** | Explicitly supported via `POST /api/models/unload` and `POST /api/models/select`. Deletes model tensors, calls `gc.collect()`, and executes `torch.cuda.empty_cache()`. |
| **Dynamic Batching** | **NOT SUPPORTED** | There is no continuous batching or paged attention engine (e.g. vLLM). Batch size is strictly 1 during inference. |
| **KV-Cache Optimization** | **NOT IMPLEMENTED** | In `qorin_provider.py:380`, the model performs a full context forward pass at every token step: `logits = model(inp[:, -context_len:])[:, -1, :]`. Past KV states are not cached between autoregressive steps. |
