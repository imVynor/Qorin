# QORIN Model Provider Runtime Audit

**Audit Date:** November 17, 2026  
**Target Module:** `backend/models/qorin_provider.py`  
**Provider Class:** `QorinModelProvider`  
**Audit Status:** **VERIFIED (Fully Functional with Documented Architectural Notes)**

---

## 1. Executive Summary

This audit evaluates `backend/models/qorin_provider.py`, the core inference engine powering the QORIN web harness, REST API, and Server-Sent Events (SSE) streaming infrastructure. The loader, configuration resolver, tensor device placement, ChatML formatting, and autoregressive generation loops were inspected against all 9 native QORIN checkpoints.

### Key Audit Findings:
1. **100% Checkpoint Load Success:** All 9 registered QORIN native models successfully instantiate and load their state dictionaries into memory without unhandled exceptions.
2. **Correct 500M Architecture Instantiation:** `ModelConfig.get_500m_config()` correctly instantiates the 20-layer, 24-head, 4-KV head, 2048-context architecture matching `ckpt_experiment_c_final.pt` and its lineage.
3. **Robust RoPE Support:** The provider correctly differentiates between base RoPE ($\text{scale}=1.0$) and extended linear RoPE ($\text{scale}=8.0$) for 4K context models (`qorin-100m-sft-v3` and `qorin-100m-4k-base`).
4. **Identified Operational Caveats:**
   - Single-token stop sequence evaluation: string-matching on individual decoded tokens can miss multi-token stop patterns if split across token boundaries.
   - Non-strict weight loading: `strict=False` in `load_state_dict` permits graceful loading across checkpoint variants but masks unexpected tensor omissions unless audited.
   - Autoregressive $O(T^2)$ recomputation: The generation loop currently evaluates a full forward pass without a persistent KV-cache, causing quadratic compute on long context responses.

---

## 2. Checkpoint Loading & Architecture Resolution

### 2.1 Model Instantiation Logic
In `QorinModelProvider.load_model(model_id, device=None)`:
```python
# Resolve config parameters
if meta.get("generation") == "500M":
    config = ModelConfig.get_500m_config(gradient_checkpointing=False)
else:
    # Non-500M dynamic configuration from checkpoint state
    cfg_dict = state.get("model_config", state.get("config", {}))
    ...
```

#### Analysis:
- For **500M generation models** (`qorin-500m-sft-exp-c`, `qorin-500m-sft-exp-b`, `qorin-500m-sft-exp-a`, and `qorin-500m-pretrain-pilot`), the provider invokes `ModelConfig.get_500m_config(gradient_checkpointing=False)`.
- This factory method returns:
  - `n_layers = 20`
  - `emb_dim = 1536`
  - `n_heads = 24` ($d_{\text{head}} = 64$)
  - `num_kv_heads = 4` (GQA 6:1 ratio)
  - `ffn_hidden_dim = 3840` (SwiGLU 2.5x)
  - `context_length = 2048`
  - `vocab_size = 32768`
  - `qkv_bias = False`, `out_proj_bias = False`
  - `rope_theta = 10000.0`, `rope_scale = 1.0`, `rope_type = "default"`
- **Verdict:** This configuration is 100% structurally identical to the state dictionaries of `ckpt_experiment_c_final.pt` and `ckpt_experiment_b_final.pt`.
- **Architectural Caveat:** By hardcoding `get_500m_config()`, the provider overrides any experimental parameters serialized inside `state['config']`. For current checkpoints this is safe, but best practice is to prioritize serialized `state['config']` when present.

### 2.2 Dynamic Configuration for Non-500M Models
For models of 16M, 50M, and 100M generations, the provider dynamically parses `state['model_config']` or `state['config']`:
- Correctly parses `num_kv_heads` vs `kv_heads`.
- Handles RoPE scale: if `context_length > 512`, defaults `rope_scale = 8.0` and `rope_type = "linear"`.
- This enables seamless execution of `qorin-100m-sft-v3` at 4,096 context length.

### 2.3 Weights Loading (`strict=False`)
- The provider invokes `model.load_state_dict(weights, strict=False)`.
- **Why it is necessary:** Training checkpoints frequently store RoPE persistent buffers (`rope.cos`, `rope.sin`) or explicit `lm_head.weight` tensors alongside tied embeddings. `strict=False` ensures these do not throw PyTorch runtime exceptions.
- **Risk:** If a damaged checkpoint missing an entire transformer block were loaded, `strict=False` would allow silent failure.
- **Verification:** During our audit, all 9 checkpoints were confirmed to have 0 missing structural layer keys.

---

## 3. Device Placement & Memory Management

### 3.1 VRAM Allocation & Cleanup
- Prior to loading any new model, `self.unload_model()` is strictly called:
  ```python
  def unload_model(self, model_id: Optional[str] = None) -> bool:
      if self.active_model is not None:
          del self.active_model
          del self.active_tokenizer
          del self.active_config
          self.active_model = None
          if torch.cuda.is_available():
              torch.cuda.empty_cache()
          gc.collect()
          return True
      return False
  ```
- **Single Active Model Policy:** Only one model resides in VRAM at any given time, preventing CUDA Out-of-Memory (OOM) errors on consumer GPUs (e.g. 8GB RTX 4070 Laptop).
- **Precision:** Automatically uses `torch.bfloat16` when targeting CUDA, and `torch.float32` when targeting CPU.

---

## 4. Generation Loop & Streaming Contract

### 4.1 ChatML Formatting
The provider formats all conversation turns into the canonical ChatML prompt structure:
```text
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
```
- Handles system prompt deduplication if already present in messages.
- Appends `<|im_start|>assistant\n` as the autoregressive generation prefix.

### 4.2 Context Window Management
- `input_ids = input_ids[-(context_len - max_new):]`
- Strictly truncates prompt history from the left (sliding window) to ensure that the input sequence plus requested `max_new_tokens` never exceeds `context_length` (2048 for 500M, 4096 for 100M-4K).
- Prevents RoPE index overflow.

### 4.3 Sampling Implementation
The provider supports a standard generation pipeline:
1. **Greedy decoding:** Activated when $\text{temperature} \le 0.05$ via `torch.argmax`.
2. **Repetition Penalty:** Applied via multiplicative scaling to positive and negative logits of previously generated tokens.
3. **Top-K Truncation:** Clamps logits below the $k$-th highest value to $-\infty$.
4. **Top-P (Nucleus) Sampling:** Cumulative softmax probability mass thresholding.

### 4.4 Stop Token & Sequence Handling
```python
is_eos = (token_id == eos_id) or any(s in decoded_text for s in stop_seqs)
```
- **Potential Edge Case:** `decoded_text = tok.decode([token_id])` only decodes the single most recent token. If a multi-token stop sequence like `<|im_end|>` is split into sub-tokens by the BPE tokenizer, checking `s in decoded_text` on a single sub-token will not detect the sequence until the final sub-token arrives, or may fail if the sequence spans across tokens.
- **Mitigation in QORIN:** In `tokenizer_32k`, `<|im_end|>` is an atomic single-token special token (`token_id` = 32761), so it triggers `token_id == eos_id` reliably.

### 4.5 Tool Syntax Handling
- The provider does not alter or intercept tool calls during token emission; it emits `<|tool_call|>` and `<|tool_result|>` tokens transparently.
- Tool extraction, argument parsing, sandboxed execution, and multi-turn re-prompting are cleanly isolated in `backend/app.py` via `ToolOrchestrator`, maintaining proper architectural separation of concerns.

---

## 5. Verification Matrix of All 9 Registered Models

All 9 native checkpoints were empirically loaded via `QorinModelProvider.load_model()` on CPU:

| Model ID | Generation | Layers | Dim | Context | Provider Load Test | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `qorin-100m-sft-v3` | 100M | 14 | 768 | 4096 | **SUCCESS** (14L, 768D) | Verified |
| `qorin-100m-4k-base` | 100M | 14 | 768 | 4096 | **SUCCESS** (14L, 768D) | Verified |
| `qorin-100m-base` | 100M | 14 | 768 | 512 | **SUCCESS** (14L, 768D) | Verified |
| `qorin-50m-sft` | 50M | 12 | 512 | 512 | **SUCCESS** (12L, 512D) | Verified |
| `qorin-16m-sft` | 16M | 6 | 384 | 512 | **SUCCESS** (6L, 384D) | Verified |
| `qorin-500m-sft-exp-c` | 500M | 20 | 1536 | 2048 | **SUCCESS** (20L, 1536D) | Verified |
| `qorin-500m-sft-exp-b` | 500M | 20 | 1536 | 2048 | **SUCCESS** (20L, 1536D) | Verified |
| `qorin-500m-sft-exp-a` | 500M | 20 | 1536 | 2048 | **SUCCESS** (20L, 1536D) | Verified |
| `qorin-500m-pretrain-pilot` | 500M | 20 | 1536 | 2048 | **SUCCESS** (20L, 1536D) | Verified |

---

## 6. Recommendations for Future Hardening

1. **Prioritize Checkpoint Config Metadata:** Update line 228 to read `state.get('config', {})` for 500M models rather than unconditionally calling `get_500m_config()`, ensuring experimental flags are respected.
2. **Key Validation Warning:** Add explicit logging if `load_state_dict` encounters missing keys that are not RoPE buffers or tied heads.
3. **Multi-Token Stop Buffer:** Maintain a sliding string buffer of the last $N$ tokens to ensure multi-token stop sequences that cross token boundaries are caught before emission.
4. **KV-Cache Implementation:** Introduce an optional static KV-cache in `ModernGPT` for inference generation to reduce decode time from $O(T^2)$ to $O(T)$.
