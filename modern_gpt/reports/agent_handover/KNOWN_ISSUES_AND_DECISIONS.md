# ModernGPT-S 500M — Known Issues, Technical Debt & Decisions Register

**Authority:** Permanent Engineering Log  
**Applicability:** All Future Agents & Maintainers  

---

## 1. Locked Decisions (DO NOT CHANGE CASUALLY)

1. **Production Tokenizer Identity:**
   - Vocabulary size: exactly `32,768`.
   - File: `modern_gpt/artifacts/tokenizer_32k/tokenizer.json`.
   - SHA-256: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`.
   - Tokenizer Fingerprint: `3559c7435e1d9330`.
   - **Reason:** All 108 binary training and validation shards (5.02B tokens) are packed using this exact vocabulary. Modifying or retraining the tokenizer invalidates the entire 9.35 GB binary dataset and all trained checkpoints.
2. **Binary Shard Header Format:**
   - Fixed 80-byte header format.
   - Little-endian `uint16` tokens.
   - Any modification will corrupt data loading across `final_5b/`.
3. **Architecture Parameters (500M Track):**
   - 20 layers, 1536 embedding dimension, 24 query heads, 4 KV heads (6:1 GQA), SwiGLU 3840 intermediate dimension, RMSNorm, RoPE ($\theta = 10,000.0$), tied embeddings, linear biases completely disabled (`out_proj_bias=False`, `qkv_bias=False`).
   - Trainable parameters must evaluate to exactly `514,389,504`.
4. **Mandatory 1:45 / 15 GPU Duty Cycle:**
   - 105 minutes active training followed by 15 minutes mandatory cooling.
   - Thermal emergency override at $\ge 92^\circ\text{C}$.
   - **Reason:** Enforces thermal longevity on the laptop hardware. Bypassing this schedule risks thermal throttling, clock degradation, or hardware shutdown.
5. **Frozen 100M Baseline:**
   - `modern_gpt/archive/100m_baseline/` is cryptographically sealed under SHA-256 `60040fe90df04e765b49af0cc98418fcc269d500e4777a0bece1302062163a7a`.
   - Must never be overwritten or modified.

---

## 2. Known Limitations & Forensic Findings

1. **Pretrained Base Model Nature (Phase 1D State):**
   - At 501.68M tokens, the 500M model is an early-stage pretrained base model. It is an autoregressive next-token predictor, not an instruction-following assistant.
   - It has not undergone Supervised Fine-Tuning (SFT) or RLHF. It generates natural text continuations rather than direct answers to conversational prompts unless prompted as document completion.
2. **Context Extension vs Attention Retrieval Gap:**
   - As proven in the 100M 4K pilot, extending context length via RoPE linear scaling yields low cross-entropy validation loss, but passkey needle-in-a-haystack retrieval collapsed to 4.0%.
   - In 500M pretraining, context length is natively 2,048 tokens. Full passkey retrieval evaluations should be conducted at 1B and 2B token milestones.
3. **Conversational Starvation Sensitivity:**
   - The Phase 0 audit confirmed that conversational datasets with $< 0.1\%$ greeting density cause catastrophic conversational collapse. SFT v4 candidate datasets must maintain at least 7.0% natural greeting and multi-turn conversational repair density.

---

## 3. Technical Debt & Open Work Items

1. **Single-GPU Training Throughput:**
   - Training at ~5,054 tokens/sec on an RTX 5060 Laptop GPU yields ~18.2M tokens/hour of active training.
   - Reaching 1.0B tokens requires approximately 27 hours of active GPU time (~31 hours total with cooling cycles).
   - Reaching the full 5.0B token target requires ~275 hours of active compute (~13.7 days wall-clock time).
2. **Validation Shard Size Decimation:**
   - The validation set is 20,000,000 tokens (7 shards). Evaluating the entire 20M tokens during periodic training checkpoints takes ~65 minutes.
   - In `pretrain_500m.py`, `eval_sequences` is currently bounded to 200 sequences (409,600 tokens) to keep checkpoint pauses under 90 seconds. Full 20M validation should be reserved for major milestone checkpoints (e.g. 500M, 1B, 2.5B, 5B).

