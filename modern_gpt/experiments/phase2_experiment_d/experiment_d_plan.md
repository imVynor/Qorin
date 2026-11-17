# QORIN Experiment D — Targeted Web Tool-Use Enhancement Plan

**Document Version**: `1.0.0`  
**Date**: 2026-11-17T02:55:00+05:30  
**Target Model**: `QORIN-500M SFT Experiment D`  
**Parent Model**: `QORIN-500M SFT Experiment C`  
**Parent Checkpoint**: `modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt`  
**Parent Checkpoint SHA-256**: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`  
**Production Tokenizer**: `bpe_32k_production` (`d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`)  

---

## 1. Objective and Scope

The goal of **Experiment D** is to remediate the critical intent triggering and multi-tool selection defects discovered in Experiment C:
1. **Colloquial Web Intent Recognition**: Train the neural weights to trigger `<|tool_call|>{"name": "web_research"}` when presented with natural, colloquial queries (`"grab info on"`, `"look into"`, `"who is"`, etc.) rather than only rigid formulaic strings (`"Search the web for: "`).
2. **Elimination of Multi-Tool Selection Bias**: Reduce the Python-to-Web ratio from **28.72:1** down to **0.61:1**, eliminating the model's severe bias toward selecting Python for factual queries.
3. **Multi-Step & Error-Recovery Orchestration**: Train closed-loop recovery when web searches return empty or malformed results, and train multi-step workflows (Web retrieval followed by Python computation).
4. **Preservation of Pretraining & Python Capabilities**: Maintain rigorous mathematical reasoning, code execution, and direct conversational fluency through balanced multi-task training without catastrophic forgetting.

---

## 2. Model Architecture & Tokenizer Specifications

This is an incremental SFT run strictly preserving the frozen architecture of Experiment C:

- **Base Architecture**: `QorinForCausalLM` / `ModernGPT` decoder-only Transformer
- **Layers**: 20 Transformer blocks
- **Hidden Dimension**: 1536
- **Query Attention Heads**: 24
- **Key-Value Heads (GQA)**: 4 (6:1 query-to-KV ratio)
- **Head Dimension**: 64
- **FFN Dimension**: 3840 (SwiGLU activation)
- **Context Length**: 2048 tokens
- **Vocabulary Size**: 32768
- **Embedding Tying**: `lm_head.weight` tied to `tok_emb.weight`
- **Unique Trainable Parameters**: 514,389,504
- **Positional Encoding**: Rotary Position Embeddings (RoPE), $\theta = 10000.0$
- **Normalization**: RMSNorm with $\epsilon = 10^{-5}$

---

## 3. Dataset Architecture (Stage 1 & Stage 2 Complete)

The Experiment D dataset has been generated, audited, and cryptographically frozen:

| Split | Records | Total Tokens | Assistant Tokens | SHA-256 Checksum |
|---|---|---|---|---|
| `train.jsonl` | 3,800 | 1,446,128 | 321,490 | `90aebbac462bbad01e3239a51ef4b2c33f9d49faf5d23172060d89ef0d66cd2a` |
| `val.jsonl` | 200 | 76,218 | 16,920 | `b7c10807fda6ac5553b1806afe2fb3e25df65949dfd256d13456bf4e3a1af8e6` |
| `dataset_manifest.json` | 1 | - | - | `df85327ed2acbe3066486f000405b67c01f479744d50182a67d39750df21be0c` |

### Category Distribution (4,000 Records Total):
1. **WEB Core (1,200 records / 30%)**: Colloquial requests, questions, explicit searches, and fact verifications grounded in verified encyclopedic fixtures with authentic URLs.
2. **PYTHON Core (700 records / 17.5%)**: Arithmetic, number theory, statistics, and algebraic calculations verifying execution and maintaining Experiment C skills.
3. **DIRECT Core (700 records / 17.5%)**: Conceptual CS, deep learning theory, algorithms, and general conversational interactions without tool calls.
4. **Selection Discrimination (500 records / 12.5%)**: Matched pairs contrasting external information lookups against pure calculations under the two-tool manifest.
5. **Multi-Step Orchestration (500 records / 12.5%)**: Two-step workflows (Web retrieval followed by Python data transformation and analysis).
6. **Error Recovery & Zero-Source Handling (400 records / 10%)**: Query rephrasing after 0 results returned, argument correction, and transparent acknowledgment of unavailable evidence.

---

## 4. Training Implementation & Hyperparameters (Stage 3 & 4)

### Proposed Optimizer and Schedule:
- **Optimizer**: 8-bit AdamW (`bitsandbytes.optim.AdamW8bit`)
  - $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$, Weight Decay = $0.05$
- **Loss Masking**: Assistant targets only (`labels = -100` for system, user, tool observation, and padding tokens).
- **Precision**: BF16 mixed precision (`torch.amp.autocast(dtype=torch.bfloat16)`)
- **Peak Learning Rate**: $1.2 \times 10^{-5}$
- **Min Learning Rate**: $1.2 \times 10^{-6}$ (10% of peak)
- **Warmup**: 25 steps (~5% of training steps)
- **Decay Schedule**: Cosine learning rate decay to `min_lr`
- **Batch Sizing**:
  - Micro-batch size: 1 sequence (up to 2048 tokens)
  - Gradient Accumulation Steps: 8
  - Effective Batch Size: 8 sequences (~12,000 tokens per optimizer step)
- **Total Training Steps**: ~475 updates for 1 full epoch over 3,800 train records.
- **Gradient Clipping**: Maximum norm 1.0
- **Gradient Checkpointing**: Enabled on all 20 Transformer blocks

---

## 5. Hardware Constraints & Thermal Safety

1. **Active GPU Budget**: Maximum 105 minutes continuous GPU load per training session.
2. **Thermal Monitoring**: Continuous duty-cycle monitoring via `nvidia-smi` logging temperature, power draw (W), and VRAM allocation.
3. **Emergency Safeguards**: Automatic throttle or pause if GPU junction temperature exceeds 78°C or power exceeds TDP limits.
4. **Lineage Preservation**: The parent checkpoint `ckpt_experiment_c_final.pt` is treated as read-only and immutable.

---

## 6. Pilot Phase Gate (Stage 3 Protocol)

Before authorizing full training:
- Execute a **50-step pilot run** on GPU.
- Verify:
  1. Initial validation loss matches Experiment C baseline ($\sim 1.95 - 2.10$).
  2. Loss decreases monotonically with stable gradient norms ($< 1.0$).
  3. No NaN/Inf loss values.
  4. Tool token generation is healthy.
  5. VRAM consumption remains within hardware envelope ($< 6.5$ GB on RTX 3060).
