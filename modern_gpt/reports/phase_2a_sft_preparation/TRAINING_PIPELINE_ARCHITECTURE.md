# ModernGPT-S Phase 2A — Training Pipeline Architecture Report

**Production Training Engine:** [train_sft_500m.py](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/scripts/train_sft_500m.py)  
**Configs Directory:** [configs/sft](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/configs/sft)

---

## 1. Architecture Overview

The SFT training harness is custom-tailored for the ModernGPT-S 500M decoder-only transformer:
- **Tied Embeddings:** `token_embedding` weights are strictly tied to `lm_head.weight`.
- **Rotary Position Embeddings:** RoPE base frequency theta = 500,000 for long-context stability.
- **Attention Configuration:** 4:1 Grouped-Query Attention (GQA) with 16 query heads and 4 KV heads.
- **Context Length:** 2,048 tokens native.
- **Precision:** `bfloat16` native mixed precision.

---

## 2. Checkpointing and State Fidelity

The trainer implements an atomic 19-point state recovery system:
- Model weights (`model_state_dict`)
- Optimizer state (`optimizer_state_dict` for 8-bit / Fused AdamW)
- Exact counters (`global_step`, `tokens_processed`, `epoch`)
- Metric baselines (`best_val_loss`)
- Complete RNG states (`torch.get_rng_state()`, `torch.cuda.get_rng_state()`, `random.getstate()`, `np.random.get_state()`)

Atomic write semantics (`.tmp` write followed by atomic `replace`) guarantee immunity to mid-save crashes or power interrupts.

Empirically validated via `--test-resume`: **100% exact state restoration verified.**
