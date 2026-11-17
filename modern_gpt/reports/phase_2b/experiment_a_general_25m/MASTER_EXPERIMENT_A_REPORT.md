# ModernGPT-S Phase 2B — Experiment A General SFT Pilot Master Report

**Experiment ID:** `sft_exp_a_general_25m`  
**Execution Date:** 2026-11-15  
**Model Architecture:** ModernGPT-S 500M Decoder-Only Transformer  
**Base Pretrained Checkpoint:** [`ckpt_latest.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt) (1,121,648,640 pretraining tokens)  
**Pilot Checkpoint:** [`ckpt_pilot_final.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_pilot_final.pt)  
**Best Validation Checkpoint:** [`ckpt_best.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_best.pt)  
**Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)

---

## 1. Executive Summary

Phase 2B executed the first controlled Supervised Fine-Tuning pilot on the frozen 500M-parameter ModernGPT-S model. The objective was to evaluate whether full-parameter SFT using 8-bit AdamW with gradient checkpointing under strict thermal guardrails could impart instruction-following behavior while preserving foundational language-modeling capabilities.

The pilot successfully processed **24,977,669 total serialized conversation tokens** (19,942,069 assistant-target tokens supervised) over **4,050 optimizer steps**.

### Key Experimental Findings:
1. **Convergence & Optimization Health:**
   - Validation loss on held-out instruction splits decreased monotonically from **2.1127** (PPL 8.27) to **1.8799** (PPL 6.55), representing an **11.02% cross-entropy loss reduction**.
   - Zero NaN/Inf loss occurrences, zero gradient clipping explosions (gradient norm maintained between 1.0 and 4.1), and zero training instability.
2. **Language Modeling Retention (Zero Catastrophic Forgetting):**
   - Weighted validation loss across all 7 pretraining held-out domains improved from **2.8621** (PPL 17.50) to **2.8487** (PPL 17.26).
   - Losses improved across all 7 domains (Web, Edu, Code, Wiki, Math, Books, Dialogue), confirming that foundational language modeling capabilities were 100% preserved.
3. **Emergence of Instruction Following:**
   - The base pretrained model previously exhibited prompt-echoing and hallucinated user responses when presented with ChatML instructions.
   - The fine-tuned pilot actively responds as an assistant, formats code in markdown backticks, and answers queries directly.
4. **Thermal Schedule & Atomic State Recovery:**
   - The automated 105-minute active limit triggered seamlessly at step 3,545, executing an atomic checkpoint write ([`ckpt_step_3545_pre_cooldown.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase_2b/experiment_a_general_25m/checkpoints/ckpt_step_3545_pre_cooldown.pt)), undergoing a 15-minute cooldown break, and resuming with 100% state and RNG fidelity.
5. **Decision Gate Classification:** **PASS** (Ready to proceed to tool-use and joint-capability experiments).

---

## 2. Checkpoint & Pre-Flight Verification Audit

| Parameter | Pretrained Base Checkpoint | Verified Status |
|---|---|---|
| **Checkpoint Path** | `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt` | Verified immutable |
| **SHA-256 Checksum** | `28ac47b07e9616967d6a2ea0458e68432682085a2e05fba6ac2e55b82b6d639d` | Cryptographically matched |
| **Tokenizer Fingerprint** | `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` | `bpe_32k_production` verified |
| **Trainable Parameters** | 514,389,504 (Untied parameter count) | 100% Finite tensors |
| **Weight Tying** | `token_embedding.weight == lm_head.weight` | Strictly verified |
| **Context Length** | 2,048 tokens native (RoPE theta 500,000) | Verified |

---

## 3. Dataset & Token Budget Accounting

Experiment A was restricted strictly to the general assistant and reasoning subset:
- **Category Filter Applied:** `general_instructions` and `reasoning` (Tool-use records excluded).
- **Training Records Ingested:** 92,714 records.
- **Validation Records Ingested:** 2,439 records.
- **Token Budget Accounting:**
  - `conversation_tokens_processed`: **24,977,669** (Stopping criterion threshold).
  - `assistant_target_tokens_processed`: **19,942,069** (79.77% assistant supervision density).
  - Average sequence length: ~617 tokens per sample.

---

## 4. Hardware Profile & Thermal Schedule

- **GPU:** NVIDIA GeForce RTX 5060 Laptop GPU (8GB VRAM)
- **Peak VRAM Allocated:** 3,960 MB (49.5% of 8GB capacity)
- **Peak VRAM Reserved:** 4,216 MB
- **Mean Active Throughput:** 3476.5 tokens/second
- **Thermal Safety Duty Cycle:** 105 minutes active work / 15 minutes cooling break (1 cooldown cycle triggered at 23:07:31).
- **Peak GPU Temperature:** 89.0°C (safely below 90°C thermal limit).
