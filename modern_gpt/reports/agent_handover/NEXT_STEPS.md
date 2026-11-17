# ModernGPT-S 500M — Next Steps & Prioritized Research Roadmap

**Objective:** Actionable Milestone Roadmap Following Phase 1D Evaluation  
**Target:** Engineering Execution & Campaign Continuation  

---

## 1. Prioritized Roadmap Overview

```
Phase 1D (500M Tokens) [NOW]
  │
  ├── 1. Resume 500M Pretraining Campaign ──> Target 1.0B Tokens (~27 hours GPU compute)
  │     ├── Automated 105m / 15m duty cycle
  │     └── Milestone checkpoint at ckpt_1000m.pt
  │
  ├── 2. Prepare SFT v4 Instruction Dataset
  │     ├── Curated pool: 18,322 records (1,300 greetings + 600 multi-turn repairs)
  │     └── Global token-weighted loss training pipeline
  │
  ├── 3. Execute Phase 2A Evaluation at 1.0B Tokens
  │     ├── Measure validation loss trajectory (Target: loss < 2.50)
  │     └── Re-evaluate coding AST pass rate and mathematical reasoning
  │
  └── 4. Pilot SFT v4 on 500M-1B Checkpoint
        ├── Measure conversational transition from pure continuation to assistant
        └── Benchmark external retrieval augmentation vs parametric capacity
```

---

## 2. Immediate Actions

1. **Verify Checkpoint Preservation:**
   - Confirm `modern_gpt/checkpoints/checkpoint_501M/ckpt_501m_tokens_501678080.pt` remains preserved as the frozen 500M milestone.
2. **Resume Pretraining toward 1B Tokens:**
   - Command:
     ```powershell
     & ".\.venv\Scripts\python.exe" modern_gpt/scripts/pretrain_500m.py --resume modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt --tokens 1000000000
     ```
   - Monitor `modern_gpt/reports/phase1c_pretraining/training_telemetry.jsonl` for steady loss decrease and thermal health.
3. **Format SFT v4 Dataset for 32K Tokenizer:**
   - Tokenize `modern_gpt/data/sft_v4/curated_sft_v4_pool.jsonl` using `bpe_32k_production` tokenizer (`artifacts/tokenizer_32k/tokenizer.json`).

---

## 3. Medium-Term Milestones

### Milestone 1: 1.0B Training Tokens (Phase 2A)
- Estimated requirement: ~15,250 additional optimizer updates (~498M tokens).
- Expected validation loss: `2.40 – 2.55` (Perplexity: `11.0 – 12.8`).
- Compare directly against the 100M model at 1.5B tokens to determine if 500M parameter capacity overtakes 100M at 33% token exposure.

### Milestone 2: SFT v4 Conversational Tuning
- Fine-tune 500M (1B checkpoint) on the 18,322-record SFT v4 dataset using token-weighted loss.
- Compare with historical SFT v1, v2, and v3 to verify complete resolution of conversational starvation and template collapse.

### Milestone 3: End-to-End Retrieval Augmentation Evaluation
- Test whether external TF-IDF / vector retrieval on a 100M or 500M model matches or exceeds a 1B+ parameter closed-book model on factual QA and architectural reasoning benchmarks.

