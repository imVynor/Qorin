# ModernGPT-S Phase 2A — Pilot Experiment Plan

**Status:** Ready for Execution (Awaiting User Sign-Off)  
**Trainer Script:** [train_sft_500m.py](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/scripts/train_sft_500m.py)

---

## 1. Candidate Experiment Definitions

### Experiment A: General Assistant SFT
- **Config:** [experiment_a_general.yaml](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/configs/sft/experiment_a_general.yaml)
- **Focus:** Multi-turn dialogue, conversational fluency, explanations, tutoring.
- **Pilot Token Budget:** 25,000,000 tokens (~1.1 hours wall-clock).
- **Optimizer:** 8-bit AdamW | LR: 2.5e-5 (cosine decay to 2.5e-6).

### Experiment B: General + Tool-Use SFT
- **Config:** [experiment_b_general_tools.yaml](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/configs/sft/experiment_b_general_tools.yaml)
- **Focus:** Experiment A capabilities + Tool selection, argument generation, error handling, negative controls.
- **Pilot Token Budget:** 35,000,000 tokens (~1.55 hours wall-clock).
- **Optimizer:** 8-bit AdamW | LR: 2.5e-5.

### Experiment C: Joint Multi-Domain Capability SFT
- **Config:** [experiment_c_joint_capability.yaml](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/configs/sft/experiment_c_joint_capability.yaml)
- **Focus:** Full unified fine-tuning across all 5 domains (General, Reasoning, Coding, Structured, Tool-Use).
- **Full Run Budget:** 137,164,498 tokens (1 full epoch, ~6.81 hours wall-clock).
- **Optimizer:** 8-bit AdamW | LR: 2.0e-5.

---

## 2. Recommendation & Staged Execution

1. **Stage 1 (Recommended First):** Launch **Experiment A Pilot** (25M tokens, 1.1 hrs).
   - Evaluate loss convergence and check distinct-n diversity on held-out chat probes.
2. **Stage 2:** Launch **Experiment B Pilot** (35M tokens, 1.55 hrs).
   - Evaluate tool selection accuracy and JSON argument validity.
3. **Stage 3:** Full SFT on **Experiment C** (1 full epoch, 6.81 hrs).
