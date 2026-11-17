# QORIN Phase 2D: Experiment D — Autonomous Training Report

**Run Completion**: 2026-11-17 06:25:36  
**Final Status**: Completed full 475 updates  
**Early Stopping Triggered**: False  
**Total Steps Completed**: 475 / 475 (100.0% of planned curriculum)  
**Total Epochs**: 1.000 Epochs  

---

## 1. Best Model & Checkpoint Integrity

- **Best Validation Loss**: **0.0021** (Achieved at **Step 425**)
- **Best Checkpoint Path**: [`stage_3_training/checkpoints/ckpt_experiment_d_best.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt)
- **Best Checkpoint SHA-256**: `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`
- **Final Checkpoint Path**: [`stage_3_training/checkpoints/ckpt_experiment_d_final.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_final.pt)
- **Final Checkpoint SHA-256**: `55b31a7f320d01ed94ac64a74715bf4236366b9242600859ab0b5b1940bbd849`
- **Parent Checkpoint**: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b` (Immutable, 100% Unmutated)
- **Step 50 Pilot Checkpoint**: `51aaed9c35458b01ea076f2b45026d8e6e3d96ee5f636345119f61b90262a23f` (Preserved intact)

---

## 2. Validation Loss & Early Stopping Trajectory

- **Evaluations without Improvement**: 2 consecutive evaluations (Patience: 5)

| Step | Validation Loss | Status |
|---|---|---|
| 050 | 0.3048 | Baseline |
| 075 | 0.1402 |  |
| 100 | 0.0605 |  |
| 125 | 0.0207 |  |
| 150 | 0.0101 |  |
| 175 | 0.0068 |  |
| 200 | 0.0046 |  |
| 225 | 0.0042 |  |
| 250 | 0.0033 |  |
| 275 | 0.0031 |  |
| 300 | 0.0030 |  |
| 325 | 0.0027 |  |
| 350 | 0.0024 |  |
| 375 | 0.0022 |  |
| 400 | 0.0022 |  |
| 425 | 0.0021 | BEST |
| 450 | 0.0021 |  |
| 475 | 0.0021 |  |

---

## 3. Best Model Benchmark Funnel Performance (Step 425)

Evaluated on the authoritative 30-task held-out benchmark:

- **Intent Recognition Recall**: **100.0%** (24/24)
- **Intent Recognition Precision**: **92.3%** (24/26)
- **Intent F1 Score**: **96.0%**
- **Correct Tool Selection**: **70.8%** (17/24)
- **Valid Argument Schemas**: 14/30
- **Dispatch Success**: 14/30
- **Execution Success**: 13/30
- **End-to-End Semantic Success**: **30.0%** (9/30)

---

## 4. Hardware and Telemetry Summary

- **Total Wall-Clock Time**: 99.04 minutes
- **Peak GPU Temperature**: 83.0°C
- **Peak VRAM Allocated**: 7890.0 MiB / 8,151 MiB
- **Hardware Duty-Cycle Adherence**: 100% Compliant (15-min cooldown enforced)

---

## 5. Termination & Stopping Analysis

Completed full 475 updates
