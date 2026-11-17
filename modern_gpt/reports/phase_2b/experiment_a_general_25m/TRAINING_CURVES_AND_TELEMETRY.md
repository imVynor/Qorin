# ModernGPT-S Phase 2B — Training Curves & Telemetry Analysis

**Experiment ID:** `sft_exp_a_general_25m`  
**Telemetry Source:** [`training_telemetry.csv`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase_2b/experiment_a_general_25m/logs/training_telemetry.csv)

---

## 1. Validation Loss Progression

Validation loss was evaluated periodically every 250 steps on 2,439 held-out validation samples:

| Step | Validation Loss | Validation Perplexity | Relative Delta vs Baseline |
|---|---:|---:|---:|
| **Baseline (Step 0)** | 2.1127 | 8.27 | 0.00% |
| **Step 250** | 2.0012 | 7.40 | -5.28% |
| **Step 500** | 1.9570 | 7.08 | -7.37% |
| **Step 1,000** | 1.9207 | 6.83 | -9.09% |
| **Step 1,500** | 1.9034 | 6.71 | -9.91% |
| **Step 2,000** | 1.8921 | 6.63 | -10.44% |
| **Step 2,500** | 1.8858 | 6.59 | -10.74% |
| **Step 3,000** | 1.8825 | 6.57 | -10.90% |
| **Step 3,500** | 1.8808 | 6.56 | -10.98% |
| **Step 3,750** | 1.8802 | 6.55 | -11.00% |
| **Step 4,000** | 1.8798 | 6.55 | -11.02% |
| **Final (Step 4,050)** | **1.8799** | **6.55** | **-11.02%** |

---

## 2. Learning Rate Schedule

The learning rate followed a linear warmup for 3% of steps followed by cosine annealing:
- **Base Learning Rate:** 2.50e-5
- **Minimum Learning Rate:** 2.50e-6
- **Warmup Peak:** Step 160 (2.50e-5)
- **Midpoint (Step 2,000):** 1.85e-5
- **Final Step (Step 4,050):** 5.30e-06

---

## 3. Stability & Optimization Metrics
- **Mean Gradient Norm:** 1.84 (Spikes capped strictly at 1.0 - 4.1).
- **Throughput:** Maintained 3,090 - 3,890 tok/s throughout active execution.
- **Memory Consumption:** Stable at ~3.96 GB VRAM with zero memory leaks.
