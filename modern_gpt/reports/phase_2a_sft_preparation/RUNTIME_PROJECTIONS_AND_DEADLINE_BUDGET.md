# ModernGPT-S Phase 2A — Runtime Projections & 48-Hour Deadline Budget

**Remaining Milestone Budget:** ~48 Hours  
**Primary Engine Baseline:** Config A2 (Full SFT + 8-bit AdamW @ 6,285.4 tok/s)  
**Conservative Engine Baseline:** Config A (Full SFT + 8-bit AdamW + GradCkpt @ 5,136.5 tok/s)  
**Thermal Duty Cycle:** 1h 45m active work / 15m cooling cooldown (14.28% cooldown overhead)

---

## 1. Runtime Scaling Projections

| Experiment Scope | Target Tokens | Active GPU Hours (6,285 tok/s) | Mandatory Cooldown Breaks | Total Wall-Clock Time | Status vs 48h Deadline |
|---|---:|---:|---:|---:|---|
| **Pilot Fast-Check** | 5,000,000 | 0.22 hrs (13.3 mins) | 0 breaks | **13.3 minutes** | Negligible footprint |
| **Experiment A (Pilot)** | 25,000,000 | 1.10 hrs (66.3 mins) | 0 breaks | **1.10 hours** | Easily fits |
| **Experiment B (Pilot)** | 35,000,000 | 1.55 hrs (92.8 mins) | 0 breaks | **1.55 hours** | Easily fits |
| **Full SFT (1 Epoch)** | 137,164,498 | 6.06 hrs | 3 breaks (45 mins) | **6.81 hours** | ~14.2% of deadline |
| **Full SFT (2 Epochs)** | 274,328,996 | 12.12 hrs | 6 breaks (90 mins) | **13.62 hours** | ~28.4% of deadline |
| **LoRA SFT (1 Epoch)** | 137,164,498 | 4.25 hrs (@ 8,976 tok/s) | 2 breaks (30 mins) | **4.75 hours** | Extremely rapid |

---

## 2. 48-Hour Milestone Budget Allocation

```
+-------------------------------------------------------------+
| Total Available Window: 48.0 Hours                          |
+-------------------------------------------------------------+
| [■■■■■■■] Full SFT 1 Epoch: 6.81 Hours (14.2%)              |
| [■■] Capability Pilots (A & B): 2.65 Hours (5.5%)            |
| [■■■] Post-SFT Evaluation Suite: 3.50 Hours (7.3%)          |
| [■■] Intermediate Audits & Checkpoints: 2.00 Hours (4.2%)   |
| [□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□] Buffer: 33.04 Hours (68.8%)|
+-------------------------------------------------------------+
```

Over **33 hours of safety margin** remain to handle any unexpected divergence, hyperparameter retries, or extended evaluations.
