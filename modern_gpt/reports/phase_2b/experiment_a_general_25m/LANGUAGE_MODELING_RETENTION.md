# ModernGPT-S Phase 2B — Language Modeling Retention Report

**Validation Dataset:** Held-out Pretraining Multi-Domain Shards (`modern_gpt/data_500m/final_5b/val`)  
**Sequences Evaluated:** 15 sequences of 2,048 tokens per domain (30,720 tokens per domain, 215,040 total tokens)

---

## 1. Domain-by-Domain Perplexity & Loss

| Pretraining Domain | Domain Weight | Base Pretrained Loss | Base Pretrained PPL | Experiment A Loss | Experiment A PPL | Retention Status |
|---|---:|---:|---:|---:|---:|---|
| **Web** | 0.400 | 3.3836 | 29.48 | **3.3699** | **29.08** | Fully Retained (+0.014 gain) |
| **Educational** | 0.150 | 2.3594 | 10.59 | **2.3389** | **10.37** | Fully Retained (+0.021 gain) |
| **Code** | 0.150 | 1.8604 | 6.43 | **1.8541** | **6.39** | Fully Retained (+0.006 gain) |
| **Wikipedia** | 0.075 | 3.2799 | 26.57 | **3.2666** | **26.22** | Fully Retained (+0.013 gain) |
| **Mathematics** | 0.075 | 2.8715 | 17.66 | **2.8583** | **17.43** | Fully Retained (+0.013 gain) |
| **Books** | 0.100 | 3.106 | 22.33 | **3.1014** | **22.23** | Fully Retained (+0.005 gain) |
| **Dialogue** | 0.050 | 2.0742 | 7.96 | **2.0442** | **7.72** | Fully Retained (+0.030 gain) |
| **Weighted Overall** | **1.000** | **2.8621** | **17.5** | **2.8487** | **17.26** | **Zero Degradation** |

---

## 2. Conclusion on Catastrophic Forgetting

Fine-tuning on 25M conversation tokens using a moderate peak learning rate of 2.5e-5 with cosine decay produced zero catastrophic forgetting across all pretraining domains. The foundational language model features remain intact.
