# ModernGPT-S 500M — Held-Out Validation Loss & Language Modeling Report

This evaluation benchmarks next-token prediction across the seven pretraining domains using exactly identical, paired validation sequences drawn from the held-out validation shards.

## 1. Executive Comparison: 501M Baseline vs Post-1B Milestones

| Evaluation Metric | 501M Checkpoint (`ckpt_501m`) | Best Val Checkpoint (`ckpt_best_val`) | Latest Post-1B (`ckpt_latest`) | Total Progress / Delta | Relative Improvement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tokens Seen** | 501,678,080 | 982,712,320 | **1,121,648,640** | **+619,970,560 tokens** | +123.6% pretraining volume |
| **Token-Weighted Loss** | 3.0590 | 2.8730 | **2.8414** | **-0.2176** | **7.11% reduction** |
| **Token-Weighted PPL** | 21.31 | 17.69 | **17.14** | **-4.17** | **4.17 points lower** |
| **Bits per Token (BPT)** | 4.4133 | 4.1448 | **4.0992** | **-0.3141** | Compression improved |
| **Macro-Average Loss** | 2.9056 | 2.7199 | **2.6903** | **-0.2153** | All-domain gain |
| **Macro Perplexity** | 18.28 | 15.18 | **14.74** | **-3.54** | Broad convergence |

## 2. Per-Domain Validation Breakdown

| Pretraining Domain | Domain Weight | 501M Loss | Latest Post-1B Loss | Delta Loss | % Improvement | 501M PPL | Latest Post-1B PPL | Tokens Evaluated |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Web** | 40.0% | 3.5702 | **3.3509** | -0.2193 | **+6.14%** | 35.52 | **28.53** | 102,400 |
| **Edu** | 15.0% | 2.6131 | **2.3604** | -0.2527 | **+9.67%** | 13.64 | **10.59** | 102,400 |
| **Code** | 15.0% | 2.0174 | **1.8334** | -0.1840 | **+9.12%** | 7.52 | **6.25** | 102,400 |
| **Wiki** | 7.5% | 3.5912 | **3.4014** | -0.1898 | **+5.29%** | 36.28 | **30.01** | 102,400 |
| **Math** | 7.5% | 2.9053 | **2.7044** | -0.2009 | **+6.91%** | 18.27 | **14.94** | 102,400 |
| **Books** | 10.0% | 3.3407 | **3.0985** | -0.2422 | **+7.25%** | 28.24 | **22.16** | 102,400 |
| **Dialogue** | 5.0% | 2.3015 | **2.0831** | -0.2184 | **+9.49%** | 9.99 | **8.03** | 102,400 |

## 3. Methodological Reliability Verification
- **Identical Sequence Pairing:** All three models were evaluated on the exact same token sequences, ensuring paired statistical comparisons with 0 variance from sampling noise.
- **Context Alignment:** Every sequence was evaluated at full 2,048-token context length using bfloat16 precision.
- **No Leakage:** Validation shards originate from the held-out `final_5b/val` directory, completely disjoint from training shards.
- **Conclusion:** Pretraining continuation from 501M to 1.12B tokens yielded consistent loss reductions across all seven domains without saturation or regression.