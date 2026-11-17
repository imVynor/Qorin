# ModernGPT-S / QORIN — Master Experiment Registry

**Registry Scope:** Complete Experimental Lineage Across All Generations  
**Classification:** Completed, Partial, Forensic & Active Campaigns  

---

## 1. Summary Registry Table

| Experiment ID | Model Track | Target Tokens | Actual Tokens | Best Val Loss (PPL) | Status | Artifact Directory |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `EXP-PRETRAIN-5M` | PoC 5M | 5,000,000 | 5,000,000 | 6.0346 (417.6) | Completed | `checkpoints/exp_5m/` |
| `EXP-PRETRAIN-25M` | PoC 25M | 25,000,000 | 25,000,000 | 5.0837 (161.4) | Completed | `checkpoints/exp_25m/` |
| `EXP-PRETRAIN-16M` | 16M Baseline | 100,000,000 | 102,300,000 | 3.8615 (47.5) | Completed | `checkpoints/exp_100m/` |
| `EXP-SFT-16M-MIDSCALE-500` | 16M SFT Diagnostic | 500 tasks | 500 tasks | 1.8410 | Completed | `artifacts/diagnostics/midscale_500/` |
| `EXP-ABLATION-50M-GQA4` | 50M GQA-4 | 250,000,000 | 250,000,000 | 3.1637 (22.7) | Completed | `checkpoints/modern_gpts_v2/gqa4_250m/` |
| `EXP-ABLATION-50M-DEEPTHIN` | 50M Deep-Thin | 250,000,000 | 250,000,000 | 3.1687 (22.8) | Completed | `checkpoints/modern_gpts_v2/deep_thin_250m/` |
| `EXP-PRETRAIN-100M-BASE` | 100M Baseline | 1,500,000,000 | 1,499,987,968 | 2.7137 (15.1) | Completed | `archive/100m_baseline/checkpoints/` |
| `EXP-100M-4K-PILOT` | 100M 4K Pilot | 4,000 steps | 2,000 steps | 2.5830 | Completed | `checkpoints/modern_gpts_100m_4k_pilot/` |
| `EXP-100M-SFT-V1` | 100M SFT v1 | 10,000 records | 9,997 records | 1.4420 | Completed | `archive/100m_baseline/checkpoints/` |
| `EXP-100M-SFT-V2` | 100M SFT v2 | 30,000 records | 26,734 records | 1.2410 | Completed | `archive/100m_baseline/checkpoints/` |
| `EXP-100M-SFT-V3` | 100M SFT v3 | 35,000 records | 31,949 records | 0.7648 | Forensic | `archive/100m_baseline/checkpoints/` |
| `EXP-PHASE1A-LOSS-CORR` | Loss Correction | 10 test suites | 10 test suites | Passed (10/10) | Completed | `reports/phase1a_loss_correction/` |
| `EXP-PRETRAIN-500M-5B` | 500M Pretraining | 5,000,000,000 | **501,678,080** | **2.9464 (19.0)** | **Active (1D Milestone)** | `checkpoints/checkpoint_501M/` |

---

## 2. Detailed Experiment Records

### 1. `EXP-PRETRAIN-16M` (16M Proof-of-Concept Baseline)
- **Model:** ModernGPT-16M (6 layers, 384 hidden dimension, 6 heads, GELU/SwiGLU, 16.7M total parameters).
- **Corpus:** 102.3M tokens across Wikipedia and filtered web.
- **Key Outcome:** Confirmed sub-50M scaling dynamics. Validation loss reached 3.8615. Demonstrates proper syntax generation but high factual hallucination.

### 2. `EXP-SFT-16M-MIDSCALE-500` (The 500-Example Capability Experiment)
- **Objective:** Determine whether small language models can acquire algorithmic reasoning from small, clean supervision sets.
- **Dataset:** 500 examples strictly partitioned into 68 task families (400 train / 100 validation) covering atomic Python, raw JSON, and factual QA.
- **Outcome:** AST validity reached 77.5% on held-out tasks, but functional execution pass rate was **0.0%**. Proved that sub-50M models memorize syntactic surface structures without internalizing algorithmic state transitions.

### 3. `EXP-ABLATION-50M-GQA4` vs `EXP-ABLATION-50M-DEEPTHIN` (50M Iso-Data Architecture Ablation)
- **Objective:** Evaluate KV cache memory reduction and layer depth under identical 250M-token data budgets.
- **GQA-4 Variant:** 12 layers, 8 query heads, 4 KV heads (2:1 ratio). Best Val Loss: **3.1637**.
- **Deep-Thin Variant:** 16 layers, 7 query heads, 7 KV heads. Best Val Loss: **3.1687**.
- **Outcome:** GQA-4 halved KV cache memory with identical perplexity. GQA adopted as standard for all future scaling.

### 4. `EXP-PRETRAIN-100M-BASE` (100M Baseline Pretraining)
- **Model:** ModernGPT-S 100M GQA-4 (14 layers, 768 dim, 12 query heads, 4 KV heads, 100,401,408 params).
- **Corpus:** 1.5B tokens across 30 shards.
- **Outcome:** Baseline terminal loss of 2.7137 (PPL 15.08). Serves as the frozen reference point for all 500M comparisons.

### 5. `EXP-100M-4K-PILOT` (Context Length Extension)
- **Model:** 100M with RoPE linear frequency scaling ($s = 8.0$) extending context from 512 to 4,096 tokens.
- **Outcome:** Validation loss remained flat at 2.583. However, passkey needle-in-a-haystack retrieval collapsed to **4.0%** across 4K tokens, proving pretraining loss alone does not guarantee effective deep-context attention.

### 6. `EXP-100M-SFT-V3` (Conversational Template Collapse)
- **Dataset:** 31,949 instruction records.
- **Outcome:** Lowest training loss (0.7648), but severe conversational failure. Audit uncovered greeting starvation (0.026% greeting records), template domination (56.5% of tokens in 2 templates), and unweighted batch loss. Led directly to Phase 1A loss correction.

### 7. `EXP-PRETRAIN-500M-5B` (500M 5B-Token Pretraining Campaign)
- **Model:** ModernGPT-S 500M (20 layers, 1536 dim, 24 query heads, 4 KV heads [6:1 GQA], SwiGLU 3840, 514,389,504 trainable parameters).
- **Corpus:** 5,000,000,000 tokens across 101 binary shards in 7 balanced domains.
- **Current Milestone (Phase 1D):** Reached **501,678,080 tokens** at step 15,310 with best validation loss of **2.9464**.

