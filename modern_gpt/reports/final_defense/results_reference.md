# QORIN Results Reference & Metric Provenance

**Audit Date:** November 17, 2026  
**Subject:** Authoritative Numerical Results, Denominators, & Source Artifacts  
**Authoritative Checkpoint:** `ckpt_experiment_c_final.pt` (SHA-256: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`)  
**Status:** **CERTIFIED & VERIFIED**

---

## 1. Architectural Metrics & Parameters

| Metric Name | Authoritative Value | Calculation / Derivation | Source Artifact |
| :--- | :--- | :--- | :--- |
| **Unique Trainable Parameters** | **514,389,504** | $32768 \times 1536 + 20 \times 23202816 + 1536$ (Tied LM head) | `checkpoint_registry.json` |
| **Untied Parameter Sum** | **564,721,152** | $514389504 + 32768 \times 1536$ | `architecture_audit.md` |
| **Raw Tensor Elements in Checkpoint** | **569,964,032** | $564721152 + 5242880$ (40 RoPE buffers) | `checkpoint_registry.json` |
| **Transformer Blocks / Layers** | **20** | `blocks.0` through `blocks.19` | `architecture_audit.md` |
| **Attention Query Heads** | **24** | $24 \times 64 = 1536$ hidden dimension | `canonical_model_manifest.json` |
| **Attention Key-Value Heads** | **4** | $4 \times 64 = 256$ (Grouped-Query Attention 6:1) | `canonical_model_manifest.json` |
| **Context Length** | **2048 tokens** | RoPE buffers: `[1, 1, 2048, 64]` | `checkpoint_registry.json` |
| **Vocabulary Size** | **32,768** | Byte-Pair Encoding (`tokenizer_32k`) | `canonical_model_manifest.json` |

---

## 2. Checkpoint Lineage & Pretraining Milestones

| Checkpoint Name | Tokens Processed | Loss | PPL | Verified SHA-256 | Source Checkpoint File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Pretraining Base (Pilot)** | 252M | 3.12 | 22.65 | `28ac47b07e9616967d6a2ea0458e68432682085a2e05fba6ac2e55b82b6d639d` | `modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt` |
| **Experiment A (General SFT)** | 25M | 1.62 | 5.05 | `249fb467a9cf76e732890cc0501ec39a443756cb2f1c86c66402960f4abbadc4` | `modern_gpt/experiments/phase_2b/.../ckpt_pilot_final.pt` |
| **Experiment B (General + Schemas)** | 35M | 1.49 | 4.43 | `acbecc699da76387de9ef6d6a40a64ad73b324b6eee2bd1842a6fd9ab1972c4f` | `modern_gpt/checkpoints/sft_experiment_b/ckpt_experiment_b_final.pt` |
| **Experiment C (Two-Tool SFT)** | 1.885M | 1.17 | 3.22 | `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b` | `modern_gpt/experiments/phase2_experiment_c/.../ckpt_experiment_c_final.pt` |

---

## 3. Experiment B vs. Experiment C Comparative Metrics

| Capability Dimension | Experiment B (35M General) | Experiment C (1.885M Targeted SFT) | Exact Denominator | Source Evaluation File |
| :--- | :--- | :--- | :--- | :--- |
| **Coding Syntax Validity** | 92.0% | **96.0%** | 50 test prompts | `final_evaluation_audit.md` |
| **Coding Pass@1 Accuracy** | 42.0% | **38.0%** | 50 test prompts | `final_evaluation_audit.md` |
| **Pretraining Retention Loss** | 2.89 | **2.8651** | 7 domain held-out test splits | `final_evaluation_audit.md` |
| **Autonomous Tool-Intent Recall** | **0.0%** (0/25) | **100.0%** (20/20) | Tool-intent prompt subset | `end_to_end_evaluation.json` |
| **Autonomous Tool-Intent Precision** | **0.0%** | **100.0%** (20/20) | All emitted tool calls | `end_to_end_evaluation.json` |
| **Tool-Intent F1-Score** | **0.000** | **1.0000** | Harmonic mean of P & R | `end_to_end_evaluation.json` |
| **End-to-End Task Success** | 0.0% | **96.0%** (24/25) | 25 deterministic benchmark tasks | `end_to_end_evaluation.json` |

---

## 4. Benchmark Denominators & Disaggregation

### Primary Benchmark (25 Tasks)
- **Total Tasks Evaluated:** 25
- **Tool-Intent Queries:** 20 (Expected Intent = `True`)
- **Negative Control Queries:** 5 (Expected Intent = `False`)
- **Confusion Matrix:**
  - $\text{TP} = 20$ (True Positives: Model called tool when expected)
  - $\text{FP} = 0$ (False Positives: Model called tool on negative controls)
  - $\text{TN} = 5$ (True Negatives: Model gave direct response on negative controls)
  - $\text{FN} = 0$ (False Negatives: Model failed to call tool when expected)
- **Syntax Valid Calls:** 18 / 20 = **90.0%** (2 deliberate syntax/empty query probes safely caught)
- **Dispatched Calls:** 18 / 20 = **90.0%** (Invalid calls rejected)
- **Executed Calls:** 16 / 18 = **88.89%** (2 intentional error/timeout probes caught)
- **End-to-End Semantic Success:** 24 / 25 = **96.0%**

### Additional Held-Out Blind Benchmark (25 Tasks)
- **Total Tasks Evaluated:** 25
- **Tool-Intent Queries:** 17
- **Negative Control Queries:** 8
- **Confusion Matrix:** $\text{TP} = 17, \text{FP} = 0, \text{TN} = 8, \text{FN} = 0$
- **Precision:** $17 / 17 = \mathbf{100.0\%}$
- **Recall:** $17 / 17 = \mathbf{100.0\%}$
- **F1 Score:** $\mathbf{1.0000}$
- **End-to-End Semantic Success:** 22 / 25 = **88.0%** (3 live web search rate-limit failures)
