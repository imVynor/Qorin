# QORIN Experiment E: Comprehensive Experimental Protocol & Ablation Plan

**Document**: `experimental_protocol.md`  
**Status**: Pre-Execution Experimental Protocol  
**Parent Checkpoint**: Frozen Experiment D Best Checkpoint (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388da...`)  
**Evaluation Standard**: Identical 150-task held-out Stage 5 Benchmark (`benchmark.jsonl`)

---

## 1. Lineage & Checkpoint Initialization

Experiment E is an incremental surgical fine-tuning phase that builds directly upon the activation successes of Experiment D. 

### 1.1 Checkpoint Integrity Verification
Before initializing any training branch, the parent checkpoint must be validated:
- **File**: `modern_gpt/experiments/phase2_experiment_d/stage_3_training/checkpoints/ckpt_experiment_d_best.pt`
- **Expected SHA-256**: `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`
- **Model Architecture**: 20 layers, $d_{model} = 1536$, 24 query heads, 4 KV heads, context length 2048, vocabulary 32768, 514,389,504 parameters.
- **Tokenizer**: Byte-level BPE tokenizer (`d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`).
- **Read-Only Invariance**: The parent checkpoint file will remain strictly read-only. Weights are loaded into memory and branched into separate Experiment E working directories.

---

## 2. Four-Branch Factorial Ablation Design

To isolate the individual and interaction effects of the two proposed interventions, Experiment E establishes a strict 4-branch factorial matrix. All branches share identical base architecture, decoding parameters, and evaluation harnesses.

```
                         [Parent Checkpoint: Exp D Step 425]
                                        │
        ┌───────────────────┬───────────┴───────────┬───────────────────┐
        ▼                   ▼                       ▼                   ▼
    Branch A            Branch B                Branch C            Branch D
[Frozen Control]   [Calibration Only]      [Synthesis Only]    [Combined E]
   (0 Steps)        (Corpus 1: 500 rec)     (Corpus 2: 500 rec) (Joint: 1,000 rec)
```

### Branch Specifications:

| Branch ID | Training Corpus | Target Parameter Subspace | Primary Objective |
| :--- | :--- | :--- | :--- |
| **Branch A (Baseline)** | None (Frozen Reference) | None | Provide frozen control baseline on the 150-task benchmark. |
| **Branch B (Calibration Only)** | Corpus 1 (500 records) | Stage 1 Decision Logits & Multi-Head Routing | Eliminate false-positive tool calls on direct conceptual queries. |
| **Branch C (Synthesis Only)** | Corpus 2 (500 records) | Stage 2 In-Context Cross-Attention & Output Projection | Maximize factual extraction fidelity from Structured Evidence Records. |
| **Branch D (Combined E)** | Joint Curriculum (1,000 records) | Joint Stage 1 & Stage 2 | Deliver the production-ready model with calibrated tool use and grounded synthesis. |

---

## 3. Optimization & Training Hyperparameters

All active branches (B, C, D) follow identical optimization kinematics:

| Hyperparameter | Value | Rationale |
| :--- | :---: | :--- |
| **Optimizer** | `AdamW8bit` (bitsandbytes) | Memory-efficient optimizer preserving VRAM headroom on RTX GPU. |
| **Base Learning Rate ($\eta_{max}$)** | $1.5 \times 10^{-5}$ | Conservative peak rate to prevent catastrophic forgetting of base weights. |
| **LR Scheduler** | Cosine Annealing with Warmup | Linear warmup for first 10% of steps, decaying to $1.5 \times 10^{-6}$. |
| **Weight Decay** | 0.05 | Prevents over-fitting on small, dense dialogue corpora. |
| **Betas** | $(\beta_1 = 0.9, \beta_2 = 0.95)$ | Standard transformer momentum values. |
| **Gradient Clipping** | 1.0 ($L_2$ norm) | Eliminates gradient spikes and numerical instability. |
| **Precision** | Mixed Precision (`bfloat16`) | Native Ada/Ampere tensor core throughput; prevents underflow. |
| **Batch Size per Device** | 1 sequence | Accommodates multi-turn context without memory thrashing. |
| **Gradient Accumulation** | 4 steps | Effective batch size = 4 sequences per optimizer step. |
| **Max Optimizer Steps** | 250 steps (Branches B/C), 400 steps (Branch D) | Sufficient for convergence on small high-quality curricula. |
| **Evaluation Frequency** | Every 25 optimizer steps | Tight monitoring for early stopping and loss divergence. |

---

## 4. Hardware Safety & Duty-Cycle Policy

To ensure hardware longevity and prevent thermal throttling during training runs:
1. **Maximum Continuous Training Window**: $\le 105$ minutes of active GPU compute.
2. **Mandatory Thermal Cooldown**: $\ge 15$ minutes of complete GPU idle before evaluation passes.
3. **Thermal Shutdown Threshold**: If GPU core temperature exceeds $80^\circ\text{C}$ or VRAM memory junction exceeds $90^\circ\text{C}$, the training loop immediately executes safe checkpoint serialization and terminates.

---

## 5. Early Stopping & Regression Safeguards

Training will terminate early if any of the following conditions are met:

### Criterion 1: Validation Loss Plateau
- Validation loss is evaluated every 25 optimizer steps on the held-out 100-record validation split.
- If validation loss does not achieve a new minimum for **3 consecutive evaluation points** (75 steps), training terminates.
- The best checkpoint corresponding to the minimum validation loss is automatically archived as `ckpt_branch_best.pt`.

### Criterion 2: Catastrophic Generalization Divergence
- If training loss continues to decline while validation loss increases by more than **$10.0\%$ over its recorded minimum**, training terminates immediately (overfitting tripwire).

### Criterion 3: Numerical Anomaly
- If loss is `NaN`, `Inf`, or gradient norm exceeds $5.0$, training aborts immediately.

---

## 6. Standardized Evaluation & Statistical Protocol

To guarantee scientific validity, all branches will be evaluated under strictly identical conditions:

### 6.1 Inference Settings (Frozen Across All Branches)
- **Decoding Strategy**: Deterministic Greedy Decoding (Temperature = 0.0, Top-p = 1.0).
- **Precision**: `torch.bfloat16` with `@torch.inference_mode()`.
- **Context Length**: 2,048 tokens.
- **Max Generation Budget**: Stage 1 = 128 tokens; Stage 2 = 160 tokens.
- **Harness**: QORIN Two-Stage Protocol (`two_stage_protocol.py`).

### 6.2 Primary Evaluation Metrics
1. **Unnecessary Tool-Call Rate**: False Positives / Total Direct-Expected Tasks (36 tasks).
2. **Web Selection Recall**: True Web Selections / Total Web-Expected Tasks (76 tasks).
3. **Python Selection Recall**: True Python Selections / Total Python-Expected Tasks (38 tasks).
4. **Evidence Grounding Rate**: Correct Entity & Number Extraction on Executed Web Tasks.
5. **End-to-End Semantic Pass Rate**: Strict programmatic criteria pass across all 150 tasks.

### 6.3 Paired Statistical Hypothesis Testing
- **McNemar's Paired Test (with Edwards continuity correction)**: Evaluates whether the shift in semantic passes between Branch A and Branch D is statistically significant ($p < 0.05$).
- **Wilson Score 95% Confidence Intervals**: Computed for every proportion metric.
- **Cohen's $h$ Effect Size**: Measures the magnitude of difference in success proportions.
