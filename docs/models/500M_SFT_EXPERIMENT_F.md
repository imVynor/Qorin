# QORIN-500M SFT Experiment F — Checkpoint Lineage, Training Architecture & Capability Audit

**Document**: `docs/models/500M_SFT_EXPERIMENT_F.md`  
**Checkpoint**: `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt`  
**SHA-256**: `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4`  
**Status**: VERIFIED 100-UPDATE PILOT (Exploratory Proof-of-Concept; Non-Converged)  
**Parent Checkpoint**: Experiment D Step 425 (`ckpt_experiment_d_best.pt`, SHA-256: `c6b388dab89b5762c3ad451cae30b0d5c73447a4e07bba962c745b2cb6948be7`)

---

## 1. Original 500M Base Architecture

The QORIN 500M model family is a dense decoder-only Transformer engineered with modern architectural primitives:
- **Layers ($n_{\text{layers}}$)**: 20
- **Hidden Dimension ($d_{\text{model}}$)**: 1,536
- **Query Attention Heads ($n_q$)**: 24 (head dimension 64)
- **Key/Value Attention Heads ($n_{kv}$)**: 4 (Grouped-Query Attention with 6:1 sharing ratio)
- **Feed-Forward Network**: SwiGLU activation with intermediate dimension $d_{\text{ffn}} = 3,840$ ($2.5 \times d_{\text{model}}$)
- **Normalization**: Pre-layer RMSNorm with $\epsilon = 1.0 \times 10^{-6}$
- **Positional Encoding**: Rotary Position Embeddings (RoPE) with base frequency $\theta = 10,000.0$
- **Context Length**: 2,048 tokens
- **Vocabulary**: 32,768 tokens (Byte-Pair Encoding, `modern_gpt/artifacts/tokenizer_32k/tokenizer.json`)
- **Tied Embeddings**: True at runtime ($32,768 \times 1,536 = 50,331,648$ parameters tied between token embedding and output head)
- **Unique Parameters**: Exactly **514,389,504 parameters** (Non-embedding: 464,057,856 parameters).

---

## 2. Complete Checkpoint Lineage

The Experiment F pilot is the culmination of six sequential training phases:

```
[500M Causal Pretraining Pilot] (ckpt_latest.pt, ~252M tokens, Step 7,694)
            │
            ▼
[Experiment A: 25M General SFT Pilot] (ckpt_pilot_final.pt)
            │
            ▼
[Experiment B: 35M Multi-Domain SFT] (ckpt_experiment_b_final.pt)
            │
            ▼
[Experiment C: Targeted Two-Tool SFT] (ckpt_experiment_c_final.pt)
            │
            ▼
[Experiment D: Balanced Web + Python SFT] (ckpt_experiment_d_best.pt, Step 425)
            │
            ▼
[Experiment F: Multi-Turn Chaining Pilot] (ckpt_pilot_best.pt, Step 100)
```

1. **500M Base Pretraining**: Pretrained from scratch on 5B sharded tokens (`data_500m/final_5b/train/`) to initialize general language structure.
2. **Experiment A (General Pilot)**: 25M-token exploratory instruction-tuning run establishing baseline conversational syntax.
3. **Experiment B (35M Multi-Domain)**: Scaled SFT over reasoning, coding, and basic single-turn tool calls.
4. **Experiment C (Autonomous Two-Tool)**: Introduced structured tool schemas (`web_search`, `python_interpreter`) with `<|tool_call|>` ChatML delimiters.
5. **Experiment D (Balanced Multi-Task)**: Mitigated Python bias by balancing web search with python execution over 3,800 training records, converging at Step 425.
6. **Experiment E (Architectural Branch Studies)**: Diagnosed multi-turn calibration degradation, establishing that single-turn SFT fails when encountering sequential observations.
7. **Experiment F Stage 14 (Multi-Turn Chaining Pilot)**: Branched from Experiment D Step 425 to test explicit multi-turn sequential tool supervision (`User → Web Search → Observation → Python Execution → Observation → Grounded Synthesis`).

---

## 3. Why Experiment F Was Conducted

Previous experiments (Exp C and Exp D) demonstrated that while the 500M model could trigger single-turn tool calls (e.g., executing a python snippet or searching a query), it suffered from two critical architectural failures:
1. **Observation Blindness**: When presented with an external search observation, the model either repeated the search or immediately hallucinated an answer instead of using the numeric facts in subsequent computation.
2. **Sequential Chaining Failure**: The model could not autonomously execute a two-step pipeline (e.g. *Query Demographics &rarr; Parse Population &rarr; Compute Growth Rate via Python*).

Experiment F was designed to test whether supervised multi-turn trajectories could bridge this gap without causing catastrophic forgetting of calibration and synthesis.

---

## 4. Training Dataset: Stage 13 Composition & Provenance

Experiment F trained strictly on the **Stage 13 Multi-Turn Dataset** (`modern_gpt/experiments/phase2_experiment_f/stage_13_dataset/`):
- **Total Dataset Size**: Exactly 1,000 curated records (800 Train, 100 Validation, 100 Test).
- **Corpus A — Multi-Turn Sequential Tool Use (400 total / 320 train / 40 test)**:
  - 300 records of standard sequential `Web Search → Observation → Python Code → Execution Result → Grounded Synthesis`.
  - 35 records of web search sufficient for direct answer (testing early exit).
  - 35 records of web tool failure recovery (fallback handling).
  - 30 records of contradictory source resolution.
- **Corpus B — Decision Calibration (300 total / 240 train / 30 test)**:
  - 75 direct conceptual questions (no tools allowed).
  - 75 provided calculation tasks (direct Python execution).
  - 75 factual web lookups.
  - 75 contrastive pairs.
- **Corpus C — Evidence-Grounded Synthesis (300 total / 240 train / 30 test)**:
  - 150 full support cases with exact bracketed numeric citations.
  - 50 partial support cases.
  - 50 contradictory evidence cases.
  - 50 insufficient evidence cases.
- **Provenance & Hygiene**: All 437 code snippets in the dataset were validated against a sandboxed Python execution engine with zero timeouts. 100% of records passed decontamination audits (peak 3-gram Jaccard overlap < 0.333 against held-out benchmarks).

---

## 5. Training Protocol, Hyperparameters & Optimizer Configuration

- **Script**: `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/train_pilot_stage14.py`
- **Optimizer**: AdamW 8-bit (`bitsandbytes`) with $\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $0.1$
- **Learning Rate**: Peak $1.5 \times 10^{-5}$ with linear warmup for 20 updates, decaying to minimum $1.5 \times 10^{-6}$
- **Precision**: BF16 mixed precision
- **Batch Size**: Micro-batch size 1, gradient accumulation steps 4 (Effective batch size = 4)
- **Gradient Clipping**: Maximum norm 1.0
- **Loss Masking**: Cross-entropy computed **strictly on assistant target tokens** (`loss_mask = 1.0`); system prompts, user turns, and tool observations masked to 0.0.

---

## 6. Execution Budget & Step Count

- **Total Updates Executed**: Exactly **100 optimizer steps**.
- **Epochs Completed**: 0.5 epochs over the 800-record training set ($100 \times 4 = 400\text{ records processed}$).
- **Supervised Tokens Supervised**: ~102,400 assistant tokens.
- **Wall-Clock Time**: ~14.2 minutes on an NVIDIA GeForce RTX 5060 Laptop GPU (including periodic validation checks).

---

## 7. Pilot Nature & Non-Convergence Notice

> [!WARNING]
> **Exploratory Pilot Notice**: `ckpt_pilot_best.pt` was halted at Step 100 pursuant to the mandatory pilot safety gate. It is **NOT** a fully converged production model. The full training plan required 800 updates (4 epochs). While 100 updates were sufficient to demonstrate the emergence of multi-turn tool chaining, the model exhibits incomplete calibration and occasional format brittleness.

---

## 8. Validation Loss Trajectory

Validation loss was evaluated periodically across the 100 validation records:

| Step | Validation Loss | Perplexity (PPL) | Corpus A Loss | Corpus B Loss | Corpus C Loss | Notes |
|---|---:|---:|---:|---:|---:|---|
| **0** (Parent) | 3.0854 | 21.88 | 3.4210 | 2.8120 | 3.0231 | Frozen Exp D baseline |
| **20** | 1.8421 | 6.31 | 1.9540 | 1.7210 | 1.8512 | Rapid adaptation to ChatML delimiters |
| **50** | 1.2150 | 3.37 | 1.2840 | 1.1520 | 1.2090 | Emergence of Turn-2 Python generation |
| **80** | 0.9412 | 2.56 | 0.9850 | 0.8920 | 0.9466 | Numerical extraction stabilization |
| **100** (Best) | **0.8947** | **2.45** | **0.9320** | **0.8540** | **0.8981** | Lowest validation loss recorded |

---

## 9. Held-Out Benchmark Results (Stage 15 Blind Evaluation)

In Stage 15, `ckpt_pilot_best.pt` was evaluated against its parent (`ckpt_experiment_d_best.pt`) on the untouched 100-record test split:

| Evaluation Dimension | Parent (Exp D Step 425) | Candidate (Exp F Step 100) | Net Delta ($\Delta$) | Status |
|---|---:|---:|---:|---|
| **Turn 1 Web Selection Accuracy** | 17.5% (7/40) | **70.0% (28/40)** | **+52.5%** | Statistically Significant |
| **Turn 2 Python Selection Accuracy** | 0.0% (0/30) | **50.0% (15/30)** | **+50.0%** | Emergence of Chaining |
| **Python Argument Validity** | 0.0% (0/30) | **43.3% (13/30)** | **+43.3%** | Valid AST Syntax |
| **Observation Value Extraction** | 0.0% (0/30) | **23.3% (7/30)** | **+23.3%** | Extracted empirical search facts |
| **Evidence Synthesis Support Precision**| 33.3% (5/15) | **100.0% (15/15)** | **+66.7%** | Zero Hallucinated Claims |
| **Citation Correctness Rate** | 0.0% (0/15) | **93.3% (14/15)** | **+93.3%** | Exact Source Attribution |
| **End-to-End Trajectory Completion** | 0.0% (0/30) | **3.3% (1/30)** | **+3.3%** | Non-zero fully autonomous completion |

---

## 10. Observed Capabilities & Failure Modes

### Emerging Capabilities:
1. **Autonomous Chaining**: Successfully executed sequential web search &rarr; observation ingestion &rarr; Python AST generation &rarr; numerical calculation &rarr; citation-grounded final answer on selected domains (chemistry, economics).
2. **Direct Extraction**: Demonstrated parsing of numbers from raw search snippets (e.g. electrical conductivity of metals) and writing arithmetic expressions to compare them.
3. **Calibrated Restraint**: Maintained 0.0% false positive tool triggers on conceptual questions where internal parametric knowledge was sufficient.

### Persistent Failure Modes:
1. **Value Extraction Brittleness**: Extracted values with only 23.3% accuracy; frequently misidentified dates or population years when multiple numbers were present in search results.
2. **Syntax Formatting Flaws**: Under greedy decoding, 2 out of 15 generated Python snippets contained minor variable scope mismatches.
3. **Early Exit Reluctance**: Difficulty recognizing when a web search observation already answered the query directly without requiring Python.

---

## 11. Known Limitations & Intended Use Cases

- **Intended Use**: Research evaluation of multi-step tool orchestration, ablation studies on parameter scaling vs tool retrieval, and demonstration of multi-turn ChatML parsing.
- **Unintended Use**: Autonomous financial calculation, critical decision-making without human-in-the-loop review, or unconstrained open-domain conversation.
- **Hardware Profile**: Runs in CPU mode using standard PyTorch causal inference. Peak RAM usage during greedy decoding is ~2.1 GB.

---

## 12. Artifact Files & Cryptographic Verification

| Artifact | File Path | Size (bytes) | SHA-256 Hash |
|---|---|---:|---|
| **Checkpoint** | `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/checkpoints/ckpt_pilot_best.pt` | 2,078,616,611 | `bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4` |
| **Tokenizer Config** | `modern_gpt/artifacts/tokenizer_32k/tokenizer.json` | 2,367,882 | `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` |
| **Training Log** | `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/training_log.jsonl` | 84,204 | `96ae0fa4a9df3be32448ca3327f12e8ba1dbe1ba419f79ceaa9b47e852a32dbb` |
| **Validation Report**| `modern_gpt/experiments/phase2_experiment_f/stage_14_training_pilot/stage_14_pilot_report.md` | 18,944 | `94e8243be12188ab66ba727dfec96f3069b62dc73c3ad5bb956b9c9fec7c5d41` |

---

## 13. Integration into QORIN Tool Harness

In the QORIN backend (`backend/models/qorin_provider.py`), this model is registered as `qorin-500m-sft-exp-f`. It is lazily loaded upon first request, consumes ~2,100 MB of system RAM on CPU, supports chunked token streaming, and directly hooks into `backend/tools/` (Web Search, Python Execution Sandbox, and Weather) via standard ChatML event parsing.
