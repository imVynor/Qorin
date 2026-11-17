# QORIN — Authoritative Results Provenance & Metric Audit

**Document**: `docs/RESULTS_PROVENANCE.md`  
**Status**: OFFICIALLY AUDITED & VERIFIED  
**Date**: 2026-11-17  

---

## 1. Provenance Architecture

This document establishes the end-to-end traceability of every numerical result published on the QORIN research website and GitHub repository. The chain of custody follows:

$$\text{Published Metric} \longrightarrow \text{Evaluation Report} \longrightarrow \text{Raw Execution JSONL} \longrightarrow \text{Dataset Split Hash} \longrightarrow \text{Checkpoint SHA-256}$$

---

## 2. Core Experimental Result Matrix

### 2.1 Controlled Tool-Augmentation Ablation (N = 100 Held-Out Tasks)

| Field | Condition A: Tools Disabled (Parametric Only) | Condition B: Tools Enabled (Autonomous Dispatch) | Net Delta ($\Delta$) | Statistical Test |
|---|---|---|---|---|
| **Overall Task Accuracy** | **43.0%** ($43 / 100$) | **64.0%** ($64 / 100$) | **+21.0%** (+48.8% relative) | **McNemar's $\chi^2 = 12.9032$** ($p = 0.000328$) |
| **95% Confidence Interval**| $[33.6\%, 52.9\%]$ | $[54.2\%, 72.7\%]$ | — | Wilson Score Interval |
| **Contingency Matrix** | $n_{00} = 31$ (Both Failed) | $n_{01} = 26$ (Tools Enabled Succeeded Only) | — | $n_{01} / n_{10} = 5.2$ Odds Ratio |
| | $n_{10} = 5$ (Parametric Succeeded Only) | $n_{11} = 38$ (Both Succeeded) | — | Continuity corrected $\chi^2 = 12.9032$ |
| **Corpus A (Multi-Turn, N=40)** | 25.0% ($10 / 40$) | **55.0%** ($22 / 40$) | **+30.0%** | Massive gain on multi-step math/web |
| **Corpus B (Calibration, N=30)** | 53.3% ($16 / 30$) | **66.7%** ($20 / 30$) | **+13.4%** | Selective tool dispatch |
| **Corpus C (Synthesis, N=30)** | 56.7% ($17 / 30$) | **73.3%** ($22 / 30$) | **+16.6%** | Grounded citation accuracy |
| **Evaluated Checkpoint** | `ckpt_pilot_best.pt` (`bb5b7091e951e49263e7b55e78b1f6837b5713ffb2f197c20366f39f3cd9b1e4`) | Same Checkpoint | Constant | Bit-for-bit identical weights |
| **Dataset Hash** | `test.jsonl` (`523ec4962c1de25a515fffc10649d7a06da720006a361540fddfd21c1fcab014`) | Same Dataset | Constant | N = 100 records |
| **Supporting Artifact** | `modern_gpt/tests/test_final_tool_ablation.py` | `artifacts/diagnostics/tool_ablation_results.json` | — | Verified in automated test suite |

---

### 2.2 Stage 15 Blind Capability Evaluation (Exp F Step 100 vs. Exp D Step 425 Parent)

| Evaluation Dimension | Parent (Exp D Step 425) | Candidate (Exp F Step 100) | Net Delta ($\Delta$) | Supporting Raw Record Path |
|---|---:|---:|---:|---|
| **Turn 1 Web Selection Acc** | 17.5% ($7 / 40$) | **70.0%** ($28 / 40$) | **+52.5%** | `modern_gpt/experiments/phase2_experiment_f/stage_15_blind_evaluation/per_corpus_metrics.json` |
| **Turn 2 Python Selection Acc** | 0.0% ($0 / 30$) | **50.0%** ($15 / 30$) | **+50.0%** | (Same file, lines 6-11 vs 48-53) |
| **Python Argument Validity** | 0.0% ($0 / 30$) | **43.3%** ($13 / 30$) | **+43.3%** | (Same file, lines 12-14 vs 54-56) |
| **Value Extraction Accuracy** | 0.0% ($0 / 30$) | **23.3%** ($7 / 30$) | **+23.3%** | (Same file, lines 15-17 vs 57-59) |
| **Python Execution Success Rate**| 0.0% ($0 / 30$) | **6.7%** ($2 / 30$) | **+6.7%** | (Same file, lines 18-20 vs 60-62) |
| **Numerical Answer Correctness** | 36.7% ($11 / 30$) | **70.0%** ($21 / 30$) | **+33.3%** | (Same file, lines 21-23 vs 63-65) |
| **Final Evidence Synthesis Acc** | 0.0% ($0 / 30$) | **53.3%** ($16 / 30$) | **+53.3%** | (Same file, lines 24-26 vs 66-68) |
| **End-to-End Trajectory Rate** | 0.0% ($0 / 30$) | **3.3%** ($1 / 30$) | **+3.3%** | (Same file, lines 27-29 vs 69-71) |
| **Direct Concept Answer Rate** | 87.5% ($7 / 8$) | **100.0%** ($8 / 8$) | **+12.5%** | (Corpus B calibration, lines 34-36 vs 76-78) |
| **Evidence Support Precision** | 33.3% ($5 / 15$) | **100.0%** ($15 / 15$) | **+66.7%** | (Corpus C synthesis, lines 40-42 vs 82-84) |
| **Citation Correctness Rate** | 0.0% ($0 / 15$) | **93.3%** ($14 / 15$) | **+93.3%** | (Corpus C synthesis, lines 43-45 vs 85-87) |
| **Raw Per-Record Traces** | `modern_gpt/experiments/phase2_experiment_f/stage_15_blind_evaluation/per_record_results.jsonl` (200 records: 100 parent, 100 candidate) | | | Complete generation & tool logs |

---

### 2.3 Four-Configuration Comparative Assessment

| Configuration | Checkpoint Verified | Training Budget Known | Evaluation Available | Comparable to Tool Ablation? | Status Label |
|---|---|---|---|---|---|
| **1. QORIN-100M Baseline Pretrained** | `best_val.pt` (`bded509c...`) | 1.5B tokens / 30k steps | Pretraining Val Loss ($1.564$) | **No** (Raw completion model; lacks ChatML instruction syntax) | **VERIFIED BASELINE** (Not evaluated under tool ablation protocol) |
| **2. QORIN-100M SFT v3** | `best_val.pt` (`d771ea9f...`) | 29M tokens / 8,764 steps | SFT Step 8D Suite (AST Valid 96.0%, Func Pass 38.0%, Retrieval Norm 96.0%) | **Partially** (Single-turn tool calls evaluated; multi-turn chaining not trained) | **VERIFIED SFT** (Historical single-turn benchmark) |
| **3. QORIN-500M SFT Exp F** | `ckpt_pilot_best.pt` (`bb5b7091...`) | 100k tokens / 100 steps | Stage 15 Blind Eval & Controlled Tool Ablation | **Yes** (Primary subject of controlled tool ablation protocol) | **VERIFIED PILOT** (Evaluated under full protocol) |
| **4. QORIN-500M Pretrain Pilot** | `ckpt_latest.pt` (`28ac47b0...`) | 252M tokens / 7,694 steps | Pretraining Val Loss ($2.14$) | **No** (Unconverged causal base model; lacks instruction tuning) | **VERIFIED UNCONVERGED** (Not evaluated under tool ablation protocol) |

> [!IMPORTANT]
> **Scientific Integrity Disclosure**: Configurations 1 and 4 are foundational base pretraining checkpoints. They cannot be evaluated on ChatML tool-dispatch benchmarks without instruction fine-tuning because they do not output structured delimiter tags (`<|tool_call|>`). Presenting them as "failing" tool dispatch would be scientifically unsound. They are properly labeled: *"Not evaluated under this protocol."*

---

## 3. Discrepancy Reconciliation Table

| Historical Document Claim | Audited Reality | Resolution & Provenance |
|---|---|---|
| "90 benchmark examples" | 100 total records in `test.jsonl` | Exactly 90 tasks evaluate core tool chaining and calculation ($30\text{ standard} + 30\text{ cal} + 30\text{ syn}$). 10 tasks evaluate non-chaining edge cases (early direct answer, tool failure). Both numbers are preserved and documented in `DATASET_REGISTRY.md`. |
| "150M parameter dense decoder" in Stage 14 header | 514M unique parameters | Copy-paste typographic error in Stage 14 report markdown header. Checkpoint config, layer dimensions ($n_l=20, d=1536$), and state dict verify exact 514M parameter count. Documented in `500M_SFT_EXPERIMENT_F.md`. |
| "Validation tokens: 11.2M vs 11.18M" | Exactly $11,184,395\text{ tokens}$ | Binary file `shards_250m/val.bin` is $22,368,790\text{ bytes}$ ($11,184,395\times 2\text{ bytes}$ per uint16 token). Rounding to 11.2M explains the apparent discrepancy. |
