# ModernGPT-S Phase 2C: Experiment B — Stage 0 Audit Report

**Date:** 2026-11-15  
**Experiment ID:** `sft_exp_b_general_tools_35m`  
**Target Milestone:** 35,000,000 Serialized Conversation Tokens  
**Pretrained Checkpoint:** [`ckpt_latest.pt`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt) (1,121,648,640 pretraining tokens)  
**Report Location:** [`reports/phase_2c/`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase_2c)

---

## 1. Executive Status & Critical Blockers

- **Stage Status:** **Stage 0 Complete — All Integrity Checks Passed**
- **Critical Blockers Identified:** **0 Blockers**
- **Readiness Verdict:** Ready for Stage 1 Preflight and Smoke Testing upon user review.

---

## 2. Checkpoint & Architecture Discrepancy Resolution

### Verification Summary
- **Checkpoint SHA-256:** `28ac47b07e9616967d6a2ea0458e68432682085a2e05fba6ac2e55b82b6d639d` (Exact match).
- **Tokenizer SHA-256:** `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` (Exact match).
- **Tokenizer Fingerprint:** `3559c7435e1d9330` (Exact match).
- **Unique Trainable Parameters:** `514,389,504` (All finite tensors).
- **Weight Tying:** Strictly verified (`token_embedding.weight == lm_head.weight`).
- **State Dict Keys:** 223 tensors loaded with zero missing and zero unexpected keys.

### Resolution of the Architecture Discrepancy
- **The Issue:** An earlier SFT preparation report text casually described "16 query heads, 4:1 GQA, and RoPE theta 500,000", conflicting with the originally approved 24 query heads.
- **The Finding:** We directly inspected:
  1. `ckpt_latest.pt['config']`
  2. The weight tensor shapes in `ckpt_latest.pt` (`query.weight` is 1536x1536 -> 1536/64 = **24 query heads**; `key.weight` is 256x1536 -> 256/64 = **4 KV heads**)
  3. `ModelConfig.get_500m_config()` in [`modern_gpt/model/config.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/model/config.py)
  4. The pretraining script [`pretrain_500m.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/scripts/pretrain_500m.py).
- **The Ground Truth:** All four sources strictly and authoritatively agree:
  - **Layers:** 20
  - **Hidden Dimension:** 1,536
  - **Query Heads:** 24
  - **KV Heads:** 4 (6:1 GQA ratio)
  - **Head Dimension:** 64
  - **SwiGLU FFN Dimension:** 3,840
  - **RoPE Base Theta:** `10,000.0`
  - **Context Length:** 2,048
  - **Vocab Size:** 32,768
- **Conclusion:** The mention of 16 heads / theta 500k was purely a documentation typo. The frozen checkpoint, model implementation, and training code are 100% synchronized on the 24-head architecture.

---

## 3. Experiment A Throughput Accounting Reconciliation

The user noted that Experiment A's reported numbers (25M tokens, 81 minutes, 3,110 tok/s) presented an arithmetic discrepancy, as 25M / (81 * 60) ≈ 5,144 tok/s.

We independently audited [`training_telemetry.csv`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase_2b/experiment_a_general_25m/logs/training_telemetry.csv) and reconciled the exact numerator and denominator:

| Metric | Measured Value | Explanatory Analysis |
|---|---|---|
| **Total Conversation Tokens** | 24,924,426 tokens | Delta from Step 10 to Step 4,050 |
| **Total Assistant Target Tokens** | 19,900,296 tokens | Supervised target tokens |
| **Total Wall-Clock Elapsed Time** | 8,012.6 s (133.54 min) | From first to last telemetry timestamp |
| **Automated Cooldown Break** | 918.0 s (15.30 min) | Between Step 3,540 and Step 3,550 |
| **Active Training Wall-Clock Time** | 7,094.7 s (118.24 min) | Total elapsed minus 15.3m cooldown |
| **Reported `tok_per_sec` (Script)** | **3,110.6 tok/s** | **24.9M tokens / 8,012.6 total elapsed seconds** |
| **Pure Active Conversation Throughput** | **3,513.1 tok/s** | **24.9M tokens / 7,094.7 active seconds** |
| **Pure Active Assistant Throughput** | **2,805.0 tok/s** | **19.9M tokens / 7,094.7 active seconds** |
| **Isolated GPU Microbenchmark** | **5,136.5 tok/s** | Pure compute forward/backward step loop |

### Reconciled Root Cause:
1. `81 minutes` was the pre-run theoretical projection from isolated microbenchmarks (25M / 5,136 tok/s).
2. The trainer script's reported metric `tok_per_sec` divided tokens by total elapsed project time from boot (`time.time() - start_time`), which automatically accumulated:
   - 15.3 minutes of mandatory thermal cooldown sleep.
   - 16 periodic evaluation passes across 2,439 validation sequences (~40s).
   - 8 periodic atomic checkpoint disk writes (~13s).
   - Dataloader collation and JSON parsing overhead (~390ms/step).
3. Reconciled true active training time was **118.24 minutes** (1 hr 58 mins) at an effective throughput of **3,513.1 conversation tokens/second**.

---

## 4. Dataset Release Integrity & Tool Provenance Audit

### Cryptographic Dataset Hashes
In [`modern_gpt/data_sft_release/splits/`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_release/splits):
- `train.jsonl` (226,486 records): `622240e93f850e33bfeaae719fb290c6156751df60e29e7d344a720e3aa219e9` (Verified).
- `val.jsonl` (5,959 records): `77d7f8d19867f071b09b4895aa48454a7a6c9e2d5a1ed16a214f117b15e637dc` (Verified).
- `test.jsonl` (5,959 records): `b3798c6fcf1213c473a445fce1575e7dd11ff2863949e18547eb85bc7d641923` (Verified).

### Physical Ground Truth of Tool Records (74,368 Records)
An independent audit of every record in `train.jsonl` classified the true provenance:
1. **APIGen Schema Demonstrations:** **73,621 records (98.99%)** (`HuggingFaceTB/smoltalk:apigen-80k`). Schema-validated function-calling demonstrations (synthetic generation; not executed in live subprocesses).
2. **Authentic Live Python Subprocess Executions:** **543 records (0.73%)** (`ModernGPT_Harness:live_sandbox_execution` [353] + `qorin_harness:live_sandbox_execution` [190]). Authentic Python code executed in isolated subprocesses with real stdout/stderr captured.
3. **Multi-Step Workflows:** **96 records (0.13%)** (`qorin_harness:multistep_execution`). Sequential multi-turn tool interaction.
4. **Error Recovery Trajectories:** **99 records (0.13%)** (`qorin_harness:error_recovery` [96] + `ModernGPT_Harness:error_recovery_trajectory` [3]). Tool exception handling and recovery.
5. **Negative Controls (No-Tool Decisions):** **7 records (0.01%)** (`qorin_harness:negative_control` [4] + `ModernGPT_Harness:negative_tool_control` [3]). Direct answering without tool invocation.
6. **Tool Safety / Prompt Injection Resistance:** **2 records (<0.01%)** (`qorin_harness:tool_safety`).

*Audit Conclusion:* Speculative claims of "32k authentic Python executions" in earlier reports are reconciled against actual repository files. Demonstrations will never be misrepresented as authentic live executions in Experiment B sampling or reporting.

---

## 5. Tool Execution Harness & Sandbox Audit

- **Subprocess Sandbox:** [`backend/tools/python_sandbox.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/python_sandbox.py)
  - Execution mode: Isolated subprocess with 5.0-second timeout.
  - Security whitelist: Blocks sensitive dangerous modules (`socket`, `urllib`, `http`, `subprocess`, `shutil`).
- **Tool Orchestrator:** [`backend/tools/orchestrator.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/backend/tools/orchestrator.py)
  - Parses canonical `<|tool_call|>{...}<|im_end|>` delimiters.
  - Dispatches calls to `PythonSandbox` or `WebResearcher`.
- **Loss Masking Protocol:**
  - Assistant tool call requests (`<|tool_call|>`) are **supervised**.
  - Environment observations (`<|tool_result|>`) are **strictly masked (-100)**.

---

## 6. Proposed Experiment B Sampling & Runtime Projection

### Proposed 35M Token Mixture:
- **General Instructions:** 50% (~17.5M tokens)
- **Reasoning:** 20% (~7.0M tokens)
- **Coding:** 15% (~5.25M tokens)
- **Tool-Use:** 15% (~5.25M tokens)
  *(Including all 543 authentic live sandbox traces + sampled APIGen schema demonstrations)*

### Reconciled Runtime Projection (At 3,513 active tok/s):
- **Pure Active GPU Training Time:** `35,000,000 / 3,513.1` = **9,963 seconds = 166.0 minutes (2.77 hours)**.
- **Thermal Safety Schedule:**
  - Session 1: 105 minutes active (~22.1M tokens)
  - Cooldown 1: 15 minutes mandatory cooling break
  - Session 2: 61 minutes active (~12.9M tokens)
- **Total Estimated Wall-Clock Time:** **181.0 minutes (3.02 hours)**.
- **Milestone Feasibility:** ~45 hours remaining, consuming only **6.7%** of available deadline budget.
