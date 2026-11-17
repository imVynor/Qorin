# ModernGPT-S / QORIN — Phase 2C: Experiment B
## Stage 1: Preflight Verification & Bounded Smoke Test Report

**Execution Timestamp:** 2026-11-15T18:32:25Z  
**Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (8123.4 MB VRAM)  
**PyTorch Version:** 2.16.0.dev20261022+cu134 | **CUDA Version:** 13.4  
**Frozen Base Checkpoint:** [`ckpt_latest.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt)  
**Report Artifacts Location:** [`experiments/phase2_experiment_b/stage_1_preflight/`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_b/stage_1_preflight)

---

## 1. Executive Summary & Decision Matrix

Stage 1 preflight verification and the bounded training smoke test have been executed with strict adherence to all verification protocols. Zero destructive actions were performed, and no full-scale training was initiated.

| Area | Result | Measured Evidence & Findings |
|---|---|---|
| **Frozen Checkpoint Integrity** | **PASS** | SHA-256 (`28ac47b0...`) strictly verified; 223 tensors all finite; 0 missing / 0 unexpected keys. |
| **Architecture Consistency** | **PASS** | 20 layers, 1536 dim, 24 query heads, 4 KV heads (6:1 GQA), RoPE theta 10,000, 514,389,504 params. |
| **Tokenizer Integrity** | **PASS** | SHA-256 (`d10624a9...`) and fingerprint `3559c7435e1d9330` verified. |
| **Dataset Release Integrity** | **PASS** | Train, validation, and test split hashes strictly verified against 1.1.0-production-release. |
| **Loss Masking Protocol** | **PASS** | System/User/Tool observations masked (-100); Assistant response & `<|tool_call|>` supervised. |
| **Tool Execution Sandbox** | **PASS** | Subprocess executed deterministic Python; stdout captured; 5.0s timeout enforced. |
| **Tool Error Handling** | **PASS** | Syntax errors and ZeroDivisionError handled cleanly; restricted imports safely blocked. |
| **Tool Protocol Cycle** | **PASS** | Full cycle: `<|tool_call|>` parsed -> sandbox executed -> `<|tool_result|>` formatted -> assistant continuation. |
| **Checkpoint Resume Fidelity** | **PASS** | Deterministic reload test achieved 0.00e+00 loss delta and 0.00e+00 weight delta. |
| **Training Smoke Test** | **PASS** | Exactly 2 updates completed (5,232 conv tokens; <= 10,000 budget); loss 1.7062 -> 1.7503. |
| **Thermal Safety** | **PASS** | Peak GPU temperature during smoke test: 72.0°C; VRAM: 2,112.9 MB allocated. |
| **Runtime Projection** | **CONFIDENT** | 35M tokens: 166 min active + 15 min cooldown = 181 min total wall-clock (6.8% of remaining budget). |

---

## 2. Preflight A: Checkpoint Integrity Audit
- **Path:** [`ckpt_latest.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/checkpoints/500m_pretrain/ckpt_latest.pt)
- **SHA-256 Checksum:** `28ac47b07e9616967d6a2ea0458e68432682085a2e05fba6ac2e55b82b6d639d` (Verified identical).
- **Parameters:** 514,389,504 untied unique trainable parameters.
- **Weight Tying:** Strictly verified (`token_embedding.weight.data_ptr() == lm_head.weight.data_ptr()`).
- **Pretraining Tokens:** 1,121,648,640 tokens (Step 34,230).
- **Tensors Finite:** 100% of state dict parameters are finite (zero NaN / Inf).

---

## 3. Preflight B: Architecture & Forward-Pass Verification
- **Model Config:** 20 layers, hidden dim 1536, 24 query heads, 4 KV heads (6:1 GQA), head dim 64, SwiGLU intermediate dim 3840, RoPE theta 10,000, context length 2048, vocab size 32768, tied embeddings, no bias.
- **Inference Forward Pass:** Executed test tensor `[1, 5]` on `cuda:0`. Output logits shape: `[1, 5, 32768]`. Logits 100% finite. Memory footprint: 1,029.0 MB allocated.

---

## 4. Preflight C: Dataset & Loss-Masking Verification
Hashes verified for [`data_sft_release/splits/`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_release/splits):
- `train.jsonl`: `622240e93f850e33bfeaae719fb290c6156751df60e29e7d344a720e3aa219e9`
- `val.jsonl`: `77d7f8d19867f071b09b4895aa48454a7a6c9e2d5a1ed16a214f117b15e637dc`
- `test.jsonl`: `b3798c6fcf1213c473a445fce1575e7dd11ff2863949e18547eb85bc7d641923`

### Real Serialized Record Diagnostic (8 Categories Tested)
1. **General Instructions:** System/User masked (-100); Assistant supervised.
2. **Reasoning:** System/User masked (-100); Step-by-step reasoning & answer supervised.
3. **Coding:** System/User masked (-100); Code implementation & explanations supervised.
4. **APIGen Tool Demonstrations:** System/User masked; Assistant `<|tool_call|>` request supervised; Environment `<|tool_result|>` observation masked (-100).
5. **Authentic Live Tool Traces:** System/User masked; Tool request supervised; Sandbox observation masked (-100); Final synthesis supervised.
6. **Multi-Step Workflows:** Interleaved observations masked; all model calls and final answers supervised.
7. **Error-Recovery Examples:** `<|tool_error|>` masked (-100); model's adaptive corrective call supervised.
8. **Negative Controls:** Direct assistant response supervised without emitting tool calls.

---

## 5. Preflight D: Tool Execution Harness Verification
All 9 test cases passed in [`backend/tools/python_sandbox.py`](file:///C:/Users/Anish/Desktop/Projects/ML-course/backend/tools/python_sandbox.py) and [`backend/tools/orchestrator.py`](file:///C:/Users/Anish/Desktop/Projects/ML-course/backend/tools/orchestrator.py):
- **Test 1 (Valid Execution):** `print(42 * 2)` executed in subprocess; output `84`; exit code 0; 82ms.
- **Test 2 (Syntax Error):** `def foo(:` caught cleanly in subprocess; captured in stderr; no crash.
- **Test 3 (Runtime Exception):** `10 / 0` captured `ZeroDivisionError` cleanly.
- **Test 4 (Timeout Enforcement):** `time.sleep(6.0)` terminated at 5.0s timeout limit; `timed_out=True`.
- **Test 5 (Restricted Imports):** `import socket` safely blocked by restricted header check.
- **Test 6 (Malformed Tool Calls):** Unparseable JSON and missing tool names rejected gracefully.
- **Test 7 (Multi-step Interaction):** Sequential execution verified (Step 1 output 40 passed to Step 2 -> 80).
- **Test 8 (No-Tool Control):** Plain conversational text parsed as no-tool decision.
- **Test 9 (End-to-End Protocol):** Canonical cycle verified: `<|tool_call|>` parsed -> sandbox executed -> `<|tool_result|>` formatted -> assistant continuation.

---

## 6. Preflight E: Checkpoint Resume Fidelity Test
- **Method:** 2 deterministic updates on Config A 8-bit AdamW saved to disposable checkpoint. Resumed in a clean instance and compared against uninterrupted 3rd update.
- **Results:**
  - `Loss difference`: **0.00e+00**
  - `Max weight difference`: **0.00e+00**
  - State restoration fidelity: **100.0% exact numerical match**.

---

## 7. Bounded Training Smoke Test Metrics
Detailed in [`smoke_test_metrics.json`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_b/stage_1_preflight/smoke_test_metrics.json):
- **Updates Completed:** Exactly 2 updates (8 microbatches, batch size 1, grad accum 4).
- **Total Conversation Tokens Processed:** **5,232 tokens** (Budget constraint <= 10,000 verified).
- **Assistant Target Tokens Supervised:** **3,232 tokens** (61.77% supervision density).
- **Update 1:** Loss = 1.7062 | Grad norm before/after = 3.06 / 3.06 | VRAM alloc = 2,112.9 MB | Temp = 72.0°C.
- **Update 2:** Loss = 1.7503 | Grad norm before/after = 5.03 / 5.03 | VRAM alloc = 2,092.0 MB | Temp = 70.0°C.
- **Atomic Checkpoint Save/Reload:** Verified successfully at [`smoke_checkpoints/smoke_ckpt_update_2.pt`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_b/stage_1_preflight/smoke_checkpoints/smoke_ckpt_update_2.pt).

---

## 8. Reconciled 35M Token Runtime Projection
Detailed in [`updated_runtime_projection.json`](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/experiments/phase2_experiment_b/stage_1_preflight/updated_runtime_projection.json):
- **Target Budget:** 35,000,000 serialized conversation tokens.
- **Authoritative Active Throughput:** 3,513.1 tok/s (Reproduced over 4,050 steps in Experiment A).
- **Pure Active GPU Training Time:** 9,963 seconds = **166.0 minutes (2.77 hours)**.
- **Thermal Safety Duty Cycle:**
  - Session 1 (Active): 105 minutes (~22.1M tokens)
  - Cooldown 1 (Mandatory cooling): 15 minutes
  - Session 2 (Active): 61 minutes (~12.9M tokens)
- **Total Wall-Clock Time:** **181.0 minutes (3.02 hours)**.
- **Timeline Margin:** ~44.5 hours remaining in milestone window (**6.8% deadline utilization**).

---

## 9. Blockers & Technical Recommendation
- **Unresolved Blockers:** **None**.
- **Stage 1 Recommendation:** All hardware, software, dataset, loss-masking, and tool-sandbox invariants have been verified without discrepancy. The pipeline is technically ready for the official 35M-token Experiment B training run.
