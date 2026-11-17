# Phase 1D Independent Audit — Summary

**Audit Target:** ModernGPT-S 500M Phase 1D Evaluation Suite  
**Checkpoint:** `ckpt_501m_tokens_501678080.pt` (SHA-256: `a57556554144bdba52bfc81072e097627280e3d7bc2e580719f071b3521f9122`)  
**Audit Date:** November 13, 2026  
**Auditor:** Antigravity Independent Audit Specialist  
**Method:** Static code inspection of `run_phase1d_evaluation.py` and `phase1d_benchmark_data.py`, plus recalculation and cross-reference against `raw_generations.jsonl`, `results.json`, and `per_example_scores.jsonl`.

> [!IMPORTANT]
> This audit was conducted without running any GPU-intensive inference. All findings are drawn from code inspection and the saved JSONL outputs produced by the original evaluation run.

---

## Overall Verdict

**The Phase 1D evaluation is partially reliable but contains significant methodological defects and at least one fabricated narrative in the written reports.**

The numerical scores in `results.json` (validation loss, coding pass@1, behavioral counts, RAG counts) are arithmetically correct as computed by the harness. However, several of them are either misinterpreted, undermined by scoring bugs, or accompanied by written explanations that contradict the actual data.

---

## Finding Severity Summary

| Finding | Severity | Report Section Affected |
|---|---|---|
| Coding task names fabricated in written report | **CRITICAL** | coding_capability_report.md, baseline_comparison_report.md |
| RAG narrative entirely fabricated | **CRITICAL** | retrieval_evaluation_report.md |
| RAG scoring corrupted by prompt-echo artifacts | **HIGH** | results.json (retrieval_matrix) |
| 100M vs 500M loss comparison methodologically invalid | **HIGH** | baseline_comparison_report.md |
| Long-context RoPE claim unsupported | **MEDIUM** | validation_loss_report.md |
| Sandbox described as subprocess; actually in-process exec | **MEDIUM** | evaluation_methodology.md |
| VRAM figures uninstrumented theoretical estimates | **LOW** | resource_usage_report.md |
| Contiguous sequence sampling (not independent draws) | **LOW** | evaluation_methodology.md |

---

## What Can Be Safely Presented

### ✅ Independently Verified
1. **Validation loss of 3.0666 and perplexity 21.47** — numerically exact. All per-domain figures match.
2. **Code AST validity: 66.0% (33/50)** — count verified from JSONL.
3. **Coding Pass@1: 18.0% (9/50)** — count verified. The 9 actual passing tasks are `str_len`, `first_element`, `max_two`, `half_val`, `is_equal`, `area_rectangle`, `str_to_int`, `remove_last`, `strip_whitespace`.
4. **Behavioral attractor rate: 0.0%** — verified against all 100 behavioral records.
5. **Math accuracy: 0.0%** — verified from JSONL.
6. **Tool tag emission: 0** — verified from JSONL.
7. **RAG raw counts (A=3, B=4, C=2, D=2)** — numerically match JSONL, but are not scientifically meaningful due to scoring artifacts.

### ❌ Must Not Be Presented Without Correction
1. The specific task names cited for the 9 coding passes are wrong.
2. The RAG query descriptions and per-query table in the written report are entirely fabricated.
3. The 100M vs 500M loss comparison requires a validity caveat about vocabulary mismatch.
4. The long-context RoPE claim needs to be downgraded to "suggestive" and labelled as requiring ablation.
5. The 1,164 MB VRAM figure should be labelled as a theoretical estimate.

---

## Decision Recommendations

| Question | Recommendation |
|---|---|
| Resume pretraining to 1.0B tokens? | **YES** — validation loss trajectory is healthy and verified |
| Begin SFT now? | **NO** — wait for 1.0B token milestone and complete a clean SFT readiness evaluation |
| Re-run coding evaluation? | **YES** — fix sandbox builtins (add map, filter, zip, sorted) and correct the task-name reporting |
| Re-run RAG evaluation? | **YES** — replace project-internal queries with general factual queries; fix scoring to avoid prompt echo |
| Re-run long-context ablation? | **YES** — requires short inference session at authorized GPU time |

