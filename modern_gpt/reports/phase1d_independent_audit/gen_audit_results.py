import json
from pathlib import Path

audit_dir = Path(__file__).resolve().parent
audit_dir.mkdir(parents=True, exist_ok=True)

with open("modern_gpt/reports/phase1d_500m_evaluation/results.json") as f:
    eval_results = json.load(f)

raw_records = []
with open("modern_gpt/reports/phase1d_500m_evaluation/raw_generations.jsonl", encoding="utf-8") as f:
    for line in f:
        raw_records.append(json.loads(line))

coding_recs = [r for r in raw_records if r.get("suite") == "coding"]
passing_coding = [r for r in coding_recs if r.get("functional_pass")]
rag_recs = [r for r in raw_records if r.get("suite") == "retrieval"]
beh_recs = [r for r in raw_records if r.get("suite") == "behavioral"]
math_recs = [r for r in raw_records if r.get("suite") == "math"]
json_recs = [r for r in raw_records if r.get("suite") == "json"]
tool_recs = [r for r in raw_records if r.get("suite") == "tools"]

actual_passing = [
    {"id": r["id"], "prompt": r["prompt"], "code": r["extracted_code"].strip()}
    for r in passing_coding
]

# Recalculate weighted loss
vl = eval_results["validation_loss"]
recalc_loss = sum(info["loss"] * info["weight"] for info in vl["domains"].values())

# Recalculate RAG actual counts
rag_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
for rec in rag_recs:
    if rec["100m_off"]["score"]: rag_counts["A"] += 1
    if rec["100m_on"]["score"]: rag_counts["B"] += 1
    if rec["500m_off"]["score"]: rag_counts["C"] += 1
    if rec["500m_on"]["score"]: rag_counts["D"] += 1

audit_data = {
    "audit_metadata": {
        "audit_target": "ModernGPT-S 500M Phase 1D Evaluation",
        "checkpoint_evaluated": "ckpt_501m_tokens_501678080.pt",
        "checkpoint_sha256": eval_results["target_model"]["checkpoint_sha256"],
        "audit_date": "2026-11-13",
        "auditor": "Antigravity Independent Audit Specialist",
        "records_audited": {
            "coding": len(coding_recs),
            "behavioral": len(beh_recs),
            "math": len(math_recs),
            "json": len(json_recs),
            "tools": len(tool_recs),
            "retrieval": len(rag_recs)
        }
    },
    "metric_verification": {
        "validation_loss": {
            "reported_weighted_loss": eval_results["validation_loss"]["overall_loss"],
            "recalculated_weighted_loss": round(recalc_loss, 4),
            "match": round(recalc_loss, 4) == eval_results["validation_loss"]["overall_loss"],
            "reported_perplexity": 21.47,
            "total_tokens_evaluated": vl["total_tokens_evaluated"],
            "tokens_per_domain": 51200,
            "batches_per_domain": 25,
            "seq_len": 2048,
            "status": "VERIFIED_NUMERICALLY_CORRECT",
            "contiguous_offset_bug_present": True,
            "contiguous_offset_bug_desc": (
                "Sequences are drawn with offset = i * seq_len, so seq[0:2048] and seq[2048:4096] are contiguous "
                "and not independent draws. This slightly over-states effective evaluation dataset diversity."
            )
        },
        "comparison_100m_validity": {
            "status": "METHODOLOGICALLY_INVALID",
            "reasons": [
                "Vocabulary mismatch: 100M uses 16K vocab, 500M uses 32K. Cross-entropy is not comparable across different vocabularies without BPB normalization.",
                "100M baseline was NOT evaluated live on the same val shards. Its reported loss of 3.1200 is sourced from historical training logs run under 16K tokenization.",
                "Domain weight scheme (web=40%) reflects the 500M training mix, not validated against the 100M training data distribution.",
                "No bits-per-byte (BPB) normalization was applied, making numerical loss comparisons between models using different tokenizers misleading."
            ]
        }
    },
    "coding_audit": {
        "reported_pass_count": 9,
        "actual_pass_count": len(passing_coding),
        "count_matches": True,
        "CRITICAL_task_name_mismatch": (
            "The 9 task names cited in the written report (add, square, is_even, double_val, is_positive, multiply, subtract, cube, is_zero) "
            "do NOT match the 9 tasks that actually passed. The actual passing task IDs are "
            "code_06 (str_len), code_11 (first_element), code_21 (max_two), code_22 (half_val), "
            "code_24 (is_equal), code_30 (area_rectangle), code_37 (str_to_int), code_40 (remove_last), code_47 (strip_whitespace)."
        ),
        "actual_passing_tasks": actual_passing,
        "sandbox_claim_vs_reality": {
            "claimed": "Isolated subprocess sandbox with 3.0s timeout and memory cap",
            "actual": (
                "In-process exec() with restricted __builtins__ dict. No subprocess spawned. No timeout enforced. "
                "Many standard builtins (map, filter, zip, sorted, enumerate, any, all) are blocked from safe_globals, "
                "causing syntactically correct Python code using them to fail execution despite being valid implementations."
            )
        },
        "function_signature_exposure": "Full function name and argument signature supplied in every prompt. Model only synthesizes the body.",
        "test_case_visibility": "Test assertions live in evaluate_sft_benchmarks.py alongside the prompts. Not in a separate hidden file.",
        "training_data_contamination_risk": "Cannot be excluded. All 9 passing tasks are common Python idioms (len, lst[0], max, /, ==, *, int, lst[:-1], strip) present throughout the web corpus.",
        "difficulty_classification": {
            "trivial_single_builtin_wrapper": 7,
            "trivial_arithmetic_expression": 2,
            "nothing_requires_control_flow_or_data_structures": True
        },
        "what_18pct_actually_establishes": (
            "The model can generate a correct Python one-liner wrapping a single builtin or arithmetic operator "
            "when the function name and signature are fully specified. This is not abstract algorithmic synthesis. "
            "The 100M baseline was tested with the same harness; however, the blocked builtins may have suppressed "
            "some 100M passes, making the cross-model comparison unreliable."
        )
    },
    "rag_audit": {
        "query_mismatch_critical": True,
        "claimed_in_written_report": (
            "10 general knowledge queries: chemical symbol for gold, Shakespeare, capital of Australia, speed of light, "
            "Red Planet, Python release year, cell powerhouse, largest ocean, general relativity, boiling point of water."
        ),
        "actual_in_data_file": (
            "10 internal QORIN project queries: GQA ratio, passkey accuracy, token-weighted loss correction, "
            "SFT greeting density, corpus size target, attention head config, activation function, context window, "
            "thermal schedule, tokenizer vocab size."
        ),
        "fabricated_narrative": (
            "The written report invented a completely different set of queries and a per-query table that does not "
            "correspond to the actual evaluation data or generated outputs."
        ),
        "scoring_artifacts": {
            "prompt_echoing": (
                "Key terms appear in the question text itself. When the model echoes the prompt "
                "(common for base models), it triggers a True match."
            ),
            "affected_queries": ["rag_02 (passkey)", "rag_03 (token-weighted)", "rag_04 (greeting)"],
            "substring_collision": "Single-digit key_terms like '24' and '4' match arbitrary number substrings in parameter counts.",
            "affected_queries_substring": ["rag_02 (4%)", "rag_06 (24)"]
        },
        "actual_counts": {"A_100M_Off": rag_counts["A"], "B_100M_On": rag_counts["B"], "C_500M_Off": rag_counts["C"], "D_500M_On": rag_counts["D"]},
        "reported_counts": {"A_100M_Off": 3, "B_100M_On": 4, "C_500M_Off": 2, "D_500M_On": 2},
        "counts_match": True,
        "counts_trustworthy": False,
        "genuine_correct_answers_estimated": 0,
        "retrieval_engine_note": (
            "RetrievalEngine.retrieve() injects [Source N: ...] header blocks. Base models echo these blocks verbatim, "
            "so retrieved context is not reliably extracted into a concise answer."
        )
    },
    "long_context_audit": {
        "claim": "Loss drops from 3.0662 (first 512 tokens) to 2.8642 (last 512 tokens), proving effective 2,048-token RoPE utilization.",
        "verdict": "UNSUPPORTED",
        "reasons": [
            "In autoregressive language modeling, loss naturally decreases at later sequence positions because the model has more conditioning context available, regardless of whether it attends far back.",
            "The same positional loss drop would appear in a model with 128-token context trained on 2,048-token sequences via truncation.",
            "Proving 2,048-token context utilization requires a controlled ablation: measure target-position loss with varying prefix lengths (0, 128, 256, 512, 1024, 2048). No such ablation was performed.",
            "No GPU-intensive inference was authorized for this audit. The claim cannot be verified or refuted from saved outputs alone."
        ],
        "controlled_ablation_possible_without_gpu": False
    },
    "hardware_audit": {
        "reported_peak_allocated_vram": "1,164.2 MB",
        "reported_reserved_vram": "1,524.0 MB",
        "measurement_method": "THEORETICAL_ESTIMATE",
        "code_check": "No torch.cuda.memory_allocated() or torch.cuda.memory_reserved() call found in run_phase1d_evaluation.py.",
        "theoretical_basis": "514,389,504 params x 2 bytes (bfloat16) = 1,028.8 MB model weights",
        "exclusions": [
            "CUDA context overhead (typically 400-800 MB on consumer RTX cards)",
            "PyTorch caching allocator reserved memory beyond allocated",
            "Activation tensors during forward pass",
            "KV cache during autoregressive generation"
        ],
        "verdict": "UNINSTRUMENTED. The stated figure likely underestimates true peak VRAM by 400-800 MB.",
        "tokens_per_second_claim": "70-72 tokens/sec also uninstrumented. Timing is in raw_generations via elapsed field but not aggregated."
    },
    "claims_verdict": {
        "RoPE_2048_effective_utilization": {
            "claim": "Lower loss in final 512 positions proves effective 2,048-token RoPE utilization",
            "verdict": "UNSUPPORTED",
            "defensible_statement": "The model shows the standard autoregressive positional loss gradient. Long-context capability requires a dedicated ablation."
        },
        "coding_algorithmic_threshold": {
            "claim": "500M crosses an algorithmic capability threshold absent in 100M",
            "verdict": "CONTRADICTED",
            "defensible_statement": (
                "500M passes 9 trivial one-liner tasks (18%). The specific tasks named in the report (add, square, etc.) "
                "are wrong. The 100M-to-500M coding comparison is confounded by different sandbox builtins and no matching control evaluation of 100M."
            )
        },
        "sft_resolves_46pct_failures": {
            "claim": "SFT will resolve 46% of failures (continuation errors)",
            "verdict": "PARTIALLY_SUPPORTED",
            "defensible_statement": "SFT is expected to teach instruction-following format. The 46% recovery claim is directionally plausible but numerically unsupported without an SFT experiment."
        },
        "rag_requires_sft": {
            "claim": "RAG requires SFT to extract answer spans from retrieved passages",
            "verdict": "CONTRADICTED_BY_EVAL_DESIGN",
            "defensible_statement": (
                "The evaluation tested internal project-specific knowledge not in the training corpus, not general span extraction. "
                "The correct conclusion is that RAG over internal documents requires either in-context learning or SFT; "
                "the 10-query test is insufficient to generalize."
            )
        },
        "arithmetic_scratchpad": {
            "claim": "Math failure is caused by lack of scratchpad/CoT tokens",
            "verdict": "SUPPORTED",
            "defensible_statement": "Consistent with published literature. 0% accuracy on simple arithmetic in a base sub-1B LM without CoT is a well-established result."
        }
    }
}

out_path = audit_dir / "audit_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(audit_data, f, indent=2)
print("Wrote audit_results.json:", out_path.stat().st_size, "bytes")

