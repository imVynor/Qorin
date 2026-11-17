"""
Verification test for the Decontamination Infrastructure.
Injects synthetic positive control violations to verify that Gate 1, Gate 2, and Gate 3
detect and report violations accurately with complete diagnostic attribution.
"""

import json
from decontamination_audit import DecontaminationAuditor

def test_decontamination_engine():
    bench_p = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
    auditor = DecontaminationAuditor(bench_p)

    # Synthetic test records
    test_candidates = [
        # 1. Clean Record (should pass all 3 gates)
        {
            "id": "cand_clean_01",
            "prompt": "What is the difference between supervised and unsupervised learning in classical machine learning theory?",
            "target_response": "Supervised learning relies on labeled training pairs, whereas unsupervised learning discovers patterns from unlabeled data."
        },
        # 2. Gate 1 Violation: Exact Prompt Duplicate of benchmark task "direct_01"
        {
            "id": "cand_violation_exact_prompt",
            "prompt": auditor.benchmark[0]["prompt"],
            "target_response": "A duplicated response."
        },
        # 3. Gate 1 Violation: Exact Answer Duplicate
        {
            "id": "cand_violation_exact_answer",
            "prompt": "Tell me about this topic please.",
            "target_response": auditor.benchmark[1].get("expected_answer", "some answer string that is long enough")
        },
        # 4. Gate 2 Violation: High 10-gram Jaccard Overlap
        {
            "id": "cand_violation_jaccard",
            "prompt": auditor.benchmark[2]["prompt"] + " and also please tell me more about it today.",
            "target_response": "Some unrelated answer."
        },
        # 5. Gate 3 Violation: Named Entity Overlap (Shares 2+ benchmark entities)
        {
            "id": "cand_violation_entity",
            "prompt": "Look up Apple and Microsoft annual fiscal year numbers for stock research.",
            "target_response": "Looking up financial metrics."
        }
    ]

    print(f"Testing Decontamination Auditor on {len(test_candidates)} test candidates...", flush=True)
    audit_results = auditor.run_full_audit(test_candidates)

    print(f"Audited: {audit_results['candidates_audited_count']}", flush=True)
    print(f"Clean: {audit_results['clean_records_count']}", flush=True)
    print(f"Violations: {audit_results['violating_records_count']}", flush=True)
    print(f"Violations by Gate: {audit_results['violations_by_gate']}", flush=True)

    # Verify that clean candidate passed and all 4 violations were caught
    assert audit_results["clean_records_count"] == 1, "Clean record should have passed!"
    assert audit_results["violating_records_count"] == 4, "All 4 synthetic violations must be detected!"
    print("\n--- DECONTAMINATION INFRASTRUCTURE SELF-TEST PASSED ---", flush=True)

if __name__ == "__main__":
    test_decontamination_engine()
