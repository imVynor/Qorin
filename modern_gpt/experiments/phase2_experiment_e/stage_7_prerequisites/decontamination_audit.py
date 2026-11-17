"""
Dataset Decontamination Verification Engine
Implements the 3-tier decontamination gates mandated for Experiment E:
Gate 1: Exact prompt and answer deduplication (case-insensitive normalized).
Gate 2: 10-gram Jaccard similarity threshold strictly below 0.15.
Gate 3: Named entity and topic overlap checks against the frozen 150-task Stage 5 benchmark.

Identifies and details every single violation with specific candidate ID, benchmark ID,
similarity score, and violation reason.
"""

from collections import Counter
import json
import os
import re
import time
from typing import Any, Dict, List, Set, Tuple


def normalize_text(text: str) -> str:
    """Normalize text for exact and n-gram comparison."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_ngrams(words: List[str], n: int = 10) -> Set[str]:
    """Generates set of n-gram shingles."""
    if len(words) < n:
        # If fewer words than n, use full sequence as single n-gram
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i+n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def extract_key_entities(text: str) -> Set[str]:
    """Extracts distinctive capitalized proper nouns and domain entities."""
    entities = set()
    raw_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    stopwords = {
        "What", "Explain", "How", "Why", "Find", "Calculate", "Look", "According", 
        "Based", "Which", "When", "Where", "Who", "Compute", "Determine", "Evaluate",
        "Perform", "Solve", "Show", "Generate", "Please", "Write", "Tell", "Give"
    }
    for e in raw_entities:
        if e not in stopwords and len(e) > 3:
            entities.add(e.lower())
    return entities


class DecontaminationAuditor:
    def __init__(self, benchmark_path: str):
        self.benchmark_path = benchmark_path
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.benchmark = [json.loads(line) for line in f]

        self.bench_prompts = {}
        self.bench_answers = {}
        self.bench_ngrams = {}
        self.bench_entities = {}

        for b in self.benchmark:
            bid = b["id"]
            p_norm = normalize_text(b["prompt"])
            a_norm = normalize_text(b.get("expected_answer", ""))
            words = p_norm.split()
            
            self.bench_prompts[bid] = p_norm
            self.bench_answers[bid] = a_norm
            self.bench_ngrams[bid] = get_ngrams(words, n=10)
            self.bench_entities[bid] = extract_key_entities(b["prompt"])

    def audit_record(self, candidate_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Audits a single candidate record against all 150 benchmark tasks across all 3 gates."""
        violations = []
        cid = candidate_record.get("id", "unknown_candidate")
        cand_prompt = candidate_record.get("prompt", "")
        cand_ans = candidate_record.get("target_response", "") or candidate_record.get("expected_answer", "")

        cand_p_norm = normalize_text(cand_prompt)
        cand_a_norm = normalize_text(cand_ans)
        cand_words = cand_p_norm.split()
        cand_ngrams = get_ngrams(cand_words, n=10)
        cand_entities = extract_key_entities(cand_prompt)

        for b in self.benchmark:
            bid = b["id"]
            bench_p_norm = self.bench_prompts[bid]
            bench_a_norm = self.bench_answers[bid]
            bench_ng = self.bench_ngrams[bid]
            bench_ent = self.bench_entities[bid]

            # Gate 1: Exact Prompt or Answer Deduplication
            if cand_p_norm and cand_p_norm == bench_p_norm:
                violations.append({
                    "candidate_id": cid,
                    "gate_failed": "gate_1_exact_prompt_duplicate",
                    "matched_benchmark_id": bid,
                    "matched_benchmark_prompt": b["prompt"],
                    "similarity_score": 1.0,
                    "details": "Exact match with benchmark prompt string.",
                    "remediation": "REJECT_RECORD"
                })

            if cand_a_norm and bench_a_norm and cand_a_norm == bench_a_norm and len(cand_a_norm) > 10:
                violations.append({
                    "candidate_id": cid,
                    "gate_failed": "gate_1_exact_answer_duplicate",
                    "matched_benchmark_id": bid,
                    "matched_benchmark_prompt": b["prompt"],
                    "similarity_score": 1.0,
                    "details": "Exact match with benchmark target answer string.",
                    "remediation": "REJECT_RECORD"
                })

            # Gate 2: 10-Gram Jaccard Similarity (Threshold < 0.15)
            jaccard = jaccard_similarity(cand_ngrams, bench_ng)
            if jaccard >= 0.15:
                overlap_shingles = list(cand_ngrams & bench_ng)
                violations.append({
                    "candidate_id": cid,
                    "gate_failed": "gate_2_10gram_jaccard_similarity",
                    "matched_benchmark_id": bid,
                    "matched_benchmark_prompt": b["prompt"],
                    "similarity_score": round(jaccard, 4),
                    "details": f"10-gram Jaccard similarity {jaccard:.4f} >= 0.15 threshold. Overlapping shingles: {overlap_shingles[:3]}",
                    "remediation": "REWRITE_PROMPT"
                })

            # Gate 3: Named Entity & Topic Overlap (>= 2 shared specific entities)
            shared_entities = cand_entities & bench_ent
            if len(shared_entities) >= 2:
                violations.append({
                    "candidate_id": cid,
                    "gate_failed": "gate_3_named_entity_overlap",
                    "matched_benchmark_id": bid,
                    "matched_benchmark_prompt": b["prompt"],
                    "similarity_score": len(shared_entities),
                    "details": f"Shares {len(shared_entities)} named entities with benchmark task: {list(shared_entities)}",
                    "remediation": "QUARANTINE_ENTITY_REPLACE"
                })

        return violations

    def run_full_audit(self, candidate_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Runs decontamination audit over complete candidate dataset."""
        t0 = time.time()
        all_violations = []
        violating_record_ids = set()

        for rec in candidate_dataset:
            v_list = self.audit_record(rec)
            if v_list:
                all_violations.extend(v_list)
                violating_record_ids.add(rec.get("id"))

        clean_count = len(candidate_dataset) - len(violating_record_ids)
        violation_gate_counts = Counter(v["gate_failed"] for v in all_violations)

        audit_result = {
            "audit_timestamp": time.time(),
            "benchmark_reference_path": self.benchmark_path,
            "total_benchmark_tasks": len(self.benchmark),
            "candidates_audited_count": len(candidate_dataset),
            "clean_records_count": clean_count,
            "violating_records_count": len(violating_record_ids),
            "total_violation_events": len(all_violations),
            "violations_by_gate": dict(violation_gate_counts),
            "decontamination_pass": (len(violating_record_ids) == 0),
            "violations": all_violations
        }
        return audit_result


if __name__ == "__main__":
    bench_p = "modern_gpt/experiments/phase2_experiment_d/stage_5_generalization_audit/benchmark.jsonl"
    auditor = DecontaminationAuditor(bench_p)
    print(f"Decontamination Auditor initialized with {len(auditor.benchmark)} benchmark reference tasks.")
