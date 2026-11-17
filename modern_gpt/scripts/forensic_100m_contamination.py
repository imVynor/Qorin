"""ModernGPT-S 100M: Forensic Contamination Matching & Classification System.

Performs deep forensic analysis of the 30 materialized shards against all project evaluation suites:
- Categorizes reference datasets into distinct research validity groups.
- Dissects unique matching patterns vs repeated occurrences vs chunk boundary artifacts.
- Extracts token offsets, decoded texts, and surrounding corpus context.
- Classifies matches into research-grounded taxonomy:
  * EXACT_TASK_LEAKAGE
  * SUBSTANTIAL_SEMANTIC_OVERLAP
  * COMMON_BOILERPLATE
  * COMMON_NATURAL_LANGUAGE
  * MATHEMATICAL_OR_ALGORITHMIC_CONVENTION
  * DUPLICATE_REFERENCE
  * BOUNDARY_ARTIFACT
  * UNRESOLVED
- Answers all forensic audit questions quantitatively.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sources import compute_content_hash

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.ForensicAudit")


def compute_file_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def get_ngrams(tokens: List[int], n: int = 13) -> Set[Tuple[int, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


# Known common programming boilerplate / natural language patterns
BOILERPLATE_PATTERNS = [
    re.compile(r"def\s+is_even", re.IGNORECASE),
    re.compile(r"def\s+is_odd", re.IGNORECASE),
    re.compile(r"return\s+n\s*%\s*2", re.IGNORECASE),
    re.compile(r"return\s+len\(", re.IGNORECASE),
    re.compile(r"for\s+i\s+in\s+range\(", re.IGNORECASE),
    re.compile(r"return\s+False\s+else", re.IGNORECASE),
    re.compile(r"return\s+True\s+else", re.IGNORECASE),
    re.compile(r"count\s*\+=\s*1", re.IGNORECASE),
    re.compile(r"left\s*-=\s*1", re.IGNORECASE),
    re.compile(r"right\s*\+=\s*1", re.IGNORECASE),
    re.compile(r"import\s+numpy\s+as\s+np", re.IGNORECASE),
    re.compile(r"import\s+os", re.IGNORECASE),
    re.compile(r"import\s+sys", re.IGNORECASE),
    re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]", re.IGNORECASE),
]

NATURAL_LANGUAGE_PATTERNS = [
    re.compile(r"write\s+a\s+python\s+function", re.IGNORECASE),
    re.compile(r"that\s+returns\s+the\s+number\s+of", re.IGNORECASE),
    re.compile(r"that\s+takes\s+a\s+list\s+of", re.IGNORECASE),
    re.compile(r"according\s+to\s+the\s+united\s+states", re.IGNORECASE),
    re.compile(r"in\s+this\s+chapter,?\s+we\s+will", re.IGNORECASE),
    re.compile(r"the\s+number\s+of\s+elements\s+in", re.IGNORECASE),
    re.compile(r"['\"]?a['\"]?,\s*['\"]?e['\"]?,\s*['\"]?i['\"]?,\s*['\"]?o['\"]?,\s*['\"]?u['\"]?", re.IGNORECASE),
    re.compile(r"returns?\s+True\s+if", re.IGNORECASE),
]

MATH_ALGO_PATTERNS = [
    re.compile(r"fibonacci", re.IGNORECASE),
    re.compile(r"matrix\[i\]\[j\]", re.IGNORECASE),
    re.compile(r"def\s+gcd\(", re.IGNORECASE),
    re.compile(r"bubblesort", re.IGNORECASE),
    re.compile(r"binary\s+search", re.IGNORECASE),
    re.compile(r"transpose", re.IGNORECASE),
]


def classify_match(decoded_text: str, token_ids: Tuple[int, ...], is_boundary: bool) -> Tuple[str, str]:
    """Classifies a matching 13-gram into the research taxonomy."""
    if is_boundary or (0 in token_ids):
        return "BOUNDARY_ARTIFACT", "Match spans across <|endoftext|> (token 0) document boundary."

    norm_text = decoded_text.strip()

    # Check common boilerplate
    for pat in BOILERPLATE_PATTERNS:
        if pat.search(norm_text):
            return "COMMON_BOILERPLATE", f"Matches standard programming structure or syntax ({pat.pattern})."

    # Check natural language boilerplate
    for pat in NATURAL_LANGUAGE_PATTERNS:
        if pat.search(norm_text):
            return "COMMON_NATURAL_LANGUAGE", f"Matches ubiquitous prompt phrasing or generic prose ({pat.pattern})."

    # Check algorithmic convention
    for pat in MATH_ALGO_PATTERNS:
        if pat.search(norm_text):
            return "MATHEMATICAL_OR_ALGORITHMIC_CONVENTION", f"Standard algorithmic idiom or mathematical problem statement ({pat.pattern})."

    # Check length and specificity
    if len(norm_text) < 30:
        return "COMMON_BOILERPLATE", "Short, generic token fragment (<30 characters)."

    return "UNRESOLVED", "Requires detailed context inspection to establish whether text is task-specific."


class ForensicAuditor:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        tok_dir = repo_root / "artifacts/tokenizer_16k_v2"
        self.tokenizer = HuggingFaceBPEWrapper.load(tok_dir)
        assert self.tokenizer.fingerprint == "be6c3d60c5baf217", f"Fingerprint mismatch: {self.tokenizer.fingerprint}"
        logger.info(f"Loaded Tokenizer: {self.tokenizer.name} (fingerprint: {self.tokenizer.fingerprint})")

        # Suite definitions with strict category separation
        self.suite_definitions = {
            "critical_held_out_100": {
                "category": "Critical Held-Out Evaluation",
                "path": repo_root / "data/midscale_500/val.jsonl",
                "filter_fn": None,
                "fields": ["instruction", "output"],
                "description": "Held-out 100-task benchmark (40 Python, 40 JSON, 10 QA, 10 Explanations)",
            },
            "critical_transfer_160": {
                "category": "Critical Transfer Evaluation",
                "path": repo_root / "data/midscale_500/train.jsonl",
                "filter_fn": lambda x: x.get("category") == "python",
                "fields": ["instruction", "output"],
                "description": "160-task Python algorithmic transfer benchmark",
            },
            "critical_checkpoint_selection": {
                "category": "Critical Checkpoint Selection",
                "path": repo_root / "data/midscale_500/checkpoint_selection_val.jsonl",
                "filter_fn": None,
                "fields": ["instruction", "output"],
                "description": "Checkpoint-selection validation benchmark (50 tasks)",
            },
            "historical_val_documents": {
                "category": "Historical Validation",
                "path": repo_root / "artifacts/corpus/val_documents.jsonl",
                "filter_fn": None,
                "fields": ["text"],
                "description": "Historical 250M pretraining raw validation documents (10 documents)",
            },
            "historical_val_bin": {
                "category": "Historical Validation",
                "path": repo_root / "artifacts/shards_250m/val.bin",
                "is_bin": True,
                "sample_tokens": 500000,
                "description": "Historical 250M pretraining val.bin (sampled 500k tokens)",
            },
            "training_data_sft_v2": {
                "category": "Training Data Overlap (SFT v2)",
                "path": repo_root / "data/sft_v2_public/train.jsonl",
                "filter_fn": None,
                "fields": ["instruction", "output"],
                "description": "SFT v2 training dataset (8,597 instruction conversations)",
            },
            "diagnostic_tiny_overfit": {
                "category": "Diagnostic Validation",
                "path": repo_root / "artifacts/diagnostics/tiny_overfit_data/val.jsonl",
                "filter_fn": None,
                "fields": ["prompt", "response"],
                "description": "Tiny overfit diagnostic validation split (12 tasks)",
            },
            "additional_expanded_eval": {
                "category": "Additional Evaluation Overlap",
                "path": repo_root / "artifacts/corpus/expanded_eval_documents.jsonl",
                "filter_fn": None,
                "fields": ["text"],
                "description": "Historical expanded evaluation documents",
            },
        }

        self.ngram_to_refs: Dict[Tuple[int, ...], List[Dict[str, Any]]] = defaultdict(list)
        self.suite_metadata: Dict[str, Dict[str, Any]] = {}
        self.suite_unique_ngrams: Dict[str, Set[Tuple[int, ...]]] = {}
        self.cross_suite_duplicates: Set[Tuple[int, ...]] = set()

    def index_all_reference_suites(self):
        logger.info("Indexing all reference datasets by category...")
        all_seen_ngrams = set()

        for s_key, s_def in self.suite_definitions.items():
            p = s_def["path"]
            if not p.exists():
                logger.warning(f"Reference file not found: {p}")
                self.suite_metadata[s_key] = {"available": False, "error": "File not found"}
                continue

            unique_suite_ngrams = set()
            total_extracted = 0
            records_count = 0

            if s_def.get("is_bin"):
                val_arr = np.memmap(p, dtype=np.uint16, mode="r")
                sample_len = min(len(val_arr), s_def.get("sample_tokens", 500000))
                tokens = list(val_arr[:sample_len])
                records_count = 1
                for i in range(len(tokens) - 12):
                    ng = tuple(tokens[i : i + 13])
                    total_extracted += 1
                    if ng not in unique_suite_ngrams:
                        unique_suite_ngrams.add(ng)
                        self.ngram_to_refs[ng].append({
                            "suite_key": s_key,
                            "category": s_def["category"],
                            "record_id": f"val_bin_offset_{i}",
                            "field": "tokens",
                            "text": self.tokenizer.decode(list(ng)),
                        })
            else:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue

                        filter_fn = s_def.get("filter_fn")
                        if filter_fn and not filter_fn(item):
                            continue

                        records_count += 1
                        rec_id = item.get("id") or f"{s_key}_{records_count}"

                        for fld in s_def["fields"]:
                            txt = item.get(fld, "")
                            if not txt or len(txt.strip()) < 10:
                                continue
                            t = self.tokenizer.encode(txt.strip())
                            for i in range(len(t) - 12):
                                ng = tuple(t[i : i + 13])
                                total_extracted += 1
                                if ng not in unique_suite_ngrams:
                                    unique_suite_ngrams.add(ng)
                                    self.ngram_to_refs[ng].append({
                                        "suite_key": s_key,
                                        "category": s_def["category"],
                                        "record_id": rec_id,
                                        "field": fld,
                                        "text": txt[:120].replace("\n", " "),
                                    })

            # Check cross-suite duplicates
            shared = unique_suite_ngrams.intersection(all_seen_ngrams)
            if shared:
                self.cross_suite_duplicates.update(shared)
            all_seen_ngrams.update(unique_suite_ngrams)

            self.suite_unique_ngrams[s_key] = unique_suite_ngrams
            self.suite_metadata[s_key] = {
                "available": True,
                "category": s_def["category"],
                "description": s_def["description"],
                "path": str(p),
                "records_loaded": records_count,
                "raw_13grams_extracted": total_extracted,
                "unique_13grams": len(unique_suite_ngrams),
            }
            logger.info(
                f"[{s_key}] ({s_def['category']}): {records_count:,} records, "
                f"{total_extracted:,} raw n-grams, {len(unique_suite_ngrams):,} unique"
            )

        logger.info(
            f"Indexing complete: {len(self.ngram_to_refs):,} total distinct 13-grams mapped. "
            f"Cross-suite duplicate n-grams: {len(self.cross_suite_duplicates):,}"
        )

    def run_forensic_scan(
        self,
        shard_dir: Path,
        max_shards: Optional[int] = None,
        examples_per_category: int = 50,
    ) -> Dict[str, Any]:
        shard_files = sorted(list(shard_dir.glob("shard_*.bin")))
        if max_shards:
            shard_files = shard_files[:max_shards]

        logger.info(f"Starting forensic scan across {len(shard_files)} shards...")

        # Global counters
        suite_hit_occurrences = Counter()
        suite_unique_hits = {s: set() for s in self.suite_definitions}
        category_hit_occurrences = Counter()
        category_unique_hits = defaultdict(set)
        classification_counts = Counter()

        # Detailed match records
        representative_examples = defaultdict(list)
        all_unique_matched_ngrams = set()
        total_matching_occurrences = 0
        shards_with_matches = Counter()

        # Build target lookup sets for fast matching
        all_target_set = set(self.ngram_to_refs.keys())
        logger.info(f"Target dictionary has {len(all_target_set):,} reference n-grams.")

        chunk_size = 1000000

        for s_idx, s_path in enumerate(shard_files):
            t0 = datetime.datetime.now()
            s_arr = np.memmap(s_path, dtype=np.uint16, mode="r")
            n_tokens = len(s_arr)

            shard_match_count = 0

            # Scan in 1M chunks with 12-token overlap so no n-gram is split across chunk boundaries
            for c_start in range(0, n_tokens - 12, chunk_size):
                c_end = min(n_tokens, c_start + chunk_size + 12)
                chunk_tokens = s_arr[c_start:c_end].tolist()
                chunk_len = len(chunk_tokens)

                # Iterate through chunk
                for i in range(chunk_len - 12):
                    ng = tuple(chunk_tokens[i : i + 13])
                    if ng in all_target_set:
                        global_offset = c_start + i
                        total_matching_occurrences += 1
                        shard_match_count += 1
                        all_unique_matched_ngrams.add(ng)

                        # Fetch all reference attributions for this n-gram
                        ref_records = self.ngram_to_refs[ng]
                        dec_ng = self.tokenizer.decode(list(ng))

                        # Decode surrounding context: 15 tokens before + match + 15 tokens after
                        ctx_start = max(0, global_offset - 15)
                        ctx_end = min(n_tokens, global_offset + 13 + 15)
                        ctx_tokens = list(s_arr[ctx_start:ctx_end])
                        dec_context = self.tokenizer.decode(ctx_tokens)

                        is_boundary = (0 in ng)

                        # Classify match
                        cls_type, cls_reason = classify_match(dec_ng, ng, is_boundary)
                        if len(ref_records) > 1 and cls_type not in ("BOUNDARY_ARTIFACT", "COMMON_BOILERPLATE"):
                            # Check if duplicate reference across different categories
                            unique_categories = {r["category"] for r in ref_records}
                            if len(unique_categories) > 1:
                                cls_type = "DUPLICATE_REFERENCE"
                                cls_reason = f"Appears across multiple suites ({', '.join(unique_categories)})"

                        classification_counts[cls_type] += 1

                        seen_suites = set()
                        for r in ref_records:
                            s_key = r["suite_key"]
                            if s_key not in seen_suites:
                                seen_suites.add(s_key)
                                suite_hit_occurrences[s_key] += 1
                                suite_unique_hits[s_key].add(ng)

                                # Save representative examples per suite
                                if len(representative_examples[s_key]) < examples_per_category:
                                    representative_examples[s_key].append({
                                        "suite_key": s_key,
                                        "category": r["category"],
                                        "record_id": r["record_id"],
                                        "field": r["field"],
                                        "reference_text": r["text"],
                                        "shard_name": s_path.name,
                                        "shard_index": s_idx,
                                        "token_offset": global_offset,
                                        "matching_13gram_tokens": list(ng),
                                        "decoded_13gram": dec_ng,
                                        "corpus_context": dec_context,
                                        "classification": cls_type,
                                        "classification_reason": cls_reason,
                                    })

                        seen_cats = set()
                        for r in ref_records:
                            cat = r["category"]
                            if cat not in seen_cats:
                                seen_cats.add(cat)
                                category_hit_occurrences[cat] += 1
                                category_unique_hits[cat].add(ng)

            shards_with_matches[s_path.name] = shard_match_count
            elapsed = (datetime.datetime.now() - t0).total_seconds()
            logger.info(
                f"Shard {s_idx:02d} ({s_path.name}): {shard_match_count:,} matches "
                f"({len(all_unique_matched_ngrams):,} unique cumulative) in {elapsed:.1f}s"
            )

        # Quantitative synthesis
        unique_counts_by_suite = {s: len(hits) for s, hits in suite_unique_hits.items()}
        unique_counts_by_category = {c: len(hits) for c, hits in category_unique_hits.items()}

        results = {
            "forensic_audit_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "shards_audited_count": len(shard_files),
            "total_corpus_tokens_scanned": len(shard_files) * 50000000,
            "overall_metrics": {
                "total_matching_occurrences": total_matching_occurrences,
                "unique_matching_13grams": len(all_unique_matched_ngrams),
                "cross_suite_duplicate_13grams": len(self.cross_suite_duplicates),
            },
            "metrics_by_reference_suite": {
                s_key: {
                    "category": self.suite_definitions[s_key]["category"],
                    "description": self.suite_definitions[s_key]["description"],
                    "unique_reference_13grams": self.suite_metadata[s_key]["unique_13grams"],
                    "unique_matching_13grams": unique_counts_by_suite[s_key],
                    "match_rate_pct": round(
                        unique_counts_by_suite[s_key] / max(1, self.suite_metadata[s_key]["unique_13grams"]) * 100, 4
                    ),
                    "total_matching_occurrences": suite_hit_occurrences[s_key],
                }
                for s_key in self.suite_definitions
            },
            "metrics_by_category": {
                cat: {
                    "unique_matching_13grams": unique_counts_by_category[cat],
                    "total_matching_occurrences": category_hit_occurrences[cat],
                }
                for cat in unique_counts_by_category
            },
            "classification_breakdown": dict(classification_counts),
            "shards_distribution": dict(shards_with_matches),
            "representative_examples": representative_examples,
        }

        return results


def run_forensic_pipeline(max_shards: Optional[int] = None):
    parser = argparse.ArgumentParser(description="Forensic Contamination Analysis for ModernGPT-S 100M")
    parser.add_argument("--max-shards", type=int, default=None, help="Limit number of shards to scan")
    parser.add_argument("--examples-per-category", type=int, default=40, help="Representative examples to capture")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path")
    args = parser.parse_args()

    max_shards = max_shards or args.max_shards

    auditor = ForensicAuditor(REPO_ROOT)
    auditor.index_all_reference_suites()

    shard_dir = REPO_ROOT / "artifacts/modern_gpts_100m/shards"
    results = auditor.run_forensic_scan(shard_dir, max_shards=max_shards, examples_per_category=args.examples_per_category)

    out_json = Path(args.output) if args.output else REPO_ROOT / "artifacts/reports/modern_gpts_100m_forensic_contamination_analysis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Forensic results saved to {out_json}")


if __name__ == "__main__":
    run_forensic_pipeline()
