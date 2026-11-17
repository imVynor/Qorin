"""Leakage-safe document splitting and manifest generation.

Splits documents into training and validation sets prior to tokenization.
Ensures:
- Deterministic pseudo-random assignment based on seed and document ID/hash.
- No document or duplicate hash crosses between train and validation.
- Preserves 80% FineWeb-Edu / 20% Cosmopedia v2 target ratio per split.
- Saves raw documents and split manifest to disk for reproducible downstream processing.
"""

import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .sources import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    total_documents: int
    total_bytes: int
    total_chars: int
    documents_by_source: Dict[str, int]
    bytes_by_source: Dict[str, int]
    source_proportions_by_bytes: Dict[str, float]
    source_proportions_by_docs: Dict[str, float]
    measured_tokens: Optional[int] = None
    tokens_by_source: Optional[Dict[str, int]] = None
    source_proportions_by_tokens: Optional[Dict[str, float]] = None


@dataclass
class CorpusManifest:
    seed: int
    val_ratio: float
    deduplication_scope: str
    unique_hashes_seen: int
    train_stats: SplitStats
    val_stats: SplitStats
    dataset_revisions: Dict[str, str]
    created_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


def split_documents_leakage_safe(
    documents: List[NormalizedDocument],
    val_ratio: float = 0.1,
    seed: int = 123,
    tokenizer: Optional[object] = None,
) -> Tuple[List[NormalizedDocument], List[NormalizedDocument], Dict[str, SplitStats]]:
    """Split documents deterministically by source to maintain proportions without leakage."""
    rng = random.Random(seed)

    # Group unique documents by source
    seen_hashes: Set[str] = set()
    by_source: Dict[str, List[NormalizedDocument]] = {}

    for doc in documents:
        if doc.content_hash in seen_hashes:
            continue
        seen_hashes.add(doc.content_hash)
        by_source.setdefault(doc.source, []).append(doc)

    train_docs: List[NormalizedDocument] = []
    val_docs: List[NormalizedDocument] = []

    for source_name, source_doc_list in by_source.items():
        # Shuffle deterministically per source
        shuffled = list(source_doc_list)
        # Use document ID as secondary tie-breaker for strict reproducibility across environments
        shuffled.sort(key=lambda d: d.document_id)
        rng.shuffle(shuffled)

        num_val = int(len(shuffled) * val_ratio)
        if len(shuffled) > 1 and num_val == 0 and val_ratio > 0:
            num_val = 1  # ensure at least 1 document in validation if requested

        val_subset = shuffled[:num_val]
        train_subset = shuffled[num_val:]

        val_docs.extend(val_subset)
        train_docs.extend(train_subset)

    # Shuffle train and val deterministically
    train_docs.sort(key=lambda d: d.document_id)
    val_docs.sort(key=lambda d: d.document_id)
    rng.shuffle(train_docs)
    rng.shuffle(val_docs)

    # Verify 0 hash or doc_id overlap
    train_hashes = {d.content_hash for d in train_docs}
    val_hashes = {d.content_hash for d in val_docs}
    train_ids = {d.document_id for d in train_docs}
    val_ids = {d.document_id for d in val_docs}

    hash_overlap = train_hashes.intersection(val_hashes)
    id_overlap = train_ids.intersection(val_ids)

    if hash_overlap:
        raise ValueError(f"Content hash leakage detected between train and val: {hash_overlap}")
    if id_overlap:
        raise ValueError(f"Document ID leakage detected between train and val: {id_overlap}")

    stats = {
        "train": compute_split_stats(train_docs, tokenizer=tokenizer),
        "val": compute_split_stats(val_docs, tokenizer=tokenizer),
    }

    return train_docs, val_docs, stats


def compute_split_stats(
    docs: List[NormalizedDocument],
    tokenizer: Optional[object] = None,
) -> SplitStats:
    total_docs = len(docs)
    total_bytes = sum(d.byte_count for d in docs)
    total_chars = sum(d.char_count for d in docs)

    docs_by_source: Dict[str, int] = {}
    bytes_by_source: Dict[str, int] = {}
    tokens_by_source: Dict[str, int] = {}

    for d in docs:
        docs_by_source[d.source] = docs_by_source.get(d.source, 0) + 1
        bytes_by_source[d.source] = bytes_by_source.get(d.source, 0) + d.byte_count
        if tokenizer is not None:
            tok_count = len(tokenizer.encode(d.text)) + 1  # +1 for EOS
            tokens_by_source[d.source] = tokens_by_source.get(d.source, 0) + tok_count

    prop_bytes = {
        k: round(v / total_bytes, 4) if total_bytes > 0 else 0.0
        for k, v in bytes_by_source.items()
    }
    prop_docs = {
        k: round(v / total_docs, 4) if total_docs > 0 else 0.0
        for k, v in docs_by_source.items()
    }

    measured_tokens = sum(tokens_by_source.values()) if tokenizer is not None else None
    prop_tokens = None
    if tokenizer is not None and measured_tokens and measured_tokens > 0:
        prop_tokens = {
            k: round(v / measured_tokens, 4)
            for k, v in tokens_by_source.items()
        }

    return SplitStats(
        total_documents=total_docs,
        total_bytes=total_bytes,
        total_chars=total_chars,
        documents_by_source=docs_by_source,
        bytes_by_source=bytes_by_source,
        source_proportions_by_bytes=prop_bytes,
        source_proportions_by_docs=prop_docs,
        measured_tokens=measured_tokens,
        tokens_by_source=tokens_by_source if tokenizer is not None else None,
        source_proportions_by_tokens=prop_tokens,
    )


def save_split_corpus(
    train_docs: List[NormalizedDocument],
    val_docs: List[NormalizedDocument],
    output_dir: Path,
    seed: int = 123,
    val_ratio: float = 0.1,
    revisions: Optional[Dict[str, str]] = None,
    tokenizer: Optional[object] = None,
) -> Path:
    """Save normalized train and validation documents as jsonlines and output manifest."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file = output_dir / "train_documents.jsonl"
    val_file = output_dir / "val_documents.jsonl"
    manifest_file = output_dir / "corpus_manifest.json"

    with train_file.open("w", encoding="utf-8") as f:
        for doc in train_docs:
            f.write(json.dumps(doc.to_dict()) + "\n")

    with val_file.open("w", encoding="utf-8") as f:
        for doc in val_docs:
            f.write(json.dumps(doc.to_dict()) + "\n")

    import datetime

    train_stats = compute_split_stats(train_docs, tokenizer=tokenizer)
    val_stats = compute_split_stats(val_docs, tokenizer=tokenizer)

    manifest = CorpusManifest(
        seed=seed,
        val_ratio=val_ratio,
        deduplication_scope="Bounded-sample exact SHA-256 content deduplication",
        unique_hashes_seen=len(train_docs) + len(val_docs),
        train_stats=train_stats,
        val_stats=val_stats,
        dataset_revisions=revisions or {},
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info("Saved corpus: %s (%d train docs, %d val docs)", output_dir, len(train_docs), len(val_docs))
    return manifest_file


def load_documents_jsonl(path: Path) -> List[NormalizedDocument]:
    """Load normalized documents from a jsonlines file."""
    docs = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                docs.append(NormalizedDocument.from_dict(json.loads(line)))
    return docs
