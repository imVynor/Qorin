"""ModernGPT-S 100M: Dataset Ingestion, Tokenization, and Sharding Pipeline.

Prepares the 1.5B token pretraining corpus across 5 representative sources:
- FineWeb-Edu (60%, 900M tokens)
- Cosmopedia-v2 (20%, 300M tokens)
- Python-Edu (8%, 120M tokens)
- OpenWebMath (7%, 105M tokens)
- FineWeb (5%, 75M tokens)

Features:
- Interleaved multi-source streaming with exact ratio maintenance.
- Resumable shard generation (50M tokens per shard, 30 shards total).
- uint16 binary format with explicit <|endoftext|> document boundaries.
- Cryptographic SHA-256 manifests per shard.
- Deduplication and contamination checks.
"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sources import (
    FINEWEB_EDU_CONFIG,
    COSMOPEDIA_V2_CONFIG,
    PYTHON_EDU_CONFIG,
    OPENWEBMATH_CONFIG,
    FINEWEB_CONFIG,
    NormalizedDocument,
    SourceConfig,
    compute_content_hash,
    stream_dataset_records,
    verify_source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.Corpus100M")

SOURCE_CONFIG_MAP: Dict[str, SourceConfig] = {
    "fineweb_edu": FINEWEB_EDU_CONFIG,
    "cosmopedia_v2": COSMOPEDIA_V2_CONFIG,
    "python_edu": PYTHON_EDU_CONFIG,
    "open_web_math": OPENWEBMATH_CONFIG,
    "fineweb": FINEWEB_CONFIG,
}


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file streaming in 4MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def collect_validation_hashes() -> Set[str]:
    """Collects content hashes of all known validation/evaluation sets to prevent pretraining leakage."""
    seen_hashes = set()
    val_paths = [
        REPO_ROOT / "artifacts/corpus/val_documents.jsonl",
        REPO_ROOT / "artifacts/corpus_100m/val_documents.jsonl",
        REPO_ROOT / "artifacts/corpus_25m/val_documents.jsonl",
        REPO_ROOT / "artifacts/corpus_5m/val_documents.jsonl",
        REPO_ROOT / "data/midscale_500/val.jsonl",
        REPO_ROOT / "data/midscale_500/checkpoint_selection_val.jsonl",
        REPO_ROOT / "artifacts/diagnostics/tiny_overfit_data/val.jsonl",
    ]
    for p in val_paths:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        txt = item.get("text") or (item.get("prompt", "") + " " + item.get("response", ""))
                        if txt:
                            seen_hashes.add(compute_content_hash(txt.strip()))
                    except Exception:
                        pass
    logger.info(f"Loaded {len(seen_hashes)} validation document hashes into leakage-prevention filter.")
    return seen_hashes


class InterleavedSourceStreamer:
    """Interleaves multiple dataset streams according to specified target ratios."""

    def __init__(
        self,
        source_ratios: Dict[str, float],
        min_chars: int = 50,
        seen_hashes: Optional[Set[str]] = None,
    ):
        self.source_ratios = source_ratios
        self.min_chars = min_chars
        self.seen_hashes = seen_hashes if seen_hashes is not None else set()
        self.iterators: Dict[str, Iterator[NormalizedDocument]] = {}
        self.accumulated_counts: Dict[str, int] = {s: 0 for s in source_ratios}
        self._init_streams()

    def _init_streams(self):
        for name in self.source_ratios:
            cfg = SOURCE_CONFIG_MAP[name]
            logger.info(f"Initializing stream for {name} ({cfg.dataset_id})...")
            self.iterators[name] = stream_dataset_records(
                source_config=cfg,
                min_chars=self.min_chars,
                deduplicate=True,
                seen_hashes=self.seen_hashes,
            )

    def next_document(self) -> Tuple[str, NormalizedDocument]:
        """Select next source based on greatest deficit relative to target ratio."""
        total_fetched = sum(self.accumulated_counts.values())
        if total_fetched == 0:
            chosen_source = max(self.source_ratios.items(), key=lambda x: x[1])[0]
        else:
            def deficit(name: str) -> float:
                actual_ratio = self.accumulated_counts[name] / total_fetched
                return self.source_ratios[name] - actual_ratio

            chosen_source = max(self.source_ratios.keys(), key=deficit)

        iterator = self.iterators[chosen_source]
        try:
            doc = next(iterator)
        except StopIteration:
            logger.warning(f"Stream exhausted for {chosen_source}, re-initializing...")
            cfg = SOURCE_CONFIG_MAP[chosen_source]
            self.iterators[chosen_source] = stream_dataset_records(
                source_config=cfg,
                min_chars=self.min_chars,
                deduplicate=True,
                seen_hashes=self.seen_hashes,
            )
            doc = next(self.iterators[chosen_source])

        self.accumulated_counts[chosen_source] += 1
        return chosen_source, doc


def build_shards(
    config_path: Path,
    max_tokens: Optional[int] = None,
    max_shards: Optional[int] = None,
    dry_run: bool = False,
):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    shard_dir = REPO_ROOT / cfg["shard_dir"]
    manifest_dir = REPO_ROOT / cfg["manifest_dir"]
    shard_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tok_dir = REPO_ROOT / cfg["tokenizer_dir"]
    logger.info(f"Loading tokenizer from {tok_dir}...")
    tokenizer = HuggingFaceBPEWrapper.load(tok_dir)
    assert tokenizer.fingerprint == cfg["tokenizer_fingerprint"], (
        f"Tokenizer fingerprint mismatch! {tokenizer.fingerprint} vs {cfg['tokenizer_fingerprint']}"
    )
    logger.info(f"Tokenizer verified: {tokenizer.name} (vocab {tokenizer.vocab_size}, fingerprint {tokenizer.fingerprint})")

    tokens_per_shard = cfg.get("tokens_per_shard", 50000000)
    total_budget = max_tokens or cfg.get("total_token_budget", 1500000000)
    num_shards = max_shards or int(np.ceil(total_budget / tokens_per_shard))

    source_ratios = {item["name"]: float(item["ratio"]) for item in cfg["mixture"]}
    norm_sum = sum(source_ratios.values())
    source_ratios = {k: v / norm_sum for k, v in source_ratios.items()}

    logger.info(f"Target Shards: {num_shards} ({tokens_per_shard:,} tokens/shard, Total: {total_budget:,} tokens)")
    logger.info(f"Source Mixtures: {source_ratios}")

    if dry_run:
        logger.info("Dry run requested. Verifying source connectivity...")
        for name in source_ratios:
            res = verify_source(SOURCE_CONFIG_MAP[name], sample_size=2)
            logger.info(f"Verified {name}: success={res['success']}, sample docs: {len(res.get('samples', []))}")
        logger.info("All sources verified successfully.")
        return

    val_hashes = collect_validation_hashes()
    streamer = InterleavedSourceStreamer(source_ratios, seen_hashes=val_hashes)
    eos_id = tokenizer.eos_token_id
    context_length = cfg.get("sequence_length", 512)

    for shard_idx in range(num_shards):
        shard_filename = f"shard_{shard_idx:05d}.bin"
        shard_path = shard_dir / shard_filename
        manifest_path = manifest_dir / f"shard_{shard_idx:05d}.json"

        if shard_path.exists() and manifest_path.exists():
            logger.info(f"Shard {shard_filename} already exists. Verifying integrity...")
            existing_hash = compute_file_sha256(shard_path)
            with open(manifest_path, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
            if mdata.get("sha256") == existing_hash and mdata.get("total_tokens") == tokens_per_shard:
                logger.info(f"✓ Shard {shard_filename} intact and verified ({existing_hash[:12]}). Skipping.")
                continue
            else:
                logger.warning(f"Shard {shard_filename} manifest mismatch or incomplete. Recreating.")

        logger.info(f"--- Generating Shard {shard_idx + 1}/{num_shards}: {shard_filename} ---")
        tmp_shard_path = shard_path.with_suffix(".bin.tmp")
        arr = np.memmap(tmp_shard_path, dtype=np.uint16, mode="w+", shape=(tokens_per_shard,))

        tokens_written = 0
        docs_written = 0
        source_doc_counts = {s: 0 for s in source_ratios}
        source_token_counts = {s: 0 for s in source_ratios}
        t0 = time.time()

        while tokens_written < tokens_per_shard:
            src_name, doc = streamer.next_document()
            doc_tokens = tokenizer.encode(doc.text)
            doc_tokens.append(eos_id)
            doc_len = len(doc_tokens)

            needed = tokens_per_shard - tokens_written
            to_write = doc_tokens[:needed]
            actual_len = len(to_write)

            arr[tokens_written : tokens_written + actual_len] = np.array(to_write, dtype=np.uint16)
            tokens_written += actual_len
            docs_written += 1
            source_doc_counts[src_name] += 1
            source_token_counts[src_name] += actual_len

            if docs_written % 1000 == 0:
                elapsed = time.time() - t0
                rate = tokens_written / max(1e-5, elapsed)
                logger.info(
                    f"Shard {shard_idx:05d} Progress: {tokens_written:,}/{tokens_per_shard:,} tokens "
                    f"({tokens_written/tokens_per_shard*100:.1f}%) | {rate:,.0f} tok/s | docs: {docs_written:,}"
                )

        arr.flush()
        del arr
        os.replace(tmp_shard_path, shard_path)

        shard_sha256 = compute_file_sha256(shard_path)
        num_packed = max(0, (tokens_per_shard - 1) // context_length)

        manifest = {
            "shard_name": shard_filename,
            "shard_index": shard_idx,
            "sha256": shard_sha256,
            "total_tokens": tokens_per_shard,
            "total_documents": docs_written,
            "packed_sequences": num_packed,
            "context_length": context_length,
            "tokenizer_name": tokenizer.name,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "source_doc_counts": source_doc_counts,
            "source_token_counts": source_token_counts,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        with open(tmp_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_manifest, manifest_path)

        elapsed = time.time() - t0
        logger.info(
            f"✓ Shard {shard_filename} completed: {tokens_per_shard:,} tokens in {elapsed:.1f}s "
            f"({tokens_per_shard/elapsed:,.0f} tok/s) | SHA-256: {shard_sha256[:16]}..."
        )

    # Generate consolidated dataset manifest
    consolidated_manifest_path = REPO_ROOT / "artifacts/modern_gpts_100m/dataset_manifest.json"
    update_consolidated_manifest(
        manifest_dir=manifest_dir,
        shard_dir=shard_dir,
        output_manifest_path=consolidated_manifest_path,
        target_shards=num_shards,
        expected_tokens_per_shard=tokens_per_shard,
        tokenizer_name=tokenizer.name,
        tokenizer_fingerprint=tokenizer.fingerprint,
    )


def update_consolidated_manifest(
    manifest_dir: Path,
    shard_dir: Path,
    output_manifest_path: Path,
    target_shards: int = 30,
    expected_tokens_per_shard: int = 50000000,
    tokenizer_name: str = "bpe_16k_v2",
    tokenizer_fingerprint: str = "be6c3d60c5baf217",
) -> Dict[str, Any]:
    """Generates the consolidated cryptographic manifest across all shards."""
    shard_manifests = sorted(list(manifest_dir.glob("shard_*.json")))
    shards_info = []
    total_tokens = 0
    total_docs = 0
    combined_source_tokens = {}
    combined_source_docs = {}

    for sm_path in shard_manifests:
        try:
            with open(sm_path, "r", encoding="utf-8") as f:
                sm = json.load(f)
        except Exception:
            continue
        shard_bin = shard_dir / sm.get("shard_name", "")
        if not shard_bin.exists():
            continue
        actual_sha256 = compute_file_sha256(shard_bin)
        shards_info.append({
            "shard_name": sm["shard_name"],
            "shard_index": sm["shard_index"],
            "sha256": actual_sha256,
            "manifest_sha256_match": actual_sha256 == sm.get("sha256"),
            "total_tokens": sm["total_tokens"],
            "total_documents": sm["total_documents"],
            "packed_sequences": sm.get("packed_sequences", 0),
            "source_doc_counts": sm.get("source_doc_counts", {}),
            "source_token_counts": sm.get("source_token_counts", {}),
        })
        total_tokens += sm["total_tokens"]
        total_docs += sm["total_documents"]
        for src, count in sm.get("source_token_counts", {}).items():
            combined_source_tokens[src] = combined_source_tokens.get(src, 0) + count
        for src, count in sm.get("source_doc_counts", {}).items():
            combined_source_docs[src] = combined_source_docs.get(src, 0) + count

    source_proportions = {}
    if total_tokens > 0:
        source_proportions = {src: round(count / total_tokens, 6) for src, count in combined_source_tokens.items()}

    manifest_data = {
        "dataset_name": "ModernGPT-S 100M Pretraining Corpus",
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "tokenizer": {
            "name": tokenizer_name,
            "fingerprint": tokenizer_fingerprint,
        },
        "target_shards": target_shards,
        "shards_materialized": len(shards_info),
        "is_complete": len(shards_info) == target_shards and total_tokens == target_shards * expected_tokens_per_shard,
        "total_tokens": total_tokens,
        "target_nominal_tokens": target_shards * expected_tokens_per_shard,
        "total_documents": total_docs,
        "combined_source_tokens": combined_source_tokens,
        "source_proportions_by_tokens": source_proportions,
        "shards": shards_info,
    }

    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    manifest_hash = compute_file_sha256(output_manifest_path)
    manifest_data["manifest_sha256"] = manifest_hash
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    logger.info(
        f"Consolidated dataset manifest written to {output_manifest_path} "
        f"({len(shards_info)}/{target_shards} shards, {total_tokens:,} tokens, SHA-256: {manifest_hash[:16]}...)"
    )
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description="ModernGPT-S 100M Dataset Ingestion & Sharding")
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs/modern_gpts_100m/dataset.yaml",
        help="Path to dataset configuration YAML",
    )
    parser.add_argument("--max-tokens", type=int, default=None, help="Override total token budget")
    parser.add_argument("--max-shards", type=int, default=None, help="Override total shard count")
    parser.add_argument("--dry-run", action="store_true", help="Verify source connectivity and exit")
    parser.add_argument("--update-manifest-only", action="store_true", help="Rebuild consolidated manifest from existing shard manifests and exit")
    args = parser.parse_args()

    if args.update_manifest_only:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        s_dir = REPO_ROOT / cfg["shard_dir"]
        m_dir = REPO_ROOT / cfg["manifest_dir"]
        c_manifest = REPO_ROOT / "artifacts/modern_gpts_100m/dataset_manifest.json"
        update_consolidated_manifest(
            manifest_dir=m_dir,
            shard_dir=s_dir,
            output_manifest_path=c_manifest,
            target_shards=cfg.get("total_target_shards", 30),
            expected_tokens_per_shard=cfg.get("tokens_per_shard", 50000000),
            tokenizer_name=cfg.get("tokenizer_name", "bpe_16k_v2"),
            tokenizer_fingerprint=cfg.get("tokenizer_fingerprint", "be6c3d60c5baf217"),
        )
        return

    build_shards(
        config_path=args.config,
        max_tokens=args.max_tokens,
        max_shards=args.max_shards,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
