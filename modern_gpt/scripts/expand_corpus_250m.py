"""ModernGPT-S: 250M Training Corpus Expansion & Shard Assembly Pipeline.

Expands pretraining corpus from 102.27M to >= 250.0M usable training tokens:
1. Validates immutable historical artifacts (100M shards, corpus, tokenizer).
2. Verifies dataset provenance (FineWeb-Edu, Cosmopedia-v2, CodeParrot-Clean).
3. Enforces global deduplication against historical train (84,144) + val (9,348) documents + newly ingested docs.
4. Streams, normalizes, tokenizes (bpe_16k_v2), and buffers new training documents with resume capability.
5. Assembles uint16 binary shard `artifacts/shards_250m/train.bin` (historical + new) and mirrors `val.bin`.
6. Verifies exact token bounds, packing, target shifts, and dataset isolation.
"""

import argparse
import datetime
import hashlib
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from data.bpe_tokenizer import HuggingFaceBPEWrapper
from data.sharding import ShardedPackedDataset
from data.sources import (
    CODEPARROT_CLEAN_CONFIG,
    COSMOPEDIA_V2_CONFIG,
    FINEWEB_EDU_CONFIG,
    NormalizedDocument,
    SourceConfig,
    compute_content_hash,
    normalize_record,
    stream_dataset_records,
    verify_source,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ModernGPT.Corpus250M")

# Expected immutable hashes of historical shards
EXPECTED_HISTORICAL_TRAIN_SHA256 = "0d4cc2a74eff040895bea88f21b75aef40e588ff9e0713b84ce4779aa03e1e32"
EXPECTED_HISTORICAL_VAL_SHA256 = "d97678fc0b8923c763c69c0e35271662b3a4aef1d9d43589bc5b416f5916280d"
EXPECTED_TOKENIZER_FINGERPRINT = "be6c3d60c5baf217"
EXPECTED_VOCAB_SIZE = 16000


def get_file_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file streaming in 4MB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest().lower()


def preflight_integrity_check(shards_100m_dir: Path, tok_dir: Path):
    """Verify that historical artifacts exist, match expected checksums, and disk has ample space."""
    logger.info("=" * 70)
    logger.info("PRE-FLIGHT INTEGRITY & PROVENANCE VERIFICATION")
    logger.info("=" * 70)

    train_bin = shards_100m_dir / "train.bin"
    val_bin = shards_100m_dir / "val.bin"

    if not train_bin.exists() or not val_bin.exists():
        raise FileNotFoundError(f"Missing historical shards in {shards_100m_dir}")

    logger.info("Verifying historical train.bin SHA-256...")
    train_hash = get_file_sha256(train_bin)
    if train_hash != EXPECTED_HISTORICAL_TRAIN_SHA256:
        raise ValueError(
            f"Historical train.bin SHA-256 mismatch! Got {train_hash}, expected {EXPECTED_HISTORICAL_TRAIN_SHA256}"
        )
    logger.info("✓ Historical train.bin verified: %s", train_hash)

    logger.info("Verifying historical val.bin SHA-256...")
    val_hash = get_file_sha256(val_bin)
    if val_hash != EXPECTED_HISTORICAL_VAL_SHA256:
        raise ValueError(
            f"Historical val.bin SHA-256 mismatch! Got {val_hash}, expected {EXPECTED_HISTORICAL_VAL_SHA256}"
        )
    logger.info("✓ Historical val.bin verified: %s", val_hash)

    logger.info("Verifying active tokenizer...")
    tok = HuggingFaceBPEWrapper.load(tok_dir)
    if tok.fingerprint != EXPECTED_TOKENIZER_FINGERPRINT:
        raise ValueError(
            f"Tokenizer fingerprint mismatch! Got {tok.fingerprint}, expected {EXPECTED_TOKENIZER_FINGERPRINT}"
        )
    if tok.vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(f"Tokenizer vocab size mismatch! Got {tok.vocab_size}, expected {EXPECTED_VOCAB_SIZE}")
    logger.info("✓ Active tokenizer verified: %s (vocab: %d)", tok.fingerprint, tok.vocab_size)

    # Check free disk space
    import shutil
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    logger.info("Available disk space: %.2f GB", free_gb)
    if free_gb < 5.0:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.2f} GB free, require at least 5.0 GB")
    logger.info("✓ Disk space sufficient (> 5 GB).")


def load_historical_dedup_hashes(corpus_100m_dir: Path) -> Tuple[Set[str], Set[str]]:
    """Load all historical train and validation hashes from corpus_100m."""
    logger.info("Loading historical document hashes for global deduplication...")
    train_file = corpus_100m_dir / "train_documents.jsonl"
    val_file = corpus_100m_dir / "val_documents.jsonl"

    if not train_file.exists() or not val_file.exists():
        raise FileNotFoundError(f"Missing corpus documents in {corpus_100m_dir}")

    train_hashes: Set[str] = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            train_hashes.add(data["content_hash"])

    val_hashes: Set[str] = set()
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            val_hashes.add(data["content_hash"])

    logger.info(
        "✓ Loaded %d historical train hashes and %d historical val hashes (Total unique: %d, Overlap: %d)",
        len(train_hashes),
        len(val_hashes),
        len(train_hashes | val_hashes),
        len(train_hashes & val_hashes),
    )
    if len(train_hashes & val_hashes) > 0:
        raise RuntimeError("Integrity violation: historical train and val sets have hash overlap!")

    return train_hashes, val_hashes


def stream_source_with_budget(
    source_cfg: SourceConfig,
    token_target: int,
    tokenizer: HuggingFaceBPEWrapper,
    known_hashes: Set[str],
    doc_out_file,
    min_chars: int = 50,
) -> Dict[str, Any]:
    """Stream documents from a Hugging Face source until target token budget is met.
    
    Filters, deduplicates globally, validates token bounds, and appends to output file.
    """
    logger.info("-" * 70)
    logger.info(
        "Ingesting %s: Target = %d tokens (%s, rev: %s)...",
        source_cfg.source_name,
        token_target,
        source_cfg.dataset_id,
        source_cfg.revision[:10] if source_cfg.revision else "None",
    )

    stats = {
        "candidate_docs": 0,
        "accepted_docs": 0,
        "rejected_quality": 0,
        "rejected_dedup": 0,
        "tokens_accepted": 0,
        "bytes_accepted": 0,
        "chars_accepted": 0,
    }

    start_t = time.perf_counter()
    last_log_t = start_t
    eos_id = tokenizer.eos_token_id

    from datasets import load_dataset

    max_retries = 5
    retry_count = 0

    while stats["tokens_accepted"] < token_target and retry_count < max_retries:
        try:
            dataset = load_dataset(
                source_cfg.dataset_id,
                name=source_cfg.config_name,
                split=source_cfg.split,
                revision=source_cfg.revision,
                streaming=True,
            )

            for raw_record in dataset:
                stats["candidate_docs"] += 1

                doc = normalize_record(raw_record, source_cfg, min_chars=min_chars)
                if doc is None:
                    stats["rejected_quality"] += 1
                    continue

                if doc.content_hash in known_hashes:
                    stats["rejected_dedup"] += 1
                    continue

                # Tokenize with bpe_16k_v2
                tokens = tokenizer.encode(doc.text)
                # Check token ID bounds
                for tid in tokens:
                    if not (0 <= tid < tokenizer.vocab_size):
                        raise ValueError(f"Token ID {tid} out of vocab range [0, {tokenizer.vocab_size})")

                token_count = len(tokens) + 1  # includes EOS token

                # Register document
                known_hashes.add(doc.content_hash)
                stats["accepted_docs"] += 1
                stats["tokens_accepted"] += token_count
                stats["bytes_accepted"] += doc.byte_count
                stats["chars_accepted"] += doc.char_count

                record_line = {
                    "text": doc.text,
                    "source": doc.source,
                    "document_id": doc.document_id,
                    "char_count": doc.char_count,
                    "byte_count": doc.byte_count,
                    "content_hash": doc.content_hash,
                    "token_count": token_count,
                    "metadata": doc.metadata,
                }
                doc_out_file.write(json.dumps(record_line) + "\n")

                now = time.perf_counter()
                if stats["accepted_docs"] % 2000 == 0 or (now - last_log_t) > 30.0:
                    doc_out_file.flush()
                    elapsed = now - start_t
                    rate = stats["tokens_accepted"] / max(0.1, elapsed)
                    pct = (stats["tokens_accepted"] / token_target) * 100
                    logger.info(
                        "[%s] Progress: %d / %d tokens (%.1f%%) | Docs: %d (Dedup rej: %d, Qual rej: %d) | Rate: %.0f tok/s",
                        source_cfg.source_name,
                        stats["tokens_accepted"],
                        token_target,
                        pct,
                        stats["accepted_docs"],
                        stats["rejected_dedup"],
                        stats["rejected_quality"],
                        rate,
                    )
                    last_log_t = now

                if stats["tokens_accepted"] >= token_target:
                    break

        except Exception as err:
            retry_count += 1
            logger.warning(
                "[%s] Stream exception (retry %d/%d): %s. Reconnecting in 5s...",
                source_cfg.source_name,
                retry_count,
                max_retries,
                err,
            )
            time.sleep(5.0)

    if stats["tokens_accepted"] < token_target:
        raise RuntimeError(
            f"Failed to reach token target for {source_cfg.source_name} after {max_retries} retries: "
            f"got {stats['tokens_accepted']} / {token_target}"
        )

    doc_out_file.flush()
    elapsed = time.perf_counter() - start_t
    logger.info(
        "✓ Completed %s: %d tokens accepted in %d docs (%.1f sec, %.0f tok/s). Dedup rej: %d, Qual rej: %d",
        source_cfg.source_name,
        stats["tokens_accepted"],
        stats["accepted_docs"],
        elapsed,
        stats["tokens_accepted"] / max(0.1, elapsed),
        stats["rejected_dedup"],
        stats["rejected_quality"],
    )
    return stats


def assemble_shards_250m(
    shards_100m_dir: Path,
    shards_250m_dir: Path,
    new_docs_file: Path,
    tokenizer: HuggingFaceBPEWrapper,
    context_length: int = 512,
) -> Dict[str, Any]:
    """Assemble the unified 250M uint16 binary shard and mirror the validation shard."""
    logger.info("=" * 70)
    logger.info("ASSEMBLING UNIFIED 250M SHARDS")
    logger.info("=" * 70)

    train_bin_100m = shards_100m_dir / "train.bin"
    val_bin_100m = shards_100m_dir / "val.bin"

    out_train_bin = shards_250m_dir / "train.bin"
    out_train_tmp = shards_250m_dir / "train.bin.tmp"
    out_val_bin = shards_250m_dir / "val.bin"

    # 1. Mirror validation shard byte-for-byte
    logger.info("Mirroring validation shard to %s...", out_val_bin)
    shutil.copyfile(val_bin_100m, out_val_bin)
    val_sha = get_file_sha256(out_val_bin)
    assert val_sha == EXPECTED_HISTORICAL_VAL_SHA256, "Validation shard mirroring corrupted file!"
    shutil.copyfile(shards_100m_dir / "val.json", shards_250m_dir / "val.json")
    logger.info("✓ Validation shard mirrored successfully (SHA-256: %s)", val_sha)

    # 2. Assemble training shard: copy historical train.bin + append new tokenized docs
    logger.info("Streaming historical 100M tokens into train.bin.tmp...")
    historical_tokens = train_bin_100m.stat().st_size // 2  # uint16 = 2 bytes
    with open(out_train_tmp, "wb") as out_f:
        with open(train_bin_100m, "rb") as hist_f:
            while chunk := hist_f.read(4 * 1024 * 1024):
                out_f.write(chunk)

        logger.info("Tokenizing and appending new documents from %s...", new_docs_file)
        new_docs_count = 0
        new_tokens_count = 0
        eos_id = tokenizer.eos_token_id

        chunk_tokens = []
        with open(new_docs_file, "r", encoding="utf-8") as in_f:
            for line in in_f:
                data = json.loads(line)
                toks = tokenizer.encode(data["text"])
                toks.append(eos_id)
                chunk_tokens.extend(toks)
                new_docs_count += 1
                new_tokens_count += len(toks)

                if len(chunk_tokens) >= 500_000:
                    arr = np.array(chunk_tokens, dtype=np.uint16)
                    out_f.write(arr.tobytes())
                    chunk_tokens = []

            if chunk_tokens:
                arr = np.array(chunk_tokens, dtype=np.uint16)
                out_f.write(arr.tobytes())
                chunk_tokens = []

    # Rename atomically
    if out_train_bin.exists():
        out_train_bin.unlink()
    out_train_tmp.rename(out_train_bin)

    total_tokens = historical_tokens + new_tokens_count
    num_packed_sequences = max(0, (total_tokens - 1) // context_length)
    usable_tokens = num_packed_sequences * context_length

    train_sha = get_file_sha256(out_train_bin)
    logger.info("✓ Created unified train.bin: %d tokens (Usable: %d in %d packed seqs)", total_tokens, usable_tokens, num_packed_sequences)
    logger.info("  SHA-256: %s", train_sha)

    # Write train.json
    train_meta = {
        "tokenizer_name": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
        "split": "train",
        "total_tokens": total_tokens,
        "usable_tokens": usable_tokens,
        "num_packed_sequences": num_packed_sequences,
        "context_length": context_length,
        "shard_file": "train.bin",
        "sha256": train_sha,
        "dtype": "uint16",
        "eos_token_id": eos_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(shards_250m_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_meta, f, indent=2)

    return {
        "train_sha256": train_sha,
        "val_sha256": val_sha,
        "historical_tokens": historical_tokens,
        "new_tokens": new_tokens_count,
        "total_tokens": total_tokens,
        "usable_tokens": usable_tokens,
        "num_packed_sequences": num_packed_sequences,
    }


def verify_dataset_loader(shards_250m_dir: Path, context_length: int = 512):
    """Verify ShardedPackedDataset can open the newly generated shards and yield valid batches."""
    logger.info("=" * 70)
    logger.info("VALIDATING SHARDEDPACKEDDATASET READINESS")
    logger.info("=" * 70)

    train_dataset = ShardedPackedDataset(shards_250m_dir / "train.bin", context_length=context_length)
    val_dataset = ShardedPackedDataset(shards_250m_dir / "val.bin", context_length=context_length)

    logger.info("Train dataset sequences: %d", len(train_dataset))
    logger.info("Val dataset sequences:   %d", len(val_dataset))

    assert len(train_dataset) >= 488_282, f"Expected >= 488,282 sequences for 250M tokens, got {len(train_dataset)}"
    assert len(val_dataset) == 21_844, f"Expected 21,844 validation sequences, got {len(val_dataset)}"

    # Test sample item
    inp, tgt = train_dataset[0]
    assert inp.shape == (context_length,)
    assert tgt.shape == (context_length,)
    assert torch.equal(inp[1:], tgt[:-1]), "Target shift mismatch: targets[i] != inputs[i+1]"
    assert inp.min().item() >= 0 and inp.max().item() < 16000
    assert tgt.min().item() >= 0 and tgt.max().item() < 16000

    # Test last item
    last_inp, last_tgt = train_dataset[len(train_dataset) - 1]
    assert last_inp.shape == (context_length,)
    assert torch.equal(last_inp[1:], last_tgt[:-1])

    logger.info("✓ ShardedPackedDataset passed all tensor shape, target shift, and vocab bounds checks.")


def main():
    parser = argparse.ArgumentParser(description="Expand ModernGPT-S corpus to 250M tokens")
    parser.add_argument("--shards-100m", default="artifacts/shards_100m", help="Path to historical 100M shards")
    parser.add_argument("--corpus-100m", default="artifacts/corpus_100m", help="Path to historical 100M corpus")
    parser.add_argument("--shards-250m", default="artifacts/shards_250m", help="Path to output 250M shards")
    parser.add_argument("--tokenizer-dir", default="artifacts/tokenizer_16k_v2", help="Path to tokenizer")
    args = parser.parse_args()

    shards_100m_dir = Path(args.shards_100m)
    corpus_100m_dir = Path(args.corpus_100m)
    shards_250m_dir = Path(args.shards_250m)
    tok_dir = Path(args.tokenizer_dir)

    shards_250m_dir.mkdir(parents=True, exist_ok=True)

    # 1. Preflight
    preflight_integrity_check(shards_100m_dir, tok_dir)

    # 2. Tokenizer
    tok = HuggingFaceBPEWrapper.load(tok_dir)

    # 3. Load historical hashes for global deduplication
    train_hashes, val_hashes = load_historical_dedup_hashes(corpus_100m_dir)
    known_hashes = train_hashes | val_hashes

    # 4. Ingestion targets for 250.5M tokens with 70/20/10 blend
    # Historical training tokens by source:
    # FineWeb-Edu: 71,672,802 (70.08%)
    # Cosmopedia-v2: 20,397,525 (19.94%)
    # CodeParrot-Clean: 10,203,447 (9.98%)
    # Total historical: 102,273,774 tokens
    # Total target: 250,500,000 raw tokens
    # Total FineWeb-Edu target (70%): 175,350,000 -> Incremental: 103,677,198
    # Total Cosmopedia target (20%): 50,100,000 -> Incremental: 29,702,475
    # Total CodeParrot target (10%): 25,050,000 -> Incremental: 14,846,553
    # Total Incremental: 148,226,226 tokens
    source_incremental_targets = {
        "codeparrot-clean": (CODEPARROT_CLEAN_CONFIG, 14_846_553),
        "cosmopedia-v2": (COSMOPEDIA_V2_CONFIG, 29_702_475),
        "fineweb-edu": (FINEWEB_EDU_CONFIG, 103_677_198),
    }

    new_docs_file = shards_250m_dir / "new_train_documents.jsonl"
    ingestion_stats = {}

    # Check for resume
    already_collected_tokens = {src: 0 for src in source_incremental_targets}
    already_collected_docs = {src: 0 for src in source_incremental_targets}
    if new_docs_file.exists():
        logger.info("Found existing %s, checking resume state...", new_docs_file)
        with open(new_docs_file, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                known_hashes.add(d["content_hash"])
                s = d["source"]
                if s in already_collected_tokens:
                    already_collected_tokens[s] += d.get("token_count", 0)
                    already_collected_docs[s] += 1
        logger.info(
            "Resumed document state: CodeParrot: %d tok (%d docs), Cosmo: %d tok (%d docs), FineWeb: %d tok (%d docs)",
            already_collected_tokens["codeparrot-clean"],
            already_collected_docs["codeparrot-clean"],
            already_collected_tokens["cosmopedia-v2"],
            already_collected_docs["cosmopedia-v2"],
            already_collected_tokens["fineweb-edu"],
            already_collected_docs["fineweb-edu"],
        )

    with open(new_docs_file, "a", encoding="utf-8") as out_f:
        for src_name, (cfg, tgt_tokens) in source_incremental_targets.items():
            curr_tokens = already_collected_tokens[src_name]
            needed = tgt_tokens - curr_tokens
            if needed <= 0:
                logger.info("✓ %s target already satisfied (%d tokens).", src_name, curr_tokens)
                ingestion_stats[src_name] = {
                    "accepted_docs": already_collected_docs[src_name],
                    "tokens_accepted": curr_tokens,
                    "status": "already_satisfied",
                }
                continue

            logger.info("Starting streaming for %s (need %d more tokens)...", src_name, needed)
            src_stats = stream_source_with_budget(
                cfg,
                token_target=needed,
                tokenizer=tok,
                known_hashes=known_hashes,
                doc_out_file=out_f,
            )
            ingestion_stats[src_name] = src_stats

    # 5. Assemble unified shards
    assembly_res = assemble_shards_250m(
        shards_100m_dir=shards_100m_dir,
        shards_250m_dir=shards_250m_dir,
        new_docs_file=new_docs_file,
        tokenizer=tok,
        context_length=512,
    )

    # 6. Validate loader
    verify_dataset_loader(shards_250m_dir, context_length=512)

    # 7. Post-flight verification: Check historical shards were NOT modified
    logger.info("Verifying historical shards immutability post-assembly...")
    post_train_hash = get_file_sha256(shards_100m_dir / "train.bin")
    post_val_hash = get_file_sha256(shards_100m_dir / "val.bin")
    assert post_train_hash == EXPECTED_HISTORICAL_TRAIN_SHA256, "Historical train.bin was modified!"
    assert post_val_hash == EXPECTED_HISTORICAL_VAL_SHA256, "Historical val.bin was modified!"
    logger.info("✓ Historical shards verified 100% immutable and unchanged.")

    # 8. Write complete corpus expansion manifest
    manifest_data = {
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "target_corpus": "250M tokens pretraining",
        "tokenizer": {
            "name": tok.name,
            "vocab_size": tok.vocab_size,
            "fingerprint": tok.fingerprint,
        },
        "historical_inventory": {
            "train_tokens_raw": 102_273_774,
            "train_tokens_usable": 102_273_536,
            "val_tokens_raw": 11_184_395,
            "val_tokens_usable": 11_184_128,
            "train_documents": 84_144,
            "val_documents": 9_348,
        },
        "new_tokens_ingested": assembly_res["new_tokens"],
        "combined_inventory": {
            "train_tokens_raw": assembly_res["total_tokens"],
            "train_tokens_usable": assembly_res["usable_tokens"],
            "num_packed_sequences": assembly_res["num_packed_sequences"],
            "context_length": 512,
            "train_sha256": assembly_res["train_sha256"],
            "val_tokens_raw": 11_184_395,
            "val_tokens_usable": 11_184_128,
            "val_packed_sequences": 21_844,
            "val_sha256": assembly_res["val_sha256"],
        },
        "source_contributions": {
            "fineweb-edu": {
                "dataset_id": FINEWEB_EDU_CONFIG.dataset_id,
                "config": FINEWEB_EDU_CONFIG.config_name,
                "split": FINEWEB_EDU_CONFIG.split,
                "revision": FINEWEB_EDU_CONFIG.revision,
                "historical_tokens": 71_672_802,
                "new_tokens": ingestion_stats.get("fineweb-edu", {}).get("tokens_accepted", 0),
                "total_tokens": 71_672_802 + ingestion_stats.get("fineweb-edu", {}).get("tokens_accepted", 0),
            },
            "cosmopedia-v2": {
                "dataset_id": COSMOPEDIA_V2_CONFIG.dataset_id,
                "config": COSMOPEDIA_V2_CONFIG.config_name,
                "split": COSMOPEDIA_V2_CONFIG.split,
                "revision": COSMOPEDIA_V2_CONFIG.revision,
                "historical_tokens": 20_397_525,
                "new_tokens": ingestion_stats.get("cosmopedia-v2", {}).get("tokens_accepted", 0),
                "total_tokens": 20_397_525 + ingestion_stats.get("cosmopedia-v2", {}).get("tokens_accepted", 0),
            },
            "codeparrot-clean": {
                "dataset_id": CODEPARROT_CLEAN_CONFIG.dataset_id,
                "config": CODEPARROT_CLEAN_CONFIG.config_name,
                "split": CODEPARROT_CLEAN_CONFIG.split,
                "revision": CODEPARROT_CLEAN_CONFIG.revision,
                "historical_tokens": 10_203_447,
                "new_tokens": ingestion_stats.get("codeparrot-clean", {}).get("tokens_accepted", 0),
                "total_tokens": 10_203_447 + ingestion_stats.get("codeparrot-clean", {}).get("tokens_accepted", 0),
            },
        },
        "deduplication": {
            "scope": "Global exact SHA-256 content hash across historical train + historical val + all new incoming documents",
            "historical_hashes_count": len(train_hashes) + len(val_hashes),
            "final_unique_hashes_count": len(known_hashes),
            "rejection_stats": {
                src: {
                    "dedup_rejected": ingestion_stats.get(src, {}).get("rejected_dedup", 0),
                    "quality_rejected": ingestion_stats.get(src, {}).get("rejected_quality", 0),
                }
                for src in source_incremental_targets
            },
        },
        "verification_status": {
            "historical_shards_immutable": True,
            "target_shift_verified": True,
            "vocab_bounds_verified": True,
            "loader_verified": True,
            "gate_decision": "GO FOR PRETRAINING",
        },
    }

    manifest_file = shards_250m_dir / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved complete manifest to %s", manifest_file)
    logger.info("=" * 70)
    logger.info("250M CORPUS EXPANSION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
