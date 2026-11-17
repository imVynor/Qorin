# ModernGPT-S 500M — Phase 1B.3: Reproducibility and Determinism Report

## Executive Summary

Phase 1B.3 establishes complete, end-to-end reproducibility for the ModernGPT-S 500M data processing and binary sharding pipeline.

Given the identical source file inputs, random seed, production tokenizer artifact, and pipeline configuration, the pipeline generates identical logical token sequences, exact sequence counts, and bitwise deterministic binary shard payloads.

---

## 1. Deterministic Pipeline Configuration

### 1.1 Core Seeds and Randomness
- **Global Seed**: `42`
- **Partition Seed**: `1337`
- **MinHash Permutation Seed**: `42` (Fixed coefficient arrays $a_i, b_i$ generated with Mersenne Twister seed 42)
- **Hash Function**: Cryptographic MD5/SHA-256 (strictly platform-independent, immune to Python's randomized hash seed `PYTHONHASHSEED`).

### 1.2 Frozen Tokenizer Verification
- **Artifact Location**: `modern_gpt/artifacts/tokenizer_32k/`
- **Short Fingerprint**: `3559c7435e1d9330`
- **`tokenizer.json` SHA-256**: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`
- **`tokenizer_config.json` SHA-256**: `295ff66f1fc953b00feffebfcb87d65f57348e3cf3d7dfd313be825227aa71c4`
- **Vocabulary Size**: 32,768 entries

### 1.3 Packing and Sharding Invariants
- **Sequence Length**: 2,048 tokens
- **Delimiter Token ID**: 9 (`<|endoftext|>`)
- **Padding Token ID**: 0 (`<|pad|>`)
- **Shard Capacity**: Exactly 24,414 sequences per shard ($24,414 \times 2,048 = 49,999,872$ tokens)
- **Serialization Format**: Unsigned 16-bit integer, little-endian (`<u2`)
- **Header Length**: 80 bytes fixed with magic `b"MGPT"`

---

## 2. Source Code and Module Hashes

All data processing logic is encapsulated in `modern_gpt/data_500m/`:

| Module | Purpose | SHA-256 Digest |
|---|---|---|
| `filtering.py` | Quality and heuristic filtering | `bf6e3aa12574bb4bf6f34ae3bc6793f18e952672fc921a2245b0d00fca107865` |
| `dedup.py` | Exact MD5 and MinHash LSH deduplication | `bcae285d0d828a113bc321c17fae51c8cfdcf6ea1c94411136b360b64fec289c` |
| `packing.py` | Greedy contiguous document packing | `a200cf544b6fc76d7ae523364f9b882352faee5eeeaebfc607ea0ba0beea8a04` |
| `sharding.py` | Binary shard serialization and header packing | `2a8dc96a5127ef672f2e519c5c2fcbebb39562725fa84860cc3919b38ff13a37` |
| `dataloader.py` | PyTorch streaming iterable dataset | `fba97312ee255c2d33454ae1b46a36f86b1bb0fa3f6ce46ef2825a072c4ca9bf` |
| `adapters.py` | Source schema mapping and provenance tags | `13264ad97816be36209a25b3faeefdfeb9f31ab28d115aa0c5e93348c4ae0b61` |

---

## 3. Runtime Environment and Dependencies

- **Host OS**: Microsoft Windows 11 Pro (Build 10.0.26100)
- **Python Runtime**: CPython 3.11.9 64-bit
- **PyTorch**: 2.6.0+cu124
- **Hugging Face Tokenizers**: 0.21.0
- **NumPy**: 1.26.4
- **PSUtil**: 5.9.8

---

## 4. Interruption Recovery & Shard Immutability

The pipeline employs atomic, stateful shard generation:
1. Shards are written sequentially as numbered units (`pilot_g3_train_shard_00000.bin`, `pilot_g3_train_shard_00001.bin`, etc.).
2. The shard header stores an internal payload SHA-256 computed on the finished byte stream.
3. In the event of process interruption, existing completed shards are verified via `BinaryShardValidator.validate_shard()` and skipped during resumption.
4. No documents or tokens are duplicated across shards upon pipeline restart.

