# ModernGPT-S 500M — Reproducible Corpus Pipeline Architecture

**Phase:** Phase 1B.1 Tokenizer Finalization & Pretraining Corpus Audit  
**Date:** November 11, 2026  
**Status:** **PIPELINE ARCHITECTURE SPECIFIED**  
**Target Corpus:** 5.00-Billion-Token Pretraining Mixture (2,048-Token Packed Shards)  

---

## 1. Executive Summary & Design Principles

To ensure complete reproducibility, auditability, and fault-tolerant data streaming during 500M pretraining, we specify the architectural blueprint for the ModernGPT-S 500M pretraining corpus pipeline.

Core Architecture Principles:
1. **Immutability:** Raw downloaded source records are write-protected and cryptographically fingerprinted.
2. **Determinism:** All filtering, deduplication, domain interleaving, and sequence packing steps operate with fixed pseudo-random seeds (Seed: 42) and reproducible deterministic sort keys.
3. **Fault-Tolerant Resumption:** The PyTorch data loader tracks exact shard index and token offsets, enabling seamless resumption following system reboots.
4. **Strict Tokenizer Version Binding:** Binary shards embed the tokenizer's SHA-256 fingerprint in the file header, preventing accidental cross-tokenizer training.

---

## 2. Directory Hierarchy & Storage Layout

```
modern_gpt/data/corpus_500m/
├── raw/                                # Raw immutable source downloads
│   ├── web_fineweb_edu/
│   ├── edu_cosmopedia_v2/
│   ├── code_the_stack_v2/
│   ├── wiki_en_2024/
│   ├── math_openwebmath/
│   ├── books_standard_ebooks/
│   └── dialogue_ultrachat/
├── clean/                              # Filtered & deduplicated JSONL
│   ├── 01_web.jsonl
│   ├── 02_edu.jsonl
│   ├── 03_code.jsonl
│   ├── 04_wiki.jsonl
│   ├── 05_math.jsonl
│   ├── 06_books.jsonl
│   └── 07_dialogue.jsonl
├── shards/                             # Production binary token shards
│   ├── train/                          # 5.00B tokens (100 shards x 50M tokens)
│   │   ├── shard_00000.bin
│   │   ├── shard_00001.bin
│   │   └── ...
│   └── val/                            # 100M tokens (2 shards x 50M tokens)
│       ├── shard_val_00000.bin
│       └── shard_val_00001.bin
└── manifests/                          # Audit trails, fingerprints & checksums
    ├── source_manifest.json
    ├── filtering_report.json
    ├── deduplication_summary.json
    └── shard_index.json
```

---

## 3. Binary Shard Format Specification

To achieve $>100,000\text{ tokens/second}$ zero-copy streaming from SSD into GPU memory without JSON parsing overhead, tokenized data is compiled into contiguous binary files:

### 3.1 Binary Shard Header (64 Bytes)
```
Bytes 00-03 : Magic Number (0x4D475054 -> "MGPT")
Bytes 04-05 : Version (uint16: 0x0001 -> v1.0)
Bytes 06-07 : Data Type (uint16: 0x0001 -> uint16 tokens)
Bytes 08-15 : Total Tokens in Shard (uint64: 50,000,000)
Bytes 16-23 : Sequence Length (uint64: 2,048)
Bytes 24-39 : Tokenizer Fingerprint (16-byte ASCII hex: e.g. "a1274e6fd57b5b63")
Bytes 40-47 : Domain Mixture Hash (uint64)
Bytes 48-63 : Reserved for alignment / future flags
```

### 3.2 Token Payload
- Following the 64-byte header, tokens are stored as contiguous little-endian `uint16` integers ($2\text{ bytes per token}$).
- Shard size: $64\text{ bytes} + (50,000,000 \times 2\text{ bytes}) = 100,000,064\text{ bytes} \approx \mathbf{95.37\text{ MB}}$.
- Sequence boundary handling: Sequences of length 2,048 are packed sequentially. If a document ends mid-sequence, a `<|endoftext|>` token is inserted, and the next document immediately begins.

---

## 4. Checkpointable Streaming Dataloader Design

During long pretraining runs (which take several days on a laptop GPU), training runs may pause, restart, or reboot.

```python
class CheckpointableShardedDataset(torch.utils.data.IterableDataset):
    def __init__(self, shard_dir: Path, current_shard: int = 0, current_offset: int = 0):
        self.shard_files = sorted(list(shard_dir.glob("shard_*.bin")))
        self.current_shard_idx = current_shard
        self.current_token_offset = current_offset

    def state_dict(self) -> Dict[str, Any]:
        """Save exact streaming state to model checkpoint."""
        return {
            "current_shard_idx": self.current_shard_idx,
            "current_token_offset": self.current_token_offset,
        }

    def load_state_dict(self, state: Dict[str, Any]):
        """Resume streaming without skipping or repeating tokens."""
        self.current_shard_idx = state["current_shard_idx"]
        self.current_token_offset = state["current_token_offset"]
```

---

## 5. Dataset Versioning Protocol

To protect experimental integrity, dataset releases follow semantic corpus versioning:

`corpus_500m_v<MAJOR>.<MINOR>`

- **MAJOR Version Bump (e.g. v1.0 $\to$ v2.0):**
  Triggered when altering domain proportions, switching tokenizer vocabulary, or adding new raw sources. Incompatible with ongoing training runs.
- **MINOR Version Bump (e.g. v1.0 $\to$ v1.1):**
  Triggered when fixing corrupted documents or re-ordering shards. Requires a new experiment baseline.
- **Run Immutability:** Every training run configuration locks to an explicit immutable dataset manifest checksum (`source_manifest.json` SHA-256), guaranteeing that historical training curves can always be reproduced exactly.

