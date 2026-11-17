# ModernGPT-S 500M — Phase 1B.3: Independent Shard Integrity Report

## Executive Summary

Every generated binary shard was subjected to an independent 15-point data integrity evaluation via `modern_gpt/scratch/verify_g3_shards.py`.

All 4 shards passed all 15 integrity tests ($100\%$ pass rate, 60/60 test assertions).

---

## 1. 15-Point Integrity Test Suite Matrix

| Test ID | Verification Scope | Invariant / Expected Property | Shard 0 | Shard 1 | Shard 2 | Val Shard | Status |
|---|---|---|:---:|:---:|:---:|:---:|:---:|
| **T01** | Magic & Version | Magic `b"MGPT"`, Version `1` | PASS | PASS | PASS | PASS | **PASS** |
| **T02** | Header Length | Fixed 80 bytes | PASS | PASS | PASS | PASS | **PASS** |
| **T03** | Byte Order & Dtype | `dtype == 1` (`<u2`, Little-Endian uint16) | PASS | PASS | PASS | PASS | **PASS** |
| **T04** | Tokenizer Fingerprint | Exactly `3559c7435e1d9330` | PASS | PASS | PASS | PASS | **PASS** |
| **T05** | Sequence Length | Exactly 2,048 tokens | PASS | PASS | PASS | PASS | **PASS** |
| **T06** | Token Count Invariant | `total_tokens == num_seqs * 2048` | PASS | PASS | PASS | PASS | **PASS** |
| **T07** | Token ID Range | All IDs $\in [0, 32767]$ | PASS | PASS | PASS | PASS | **PASS** |
| **T08** | Payload Byte Count | `len(payload) == total_tokens * 2` | PASS | PASS | PASS | PASS | **PASS** |
| **T09** | Payload SHA-256 Checksum | Matches internal header payload hash | PASS | PASS | PASS | PASS | **PASS** |
| **T10** | Full File SHA-256 | Bitwise immutable file digest | PASS | PASS | PASS | PASS | **PASS** |
| **T11** | Sequence Alignment | First, middle, and last seqs len 2,048 | PASS | PASS | PASS | PASS | **PASS** |
| **T12** | DataLoader Compatibility | Successfully streamed by PyTorch loader | PASS | PASS | PASS | PASS | **PASS** |
| **T13** | Domain Accounting | Traceable to verified domain pool | PASS | PASS | PASS | PASS | **PASS** |
| **T14** | Train/Val Split Isolation | 0 validation collisions in train shards | PASS | PASS | PASS | PASS | **PASS** |
| **T15** | Sequence Uniqueness | 0 duplicate sequences in sample window | PASS | PASS | PASS | PASS | **PASS** |

---

## 2. Cryptographic Checksum Inventory

### Training Shard 0 (`pilot_g3_train_shard_00000.bin`)
- **Payload SHA-256**: `8304eead1915a35c087d58c7d34f1d533157449c0414ba9b2228f3e64eecfda7`
- **File SHA-256**: `76aed0d63dbfbe346aa1c65a48b07c6ea2971115c9019afa33d6322c4c37aecb`
- **Token ID Range**: Min 9 (`<|endoftext|>`), Max 32,767
- **Verification Status**: **VALID**

### Training Shard 1 (`pilot_g3_train_shard_00001.bin`)
- **Payload SHA-256**: `21898c0d570a71ee7e356b73849c6bc8a14fc4b93d8c7eff72100bdf056e5464`
- **File SHA-256**: `2e285eea9defbf8444cc5a2a019b40c8d1aa56e55edc0b7462ecaef4e454ac80`
- **Token ID Range**: Min 9, Max 32,767
- **Verification Status**: **VALID**

### Training Shard 2 (`pilot_g3_train_shard_00002.bin`)
- **Payload SHA-256**: `ffebac8e79aed12df621dabb5df37ceee76c0797a6899f4bb643745e51f0c89f`
- **File SHA-256**: `133e44fbd6cdbbb23caaae3ca61d408e5afdad029f15647d06bdf3b5f7e9c4e0`
- **Token ID Range**: Min 9, Max 32,767
- **Verification Status**: **VALID**

### Validation Shard 0 (`pilot_g3_val_shard_00000.bin`)
- **Payload SHA-256**: `de01387538dc921cfea3b2214d1d15949646e28ab6e3ac2025cbc709e1f3357d`
- **File SHA-256**: `cbc3ca6b259b6c4e93e56c06f2ae36e47258a0b7e7566c15a73fd479372239d3`
- **Token ID Range**: Min 0 (`<|pad|>`), Max 32,767
- **Verification Status**: **VALID**

---

## 3. Boundary & Sequence Alignment Invariants

For each shard, the first sequence ($S_0$), middle sequence ($S_{\lfloor N/2 \rfloor}$), and final sequence ($S_{N-1}$) were inspected directly:
- Every sequence unpacked into exactly 2,048 discrete tokens.
- No sequence contained NaN, null bytes, negative values, or values $\ge 32,768$.
- Byte alignment was strictly verified: byte offset is divisible by 4,096 ($2,048 \times 2\text{ bytes}$).

