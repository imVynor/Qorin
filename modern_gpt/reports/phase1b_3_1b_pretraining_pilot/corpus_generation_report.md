# ModernGPT-S 500M — Phase 1B.3: Corpus Generation and Shard Inventory Report

## Executive Summary

Phase 1B.3 generated a multi-shard pretraining pilot and held-out validation corpus using the production pipeline (`modern_gpt/data_500m/`).

All shards were written to the dedicated directory `modern_gpt/data_500m/pilot_shards/g3_1b/`, preserving existing G1 and G2 pilots without overwriting.

---

## 1. Corpus Shard Inventory

| Shard Identifier | Shard File Name | Partition | Sequences | Token Count | Payload Bytes | Total File Size (Bytes) | Full File SHA-256 |
|---|---|:---:|---:|---:|---:|---:|---|
| **Shard 0** | `pilot_g3_train_shard_00000.bin` | Train | 24,414 | 49,999,872 | 99,999,744 | 99,999,824 | `76aed0d63dbfbe346aa1c65a48b07c6ea2971115c9019afa33d6322c4c37aecb` |
| **Shard 1** | `pilot_g3_train_shard_00001.bin` | Train | 24,414 | 49,999,872 | 99,999,744 | 99,999,824 | `2e285eea9defbf8444cc5a2a019b40c8d1aa56e55edc0b7462ecaef4e454ac80` |
| **Shard 2** | `pilot_g3_train_shard_00002.bin` | Train | 24,414 | 49,999,872 | 99,999,744 | 99,999,824 | `133e44fbd6cdbbb23caaae3ca61d408e5afdad029f15647d06bdf3b5f7e9c4e0` |
| **Val Shard** | `pilot_g3_val_shard_00000.bin` | Val | 4,219 | 8,640,512 | 17,281,024 | 17,281,104 | `cbc3ca6b259b6c4e93e56c06f2ae36e47258a0b7e7566c15a73fd479372239d3` |
| **Total** | **4 Shards** | — | **77,461** | **158,640,128** | **317,280,256** | **317,280,576** | — |

---

## 2. Sequence and Payload Verification

### 2.1 Fixed Sequence Length
Every sequence in the generated corpus has an exact length of 2,048 tokens.
- Training sequences: $24,414 \times 3 = 73,242$ sequences
- Validation sequences: $4,219$ sequences
- Total sequences: $77,461$ sequences

### 2.2 Payload Sizing Invariant
For uint16 token IDs, payload byte count strictly satisfies:
$$\text{Payload Bytes} = 2 \times \text{Token Count}$$
- 50M Shards: $2 \times 49,999,872 = 99,999,744\text{ bytes}$
- Val Shard: $2 \times 8,640,512 = 17,281,024\text{ bytes}$
- Adding the fixed 80-byte header:
  - $99,999,744 + 80 = 99,999,824\text{ bytes}$
  - $17,281,024 + 80 = 17,281,104\text{ bytes}$
- Bitwise verification confirmed zero padding waste and zero payload truncation.

---

## 3. Storage Footprint

- **Total Disk Space Consumed**: 317,280,576 bytes (~317.28 MB)
- **Host Free Disk Capacity**: $>308\text{ GB}$ available on drive C.
- **Directory Layout**: Dedicated isolation under `modern_gpt/data_500m/pilot_shards/g3_1b/`.

