# ModernGPT-S 500M — Binary uint16 Shard Format Specification

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **SPECIFICATION FROZEN & OPERATIONAL**  
**Module:** `modern_gpt/data_500m/sharding.py`  

---

## 1. Executive Summary

To achieve high-throughput zero-copy streaming from SSD into GPU memory during pretraining, ModernGPT-S 500M utilizes a custom, versioned binary shard format.

Because the production vocabulary contains exactly $32,768$ entries ($0$ to $32,767$), every token ID fits compactly in an unsigned 16-bit integer (`uint16`). This halves the storage footprint and memory bus bandwidth compared to standard 32-bit (`int32` / `int64`) token arrays.

---

## 2. Binary File Layout (80-Byte Header + Contiguous Payload)

```
Offset (Bytes)  Size     Type     Field Name               Description
────────────────────────────────────────────────────────────────────────────────────────
00 - 03         4 B      char[4]  magic_number             Magic bytes: b"MGPT" (0x4D475054)
04 - 05         2 B      uint16   format_version           Version identifier (0x0001 -> v1)
06 - 07         2 B      uint16   data_type                Dtype identifier (0x0001 -> uint16)
08 - 15         8 B      uint64   total_tokens             Total token count in payload
16 - 23         8 B      uint64   sequence_length          Packed sequence length (2048)
24 - 39         16 B     char[16] tokenizer_fingerprint    16-char ASCII hex ("3559c7435e1d9330")
40 - 71         32 B     bytes[32]payload_sha256           Cryptographic SHA-256 of payload
72 - 79         8 B      bytes[8] reserved                 Reserved alignment padding (0x00)
────────────────────────────────────────────────────────────────────────────────────────
80 - EOF        N*2 B    uint16[] token_payload            Contiguous little-endian uint16 tokens
```

---

## 3. Shard Sizing & Physical Geometry

- **Target Tokens per Production Shard:** **$50,000,000\text{ tokens}$**.
- **Header Size:** Exactly $80\text{ bytes}$.
- **Payload Size:** $50,000,000 \times 2\text{ bytes} = 100,000,000\text{ bytes}$.
- **Total Physical Shard Size:** **$100,000,080\text{ bytes}$** ($\approx 95.37\text{ MB}$).
- **Target 5B-Token Corpus:** Exactly **$100\text{ shards}$** totaling $\approx 9.54\text{ GB}$.

---

## 4. Verification Protocol (`BinaryShardValidator`)

Before any binary shard is accepted for dataloader ingestion, `BinaryShardValidator.validate_shard()` performs seven automated checks:
1. **Magic Bytes Verification:** Validates `magic == b"MGPT"`.
2. **Version Compatibility:** Validates `version == 1`.
3. **Data Type Check:** Validates `dtype == 1` (`uint16`).
4. **Fingerprint Match:** Validates that the shard's embedded fingerprint strictly matches `model_500m.yaml` (`3559c7435e1d9330`).
5. **Length Consistency:** Validates that physical payload bytes equal `total_tokens * 2`.
6. **Payload Cryptographic Checksum:** Recomputes the SHA-256 digest over the binary payload and compares against header bytes $40$–$71$.
7. **Token Range Audit:** Validates that all values fall strictly within $[0, 32767]$.

