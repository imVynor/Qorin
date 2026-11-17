# Phase 1B.7: Binary uint16 Shard Specification & Header Layout

## 1. 80-Byte Structured Header Layout

Every `.bin` shard begins with an exact 80-byte little-endian header:

| Offset (Bytes) | Field Name | Type | Format String | Standard Value | Description |
|---|---|:---:|:---:|:---:|---|
| `00 - 03` | `magic` | `char[4]` | `4s` | `b"MGPT"` | ModernGPT file format identifier |
| `04 - 05` | `version` | `uint16` | `H` | `1` | Format specification revision |
| `06 - 07` | `dtype` | `uint16` | `H` | `1` | Data type enum (`1` = little-endian uint16) |
| `08 - 15` | `total_tokens` | `uint64` | `Q` | Variable | Total token count in shard payload |
| `16 - 23` | `sequence_length` | `uint64` | `Q` | `2048` | Target context length |
| `24 - 39` | `tokenizer_fp` | `char[16]` | `16s` | `b"3559c7435e1d9330"` | Production tokenizer fingerprint |
| `40 - 71` | `payload_sha256` | `byte[32]` | `32s` | Variable | SHA-256 digest of payload bytes |
| `72 - 79` | `reserved` | `byte[8]` | `8s` | `b"\x00" * 8` | Reserved alignment padding |

## 2. Payload Serialization

- Following the 80-byte header, the token payload is stored as contiguous little-endian `uint16` values.
- $\text{Payload Bytes} = 2 \times \text{Total Tokens}$
- $\text{File Size} = 80 + (2 \times \text{Total Tokens})$
- Standard 50M Shard: $80 + 100,000,000 = 100,000,080\text{ bytes}$ (~95.37 MiB).
