# ModernGPT-S 500M — Phase 1B.3: Production Tokenizer Integrity Report

## Executive Summary

As part of the mandatory preflight audit for Phase 1B.3 (1B-Token Pretraining Corpus Pilot), the canonical production tokenizer was independently verified for cryptographic integrity, vocabulary bounds, special token alignment, and shard header compatibility.

All tests passed with zero deviations.

---

## 1. Cryptographic Hashes and Identifiers

The production tokenizer artifact is frozen at `modern_gpt/artifacts/tokenizer_32k/`.

| Artifact File | Size (Bytes) | SHA-256 Digest |
|---|---:|---|
| `tokenizer.json` | 1,845,459 | `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` |
| `tokenizer_config.json` | 772 | `295ff66f1fc953b00feffebfcb87d65f57348e3cf3d7dfd313be825227aa71c4` |
| `tokenizer_manifest.json` | 1,512 | `0535ea247858c704caae4b2c14041e17d2da2ef124b2257d00fcfef2ba2f2ff8` |

- **Short Fingerprint**: `3559c7435e1d9330`
- **Derivation**: First 16 hex characters of the MD5/SHA representation of the frozen vocabulary configuration.
- **Shard Header Field**: Exactly 16 ASCII bytes stored at offset 32–47 in the 80-byte binary shard header (`fp_bytes.decode('ascii') == '3559c7435e1d9330'`).

---

## 2. Vocabulary Bounds and ID Integrity

- **Declared Vocabulary Size**: 32,768
- **Actual Vocabulary Size**: 32,768
- **Minimum Token ID**: 0 (`<|pad|>`)
- **Maximum Token ID**: 32,767
- **Contiguity**: Complete, contiguous range $[0, 32767]$ with zero gaps and zero collisions.
- **Data Type Compatibility**: All IDs fit strictly within an unsigned 16-bit integer (`uint16`, range 0–65,535). Shard serialization allocates exactly 2 bytes per token without overflow.

---

## 3. Canonical Special Token Mapping

ModernGPT-S 500M requires ten reserved special tokens in positions 0–9 to support pretraining, conversational ChatML, and agentic tool invocation.

| Special Token | Assigned ID | Semantic Function |
|---|---:|---|
| `<|pad|>` | 0 | Sequence padding token (loss-masked) |
| `<|unk|>` | 1 | Out-of-vocabulary fallback |
| `<|bos|>` | 2 | Beginning-of-sequence delimiter |
| `<|eos|>` | 3 | End-of-sequence delimiter |
| `<|im_start|>` | 4 | ChatML message start delimiter |
| `<|im_end|>` | 5 | ChatML message end delimiter |
| `<|tool_call|>` | 6 | Agentic tool call invocation boundary |
| `<|tool_result|>` | 7 | Environment tool return boundary |
| `<|tool_error|>` | 8 | Environment tool execution exception boundary |
| `<|endoftext|>` | 9 | Pretraining document boundary delimiter |

All 10 tokens match their specified integer IDs bit-for-bit.

---

## 4. Model Architecture & Config Synchronization

The model configuration file `modern_gpt/configs/modern_gpts_500m/model_500m.yaml` declares:
- `vocab_size: 32768`
- `pad_token_id: 0`
- `bos_token_id: 2`
- `eos_token_id: 3`

Independent checks confirmed:
- Model embedding matrix dimensions: `[32768, 1536]`
- Tied output projection matrix dimensions: `[32768, 1536]`
- Parameter budget: Exactly 514,389,504 tied parameters.

---

## 5. Audit Verdict

- **Integrity Status**: **PASS**
- **Tokenizer Freezing**: Confirmed immutable. No modifications were performed during Phase 1B.3.

