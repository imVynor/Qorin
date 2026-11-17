# ModernGPT-S 500M — Production Tokenizer Specification & Freeze Report

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **FROZEN & VERIFIED**  
**Name:** `bpe_32k_production`  
**Vocabulary Size:** Exactly $32,768$ (IDs $0$ to $32,767$)  
**Fingerprint:** `3559c7435e1d9330`  
**Artifact Directory:** `modern_gpt/artifacts/tokenizer_32k/`  

---

## 1. Executive Summary & Construction Strategy

In accordance with Phase 1B.2 Part A requirements, the production Byte-Pair Encoding (BPE) tokenizer for ModernGPT-S 500M has been trained, verified, and frozen.

The production tokenizer preserves the proven subword merge behavior of the candidate evaluated in Phase 1B.1 while formally reserving all ten atomic special tokens directly into the first ten contiguous IDs ($0$ through $9$).

### Exact Architectural Alignment:
- **Parameter Count Match:** Exactly $32,768$ vocabulary entries, matching the tied-embedding budget of **$514,389,504$ parameters** to the single parameter.
- **Tool Token Retention:**
  - `<|tool_call|>` = **ID 6** (Strict Phase 1A match)
  - `<|tool_result|>` = **ID 7** (Strict Phase 1A match)
  - `<|tool_error|>` = **ID 8** (Strict Phase 1A match)
- **Zero Sequence Length Drift:** Regression benchmarking on 365 held-out documents confirmed **$0.000\%$ sequence length variance** against the Phase 1A candidate.

---

## 2. Special-Token Registry & ID Map

| Token String | Assigned ID | Type / Function | Training Loss Mask | Semantics & Boundaries |
| :--- | :---: | :--- | :---: | :--- |
| `<|pad|>` | 0 | Padding | 0 (Masked) | Batch alignment and sequence padding |
| `<|unk|>` | 1 | Out-of-Vocabulary | 0 (Masked) | Safety fallback (unused due to byte-level base) |
| `<|bos|>` | 2 | Sequence Start | 0 (Masked) | Document opening delimiter |
| `<|eos|>` | 3 | Sequence End | 1 (Supervised) | Autoregressive generation completion |
| `<|im_start|>` | 4 | Turn Boundary | 1 (Assistant) / 0 (Others) | ChatML message opening |
| `<|im_end|>` | 5 | Turn Boundary | 1 (Assistant) / 0 (Others) | ChatML message closing |
| `<|tool_call|>` | **6** | Control Delimiter | **1 (Supervised)** | Invokes structured JSON tool call |
| `<|tool_result|>` | **7** | Control Delimiter | 0 (Masked) | Injects external tool execution output |
| `<|tool_error|>` | **8** | Control Delimiter | 0 (Masked) | Injects tool execution or validation failure |
| `<|endoftext|>` | 9 | Pretraining Boundary | 1 (Supervised) | Concatenation boundary between packed documents |

---

## 3. Vocabulary Geometry

$$\text{Total Vocabulary} = \mathbf{32,768\text{ tokens}}$$

- **Special Control Tokens (IDs $0$–$9$):** $10$ tokens.
- **Initial Byte Alphabet (IDs $10$–$265$):** $256$ byte values covering all valid UTF-8 sequences.
- **Learned BPE Merges (IDs $266$–$32,767$):** $32,502$ subword merges derived from $25,000$ representative pretraining documents.

---

## 4. Special-Token Verification & Trust Boundaries

The production tokenizer was subjected to 20 dedicated unit test probes:
1. **Atomic Encoding:** All 10 special tokens encode as single-element integer arrays (e.g. `"<|tool_call|>"` $\to$ `[6]`).
2. **Atomic Decoding:** Verbatim decoding (`skip_special_tokens=False`) reproduces all special tokens without character alteration.
3. **Punctuation & Unicode Adjacency:** Special tokens adjacent to colons, parentheses, or emojis (e.g. `":<|tool_call|>:"`, `"⚡<|tool_call|>🔥"`) encode atomically without splitting.
4. **Near-Special-Token Resistance:** User inputs containing near-matches (e.g. `"<|tool_call"`, `"<|tool_call_v2|>"`, `"<|im_start_new|>"`) do **not** trigger control token IDs, preventing prompt-injection spoofing.
5. **Multi-Turn ChatML Integrity:** Complex turns with nested JSON payloads, multiple tool calls, and error handling decode with 100% character fidelity.

---

## 5. Artifact Inventory & Cryptographic Checksums

Located at `modern_gpt/artifacts/tokenizer_32k/`:

| File Name | Size (Bytes) | SHA-256 Checksum | Description |
| :--- | :---: | :--- | :--- |
| `tokenizer.json` | $2,325,321$ | `0535ea247858c704caae4b2c14041e17d2da2ef124b2257d00fcfef2ba2f2ff8` | Full Hugging Face BPE model & merges |
| `tokenizer_config.json` | $427$ | `295ff66f1fc953b00feffebfcb87d65f57348e3cf3d7dfd313be825227aa71c4` | Production metadata and special-token map |
| `tokenizer_manifest.json`| $1,059$ | `5d48c894d84fbbec862fe8347f3b890f05f778736e6c27fc5e1141df90eef530` | Versioned provenance manifest |

**Permanent Tokenizer Fingerprint:** `3559c7435e1d9330`

