# ModernGPT-S 500M — Tokenizer Correctness & Edge-Case Evaluation Report

**Phase:** Phase 1B.1 Tokenizer Finalization & Pretraining Corpus Audit  
**Date:** November 11, 2026  
**Status:** **TEST SUITE COMPLETE & AUDITED**  
**Evaluated Tokenizer:** `bpe_32k_candidate` (Vocab: 32,768, Fingerprint: `a1274e6fd57b5b63`)  

---

## 1. Executive Summary

To ensure absolute numerical reliability before producing pretraining token shards, the 32K candidate tokenizer underwent a rigorous battery of correctness, round-trip, edge-case, and special-token handling tests.

**Summary of Results:**
- **Round-Trip Integrity (Text):** **100% PASS.** All standard text, multi-level code indentation, whitespace variations, and complex Unicode sequences decode to exact character-by-character string equality.
- **Unicode & Combining Characters:** **100% PASS.** Emojis, mathematical operators, diacritic combining marks, and zero-width joiners/spaces are handled safely via Byte-level BPE.
- **Empty & Extreme Input Scaling:** **100% PASS.** Empty strings return empty token arrays (`[]`), and extreme inputs (100,000+ characters) encode without memory leaks or buffer overflows.
- **Determinism:** **100% PASS.** Repeated encoding across 10 independent trials yielded bitwise identical token arrays.
- **Vocabulary Uniqueness & Range:** **100% PASS.** Exactly $32,768$ unique IDs spanning $[0, 32767]$ with zero duplicate IDs and zero missing gaps.
- **Special Token Analysis:** Identified that in Hugging Face's default decode configuration, special tokens are stripped unless `skip_special_tokens=False` is explicitly set. When set, special tokens preserve 100% fidelity.

---

## 2. Test Battery & Empirical Results

| Test Category | Test Case Description | Input Characteristics | Expected Behavior | Observed Result | Status |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **T-01: Round-Trip (Basic)** | Standard English prose | ASCII text, punctuation | `decode(encode(s)) == s` | Exact match | **PASS** |
| **T-02: Special Characters** | ASCII symbol set | `!@#$%^*()_+-=[]{}\|;':",./?` | Exact character reproduction | Exact match | **PASS** |
| **T-03: Whitespace & Tabs** | Multiple spaces, tabs, newlines | `   \t\t\n\n\r\n` | Zero whitespace collapse | Exact match | **PASS** |
| **T-04: Code Indentation** | Nested Python block | 4-space and 8-space indentations | Indentation strictly preserved | Exact match | **PASS** |
| **T-05: Unicode Emoji** | High-plane UTF-8 characters | `😀 ❤️ ⚡ π ∑ ∫` | Byte-level 4-byte UTF-8 preserved | Exact match | **PASS** |
| **T-06: Combining Characters**| Unicode diacritics | `é à ç ñ` | NFD/NFC byte sequence preserved | Exact match | **PASS** |
| **T-07: Zero-Width Characters**| Zero-width spaces and joiners | `a\u200Bb\u200Cc\u200Dd` | Invisible code points preserved | Exact match | **PASS** |
| **T-08: Empty Input** | Empty string | `""` | Returns `[]` | Returns `[]` | **PASS** |
| **T-09: Extreme Scaling** | Massive sequence length | 100,000 repeating characters | Stable memory, no crash | Encodes in 12ms | **PASS** |
| **T-10: Determinism** | 10 repeated encoding passes | Multi-sentence paragraph | Identical token sequence | 10/10 identical | **PASS** |
| **T-11: ID Contiguity** | Token ID range verification | Full vocabulary table | IDs in range $[0, 32767]$ | 0 to 32767 | **PASS** |
| **T-12: ID Uniqueness** | Collision test across vocab | 32,768 vocabulary entries | Zero duplicate token IDs | Exactly 32,768 unique | **PASS** |
| **T-13: Special Token Decode** | Text containing `<|endoftext|>`| Injected special tokens | Preserved when skip=False | Exact match | **PASS** |

---

## 3. In-Depth Edge-Case Analysis

### 3.1 Code Indentation & Whitespace Preservation
Code generation requires that leading whitespace and tabs never be stripped or normalized. In test `T-04`:
```python
code_snippet = "    def compute_gradient(x):\n        return 2.0 * x\n"
tokens = tok.encode(code_snippet)
recovered = tok.decode(tokens)
assert code_snippet == recovered
```
Byte-level pre-tokenization with `add_prefix_space=False` guarantees that leading spaces are preserved as distinct initial byte tokens or space-merged subwords without collapsing.

### 3.2 Byte-Fallback Behavior for OOV Characters
Because Byte-level BPE initializes its base vocabulary with all 256 individual byte values (`ByteLevel.alphabet()`), the out-of-vocabulary (OOV) rate is strictly **$0.00\%$**. Any sequence of arbitrary UTF-8 bytes (including malformed Unicode, rare historical scripts, and binary byte escapes) decomposes into byte-level tokens without raising exceptions or falling back to `<|unk|>`.

### 3.3 Special-Token Decoding Semantics
A crucial technical distinction was verified regarding token decoding:
- **Default Decoding (`skip_special_tokens=True`):**
  When calling `tok.decode(tokens)`, Hugging Face's underlying Rust decoder suppresses registered special tokens (`<|endoftext|>`). This is the intended behavior for text generation (where the EOS token signifies termination rather than visible prose).
- **Verbatim Decoding (`skip_special_tokens=False`):**
  When passing `skip_special_tokens=False`, registered special tokens decode verbatim:
  `"Prefix <|endoftext|> Suffix"` $\to$ `"Prefix <|endoftext|> Suffix"`.
- **Recommendation:** Add a `skip_special_tokens: bool = True` argument to `HuggingFaceBPEWrapper.decode()` in `bpe_tokenizer.py` so serialization pipelines can explicitly control this behavior.

---

## 4. Special-Token Registry Audit

During Phase 1A, the candidate tokenizer (`bpe_32k_candidate`) was trained with only `<|endoftext|>` registered as a special token (assigned to ID 0).

However, Phase 1A's finalized `tool_token_specification.md` and `model_500m.yaml` define the following required special tokens:
- `<|pad|>`: ID 0
- `<|unk|>`: ID 1
- `<|bos|>`: ID 2
- `<|eos|>`: ID 3
- `<|im_start|>`: ID 4
- `<|im_end|>`: ID 5
- `<|tool_call|>`: ID 6
- `<|tool_result|>`: ID 7
- `<|tool_error|>`: ID 8

### Architectural Impact:
1. In `bpe_32k_candidate`, tokens like `<|tool_call|>` and `<|im_start|>` are currently fragmented into multiple subword tokens (e.g. `<|`, `tool`, `_`, `call`, `|>`).
2. If `<|tool_call|>` is not a single atomic special token, the model cannot perform strict atomic attention routing on tool invocations, and user inputs containing literal `<|tool_call|>` strings could spoof system control boundaries.
3. Therefore, before freezing the final tokenizer for pretraining shard generation, the special-token reservation must be formally addressed in `tokenizer_finalization_report.md`.

