# ModernGPT-S Phase 2A — Dataset Integrity Report

**Release Version:** 1.1.0-production-release  
**Generated:** 2026-11-15T15:36:39Z  
**Model Architecture Target:** ModernGPT-S 500M Decoder-Only Transformer  
**Canonical Tokenizer:** `bpe_32k_production` (Vocab: 32,768)

---

## 1. Executive Summary

This report establishes the data integrity and cryptographic verification of the production SFT dataset release stored at [data_sft_release](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_release). Every record has undergone complete schema validation, token delimiter compliance auditing, loss masking verification, and contamination isolation against 566 evaluation probe fingerprints.

Zero corruption, zero delimiter collisions, and zero benchmark contamination were detected across all 238,404 records.

---

## 2. Cryptographic Split Fingerprints

All dataset partitions have been immutably hashed and verified:

| Partition | Records | Assistant Tokens | Total Conversation Tokens | SHA-256 Checksum |
|---|---:|---:|---:|---|
| **Train** | 226,486 | 72,682,363 | 137,164,498 | `622240e93f850e33bfeaae719fb290c6156751df60e29e7d344a720e3aa219e9` |
| **Validation** | 5,959 | 1,885,027 | ~3,556,120 | `77d7f8d19867f071b09b4895aa48454a7a6c9e2d5a1ed16a214f117b15e637dc` |
| **Test** | 5,959 | 1,912,291 | ~3,607,410 | `b3798c6fcf1213c473a445fce1575e7dd11ff2863949e18547eb85bc7d641923` |
| **Total** | **238,404** | **76,479,681** | **~144,328,028** | — |

---

## 3. Delimiter & Loss Mask Integrity

The canonical tokenizer uses specialized ChatML control tokens:
- `<|im_start|>` (ID: 4)
- `<|im_end|>` (ID: 5)
- `<|tool_call|>` (ID: 6)
- `<|tool_result|>` (ID: 7)
- `<|tool_error|>` (ID: 8)
- `<|eos|>` (ID: 3)

### Validation Criteria Audited:
1. **Turn Delimiters:** 100% of messages open with `<|im_start|>{role}\n` and terminate with `<|im_end|>\n`.
2. **Tool Boundary Pairing:** Every `<|tool_call|>` is followed by an authentic execution result enclosed in `<|tool_result|>...<|im_end|>`.
3. **Loss Masking (-100):** System prompts, user instructions, and environment tool results are strictly assigned label `-100`. Assistant generations (including tool call requests and final explanations) are supervised.
4. **No Premature EOS:** No internal `<|im_end|>` or `<|endoftext|>` tokens appear inside user queries or tool payload strings.

For an audited token-by-token matrix, inspect [loss_mask_example.md](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase_2a_sft_preparation/loss_mask_example.md).
