# ModernGPT-S 500M — Phase 1B.3: Preflight Audit & Verification Matrix

## Executive Summary

Before initiating the G3 1B-Token Pretraining Corpus Pilot, an exhaustive preflight audit was executed across all components of the data pipeline, frozen artifacts, baseline pilots, and domain supply repositories.

The infrastructure components passed all cryptographic, structural, and semantic tests. However, the preflight audit revealed a critical supply constraint: local storage contains only ~228.6M tokens of unrepeated pretraining data across the approved seven domains, representing a 77.1% shortfall against the 1,000,000,000-token target.

In strict compliance with project guidelines ("If a domain cannot reach its allocation, stop and report the shortfall. Do not silently compensate with another domain. Do not repeat documents to artificially fill the allocations"), Phase 1B.3 generated the maximum valid single-epoch pilot from available local sources while classifying the full 1B pilot as **PARTIALLY COMPLETE / BLOCKED ON EXTERNAL DOWNLOADS**.

---

## 1. Preflight Verification Matrix

| Check ID | Verification Item | Target / Invariant | Measured Result | Status |
|---|---|---|---|:---:|
| **A1.1** | Tokenizer Vocab Size | Exactly 32,768 entries | 32,768 entries | **PASS** |
| **A1.2** | Token ID Range | IDs contiguous in $[0, 32767]$ | Range: `[0, 32767]` | **PASS** |
| **A1.3** | Special Token IDs | 10 canonical tokens in IDs 0–9 | IDs 0–9 matched exactly | **PASS** |
| **A1.4** | Tokenizer Fingerprint | `3559c7435e1d9330` | `3559c7435e1d9330` | **PASS** |
| **A1.5** | Full SHA-256 Digest | Canonical `tokenizer.json` digest | `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59` | **PASS** |
| **A1.6** | Model Config Parity | Config `vocab_size: 32768` | Matches architecture exactly | **PASS** |
| **A2.1** | Delimiter Token | `<|endoftext|>` (ID 9) delimiter | Emitted between docs | **PASS** |
| **A2.2** | Causal LM Attention Semantics | Continuous-stream causal pretraining | Standard causal lower-triangular mask; statistical separation via `<|endoftext|>` | **PASS** |
| **A3.1** | Next-Token Slicing | Input $[0:2046]$, Target $[1:2047]$ | Input (1, 2047), Target (1, 2047) | **PASS** |
| **A3.2** | Autoregressive Invariant | $x[:, 1:] \equiv y[:, :-1]$ | Bitwise identical | **PASS** |
| **A3.3** | Loss Padding Masking | Padding token 0 ignored | Verified in loss definition | **PASS** |
| **A4.1** | G1 Pilot Revalidation | 1,998,848 tokens, header valid | Verified valid shard | **PASS** |
| **A4.2** | G2 Pilot Revalidation | 49,999,872 tokens, header valid | Verified valid shard | **PASS** |
| **B.1** | Quality Web Supply | 400,000,000 tokens (40.0%) | 162,210,616 tokens (40.6%) | **SHORTFALL** |
| **B.2** | Educational Supply | 150,000,000 tokens (15.0%) | 51,223,479 tokens (34.1%) | **SHORTFALL** |
| **B.3** | Source Code Supply | 150,000,000 tokens (15.0%) | 13,018,692 tokens (8.7%) | **SHORTFALL** |
| **B.4** | Wikipedia/Reference Supply | 75,000,000 tokens (7.5%) | 406,385 tokens (0.5%) | **SHORTFALL** |
| **B.5** | Mathematics Supply | 75,000,000 tokens (7.5%) | 358,188 tokens (0.5%) | **SHORTFALL** |
| **B.6** | Books/Prose Supply | 100,000,000 tokens (10.0%) | 0 tokens (0.0%) | **SHORTFALL** |
| **B.7** | Dialogue Supply | 50,000,000 tokens (5.0%) | 1,391,761 tokens (2.8%) | **SHORTFALL** |
| **C.1** | Train/Val Contamination | Zero overlap across splits | 0 document / text hash collisions | **PASS** |

---

## 2. Supply Deficit Analysis

The 1B-token pretraining allocation requires $1.0\times 10^9$ unique tokens. The local repository state reflects the historical 100M and 250M experiments:
- **Total Local Unique Tokens**: ~228,609,121 tokens
- **Total Deficit**: 771,390,879 tokens (77.14%)

### Detailed Domain Breakdown:
1. **Quality Web (`fineweb-edu`)**:
   - Target: 400,000,000 tokens
   - Available on Disk: 162,210,616 tokens
   - Deficit: 237,789,384 tokens (59.4% missing)
2. **Educational (`cosmopedia-v2`)**:
   - Target: 150,000,000 tokens
   - Available on Disk: 51,223,479 tokens
   - Deficit: 98,776,521 tokens (65.9% missing)
3. **Source Code (`codeparrot-clean`)**:
   - Target: 150,000,000 tokens
   - Available on Disk: 13,018,692 tokens
   - Deficit: 136,981,308 tokens (91.3% missing)
4. **Wikipedia / Reference**:
   - Target: 75,000,000 tokens
   - Available on Disk: 406,385 tokens (from curated knowledge pool)
   - Deficit: 74,593,615 tokens (99.5% missing)
5. **Mathematics / Technical**:
   - Target: 75,000,000 tokens
   - Available on Disk: 358,188 tokens (from reasoning pool)
   - Deficit: 74,641,812 tokens (99.5% missing)
6. **Books / Prose**:
   - Target: 100,000,000 tokens
   - Available on Disk: 0 tokens (no book corpus currently staged)
   - Deficit: 100,000,000 tokens (100.0% missing)
7. **Curated Dialogue**:
   - Target: 50,000,000 tokens
   - Available on Disk: 1,391,761 tokens (from SFT dialogue & conversational pools)
   - Deficit: 48,608,239 tokens (97.2% missing)

---

## 3. Preflight Decision

1. **Pipeline Execution**: **APPROVED FOR STAGED MULTI-SHARD VALIDATION**. The production data pipeline in `modern_gpt/data_500m/` is fully operational, mathematically verified, and compatible with streaming training.
2. **1B Production Claim**: **BLOCKED ON RAW DATA ACQUISITION**. To prevent artificial document duplication or dishonest domain substitution, full 1B token generation is suspended pending download of the remaining 771.4M raw tokens from public Hugging Face repositories.

