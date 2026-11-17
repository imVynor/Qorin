# ModernGPT-S 500M — Phase 1B.3: Domain Token Accounting and Shortfall Audit

## Executive Summary

Phase 1B.3 establishes exact domain and token accounting for the proposed 1,000,000,000-token pretraining pilot across the seven approved domains.

In accordance with strict project rules:
1. No documents were repeated to artificially inflate token counts.
2. No domain shortfall was silently compensated by borrowing tokens from another domain.
3. Every token generated has verifiable source dataset provenance and licensing metadata.

---

## 1. 1B Target Mixture vs Achieved Local Accounting

| Domain | Target Tokens | Target Share (%) | Available Local Unique Tokens | Realized Pilot Tokens | Deficit Against 1B Target | Status |
|---|---:|---:|---:|---:|---:|:---:|
| **Quality Web** | 400,000,000 | 40.0% | 162,210,616 | 134,217,728 | 265,782,272 | **SHORTFALL** |
| **Educational** | 150,000,000 | 15.0% | 51,223,479 | 48,154,624 | 101,845,376 | **SHORTFALL** |
| **Source Code** | 150,000,000 | 15.0% | 13,018,692 | 13,018,692 | 136,981,308 | **SHORTFALL** |
| **Wikipedia / Reference** | 75,000,000 | 7.5% | 406,385 | 406,385 | 74,593,615 | **SHORTFALL** |
| **Mathematics / Technical** | 75,000,000 | 7.5% | 358,188 | 358,188 | 74,641,812 | **SHORTFALL** |
| **Books / Prose** | 100,000,000 | 10.0% | 0 | 0 | 100,000,000 | **SHORTFALL** |
| **Curated Dialogue** | 50,000,000 | 5.0% | 1,391,761 | 1,391,761 | 48,608,239 | **SHORTFALL** |
| **Total Pretraining** | **1,000,000,000** | **100.0%** | **~228,609,121** | **199,999,488** | **800,000,512** | **SHORTFALL** |

*Note: In addition to the 199,999,488 pretraining tokens, an independent held-out validation shard of 8,640,512 tokens was generated and verified.*

---

## 2. Root Cause Analysis of Domain Deficits

The target 1B pretraining mixture was planned in Phase 1A / Phase 1B.1 based on total estimated open-web supply (e.g., FineWeb-Edu contains $>10\text{B}$ tokens, Cosmopedia-v2 contains $>25\text{B}$ tokens, SmolLM contains $>600\text{M}$ curated dialogue tokens). However, only the historical 100M and 250M training subsets (amounting to ~228M unique tokens in `train_documents.jsonl` and `new_train_documents.jsonl`) had been downloaded and staged to the local developer drive.

### Specific Deficit Causes:
1. **Books / Prose (100M Target — 0 Available)**:
   No raw BookCorpus / Project Gutenberg / SmolLM books subset was staged on local disk. Local repository contains 0 book documents.
2. **Wikipedia / Reference (75M Target — 0.4M Available)**:
   Only a small held-out encyclopedic knowledge set (from SFT candidate pool) exists locally. Full English Wikipedia dumps (1.2B tokens) have not been downloaded.
3. **Mathematics / Technical (75M Target — 0.36M Available)**:
   Only synthetic math reasoning examples from SFT candidate pools exist locally. The OpenWebMath or InfiMM-Math pretraining datasets have not been downloaded.
4. **Dialogue (50M Target — 1.39M Available)**:
   The curated dialogue pool contains high-quality conversational turns from Phase 1B.3A and 1B.3B, but lacks large-scale multi-turn pretraining dialogue datasets (such as UltraChat or OpenHermes).
5. **Code (150M Target — 13M Available)**:
   `codeparrot-clean` subset locally has 13M tokens; full StarCoder / The Stack slices were not pre-fetched.

---

## 3. Compliance with Single-Epoch Directive

To preserve training distribution integrity and avoid catastrophic memorization:
- **No Document Repetition**: Every document in the generated shards was processed exactly once.
- **Strict Stop Condition**: Generation halted cleanly when unique local supply reached shard boundaries, producing exactly 4 complete 50M-token training shards (199,999,488 tokens).
- **Audit Verdict**: The full 1B pretraining pilot cannot be claimed as complete until the missing 800M tokens are downloaded from their upstream sources.

