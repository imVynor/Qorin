# ModernGPT-S — Phase 1B.2: Deduplication & Quality Report

**Dataset Target:** SFT v4 Curated Candidate Pool  
**Status:** DEDUPLICATED & QUALITY GATED  
**Date:** 2026-11-09  
**Candidate Pool Path:** `modern_gpt/data/sft_4k_v4/raw/curated_candidate_pool.jsonl`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This report documents the exact-duplicate detection, template repetition pruning, and automated programmatic quality gates executed across the candidate source pool for SFT v4.

A total of **493 duplicate records** were identified and purged, and **15,350 redundant template records** were eliminated through deterministic pruning.

---

## 2. Exact Duplicate Detection Methodology

### 2.1 Canonical Content Hashing
Every candidate conversation is canonicalized before inspection:
1. Message roles are preserved (`user`, `assistant`).
2. Content text is stripped of leading/trailing whitespace, converted to lowercase, and internal whitespace collapsed.
3. Messages are concatenated with explicit role prefixes: `role:normalized_content|...`.
4. A 256-bit hash is computed using `hashlib.sha256()`.

### 2.2 Cross-Source Duplicate Purge
When assembling candidate records across `sft_4k_v3`, `sft_v2_public`, and `custom_context_insulation_1200`, the deduplication engine detected **493 duplicate conversations**:
- 382 duplicates between `sft_4k_v3` and `sft_v2_public` (repeated Dolly instruction and GSM8K problems).
- 99 duplicate entries within multi-turn conversation pools.
- 12 duplicate retrieval test candidates.
All 493 duplicate instances were rejected at ingestion. Zero duplicate hashes exist in `curated_candidate_pool.jsonl`.

---

## 3. Template Repetition Pruning

Beyond exact text duplicates, Phase 1B.2 targeted structural template repetition:

### 3.1 Commonsense Multiple-Choice Template Reduction
- **Original Frequency**: 8,269 records in SFT v3 (100% opening with `"The correct answer is [Choice]..."`).
- **Pruned Volume**: 6,769 records removed.
- **Retained Volume**: 1,500 records selected deterministically using `random.Random(42)`.
- **Outcome**: The multiple-choice template frequency in the pool dropped from **25.89% down to 9.13%**, breaking its status as a dominant attractor.

### 3.2 Synthetic Retrieval Template Reduction
- **Original Frequency**: 9,781 records in SFT v3.
- **Pruned Volume**: 8,581 records removed.
- **Retained Volume**: 1,200 records (300 records each at 512, 1024, 2048, and 3072 context lengths).
- **Outcome**: Retrieval frequency dropped from **30.61% down to 7.31%** of candidate pool records, saving over 10 million non-informative document tokens.

---

## 4. Automated Programmatic Quality Gates

Every candidate record retained in the curated pool passed three automated programmatic gates:

1. **Python AST Syntax Gate (`ast.parse`)**:
   - Every code block delimited by ` ```python ... ``` ` inside assistant responses was parsed through Python's Abstract Syntax Tree compiler.
   - Any syntax error, unclosed string literal, or indentation error resulted in immediate rejection.
   - 2,990 valid coding records passed; 0 AST failures retained.

2. **JSON Schema Validity Gate (`json.loads`)**:
   - Every JSON payload or `<tool_call>` block inside structured output assistant responses was parsed through `json.loads()`.
   - Any unquoted key, trailing comma, or malformed delimiter was rejected.
   - 1,895 valid structured records passed; 0 malformed JSON records retained.

3. **Context Length & Non-Empty Supervision Gate**:
   - 100% of records are $\le 4,096$ total tokens.
   - 100% of records have `supervised_tokens > 0`.

