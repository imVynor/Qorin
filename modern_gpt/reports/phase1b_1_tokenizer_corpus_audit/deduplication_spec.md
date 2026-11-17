# ModernGPT-S 500M — Corpus Deduplication Specification

**Phase:** Phase 1B.1 Tokenizer Finalization & Pretraining Corpus Audit  
**Date:** November 11, 2026  
**Status:** **SPECIFICATION COMPLETE & ARCHITECTED**  
**Target Scale:** 5.00-Billion-Token Pretraining Mixture  

---

## 1. Executive Summary

Deduplication is essential to prevent memorization, reduce training loss plateauing, improve sample efficiency, and eliminate train/validation contamination. Studies in language model scaling (Lee et al., 2022; Penedo et al., 2023) show that training on duplicate documents leads to severe repetition loops and degrades factual reasoning in models under 1B parameters.

This specification outlines a multi-tier deduplication pipeline:
1. **Tier 1 — Exact Deduplication:** Bounded-sample and global SHA-256 hash collision detection.
2. **Tier 2 — Near-Duplicate Detection:** MinHash with Locality-Sensitive Hashing (LSH) targeting Jaccard similarity $\ge 0.80$.
3. **Tier 3 — Cross-Source Harmonization:** Resolving overlapping documents across web crawls, encyclopedias, and educational datasets.
4. **Tier 4 — Evaluation Contamination Shield:** Zero-leakage filtering against evaluation benchmark suites.

---

## 2. Multi-Tier Deduplication Architecture

```
Raw Processed Documents
   │
   ▼ [Tier 1: Canonical Text SHA-256 (Exact Deduplication)]
   │  - Normalize: Lowercase, collapse consecutive whitespace, strip punctuation
   │  - Compute 256-bit hash; retain first occurrence; discard exact duplicates
   │
   ▼ [Tier 2: MinHash + LSH Clustering (Near-Duplicate Detection)]
   │  - Generate 5-gram token shingles per document
   │  - Compute 128 MinHash signatures using independent linear permutation hashes
   │  - Index into 16 LSH bands (8 rows per band) -> Jaccard threshold ~ 0.80
   │  - Group duplicate candidates into connected components (Union-Find)
   │  - Retain the longest, most grammatically complete document in each cluster
   │
   ▼ [Tier 3: Cross-Source Harmonization]
   │  - Detect Wikipedia mirrors in FineWeb web text
   │  - Detect overlapping educational text between Cosmopedia and textbooks
   │  - Assign canonical priority: Curated Reference > Educational > Web
   │
   ▼ [Tier 4: Benchmark Contamination Elimination]
   │  - Filter 13-gram exact matches against MMLU, GSM8k, ARC, HumanEval, Val set
   │
Deduplicated, Pristine Pretraining Corpus
```

---

## 3. Algorithm Specifications & Mathematical Thresholds

### 3.1 Tier 1: Exact Deduplication (SHA-256)
- **Normalization Function:**
  $$f_{\text{norm}}(\text{text}) = \text{re.sub}(r'\s+', '\ ', \text{text.strip().lower()})$$
- **Hash Calculation:** $\text{SHA-256}(f_{\text{norm}}(\text{text}))$.
- **Storage & Lookups:** 64-bit integer prefixes ($8\text{ bytes}$) stored in an in-memory hash set. For a 5-billion-token corpus ($\sim 3\text{M documents}$):
  $$3,000,000 \times 8\text{ bytes} \approx 24\text{ MB RAM}$$
- **Deduplication Action:** Discard any subsequent document with an identical 64-bit hash prefix.

### 3.2 Tier 2: Near-Duplicate Detection (MinHash + LSH)
- **Shingling:** Extract contiguous character 5-grams (or word 3-grams) from normalized text:
  $$S(d) = \{w_i, w_{i+1}, w_{i+2} \mid i \in [1, |d|-2]\}$$
- **MinHash Signatures:** Compute $K = 128$ permutation hash values per document:
  $$h_k(d) = \min_{s \in S(d)} \left( (a_k \cdot \text{hash}(s) + b_k) \pmod p \right)$$
  where $p = 2^{61} - 1$ (Mersenne prime), and $a_k, b_k$ are random integers.
- **LSH Banding Geometry:**
  - Number of bands ($b$): $16$
  - Rows per band ($r$): $8$ ($b \times r = 128$)
  - S-Curve Jaccard Threshold ($t$):
    $$t \approx \left(\frac{1}{b}\right)^{1/r} = \left(\frac{1}{16}\right)^{1/8} \approx \mathbf{0.707}$$
  - Candidate pairs with Jaccard similarity $>0.80$ have a $>98.5\%$ collision probability.
- **Cluster Selection:** For each connected component of near-duplicate documents, select the canonical document maximizing:
  $$\text{Score}(d) = \text{length}(d) \times \text{alphanumeric\_ratio}(d)$$
  Discard all other variants.

### 3.3 Tier 3: Cross-Source Conflict Resolution
When identical or near-duplicate documents exist across multiple data sources, precedence is resolved deterministically according to provenance authority:
1. **Priority 1:** Curated Reference (Wikipedia, Proof-Pile-2, OpenWebMath)
2. **Priority 2:** Educational Material (Cosmopedia v2)
3. **Priority 3:** General Web Prose (FineWeb-Edu)
4. **Priority 4:** General Scraped Web

*Example:* If a FineWeb article is an uncredited mirror of a Wikipedia page, the Wikipedia record is retained and the FineWeb duplicate is discarded.

---

## 4. Resource & Computational Footprint

| Subsystem | Algorithm | Memory Footprint (RAM) | Computational Complexity | Estimated Run Time |
| :--- | :--- | :---: | :---: | :---: |
| **Exact Deduplication** | SHA-256 hash table | $< 50\text{ MB}$ | $O(N)$ linear | $\sim 5\text{ minutes}$ |
| **MinHash Generation** | 128 linear hash functions | $\sim 1.5\text{ GB}$ (streamed) | $O(N \cdot |d|)$ | $\sim 35\text{ minutes}$ (8 CPU cores) |
| **LSH Bucketing & Clustering** | Inverted index + Union-Find | $\sim 800\text{ MB}$ | $O(N)$ average | $\sim 10\text{ minutes}$ |
| **Contamination Checking** | 13-gram bloom filter / set | $\sim 200\text{ MB}$ | $O(N \cdot |d|)$ | $\sim 15\text{ minutes}$ |
| **Total Deduplication Pipeline**| Combined Multi-Tier | **$< 2.5\text{ GB RAM}$** | **$O(N)$ linear** | **$\approx 1.1\text{ hours}$** |

All deduplication steps can execute comfortably on standard developer machines with $\le 16\text{ GB}$ RAM.

