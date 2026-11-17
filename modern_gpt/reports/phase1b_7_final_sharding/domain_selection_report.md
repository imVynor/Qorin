# Phase 1B.7: Domain Selection & Boundary Truncation Policy

## 1. Selection Methodology

The selection engine deterministically traverses clean, deduplicated documents per domain:
1. **Contamination Check**: Every candidate document is matched against the 25,137-hash quarantine set. All validation matches are discarded.
2. **Deduplication Check**: Cross-session SHA-256 deduplication ensures single-epoch processing with 0 duplicate documents.
3. **Document Boundary Delimitation**: Delimiter token ID 9 (`<|endoftext|>`) is preserved at every document boundary.
4. **Deterministic Boundary Slicing**: When admitting the final document that meets the domain's exact target, the document is sliced to the exact remaining token quota $\Delta$.

## 2. Domain Selection Summary & Boundary Truncations

| Domain | Split | Target Tokens | Admitted Tokens | Documents Used | Truncated Boundary Doc | Admitted Slice | Discarded Remainder |
|---|:---:|---:|---:|---:|---|---:|---:|
| **WEB** | Train | 2,000,000,000 | 2,000,000,000 | 1,956,428 | `acq_web_topup_057159` | 569 | 639 |
| **EDU** | Train | 750,000,000 | 750,000,000 | 1,032,334 | `acq_edu_topup_063867` | 803 | 124 |
| **CODE** | Train | 750,000,000 | 750,000,000 | 436,289 | `acq_code_434634` | 932 | 977 |
| **WIKI** | Train | 375,000,000 | 375,000,000 | 426,787 | `acq_wiki_423627` | 1,653 | 743 |
| **MATH** | Train | 375,000,000 | 375,000,000 | 216,242 | `acq_math_214181` | 578 | 130 |
| **BOOKS** | Train | 500,000,000 | 500,000,000 | 24,554 | `acq_books_024645` | 3,308 | 6,803 |
| **DIALOGUE** | Train | 250,000,000 | 250,000,000 | 264,187 | `acq_dialogue_262385` | 781 | 134 |
| **WEB** | Val | 8,000,000 | 8,000,000 | 7,766 | `N/A` | 8,000,000 | 0 |
| **EDU** | Val | 3,000,000 | 3,000,000 | 4,138 | `N/A` | 3,000,000 | 0 |
| **CODE** | Val | 3,000,000 | 3,000,000 | 1,863 | `N/A` | 3,000,000 | 0 |
| **WIKI** | Val | 1,500,000 | 1,500,000 | 222 | `N/A` | 1,500,000 | 0 |
| **MATH** | Val | 1,500,000 | 1,500,000 | 847 | `N/A` | 1,500,000 | 0 |
| **BOOKS** | Val | 2,000,000 | 2,000,000 | 92 | `N/A` | 2,000,000 | 0 |
| **DIALOGUE** | Val | 1,000,000 | 1,000,000 | 847 | `N/A` | 1,000,000 | 0 |
