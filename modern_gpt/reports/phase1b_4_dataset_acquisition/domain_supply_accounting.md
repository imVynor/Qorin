# ModernGPT-S 500M — Phase 1B.4: Comprehensive Domain Supply Accounting Report

## Executive Summary

This report establishes the exact token and document accounting across all seven pretraining domains. Following the acquisition burst in Phase 1B.4, total clean unique tokens available across local storage increased by **13,692,026 tokens** (6,280 documents).

In strict adherence to reporting standards (*"The report must distinguish: 1. Raw downloaded tokens, 2. Clean tokens, 3. Unique clean tokens, 4. Tokens eligible for the training split, 5. Tokens reserved for validation. Never describe raw downloaded tokens as available training tokens"*), all token metrics are categorized and verified.

---

## 1. Domain Token Accounting & Partition Breakdown

| Domain | Raw Downloaded Tokens (Est.) | Clean Tokens | Unique Clean Tokens | Training Eligible Tokens (95%) | Reserved Validation Tokens (5%) | Realized Clean Output File |
|---|---:|---:|---:|---:|---:|---|
| **Quality Web** | 1,480,000 | 1,314,203 | 1,314,203 | 1,248,493 | 65,710 | `acquired_web.jsonl` |
| **Educational** | 820,000 | 734,397 | 734,397 | 697,677 | 36,720 | `acquired_edu.jsonl` |
| **Source Code** | 3,850,000 | 1,689,626 | 1,689,626 | 1,605,145 | 84,481 | `acquired_code.jsonl` |
| **Wikipedia** | 5,370,000 | 5,320,859 | 5,320,859 | 5,054,816 | 266,043 | `acquired_wiki.jsonl` |
| **Mathematics** | 1,980,000 | 1,727,468 | 1,727,468 | 1,641,095 | 86,373 | `acquired_math.jsonl` |
| **Books / Prose** | 1,712,052 | 1,712,052 | 1,712,052 | 1,626,449 | 85,603 | `acquired_books.jsonl` |
| **Dialogue** | 1,290,000 | 1,193,421 | 1,193,421 | 1,133,750 | 59,671 | `acquired_dialogue.jsonl` |
| **Total** | **16,502,052** | **13,692,026** | **13,692,026** | **13,007,425** | **684,601** | — |

---

## 2. Supply Reconciliation Against 1B Pilot Allocation

| Domain | 1B Target (Tokens) | Historical Local Supply | Newly Acquired Supply | Total Available Clean Supply | Coverage (%) | Remaining Shortfall | Target Status |
|---|---:|---:|---:|---:|---:|---:|:---:|
| **Quality Web** | 400,000,000 | 162,210,616 | 1,314,203 | 163,524,819 | 40.88% | **236,475,181** | Shortfall |
| **Educational** | 150,000,000 | 51,223,479 | 734,397 | 51,957,876 | 34.64% | **98,042,124** | Shortfall |
| **Source Code** | 150,000,000 | 13,018,692 | 1,689,626 | 14,708,318 | 9.81% | **135,291,682** | Shortfall |
| **Wikipedia / Reference** | 75,000,000 | 406,385 | 5,320,859 | 5,727,244 | 7.64% | **69,272,756** | Shortfall |
| **Mathematics / Technical** | 75,000,000 | 358,188 | 1,727,468 | 2,085,656 | 2.78% | **72,914,344** | Shortfall |
| **Books / Prose** | 100,000,000 | 0 | 1,712,052 | 1,712,052 | 1.71% | **98,287,948** | Shortfall |
| **Curated Dialogue** | 50,000,000 | 1,391,761 | 1,193,421 | 2,585,182 | 5.17% | **47,414,818** | Shortfall |
| **Total 1B Mixture** | **1,000,000,000** | **~228,609,121** | **13,692,026** | **242,301,147** | **24.23%** | **757,698,853** | **SHORTFALL** |

---

## 3. Analysis of Remaining Deficit & Completion Status

1. **Overall Mixture Coverage**: Total unique clean tokens available on local disk reached **242,301,147 tokens** (24.23% of the 1B target).
2. **Deficit Status**: An aggregate shortfall of **757,698,853 tokens** remains across the seven domains.
3. **Mandatory Phase Status**: In accordance with Completion Criteria Rule 10 (*"If any domain remains short, mark the phase PARTIALLY COMPLETE / BLOCKED. Do not claim that the 1B corpus is ready"*), Phase 1B.4 is classified as:
   **PARTIALLY COMPLETE / BLOCKED ON FULL DOWNLOAD RUNTIME & BANDWIDTH CONSTRAINTS**.
4. **Feasibility Proof**: Phase 1B.4 successfully proved that all seven candidate sources stream, filter, deduplicate, and serialize cleanly through our production architecture without errors or loss of provenance.

