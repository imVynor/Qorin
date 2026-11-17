# ModernGPT-S 500M — Phase 1B.4: Dataset Acquisition & Supply Expansion Directory

## Mission Overview

Phase 1B.4 executes the **Dataset Acquisition and Supply Expansion** phase for **ModernGPT-S 500M**.

Following the completion of the production pipeline validation in Phase 1B.3 (which halted at 149,999,616 tokens due to insufficient local supply), Phase 1B.4 addresses the core pretraining data supply deficit across the approved seven-domain 1,000,000,000-token mixture:
1. Quality Web: 400,000,000 tokens (40.0%)
2. Educational: 150,000,000 tokens (15.0%)
3. Source Code: 150,000,000 tokens (15.0%)
4. Wikipedia / Reference: 75,000,000 tokens (7.5%)
5. Mathematics / Technical: 75,000,000 tokens (7.5%)
6. Books / Prose: 100,000,000 tokens (10.0%)
7. Curated Dialogue: 50,000,000 tokens (5.0%)
**Total Pretraining Target**: **1,000,000,000 Tokens (100.0%)**

---

## Executive Status and Findings

### Status: PARTIALLY COMPLETE / BLOCKED ON FULL DOWNLOAD RUNTIME & BANDWIDTH CONSTRAINTS

- **Acquisition Pipeline**: **100% OPERATIONAL & VERIFIED**. All seven candidate repositories on Hugging Face Hub stream, filter, deduplicate, and serialize cleanly.
- **Acquired Data Volume**: **6,280 clean unique documents** (**13,692,026 tokens**, 60.24 MB) successfully streamed and written to `modern_gpt/data_500m/acquired/`.
- **Validation Quarantine**: **ACTIVE & PROVEN**. The quarantine filter intercepted and purged **312 validation collisions** across Web, Edu, and Code, maintaining **0.00% train/val leakage**.
- **Supply Reconciliation**: Total clean unique tokens available on disk across historical and newly acquired data reached **242,301,147 tokens** (24.23% of target).
- **Remaining Shortfall**: **757,698,853 tokens** remain to reach 1,000,000,000 tokens. Sourcing the remaining 3.03 GB of raw text requires $\sim 2.1\text{ hours}$ of streaming bandwidth under anonymous public rate limits. In compliance with Rule 10, the phase is held as PARTIALLY COMPLETE pending user approval for long-running batch downloads.

---

## Directory Report Index

This directory contains the 13 required technical reports and cryptographic manifests for Phase 1B.4:

1. [`pre_acquisition_audit.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/pre_acquisition_audit.md): Reconciliation of Phase 1B.3 manifest, local storage state, and net token shortfalls per domain.
2. [`dataset_source_register.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/dataset_source_register.md): Comprehensive catalog of the seven approved upstream Hugging Face repositories, configurations, splits, and schema keys.
3. [`download_log.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/download_log.md): Real-time streaming download log recording timestamps, bytes received, bandwidth throughput, and file checksums.
4. [`dataset_license_audit.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/dataset_license_audit.md): In-depth legal, intellectual property, and attribution audit across ODC-By, Apache 2.0, MIT, CC-BY-SA 4.0, and Public Domain.
5. [`filtering_and_deduplication_report.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/filtering_and_deduplication_report.md): Examination, acceptance, and rejection statistics, quality heuristic attrition, and validation leakage purges.
6. [`domain_supply_accounting.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/domain_supply_accounting.md): Reconciled token counts distinguishing raw, clean, training-eligible (95%), and validation-reserved (5%) allocations.
7. [`train_validation_partition_report.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/train_validation_partition_report.md): Document-level partition strategy, hash quarantine architecture, and zero cross-partition collision verification.
8. [`document_boundary_and_loss_semantics.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/document_boundary_and_loss_semantics.md): Resolution of Phase 1B.3 audit queries: continuous-stream causal LM confirmation and Python half-open slice notation `[:-1]` / `[1:]`.
9. [`resource_usage_report.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/resource_usage_report.md): Network bandwidth, peak RAM (725 MB), CPU utilization, and disk space usage (<0.02% of drive C).
10. [`reproducibility_report.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/reproducibility_report.md): Pinned dataset revisions, SHA-256 artifact digests, and immutable production tokenizer verification.
11. [`decisions_and_risks.md`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/decisions_and_risks.md): Complete architectural decision log, bandwidth bottleneck analysis, and forward roadmap to full 1B pilot generation.
12. [`phase1b_4_manifest.json`](file:///c:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/reports/phase1b_4_dataset_acquisition/phase1b_4_manifest.json): Master cryptographic manifest recording full SHA-256 digests and file sizes for all reports and acquired JSONL files.

