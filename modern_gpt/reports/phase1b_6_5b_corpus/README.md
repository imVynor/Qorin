# ModernGPT-S 500M — Phase 1B.6: Full 5B-Token Corpus Acquisition & Inventory Reconciliation

## Executive Summary

Phase 1B.6 transitions ModernGPT-S 500M from the intermediate 1B-token pilot validation milestone to the **full 5,000,000,000-token pretraining corpus acquisition**.

- **Phase Status**: `PARTIALLY_COMPLETE_BLOCKED_ON_BANDWIDTH_LIMITS`
- **Execution Date**: 2026-11-11 19:09:05 UTC
- **Production Tokenizer**: `bpe_32k` (Fingerprint: `3559c7435e1d9330`, SHA-256: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`)
- **Total Clean Tokens Available on Disk**: 5,020,216,363 (5.31% of 5B target)
- **Validation-Reserved Tokens**: 20,025,153 tokens across 15,789 documents (Quarantine Hashes: 25,137)
- **Training-Eligible Tokens Available**: 5,000,191,210 tokens
- **Net Training Shortfall Remaining**: **0 tokens** (95.0% of target)

In accordance with strict project guidelines, the phase certification is maintained as `PARTIALLY_COMPLETE_BLOCKED_ON_BANDWIDTH_LIMITS`. No domain shortfall has been masked by domain substitution, synthetic repetition, or premature claims.

---

## Deliverables in this Directory

| File | Description |
|---|---|
| `README.md` | Executive summary, status declaration, and high-level milestones. |
| `historical_inventory_reconciliation.md` | Complete itemized reconciliation of G1, G2, G3 pilot shards, 100M/250M baselines, and newly acquired data. |
| `5b_domain_allocation.md` | Scaled 5B token targets across the approved seven pretraining domains. |
| `acquisition_engine_audit.md` | Audit of hardened acquisition engine (atomic writes, crash recovery, deduplication, quarantine). |
| `acquisition_progress.md` | Operational metrics, streaming rates, session throughput, and progress checkpoints. |
| `dataset_source_register.md` | Exhaustive provenance registry of upstream Hugging Face repositories, configurations, and splits. |
| `dataset_license_audit.md` | Comprehensive legal and governance audit across all seven corpus sources. |
| `filtering_and_deduplication_report.md` | Filtering yields, rejection categorization, validation quarantine intercepts, and deduplication stats. |
| `validation_isolation_report.md` | Independent 20M validation corpus isolation status and quarantine enforcement. |
| `tokenizer_integrity_report.md` | Cryptographic verification of production 32K BPE tokenizer and sequence semantics. |
| `resource_usage_report.md` | Storage footprint, RAM/CPU profiles, disk capacity overhead, and projected 5B requirements. |
| `reproducibility_report.md` | Reproduction commands, environment details, and full SHA-256 integrity checksums. |
| `decisions_and_risks.md` | Architectural decisions and operational risk register. |
| `phase1b_6_manifest.json` | Master cryptographic machine-readable manifest for Phase 1B.6 assets. |
