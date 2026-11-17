# Architecture Metadata Correction Notice

**Notice Date:** November 17, 2026  
**Subject:** Correction of Experiment C Architecture Metadata in Sprint Artifacts  
**Authoritative Source of Truth:** [`modern_gpt/reports/checkpoint_architecture_audit/architecture_audit.md`](../checkpoint_architecture_audit/architecture_audit.md)  
**Canonical Model Manifest:** [`modern_gpt/canonical_model_manifest.json`](../../canonical_model_manifest.json)

---

## Notice

The file `modern_gpt/reports/final_8hour_sprint/website_data/checkpoint_lineage.json` contains historical metadata listing 514,480,128 parameters for the 500M checkpoints.

As established by the CPU forensic audit, the verified unique parameter count for all 500M checkpoints (`ckpt_latest.pt`, `ckpt_pilot_final.pt`, `ckpt_experiment_b_final.pt`, and `ckpt_experiment_c_final.pt`) is **514,389,504**, with **20 layers, 24 query heads, 4 KV heads, and 2048 context length**.

This artifact is preserved historically. Active applications and showcase builders must read from [`modern_gpt/canonical_model_manifest.json`](../../canonical_model_manifest.json).
