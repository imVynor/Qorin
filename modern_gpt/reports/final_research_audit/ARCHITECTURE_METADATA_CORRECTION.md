# Architecture Metadata Correction Notice

**Notice Date:** November 17, 2026  
**Subject:** Correction of Experiment C / ModernGPT-S 500M Architecture Metadata  
**Authoritative Source of Truth:** [`modern_gpt/reports/checkpoint_architecture_audit/architecture_audit.md`](../checkpoint_architecture_audit/architecture_audit.md)  
**Canonical Model Manifest:** [`modern_gpt/canonical_model_manifest.json`](../../canonical_model_manifest.json)

---

## 1. Notice of Erroneous Historical Metadata

The historical audit reports and generation scripts in this directory (`modern_gpt/reports/final_research_audit/`):
- `experiment_registry.json`
- `generate_experiment_registry.py`
- `final_evaluation_audit.md`
- `website_data/checkpoint_lineage.json`

contain an unverified metadata draft claiming that ModernGPT-S / QORIN Experiment C possesses:
- **Erroneous Claim:** 12 layers, 1536 hidden dimension, 12 attention heads, 4 KV heads (3:1 GQA), 4096 context length, and 514,480,128 parameters.

## 2. Definitive Forensic Verification

In accordance with strict research preservation guidelines, historical files in this directory are preserved unaltered as historical audit records. However, a read-only forensic CPU audit of the physical checkpoint `ckpt_experiment_c_final.pt` (SHA-256: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`) demonstrated that:

| Dimension | Erroneous Historical Draft | Verified Physical Architecture |
| :--- | :--- | :--- |
| **Layers** | 12 | **20 layers** (`blocks.0` – `blocks.19`) |
| **Query Heads** | 12 | **24 query heads** ($d_{\text{head}}=64$) |
| **KV Heads** | 4 | **4 KV heads** ($6:1$ GQA ratio) |
| **Context Length** | 4096 | **2048 tokens** |
| **Unique Trainable Parameters** | 514,480,128 | **514,389,504 parameters** |
| **Weight Tying** | Untied (presumed) | **Tied** (`lm_head` $\equiv$ `token_embedding`) |

## 3. Direction for Active Tools & Research Showcase

All future research reports, application providers, website data generators, and research showcase dashboards must consume [`modern_gpt/canonical_model_manifest.json`](../../canonical_model_manifest.json) as the sole source of truth.
