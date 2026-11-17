# QORIN Architecture Metadata Correction Report

**Audit Date:** November 17, 2026  
**Audited Target:** Architecture Metadata & Experiment Registries Across the Repository  
**Authoritative Forensic Audit:** [`modern_gpt/reports/checkpoint_architecture_audit/architecture_audit.md`](../checkpoint_architecture_audit/architecture_audit.md)  
**Canonical Source of Truth:** [`modern_gpt/canonical_model_manifest.json`](../../canonical_model_manifest.json)  
**Audit Status:** **RESOLVED & CERTIFIED**

---

## 1. Executive Summary

During the preparation of research sprint reports, an erroneous architectural metadata template was manually drafted in `modern_gpt/reports/final_research_audit/generate_experiment_registry.py`. This draft erroneously specified that the 500M model family (including Experiment C) had 12 layers, 12 attention heads, 4096 context length, and 514,480,128 parameters.

A forensic CPU inspection of the frozen Experiment C checkpoint (`ckpt_experiment_c_final.pt`, SHA-256: `6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b`) conclusively refuted this narrative, proving the physical model has **20 layers, 24 query heads, 4 KV heads (6:1 GQA), 2048 context length, and 514,389,504 unique parameters**.

This report documents the repository-wide audit, the creation of the canonical machine-readable model manifest, the isolation and annotation of historical artifacts, and the automated validation tests enforcing metadata integrity.

---

## 2. Inventory of Located Erroneous Claims

A repository-wide regex scan for `514,480,128`, `12 layers`, `12 query heads`, and `4096 context` in connection with 500M checkpoints identified the following files:

| File Path | Nature of File | Erroneous Claim Located | Remediated Action |
| :--- | :--- | :--- | :--- |
| `modern_gpt/reports/final_research_audit/generate_experiment_registry.py` | Historical generation script | 12 layers, 1536 dim, 12 heads, 4096 context, 514,480,128 params | Documented; annotated via correction notice |
| `modern_gpt/reports/final_research_audit/experiment_registry.json` | Historical audit report | Same as above | Preserved historically; annotated via correction notice |
| `modern_gpt/reports/final_research_audit/final_evaluation_audit.md` | Historical audit report | Same as above | Preserved historically; annotated via correction notice |
| `modern_gpt/reports/final_research_audit/website_data/checkpoint_lineage.json` | Generated website data | 514,480,128 parameters across all 4 nodes | **Regenerated from canonical manifest** |
| `modern_gpt/reports/final_8hour_sprint/website_data/checkpoint_lineage.json` | Historical sprint artifact | 514,480,128 parameters | Preserved historically; annotated via correction notice |
| `modern_gpt/reports/final_research_audit/prepare_website_data.py` | Active website data generator | Hardcoded 514,480,128 in `generate_checkpoint_lineage` | **Updated to load dynamically from canonical manifest** |
| `web/src/data/modelsData.ts` | Frontend model metadata | Already accurate (20 layers, 24 heads, 2048 ctx, 514,389,504 params) | Verified accurate |
| `web/src/pages/Home.tsx` & `Model.tsx` | Frontend research pages | Already accurate (20 layers, 24 heads, 2048 ctx, 514,389,504 params) | Verified accurate |

---

## 3. The Canonical Model Manifest

To eliminate hardcoded duplication, [`modern_gpt/canonical_model_manifest.json`](../../canonical_model_manifest.json) was established as the sole machine-readable source of truth for all QORIN models.

### Key Fields for QORIN-500M Experiment C:
```json
{
  "family_id": "QORIN-500M",
  "historical_name": "ModernGPT-S-500M",
  "parameters": {
    "unique_trainable_parameters": 514389504,
    "raw_tensor_elements_in_checkpoint": 569964032,
    "tied_embeddings": true
  },
  "canonical_architecture": {
    "n_layers": 20,
    "emb_dim": 1536,
    "n_heads": 24,
    "num_kv_heads": 4,
    "gqa_ratio": "6:1",
    "head_dim": 64,
    "ffn_hidden_dim": 3840,
    "activation": "SwiGLU",
    "norm_type": "RMSNorm",
    "norm_eps": 1e-05,
    "context_length": 2048,
    "vocab_size": 32768,
    "qkv_bias": false,
    "out_proj_bias": false,
    "rope_theta": 10000.0,
    "rope_scale": 1.0,
    "rope_type": "default"
  },
  "tokenizer": {
    "identity": "bpe_32k",
    "vocab_size": 32768,
    "path": "modern_gpt/artifacts/tokenizer_32k/tokenizer.json",
    "sha256": "d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59"
  },
  "checkpoints": {
    "qorin-500m-sft-exp-c": {
      "checkpoint_id": "qorin-500m-sft-exp-c",
      "display_name": "QORIN-500M SFT Experiment C (Autonomous Two-Tool)",
      "checkpoint_path": "modern_gpt/experiments/phase2_experiment_c/stage_3_training/checkpoints/ckpt_experiment_c_final.pt",
      "sha256": "6cb5125896bbe1d82b247212b2de7bc18832561c39d0be6a8dbc25724e839a1b"
    }
  }
}
```

---

## 4. Preservation & Correction Notices

In strict compliance with non-destructive audit rules, earlier historical reports were **not deleted or overwritten**. Instead, formal markdown correction notices were added to the directories containing erroneous historical reports:
1. [`modern_gpt/reports/final_research_audit/ARCHITECTURE_METADATA_CORRECTION.md`](../final_research_audit/ARCHITECTURE_METADATA_CORRECTION.md)
2. [`modern_gpt/reports/final_8hour_sprint/ARCHITECTURE_METADATA_CORRECTION.md`](../final_8hour_sprint/ARCHITECTURE_METADATA_CORRECTION.md)

These notices document the exact discrepancies, state that the earlier numbers were pre-audit drafts, and direct all researchers to the verified checkpoint architecture audit and canonical manifest.

---

## 5. Automated Validation Test Suite

To prevent future regression, an automated test suite was introduced in [`tests/test_model_manifest_validation.py`](file:///c:/Users/Anish/Desktop/Projects/ML-course/tests/test_model_manifest_validation.py).

### Test Verifications:
1. `test_manifest_exists`: Verifies canonical manifest presence.
2. `test_experiment_c_architecture_truth`: Asserts `n_layers == 20`, `n_heads == 24`, `num_kv_heads == 4`, `context_length == 2048`, `unique_trainable_parameters == 514389504`, and SHA-256 is `6cb5125896...`.
3. `test_manifest_matches_checkpoint_audit_registry`: Asserts bit-for-bit parity between the manifest and the physical checkpoint registry.
4. `test_website_lineage_data_matches_canonical`: Asserts that `website_data/checkpoint_lineage.json` contains exactly 514,389,504 parameters across all nodes.

### Test Execution Result:
```text
tests/test_model_manifest_validation.py::test_manifest_exists PASSED [ 25%]
tests/test_model_manifest_validation.py::test_experiment_c_architecture_truth PASSED [ 50%]
tests/test_model_manifest_validation.py::test_manifest_matches_checkpoint_audit_registry PASSED [ 75%]
tests/test_model_manifest_validation.py::test_website_lineage_data_matches_canonical PASSED [100%]
============================== 4 passed in 0.02s ==============================
```
