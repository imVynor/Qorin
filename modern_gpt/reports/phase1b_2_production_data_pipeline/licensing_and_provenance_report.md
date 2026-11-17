# ModernGPT-S 500M — Licensing Safeguards & Provenance Audit Report

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **AUDITED & COMPLIANT**  
**Pipeline Module:** `modern_gpt/data_500m/filtering.py` & `adapters.py`  

---

## 1. Executive Summary & Legal Compliance Framework

To ensure that ModernGPT-S 500M remains fully compliant with open-weights distribution and commercial usage standards, the data ingestion pipeline implements strict licensing classification and metadata preservation.

Every document ingested by `SourceAdapterRegistry` is wrapped in a `DocumentRecord` structure that retains:
1. `source_dataset`: The authoritative upstream repository identifier.
2. `source_url_or_ref`: The direct URL or persistent manifest reference.
3. `license_id`: Standard SPDX identifier (e.g. `ODC-By-1.0`, `Apache-2.0`, `MIT`, `CC-BY-SA-4.0`, `Public Domain`).
4. `license_category`: Explicit legal category governing data flow.
5. `attribution`: Required downstream attribution text.

---

## 2. Legal Classification Taxonomy

The pipeline implements 5 distinct licensing tiers:

| Tier | Category Identifier | Definition & Permissions | Pipeline Action | Permitted Sources |
| :---: | :--- | :--- | :---: | :--- |
| **Tier 1** | `permissive` | Unrestricted commercial and non-commercial usage; modification allowed | **Approved for Training** | Cosmopedia v2, The Stack v2 (MIT/Apache), CodeParrot |
| **Tier 2** | `attribution_required` | Permitted for training with explicit attribution in model card/reports | **Approved for Training** | FineWeb-Edu (ODC-By 1.0), OpenWebMath |
| **Tier 3** | `public_domain` | Free of copyright encumbrance (CC0, pre-1929 published works) | **Approved for Training** | Standard Ebooks, Project Gutenberg (US Public Domain) |
| **Tier 4** | `share_alike` | Copyleft obligations requiring derived dataset sharing | **Quarantined / Partitioned**| English Wikipedia (CC-BY-SA 4.0; isolated in reference partition) |
| **Tier 5** | `restricted` | Non-commercial only (CC-BY-NC), no derivatives (ND), or active copyright | **PERMANENTLY EXCLUDED** | Books3, BookCorpus, proprietary API scraped transcripts |

---

## 3. Provenance Survival through Processing

A core failure mode in data pipelines is the stripping of metadata during tokenization. In the ModernGPT-S 500M pipeline:
- **Filtering Stage:** `DataQualityFilter` modifies only `doc.text`; all licensing and source attributes remain linked to the document record.
- **Deduplication Stage:** When near-duplicates are merged, the canonical record retains a cross-reference list of all merged duplicate IDs and their source licenses.
- **Binary Shard Manifest:** Each binary shard file (`.bin`) is accompanied by a JSON sidecar recording the exact document IDs, source licenses, and attribution strings contained within that shard.

---

## 4. Licensing Audit Checklist

- [x] Zero non-commercial (NC) datasets included in pretraining streams.
- [x] Zero unlicensed scraped novel / book datasets included (Books3 completely blacklisted).
- [x] Attribution strings cataloged for ODC-By (FineWeb-Edu) and Apache-2.0 (Cosmopedia v2).
- [x] Wikipedia CC-BY-SA attribution tracked separately.
- [x] SFT v4 datasets remain isolated under their own licenses without pretraining mixing.

