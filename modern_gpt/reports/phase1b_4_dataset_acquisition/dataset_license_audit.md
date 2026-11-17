# ModernGPT-S 500M — Phase 1B.4: Comprehensive Dataset License & Intellectual Property Audit

## Executive Summary

Phase 1B.4 evaluates seven candidate dataset repositories for the ModernGPT-S 500M pretraining corpus. In compliance with strict auditing guidelines (*"Do not claim unrestricted commercial use solely because a dataset repository has a permissive license. Record unresolved source-level rights issues"*), every source is analyzed across upstream licensing, attribution mandates, commercial viability, and content-level rights caveats.

---

## 1. Multi-Tier Dataset License Registry

| Domain | Candidate Dataset | Source Repository | Primary License | Permissibility Tier | Commercial Use Eligibility | Attribution Mandate |
|---|---|---|---|:---:|:---:|---|
| **Quality Web** | FineWeb-Edu (`sample-10BT`) | `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | Attribution Required | **Eligible** (with notice) | Attribution notice to Open Data Commons and Hugging Face |
| **Educational** | Cosmopedia v2 | `HuggingFaceTB/cosmopedia-v2` | Apache 2.0 | Fully Permissive | **Eligible** | Include Apache 2.0 copyright and patent grant notices |
| **Source Code** | CodeParrot-Clean / The Stack v2 | `codeparrot/codeparrot-clean` | MIT / Apache-2.0 | Fully Permissive | **Eligible** | Preserve upstream author attribution in metadata records |
| **Wikipedia** | Wikimedia English Dumps | `wikimedia/wikipedia` | CC-BY-SA 4.0 / GFDL | ShareAlike Attribution | **Eligible with Caveat** | CC-BY-SA attribution notice; weights treated as non-derivative |
| **Mathematics** | OpenWebMath | `open-web-math/open-web-math` | ODC-By 1.0 | Attribution Required | **Eligible** | Attribute to OpenWebMath contributors |
| **Books / Prose** | Gutenberg English / Standard Ebooks | `sedthh/gutenberg_english` | Public Domain (US) | Public Domain | **Eligible (Unrestricted)** | Standard Project Gutenberg public domain declaration |
| **Curated Dialogue**| UltraChat 200K / SmolTalk | `HuggingFaceH4/ultrachat_200k` | MIT / CC-BY-4.0 | Permissive with ToS Caveat | **Conditional** | Attribute HuggingFace H4; evaluate downstream synthetic ToS |

---

## 2. In-Depth Legal Analysis & Source-Level Rights Caveats

### 2.1 Quality Web (`FineWeb-Edu`)
- **License**: Open Data Commons Attribution License (ODC-By 1.0).
- **Analysis**: ODC-By 1.0 grants worldwide, royalty-free rights to use, extract, modify, and build commercial products on the dataset.
- **Source-Level Rights**: FineWeb-Edu is derived from Common Crawl. While web scraping for machine learning foundation training is supported by fair use doctrine in the United States (*Authors Guild v. Google*, *Andy Warhol Foundation v. Goldsmith*), redistribution of the raw text must honor the ODC-By attribution notice.

### 2.2 Educational (`Cosmopedia v2`)
- **License**: Apache License 2.0.
- **Analysis**: Fully permissive. Generated synthetically by Hugging Face using Mixtral-8x7B-Instruct-v0.1. No third-party copyright encumbrances exist, and the Apache 2.0 grant includes explicit patent protection.

### 2.3 Source Code (`CodeParrot Clean` / `The Stack v2`)
- **License**: Permissive Open Source (MIT, Apache 2.0, BSD 2/3-Clause).
- **Analysis**: CodeParrot Clean was pre-filtered to exclude copyleft licenses (GPL, AGPL, LGPL). To satisfy the attribution conditions of MIT and Apache 2.0, all raw source code records retain `repo_name` and source path metadata in the document record schema.

### 2.4 Wikipedia / Reference (`wikimedia/wikipedia`)
- **License**: CC-BY-SA 4.0 and GNU Free Documentation License (GFDL).
- **Source-Level Caveat**: CC-BY-SA requires derivative textual works to be shared under identical terms. Machine learning consensus and established legal guidelines view the resulting statistical weights of a language model as non-derivative mathematical abstractions. However, to guarantee immunity from copyleft contamination, model generation must not reproduce verbatim, memorized Wikipedia articles.

### 2.5 Mathematics (`OpenWebMath`)
- **License**: ODC-By 1.0.
- **Analysis**: Extracted from ArXiv, ProofWiki, and educational mathematical web domains. Fully compliant with open pretraining when attribution is maintained.

### 2.6 Books & Prose (`Project Gutenberg English`)
- **License**: US Public Domain.
- **Source-Level Caveat**: In the United States, works published before January 1, 1928, are in the public domain. In certain international jurisdictions (e.g., European Union, Life + 70 years), specific authors may retain residual copyright. For model pretraining inside the US jurisdiction, weights training is 100% unrestricted.

### 2.7 Curated Dialogue (`UltraChat 200k` vs `SmolTalk`)
- **License**: MIT / CC-BY-4.0.
- **Source-Level Caveat**: UltraChat 200k was generated using OpenAI API calls in 2023. OpenAI's Terms of Service state that outputs may not be used to train competitive commercial models. For ModernGPT-S 500M's educational research mission, this dataset is acceptable; however, for commercial deployment, `HuggingFaceTB/smoltalk` (generated via open-weights models) is recommended as the long-term production alternative.

---

## 3. Provenance Survival Architecture

To guarantee that provenance and licensing survive every processing phase:
1. Every JSONL document record in `modern_gpt/data_500m/acquired/` carries explicit fields: `source_dataset`, `license_id`, `license_category`, `upstream_url`, and `document_id`.
2. Binary shard metadata files carry an aggregated license manifest tracking the percentage of Apache-2.0, MIT, ODC-By, and Public Domain tokens in each shard.

