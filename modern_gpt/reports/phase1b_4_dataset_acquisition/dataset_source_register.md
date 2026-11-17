# ModernGPT-S 500M — Phase 1B.4: Dataset Source Register & Acquisition Catalog

## Executive Summary

This register formally documents the seven approved upstream dataset sources evaluated, streamed, and integrated for ModernGPT-S 500M pretraining supply expansion. Every candidate repository is audited for upstream provenance, structural schema, data format, licensing, and domain alignment.

---

## 1. Upstream Dataset Catalog

| Domain | Source Repository | Configuration / Subset | Split | Primary License | Upstream Distributor | Schema Keys |
|---|---|---|:---:|---|---|---|
| **Quality Web** | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | `train` | ODC-By 1.0 | Hugging Face FW | `['text', 'id', 'url', 'score', 'int_score']` |
| **Educational** | `HuggingFaceTB/smollm-corpus` | `cosmopedia-v2` | `train` | Apache-2.0 | Hugging Face TB | `['prompt', 'text', 'format', 'token_length']` |
| **Source Code** | `codeparrot/codeparrot-clean-valid` | `default` | `train` | MIT / Apache-2.0 | CodeParrot Org | `['content', 'repo_name', 'path', 'license']` |
| **Wikipedia** | `wikimedia/wikipedia` | `20231101.en` | `train` | CC-BY-SA 4.0 | Wikimedia Foundation | `['id', 'title', 'text', 'url']` |
| **Mathematics** | `open-web-math/open-web-math` | `default` | `train` | ODC-By 1.0 | OpenWebMath Org | `['url', 'text', 'date', 'metadata']` |
| **Books / Prose** | `storytracer/LoC-PD-Books` | `default` | `train` | Public Domain (US) | Library of Congress | `['lccn', 'title', 'author', 'year', 'text']` |
| **Dialogue** | `HuggingFaceH4/ultrachat_200k` | `default` | `train_sft` | MIT / CC-BY-4.0 | Hugging Face H4 | `['prompt', 'prompt_id', 'messages']` |

---

## 2. Domain Alignment and Justification

### 2.1 Quality Web (`FineWeb-Edu`)
- **Target Mixture Share**: 40.0% (400,000,000 tokens)
- **Domain Justification**: Highest-quality educational web content filtered using an educational classifier trained on synthetic annotations. Provides general factual knowledge, diverse world understanding, and natural syntactic variety.
- **Filtering Rules**: Minimum educational score $\ge 3$, minimum character length $\ge 200$.

### 2.2 Educational (`Cosmopedia v2`)
- **Target Mixture Share**: 15.0% (150,000,000 tokens)
- **Domain Justification**: Synthetic textbooks, encyclopedic articles, and educational courses covering science, history, philosophy, engineering, and mathematics. High information density with structured explanations.

### 2.3 Source Code (`CodeParrot Clean`)
- **Target Mixture Share**: 15.0% (150,000,000 tokens)
- **Domain Justification**: Filtered, permissively licensed Python and polyglot source code. Instills procedural reasoning, formal grammar comprehension, and algorithmic syntax structure.

### 2.4 Wikipedia / Reference (`Wikimedia 20231101.en`)
- **Target Mixture Share**: 7.5% (75,000,000 tokens)
- **Domain Justification**: Verified factual encyclopedic articles covering global entities, history, geographical landmarks, and academic concepts. Crucial for grounding and reducing hallucination.

### 2.5 Mathematics & Technical (`OpenWebMath`)
- **Target Mixture Share**: 7.5% (75,000,000 tokens)
- **Domain Justification**: Mathematical proofs, ArXiv technical preprints, ProofWiki extracts, and LaTeX-annotated technical exposition. Strengthens quantitative reasoning, symbolic manipulation, and theorem parsing.

### 2.6 Books & Prose (`Library of Congress Public Domain Books`)
- **Target Mixture Share**: 10.0% (100,000,000 tokens)
- **Domain Justification**: Complete public-domain literary works, historical treatises, and narrative prose. Fosters long-range narrative continuity, expressive vocabulary, and complex grammatical subordination.

### 2.7 Curated Multi-Turn Dialogue (`UltraChat 200K`)
- **Target Mixture Share**: 5.0% (50,000,000 tokens)
- **Domain Justification**: Natural multi-turn conversations spanning diverse social, analytical, and informational user personas. Pre-conditions the base model for natural conversational flow prior to SFT.

