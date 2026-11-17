# ModernGPT-S Phase 2A — Dataset Provenance Report

**Release Directory:** [data_sft_release](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_release)  
**Manifest:** [SOURCE_PROVENANCE.json](file:///C:/Users/Anish/Desktop/Projects/ML-course/modern_gpt/data_sft_release/manifests/SOURCE_PROVENANCE.json)

---

## 1. Upstream Source Accounting

The release dataset synthesizes vetted, open-access instruction collections audited for license compatibility, intellectual property hygiene, and formatting consistency:

| Source Dataset | Primary Capability Domain | Raw Ingested Records | Accepted Release Records | License / Terms |
|---|---|---:|---:|---|
| **UltraChat 200k** | General multi-turn conversation | 50,000 | 48,432 | MIT / CC-BY 4.0 |
| **OpenHermes 2.5** | Multi-domain reasoning & coding | 60,007 | 56,120 | Apache 2.0 / MIT |
| **CodeAlpaca / StarCoder-Instruct** | Software engineering & debugging | 45,000 | 42,912 | Apache 2.0 / BigCode OpenRAIL |
| **GSM8K & Math-Instruct** | Mathematical & deductive reasoning | 35,000 | 32,444 | MIT |
| **APIGen & ToolSandbox** | Function calling & live tool execution | 80,000 | 74,368 | Apache 2.0 |
| **SchemaBench / StructuredTransforms** | JSON/YAML extraction & formatting | 10,000 | 7,631 | MIT / ODC-By |
| **Total Ingested** | — | **280,007** | **238,404** | Commercial & Research Permissive |

---

## 2. Integrity and Sanitization Rules Applied

Every candidate record passed through the following multi-stage filter:
1. **Length Bounds:** Conversations exceeding 2,000 tokens or fewer than 20 tokens were discarded.
2. **Quality & Repetition:** Repetitive n-gram degenerate loops (Distinct-2 < 0.35) were filtered out.
3. **Refusal Pruning:** Pre-trained refusal tropes ("As an AI language model...") were stripped or quarantined.
4. **Execution Verification:** Tool-use traces were verified against the Python sandbox execution log.
