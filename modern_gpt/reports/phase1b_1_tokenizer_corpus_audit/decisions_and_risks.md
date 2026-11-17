# ModernGPT-S 500M — Technical Decisions, Risk Register & Mitigations (Phase 1B.1)

**Phase:** Phase 1B.1 Tokenizer Finalization & Pretraining Corpus Audit  
**Date:** November 11, 2026  
**Status:** **AUDIT COMPLETE & RISK REGISTER UPDATED**  

---

## 1. Phase 1B.1 Decision Log

| Decision ID | Subsystem | Formal Decision | Rejected Alternative | Technical Rationale |
| :---: | :--- | :--- | :--- | :--- |
| **D-101** | **Tokenizer Scale** | **Confirm 32K Vocabulary** | Revert to 16K Baseline | Independent held-out benchmark confirmed $19.48\%$ overall reduction and $32.98\%$ code reduction, expanding context coverage by $+24.19\%$. |
| **D-102** | **Special-Token Formalization** | **Approve Special Token Inclusion Protocol** | Train with `<|endoftext|>` only | Atomic `<|tool_call|>`, `<|tool_result|>`, `<|im_start|>`, `<|im_end|>` are required for tool calling security and clean SFT loss masking. |
| **D-103** | **SFT v4 Boundary** | **Strict Quarantine from Pretraining** | Blend SFT v4 into Pretraining | SFT data skews general pretraining representations; SFT v4 is reserved for post-pretraining alignment. |
| **D-104** | **Book Prose Sourcing** | **Public Domain (Pre-1929 & CC0) Only** | Scraped BookCorpus / Books3 | Eliminates legal and copyright liability; guarantees clean open-weights distribution. |
| **D-105** | **Pretraining Epochs** | **Strictly Single-Epoch (1.0x)** | Multi-epoch looping of small datasets | Eliminates memorization and loss plateaus; verified oversupply of unique tokens ($>145\text{B available}$). |
| **D-106** | **Storage & Streaming** | **Binary Shards (`uint16`, 50M tok/shard)** | Streaming JSONL during training | Achieves $>100\text{k tokens/sec}$ streaming with zero-copy GPU memory mapping and checkpointable state. |

---

## 2. Technical Risk Register & Mitigations

### Risk 1: Disk Space Constraints during 5B-Token Acquisition
- **Risk Level:** **MEDIUM** (Severity: Medium, Likelihood: High)
- **Description:** Downloading 5B raw tokens in uncompressed JSONL format can consume $>25\text{ GB}$ of local SSD storage.
- **Mitigation Strategy:** Implement an **in-flight streaming pipeline**: download and process 500M-token domain chunks, apply filtering and deduplication, tokenize directly into binary `.bin` shards, and immediately purge the raw JSONL cache. Peak disk usage will remain under $12\text{ GB}$.

---

### Risk 2: Benchmark Contamination in Pretraining Web & Dialogue Streams
- **Risk Level:** **HIGH** (Severity: High, Likelihood: Medium)
- **Description:** Web crawls and synthetic dialogues frequently ingest evaluation benchmarks (e.g. GSM8k, MMLU, HumanEval questions).
- **Mitigation Strategy:** Enforce an automated **13-gram shingle bloom filter** across all downloaded records before tokenization. Any pretraining record matching evaluation shingles is immediately purged.

---

### Risk 3: Dialogue Domain Token Scarcity
- **Risk Level:** **LOW to MEDIUM** (Severity: Medium, Likelihood: Low)
- **Description:** Target allocation is $250\text{M tokens}$ ($5\%$ of mixture). Vetted clean multi-turn dialogue corpora total $\sim 315\text{M unique tokens}$, leaving a smaller safety buffer than other domains.
- **Mitigation Strategy:** Blend UltraChat 200k, OpenAssistant OASST1, and clean conversational partitions from LMSYS. If unique dialogue tokens fall below $250\text{M}$, scale dialogue share to $4.0\%$ ($200\text{M}$) and reallocate $1.0\%$ to educational material.

---

### Risk 4: Tokenizer Special-Token Alignment Gate
- **Risk Level:** **LOW** (Severity: High, Likelihood: Low)
- **Description:** Training shards generated with the preliminary `candidate_32k` tokenizer would lack atomic tool tokens.
- **Mitigation Strategy:** Part B identified this discrepancy early. No pretraining shards will be tokenized until the formal 10-token vocabulary is frozen in Phase 1B.2.

