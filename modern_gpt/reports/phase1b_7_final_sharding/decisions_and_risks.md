# Phase 1B.7: Key Engineering Decisions & Risk Management

## 1. Key Engineering Decisions

1. **Standard 50M Shard Layout**: Sizing training shards at 50,000,000 tokens aligns with optimal distributed I/O, yielding 101 training shards that fit comfortably in memory and facilitate granular checkpointing.
2. **Deterministic Boundary Slicing**: When meeting the exact 5B training target, the final candidate document in each domain is sliced to the exact remaining token quota $\Delta$. Truncated tokens and boundary document IDs are fully logged in `domain_selection_report.md`.
3. **Continuous Streaming Carryover**: The DataLoader preserves trailing token remainders across shard transitions, eliminating sequence truncation and avoiding artificial padding.
4. **Preservation of Previous Artifacts**: The 100M baseline archive (`archive/100m_baseline/`) and all pilot shards (G1, G2, G3) were preserved untouched in their original directories.

## 2. Risk Evaluation

- **Contamination Risk**: Intercepted with active 25,137-hash quarantine filter ($0.00\%$ contamination).
- **Duplication Risk**: Mitigated by cross-session exact text hashing.
- **I/O Bottlenecks**: Avoided by atomic sequential disk streaming.
