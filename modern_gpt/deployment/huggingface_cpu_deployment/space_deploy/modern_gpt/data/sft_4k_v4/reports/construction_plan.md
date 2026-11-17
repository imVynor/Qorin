# ModernGPT-S — Phase 1B.2: SFT v4 Construction Plan

**Dataset Target:** ModernGPT-S SFT v4  
**Status:** ARCHITECTED & ROADMAPPED  
**Date:** 2026-11-09  
**Manifest Reference:** `modern_gpt/data/sft_4k_v4/manifests/curation_manifest.json`  
**Author:** ModernGPT Research & Engineering Team  

---

## 1. Executive Summary

This document outlines the step-by-step construction plan for transforming the **Phase 1B.2 Curated Candidate Pool** (16,422 records, 2.53M supervised tokens) into the final **SFT v4 Dataset** (3.60M training supervised tokens, 360k validation supervised tokens).

By curating existing sources first, Phase 1B.2 established 70.2% of the necessary training tokens from existing verified data, eliminating over 15,000 template-monopoly records.

---

## 2. Construction Phases & Roadmap

```
[Phase 1B.1: Inspection & Audit] ---> [Phase 1B.2: Source Curation] ---> [Phase 1B.3: Targeted Supplementation] ---> [Phase 1B.4: Final Assembly & Audit]
 (SFT v3 Forensics & Rationale)      (Pruned 15k rows; 2.53M tokens)      (Greetings, Safety, Identity Probes)       (Stratified Split, Freezing)
```

### Phase 1B.2 (Completed):
- Pruned `tau/commonsense_qa` (8,269 $\rightarrow$ 1,500 records).
- Pruned `synthetic_long_retrieval_v3` (9,781 $\rightarrow$ 1,200 records).
- Reclassified 1,401 instruction records (`smol-rewrite`, `dolly_instruction`).
- Reclassified 1,881 knowledge records (`squad`, `explore-instruct`, `direct_qa`).
- Eliminated 493 cross-source exact duplicates.
- Validated 100% Python code via `ast.parse()` and 100% JSON via `json.loads()`.
- Serialized `curated_candidate_pool.jsonl` (2,528,376 supervised tokens).

### Next Phase — Targeted Supplementation:
To bridge the 1.07M token deficit without resorting to generic synthetic spam, the plan calls for targeted supplementation in three isolated categories:
1. **Natural Conversational Greetings & Casual Dialogue (~630k tokens)**:
   - ~1,200 short, natural greeting exchanges with proportionate responses.
   - Casual conversation covering everyday topics, preferences, and hobbies.
2. **Dedicated Safety & Alignment Suite (180k tokens)**:
   - ~1,200 responsible response examples covering crisis helpline resources, neutral refusal of harmful tasks, cybersecurity education, and medical consultation advice.
3. **Identity Retention & Continuity Suite (~150k tokens)**:
   - 600 multi-turn dialogues with 50+ diverse names, 2–6 intervening turns, recall queries, user corrections, and unsupplied name probes.
4. **Expanded Conceptual Explanations (~270k tokens)**:
   - Diverse science, history, and conceptual reasoning questions without multiple-choice choice syntax.

### Final Phase — Assembly, Stratification & Verification:
1. **Combine Pools**: Concatenate the curated candidate pool and targeted supplements.
2. **Deterministic Category Quotas**: Sample exactly to target percentages (±2% tolerance) measured in supervised target tokens.
3. **Stratified 90/10 Split**: Shuffle within each category independently before splitting into `train.jsonl` (3.60M tokens) and `val.jsonl` (360k tokens), guaranteeing equal representation of greetings and identity examples across both splits.
4. **Behavioral Evaluation Suite**: Create an isolated 100-probe behavioral suite (`behavioral_eval.jsonl`) verifying greeting responses, identity retention across turns, and safety refusals.
5. **Freeze & Hash**: Compute SHA-256 checksums and generate the final dataset manifest.

---

## 3. Engineering Invariants for SFT v4

- **No Architecture Changes**: 100M parameter ModernGPT backbone with 4,096 context length.
- **No Tokenizer Changes**: Vocabulary fingerprint `be6c3d60c5baf217` strictly maintained.
- **Objective Function Harmony**: Dataset constructed specifically to train under the Phase 1A token-weighted SFT loss formulation ($L = \frac{\sum S_i}{\sum N_i}$).
- **Zero Training Until Explicit Approval**: All dataset construction artifacts must be audited and verified prior to launching training.

