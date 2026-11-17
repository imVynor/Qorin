# QORIN — Team Contributions & Technical Division of Responsibilities

**Document**: `docs/TEAM_CONTRIBUTIONS.md`  
**Status**: AUTHORITATIVE TEAM ATTRIBUTION  
**Date**: 2026-11-17  
**Team Members**: **Anish Raj** & **Jnandeep**  

---

## 1. Overview & Research Scope

The QORIN investigation divides technical ownership between foundational model development and runtime orchestration harness engineering. All model checkpoints, empirical benchmarks, and research discoveries remain collective project deliverables.

---

## 2. Technical Division of Responsibilities

### 2.1 Anish Raj
**Primary Focus**: Model Architecture, Model Training, Model Evaluation, and Fine-Tuning.

- **Model Architecture**: Designed and implemented the decoder-only Transformer foundations across all scales (16M, 50M, 100M, and 514M) with Rotary Position Embeddings (RoPE), RMSNorm pre-normalization, SwiGLU activations, and Grouped-Query Attention (GQA).
- **Model Training & Pretraining**: Executed causal language pretraining pipelines from scratch under consumer GPU constraints, including 100M baseline pretraining across 1.5B tokens and the 500M causal pretraining pilot.
- **Supervised Fine-Tuning (SFT)**: Led instruction-tuning experiments, including context extension to 4,096 tokens via RoPE linear frequency scaling ($s=8.0$), conversational ChatML alignment, and multi-turn supervised loss masking.
- **Model Evaluation**: Designed, benchmarked, and executed quantitative evaluation suites, measuring validation loss curves, AST code syntax adherence, functional pass rates, and held-out test splits.

---

### 2.2 Jnandeep
**Primary Focus**: Harness Engineering, Retrieval Systems, Dataset Preparation, and Adapting Model to Use Tools.

- **Harness Engineering**: Architected the runtime orchestration harness, including the 4-state execution state machine, stateless Python subprocess sandbox with execution timeouts (5000ms), and ephemeral UUID session memory isolation.
- **Retrieval Systems**: Designed and implemented non-parametric retrieval pipelines, including BM25 sparse document retrieval and live DuckDuckGo web search integration for real-time factual augmentation.
- **Dataset Preparation**: Curated, preprocessed, tokenized, and decontaminated training and evaluation datasets across all experimental phases, including pretraining token shards and the Stage 13 multi-turn dataset.
- **Adapting Model to Use Tools**: Engineered structured tool calling schemas (`<tool_call>`), intent classification protocols, observation feedback loops, and runtime evidence grounding to prevent post-retrieval hallucination.

---

## 3. Website Representation Policy

- **About Page ([`About.tsx`](file:///web/src/pages/About.tsx))**: Contains the official team section detailing the respective technical focus areas and primary contributions of Anish Raj and Jnandeep.
- **Journey Page ([`Journey.tsx`](file:///web/src/pages/Journey.tsx))**: Dedicated strictly to the technical chronological narrative of research milestones, discoveries, and anomaly autopsies, without individual contributor badges.
