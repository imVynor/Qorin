# ModernGPT-S 500M — Phase 1B.3: Architectural Decisions and Risk Log

## 1. Key Architectural and Operational Decisions

### Decision 1B.3-D1: Continuous-Stream Causal Attention Semantics
- **Context**: In sequence packing, multiple documents are packed into single 2,048-token chunks separated by `<|endoftext|>`.
- **Options**:
  1. *Strict Document Isolation*: Enforce 2D block-diagonal attention masking so tokens in document $D_2$ cannot attend to document $D_1$.
  2. *Continuous-Stream Causal LM*: Allow standard lower-triangular causal attention across the entire 2,048 chunk, relying on `<|endoftext|>` to signal document transitions.
- **Decision**: **Option 2 (Continuous-Stream Causal LM)**.
- **Rationale**: Continuous-stream causal LM is the industry standard for modern open foundation models (GPT-2, GPT-3, LLaMA, Mistral). It maximizes GPU throughput and context packing efficiency without requiring custom sparse or nested attention kernels. Strict document isolation is reserved for SFT, where cross-turn contamination must be avoided.

---

### Decision 1B.3-D2: Single-Epoch Enforcement & Zero Artificial Repetition
- **Context**: Local storage contains ~228M unique pretraining tokens, whereas the G3 target mixture requests 1,000,000,000 tokens.
- **Decision**: Reject document repetition. Stop generation when unique single-epoch local data is exhausted (~200M training tokens across 4 full shards).
- **Rationale**: Artificially repeating documents $4.5\times$ to simulate 1B tokens would pollute empirical validation, degrade the model's perplexity scaling curve, and violate core project integrity rules.

---

### Decision 1B.3-D3: Prohibition of Cross-Domain Substitution
- **Context**: Certain domains (Books, Math, Wikipedia, Dialogue) have severe local supply shortages, while Web (`fineweb-edu`) has sufficient local data to generate more tokens.
- **Decision**: Do NOT rebalance or substitute Web data into the deficient domains.
- **Rationale**: Foundation models require high-entropy domain diversity. Silently replacing Books or Technical data with Web text causes domain skew and biases downstream representations.

---

### Decision 1B.3-D4: Formal Phase Status Classification
- **Decision**: Formally report Phase 1B.3 as **PARTIALLY COMPLETE / BLOCKED ON EXTERNAL RAW CORPUS DOWNLOADS**.
- **Rationale**: The production data pipeline (filtering, deduplication, packing, binary uint16 sharding, and streaming PyTorch DataLoader) is 100% verified and operational. However, the completion criteria require all 7 domains to meet their 1B pilot allocations. Accurately declaring the supply blocker ensures transparency.

---

## 2. Risk Assessment and Mitigation Matrix

| Risk ID | Risk Description | Severity | Likelihood | Mitigation Strategy |
|---|---|:---:|:---:|---|
| **R-1B.3-01** | **Pretraining Supply Bottleneck for Full 5B Scale** | **HIGH** | **HIGH** | Local disk lacks 4.8B tokens. Mitigation: Execute automated batch download script pulling Hugging Face open datasets (FineWeb-Edu, Cosmopedia, StarCoder, OpenWebMath, SmolLM-Corpus). |
| **R-1B.3-02** | **Evaluation Overlap & Test Leakage** | **HIGH** | **LOW** | Mitigation: Pre-generation SHA-256 text quarantine filter completely isolated 9,348 validation documents before training ingestion began (0% overlap verified). |
| **R-1B.3-03** | **Streaming DataLoader Desynchronization** | **MEDIUM** | **LOW** | Mitigation: Implemented state-dict checkpointing and strict byte-offset seeking within `BinaryShardDataset`. Verified next-token shift invariant bitwise across multiple shards. |
| **R-1B.3-04** | **RAM Exhaustion on Large JSONL Streams** | **MEDIUM** | **LOW** | Mitigation: Streamed file processing line-by-line; bounded document arrays; NumPy uint16 zero-copy serialization. Memory peaked safely below 3.5 GB. |

