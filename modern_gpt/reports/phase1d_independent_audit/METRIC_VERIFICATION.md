# Phase 1D Metric Verification

**Audit method:** Static recalculation from `results.json` and `raw_generations.jsonl`. No GPU inference.

---

## 1. Validation Loss — VERIFIED

Recalculated from per-domain figures in `results.json`:

| Domain | CE Loss | Weight | Weighted Contribution |
|---|---|---|---|
| `web` | 3.5868 | 0.400 | 1.4347 |
| `edu` | 2.6204 | 0.150 | 0.3931 |
| `code` | 2.0272 | 0.150 | 0.3041 |
| `wiki` | 3.5398 | 0.075 | 0.2655 |
| `math` | 2.8965 | 0.075 | 0.2172 |
| `books` | 3.4118 | 0.100 | 0.3412 |
| `dialogue` | 2.2169 | 0.050 | 0.1108 |
| **Total** | — | **1.000** | **3.0666** |

**Result:** Independently recalculated weighted loss = **3.0666** ✅ Exact match.  
**Perplexity:** `exp(3.0666)` = **21.47** ✅ Exact match.  
**Total tokens evaluated:** 7 domains × 51,200 = **358,400** ✅

### Implementation Details Verified
- Sequences per domain: 25 × 2,048 tokens = 51,200 tokens
- Offset: `i * seq_len` (contiguous, not strided — see §3)
- Loss: `F.cross_entropy(logits[0], y[0], reduction='mean')` on shifted targets — correct
- Early/late positional split: tokens `[:512]` vs `[-512:]` per sequence — plausible but not ablated

### Contiguous Sequence Sampling Bug (Low Severity)
```python
offset = i * seq_len  # seq 0: tokens 0–2047, seq 1: tokens 2048–4095
```
Consecutive sequences share no overlap but are contiguous, meaning they are not independent random draws from the shard. For 51,200 tokens across 25 contiguous sequences, the effective diversity is limited to the first 51,200 tokens of each shard. This likely has minimal impact on mean loss but slightly overstates diversity of the sample.

---

## 2. Cross-Model Loss Comparison — METHODOLOGICALLY INVALID

The report compares:
- 100M baseline val loss: **3.1200** (reported, historical)
- 500M milestone val loss: **3.0666** (freshly computed)

**This comparison is not valid for three reasons:**

### Reason 1: Different Vocabularies
- 100M uses a 16,000-token BPE tokenizer
- 500M uses a 32,768-token BPE tokenizer

Cross-entropy loss per token is not comparable across different vocabulary sizes. A larger vocabulary tends to produce lower cross-entropy per token (more precise token boundaries), so 32K tokenization structurally biases toward lower loss numbers.

The correct normalization is **bits-per-byte (BPB)**:
$$\text{BPB} = \frac{H \cdot \log_2 e}{\bar{c}}$$

where $H$ is cross-entropy loss and $\bar{c}$ is mean characters-per-token. This was never calculated.

### Reason 2: Different Evaluation Datasets
The 100M loss figure comes from historical training logs produced during the 100M run on its own 16K-tokenized validation shards. It was not re-evaluated live on the same `data_500m/final_5b/val` shards used for the 500M evaluation.

### Reason 3: Different Training Corpus Mixes
The domain weight scheme (web=40%, edu=15%, code=15%) represents the 500M training mix. The 100M was trained on a different corpus distribution with different domain weights. Applying the same aggregation scheme to both is not apples-to-apples.

**Defensible alternative statement:**
> "The 500M model achieves a weighted CE loss of 3.0666 on the 32K-tokenized 500M validation set. Direct numerical comparison to the 100M baseline loss is not valid without BPB normalization and identical evaluation datasets."

---

## 3. Benchmark Counts — VERIFIED

| Benchmark | Reported | JSONL Count | Match |
|---|---|---|---|
| Behavioral cases | 100 | 100 | ✅ |
| Coding tasks | 50 | 50 | ✅ |
| JSON tasks | 50 | 50 | ✅ |
| Math tasks | 50 | 50 | ✅ |
| Tool tasks | 20 | 20 | ✅ |
| RAG queries | 10 | 10 | ✅ |
| Coding Pass@1 | 9 | 9 | ✅ |
| Math correct | 0 | 0 | ✅ |
| JSON valid | 1 | 1 | ✅ |
| JSON schema | 0 | 0 | ✅ |
| RAG A | 3 | 3 | ✅ |
| RAG B | 4 | 4 | ✅ |
| RAG C | 2 | 2 | ✅ |
| RAG D | 2 | 2 | ✅ |

All counts are consistent with the JSONL outputs. The numbers are correct. The *interpretations* attached to some of these numbers are not.

