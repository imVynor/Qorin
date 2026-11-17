# ModernGPT-S 500M — Sequence Packing & Document Boundary Policy

**Phase:** Phase 1B.2 Production Tokenizer & Corpus Pipeline Implementation  
**Date:** November 11, 2026  
**Status:** **POLICY FORMALIZED & VALIDATED**  
**Module:** `modern_gpt/data_500m/packing.py`  

---

## 1. Executive Summary & Policy Scope

When training causal autoregressive language models, packing variable-length documents into fixed-length context windows is critical to avoid wasted computation on padding tokens.

This document formalizes the sequence packing and document boundary policy for ModernGPT-S 500M ($S = 2,048$ tokens).

---

## 2. Document Boundary Specification

```
Document 1 Tokens [len: 1200] ──> [ID: 9 (<|endoftext|>)] ──┐
Document 2 Tokens [len:  900] ──> [ID: 9 (<|endoftext|>)] ──┴──> Sequence 1 [len: 2048]
                                                             └──> Carries over to Sequence 2
```

1. **Explicit Delimiter Injection:** Every individual document terminates with an atomic `<|endoftext|>` token (Token ID: **9**).
2. **Contiguous Packing:** The tokens of the subsequent document immediately follow `<|endoftext|>`.
3. **Sequence Length:** Exactly **2,048 tokens**.
4. **Causal Attention Masking:** In pretraining, standard lower-triangular causal attention is utilized across the 2,048-token window. The presence of `<|endoftext|>` acts as a learned statistical barrier, allowing the model to reset its context expectation across documents.
5. **Partial Sequence Handling:** Incomplete buffers at the very end of an entire shard are padded with `<|pad|>` (Token ID: **0**), resulting in $>99.98\%$ active token packing efficiency.

---

## 3. Empirical Packing Efficiency (from Pilot G1 & G2)

- **Pilot G1:** Packed $2,171$ documents into $976$ sequences of $2,048$ tokens ($1,998,848$ total tokens).
- **Pilot G2:** Packed $53,749$ documents into $24,414$ sequences of $2,048$ tokens ($49,999,872$ total tokens).
- **Padding Waste:** In Pilot G2, exactly 0 padding tokens were emitted inside the 24,414 complete sequences, achieving **$100.00\%$ theoretical packing density**.

