# ModernGPT-S 500M — Architecture Specification

**Phase:** Phase 1A Architecture & Feasibility  
**Date:** November 11, 2026  
**Status:** **DESIGN SPECIFICATION & VERIFIED**  
**Target Model:** ModernGPT-S 500M  
**Baseline Reference:** ModernGPT-S 100M GQA-4 (`modern_gpt/archive/100m_baseline/`)  

---

## 1. Architectural Philosophy & Scaling Rationale

ModernGPT-S 500M is designed as a direct architectural scale-up of the proven ModernGPT-S 100M foundation. It maintains the core tenets of modern open-weight dense decoder-only Transformers (inspired by Llama 3, Gemma, Mistral, and Qwen):

1. **Pre-Layer Normalization (Pre-Norm):** Stable gradient flow using Root Mean Square Normalization (RMSNorm).
2. **Rotary Position Embeddings (RoPE):** Relative positional encoding enabling seamless context extension.
3. **Grouped-Query Attention (GQA):** Memory-efficient KV cache retention using 24 Query heads and 4 Key-Value heads (6:1 GQA ratio).
4. **SwiGLU Feed-Forward Networks:** Non-linear gating ($2.5\times$ expansion ratio) for expressivity per parameter.
5. **Strictly Bias-Free Linear Layers:** Eliminating additive bias parameters across QKV projections, attention output projection, and SwiGLU projections to maximize throughput and improve generalization.
6. **Tied Input-Output Embeddings:** Sharing weights between `token_embedding` and `lm_head` to maximize capacity allocated to transformer layers.

---

## 2. Global Hyperparameter Specification

| Hyperparameter | Symbol | 100M Baseline | 500M Specification | Scaling Factor | Justification / Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Transformer Layers** | $L$ | 14 | **20** | $+42.9\%$ | Deepens representational hierarchy for multi-hop reasoning |
| **Hidden Dimension** | $d$ | 768 | **1,536** | $2.00\times$ | Doubled representation width per token |
| **Query Attention Heads** | $H_q$ | 12 | **24** | $2.00\times$ | Maintains constant head dimension $d_k = 64$ |
| **Key-Value Heads** | $H_{kv}$ | 4 | **4** | $1.00\times$ | Retains ultra-compact KV cache ($6:1$ GQA ratio) |
| **Head Dimension** | $d_k$ | 64 | **64** | $1.00\times$ | Optimal granularity for GPU Tensor Cores |
| **SwiGLU Intermediate Dim** | $d_{ffn}$ | 2,048 ($2.67\times$) | **3,840** ($2.50\times$) | $1.875\times$ | Standard Llama-style $2.5\times$ expansion ratio |
| **Context Window** | $S$ | 512 (pretrain) / 4096 (SFT) | **2,048** (initial) | $4.00\times$ | Native pretraining context for long documents & tool calling |
| **Normalization** | — | RMSNorm | **RMSNorm** | — | Pre-norm with scale weight only |
| **RMSNorm Epsilon** | $\epsilon$ | $1\times 10^{-6}$ | **$1\times 10^{-6}$** | — | Numerical stability standard |
| **Position Encoding** | — | RoPE | **RoPE** | — | Rotary positional embedding |
| **Base RoPE Frequency** | $\theta$ | 10,000.0 | **10,000.0** | — | Native frequency base |
| **RoPE Scale Factor** | — | 1.0 (512) / 8.0 (4096) | **1.0** (2048) | — | Unscaled during initial pretraining |
| **Linear Layer Biases** | — | Partial (out_proj had bias) | **None (100% False)** | — | Eliminates 30,720 stray parameters; matches target counts |
| **Embedding Tying** | — | Tied | **Tied** | — | `lm_head.weight == token_embedding.weight` |
| **Dropout** | $p$ | 0.05–0.1 | **0.0** | — | Modern pretraining standard (regularization via data scale) |

---

## 3. Block-Level Tensor Flow & Layer Architecture

Each of the 20 Transformer blocks executes the following computation:

```
Input: x [Batch, SeqLen, 1536]
   │
   ├───> RMSNorm (attention_norm) ──> [Batch, SeqLen, 1536]
   │        │
   │        ├──> Q Proj [1536 -> 1536] (24 heads x 64 dim) ──> RoPE
   │        ├──> K Proj [1536 ->  256] ( 4 heads x 64 dim) ──> RoPE ──> Repeat 6x
   │        └──> V Proj [1536 ->  256] ( 4 heads x 64 dim) ─────────────> Repeat 6x
   │                 │
   │                 ▼
   │        Scaled Dot-Product Attention (Causal Mask, SDPA)
   │                 │
   │        Out Proj [1536 -> 1536] (bias=False)
   │                 │
   ▼                 ▼
   Add Residual: x = x + Attn_Out
   │
   ├───> RMSNorm (mlp_norm) ─────────> [Batch, SeqLen, 1536]
   │        │
   │        ├──> Gate Proj [1536 -> 3840] (bias=False) ──> SiLU
   │        └──> Up Proj   [1536 -> 3840] (bias=False) ───┐
   │                                                      │
   │             Hadamard Product (SiLU(Gate) * Up) <─────┘
   │                      │
   │             Down Proj [3840 -> 1536] (bias=False)
   │                      │
   ▼                      ▼
   Add Residual: x = x + MLP_Out
   │
Output: x [Batch, SeqLen, 1536]
```

---

## 4. Discrepancy Resolution: Attention Output Bias

During Part 1 inspection of `modern_gpt/model/attention.py`, line 32 was identified:
```python
self.output = nn.Linear(config.emb_dim, config.emb_dim)
```
In PyTorch, `nn.Linear` defaults to `bias=True`. In a 20-layer model with $d=1536$:
$$20 \text{ layers} \times 1,536 \text{ bias parameters} = 30,720 \text{ parameters}$$

- With `bias=True`: Parameter count is **489,254,400** (16K vocab) or **514,420,224** (32K vocab).
- With `bias=False`: Parameter count is **489,223,680** (16K vocab) or **514,389,504** (32K vocab).

**Architectural Decision:** To strictly satisfy the specification requirement *"No linear-layer biases"* and match the prompt's target parameter counts to the single parameter, `self.output` will be initialized with `bias=False` in the 500M implementation:
```python
self.output = nn.Linear(config.emb_dim, config.emb_dim, bias=False)
```

