# Phase 1B.5: Document Boundary and Loss Semantics Specification

## 1. Continuous-Stream Pretraining Semantics

In ModernGPT-S 500M pretraining, documents are packed into fixed sequence length $L = 2048$ separated by the `<|endoftext|>` token (ID: 0).

Under standard causal language modeling:
$$P(x_1, \dots, x_L) = \prod_{t=1}^L P(x_t \mid x_{<t})$$

Attention masks are lower-triangular causal masks. Documents concatenated within the same 2,048-token window share causal self-attention across `<|endoftext|>`. This continuous-stream convention is standard for large-scale pretraining (e.g. GPT-3, LLaMA, Mistral) and preserves maximum token throughput without pad token wastage.

## 2. Autoregressive Shift Invariance Proof

For any sequence of input tokens $x \in \mathbb{Z}^{B \times L}$:
- Inputs: $x_{input} = x[:, :-1]$
- Targets: $x_{target} = x[:, 1:]$

Both slices have length $L - 1 = 2047$. For each position $i \in [0, L-2]$:
$$x_{target}[b, i] = x[b, i + 1] = x_{input}[b, i + 1]$$
This maintains exact bitwise next-token prediction alignment with zero off-by-one errors.
