# Phase 1B.7: Shard Packing Configuration & Sequence Semantics

## 1. Architectural & Tokenizer Parameters

- **Tokenizer Architecture**: Byte-Pair Encoding (BPE), Byte-fallback enabled
- **Vocabulary Size**: 32,768 unique tokens
- **Context Length ($L$)**: 2,048 tokens
- **Input / Target Sequence Length**: 2,047 tokens ($x[:, :-1]$ and $y[:, 1:]$)
- **Document Delimiter**: Token ID 9 (`<|endoftext|>`)
- **Token Representation**: Little-endian `uint16` (2 bytes per token ID)
- **Token Range**: $[0, 32767]$
- **Attention Policy**: Continuous-stream cross-document attention across EOT boundaries
- **Standard Shard Size**: 50,000,000 tokens (100,000,080 bytes with 80-byte header)

## 2. Next-Token Shift Semantics

For every sequence $S = (t_0, t_1, \dots, t_2047)$ of 2,048 tokens:
- **Input Tensor**: $x = (t_0, t_1, \dots, t_2046)$ of shape $[2047]$
- **Target Tensor**: $y = (t_1, t_2, \dots, t_2047)$ of shape $[2047]$
- **Causal Invariant**: $x[1:] == y[:-1]$ holds for 100% of generated sequences.
