# Phase 1B.6: Tokenizer Integrity & Sequence Semantics Report

## 1. Tokenizer Cryptographic Verification

- **Tokenizer Architecture**: Byte-Pair Encoding (BPE)
- **Vocabulary Size**: 32,768 tokens
- **Vocabulary Fingerprint**: `3559c7435e1d9330`
- **Full File SHA-256**: `d10624a9069ffac5a660ef23ca101d5792fdc240575068adde1646b13407eb59`
- **Document Delimiter (EOS / EOT)**: Token ID 9 (`<|endoftext|>`)

## 2. Sequence Semantics Verification

ModernGPT-S 500M utilizes continuous-stream causal language modeling:
- Sequence Length: $L = 2048$
- Cross-Document Attention: Enabled across `<|endoftext|>` boundaries.
- Autoregressive Target Shift: $x_{input} = x[:, :-1]$ and $x_{target} = x[:, 1:]$.
- Bitwise Invariance: $x_{target}[b, i] = x[b, i + 1] = x_{input}[b, i + 1]$.
