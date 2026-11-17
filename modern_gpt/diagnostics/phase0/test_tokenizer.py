import sys
from pathlib import Path

sys.path.insert(0, ".")
from modern_gpt.data.bpe_tokenizer import HuggingFaceBPEWrapper

tok = HuggingFaceBPEWrapper.load("modern_gpt/artifacts/tokenizer_16k_v2")

print("Vocab size:", tok.vocab_size)
print("BOS ID:", getattr(tok, "bos_token_id", None))
print("EOS ID:", getattr(tok, "eos_token_id", None))
print("PAD ID:", getattr(tok, "pad_token_id", None))
print("UNK ID:", getattr(tok, "unk_token_id", None))
print("Special tokens:", getattr(tok, "special_tokens", None))

sample_strings = [
    "<|im_start|>",
    "<|im_end|>",
    "<|im_start|>system\nYou are ModernGPT-S, a helpful and precise AI assistant.<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<|endoftext|>",
    "Hello"
]

for s in sample_strings:
    ids = tok.encode(s)
    tokens = [tok.decode([i]) for i in ids]
    print(f"\n--- String: {repr(s)} ---")
    print(f"IDs: {ids}")
    print(f"Decoded tokens: {tokens}")
