import os
from pathlib import Path
from typing import Optional, Union

import tiktoken

from .bpe_tokenizer import BaseTokenizerWrapper, HuggingFaceBPEWrapper, TiktokenGPT2Wrapper


def get_tokenizer(name: str = "gpt2", path: Optional[Union[str, Path]] = None):
    """Retrieve a tokenizer instance.

    Supported names:
    - 'gpt2': tiktoken GPT-2 tokenizer (50,257 vocab)
    - 'bpe_16k_v2': 16,000-token Byte-level BPE with <|endoftext|> (no UNK)
    - 'bpe_16k': legacy 16,000-token BPE with UNK
    - Custom path to tokenizer directory or JSON file
    """
    if name == "gpt2" and path is None:
        return tiktoken.get_encoding("gpt2")
    if name in ("bpe_32k", "bpe_32k_production", "tokenizer_32k"):
        p = path or (Path("modern_gpt/artifacts/tokenizer_32k") if Path("modern_gpt/artifacts/tokenizer_32k").exists() else Path("artifacts/tokenizer_32k"))
        return HuggingFaceBPEWrapper.load(p, name="bpe_32k_production")
    if name in ("bpe_16k_v2", "tokenizer_16k_v2"):
        p = path or (Path("modern_gpt/artifacts/tokenizer_16k_v2") if Path("modern_gpt/artifacts/tokenizer_16k_v2").exists() else Path("artifacts/tokenizer_16k_v2"))
        return HuggingFaceBPEWrapper.load(p, name="bpe_16k_v2")
    if name in ("bpe_16k", "tokenizer_16k"):
        p = path or Path("artifacts/tokenizer_16k")
        return HuggingFaceBPEWrapper.load(p, name="bpe_16k")
    if path is not None:
        return HuggingFaceBPEWrapper.load(path, name=name)
    return tiktoken.get_encoding("gpt2")


def get_unified_tokenizer(name: str = "gpt2", path: Optional[Union[str, Path]] = None) -> BaseTokenizerWrapper:
    """Retrieve unified wrapper interface (BaseTokenizerWrapper)."""
    if name == "gpt2" and path is None:
        return TiktokenGPT2Wrapper()
    if name in ("bpe_32k", "bpe_32k_production", "tokenizer_32k"):
        p = path or (Path("modern_gpt/artifacts/tokenizer_32k") if Path("modern_gpt/artifacts/tokenizer_32k").exists() else Path("artifacts/tokenizer_32k"))
        return HuggingFaceBPEWrapper.load(p, name="bpe_32k_production")
    if name in ("bpe_16k_v2", "tokenizer_16k_v2"):
        p = path or (Path("modern_gpt/artifacts/tokenizer_16k_v2") if Path("modern_gpt/artifacts/tokenizer_16k_v2").exists() else Path("artifacts/tokenizer_16k_v2"))
        return HuggingFaceBPEWrapper.load(p, name="bpe_16k_v2")
    p = path or Path("artifacts/tokenizer_16k")
    return HuggingFaceBPEWrapper.load(p, name=name)


def encode(text, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        except TypeError:
            return tokenizer.encode(text)
    raise AttributeError(f"Tokenizer {type(tokenizer)} missing encode method")


def decode(tokens, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    return tokenizer.decode(tokens)
