"""Byte-level BPE Tokenizer training and wrapper for ModernGPT-S.

Supports 16,000-token BPE tokenizer without unnecessary <|unk|> token.
Because ByteLevel pre-tokenization with initial byte alphabet covers all 256
UTF-8 byte values, any valid UTF-8 string is representable without out-of-vocabulary fallback.
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

logger = logging.getLogger(__name__)

# Default special tokens for v2: only EOS is needed for causal autoregressive modeling
DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def compute_tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    """Compute deterministic SHA-256 fingerprint from tokenizer serialization."""
    raw_str = tokenizer.to_str()
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


class BaseTokenizerWrapper:
    """Unified tokenizer interface for ModernGPT-S."""

    @property
    def name(self) -> str:
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    @property
    def eos_token_id(self) -> int:
        raise NotImplementedError

    @property
    def pad_token_id(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def bos_token_id(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def unk_token_id(self) -> Optional[int]:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        raise NotImplementedError

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int]) -> str:
        raise NotImplementedError


class TiktokenGPT2Wrapper(BaseTokenizerWrapper):
    """Wrapper around tiktoken GPT-2 tokenizer (50,257 tokens)."""

    def __init__(self):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self._eos_id = self._enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    @property
    def name(self) -> str:
        return "gpt2"

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab  # 50257

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._eos_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return None

    @property
    def unk_token_id(self) -> Optional[int]:
        return None

    @property
    def fingerprint(self) -> str:
        return "tiktoken_gpt2_50257"

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        if allowed_special is None:
            allowed_special = {"<|endoftext|>"}
        return self._enc.encode(text, allowed_special=allowed_special)

    def decode(self, tokens: List[int]) -> str:
        return self._enc.decode(tokens)


class HuggingFaceBPEWrapper(BaseTokenizerWrapper):
    """Wrapper around Hugging Face Byte-level BPE tokenizer (e.g. 16,000 tokens)."""

    def __init__(self, tokenizer: Tokenizer, name: str = "bpe_16k_v2"):
        self._tokenizer = tokenizer
        self._name = name

        # Dynamically resolve special tokens
        self._eos_id = self._tokenizer.token_to_id("<|endoftext|>")
        if self._eos_id is None:
            raise ValueError("Tokenizer must define <|endoftext|> as EOS token.")

        self._unk_id = self._tokenizer.token_to_id("<|unk|>")
        self._bos_id = self._tokenizer.token_to_id("<|bos|>")
        self._pad_id = self._tokenizer.token_to_id("<|pad|>")

        self._fingerprint = compute_tokenizer_fingerprint(self._tokenizer)

    @property
    def name(self) -> str:
        return self._name

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def eos_token_id(self) -> int:
        return self._eos_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_id

    @property
    def bos_token_id(self) -> Optional[int]:
        return self._bos_id

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._unk_id

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def encode(self, text: str, allowed_special: Optional[set] = None) -> List[int]:
        return self._tokenizer.encode(text).ids

    def decode(self, tokens: List[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def save(self, output_dir: Union[str, Path]):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(output_dir / "tokenizer.json"))
        config = {
            "name": self._name,
            "vocab_size": self.vocab_size,
            "fingerprint": self._fingerprint,
            "special_tokens": {
                "eos": "<|endoftext|>",
                "eos_id": self._eos_id,
                "unk": "<|unk|>" if self._unk_id is not None else None,
                "unk_id": self._unk_id,
                "bos": "<|bos|>" if self._bos_id is not None else None,
                "bos_id": self._bos_id,
                "pad": "<|pad|>" if self._pad_id is not None else None,
                "pad_id": self._pad_id,
            },
        }
        with (output_dir / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path], name: Optional[str] = None) -> "HuggingFaceBPEWrapper":
        path = Path(path)
        if path.is_dir():
            tokenizer_file = path / "tokenizer.json"
            cfg_file = path / "tokenizer_config.json"
        else:
            tokenizer_file = path
            cfg_file = path.parent / "tokenizer_config.json"

        if not tokenizer_file.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

        tokenizer = Tokenizer.from_file(str(tokenizer_file))

        tok_name = name
        if tok_name is None:
            if cfg_file.exists():
                try:
                    with cfg_file.open("r", encoding="utf-8") as f:
                        tok_name = json.load(f).get("name", "bpe_16k_v2")
                except Exception:
                    tok_name = "bpe_16k_v2"
            else:
                tok_name = path.name if path.is_dir() else "bpe_16k_v2"

        return cls(tokenizer, name=tok_name)


def train_bpe_tokenizer(
    text_iterator: Iterator[str],
    vocab_size: int = 16000,
    special_tokens: Optional[List[str]] = None,
    name: str = "bpe_16k_v2",
) -> Tuple[HuggingFaceBPEWrapper, Dict]:
    """Train a byte-level BPE tokenizer without unnecessary UNK token."""
    if special_tokens is None:
        special_tokens = list(DEFAULT_SPECIAL_TOKENS)

    unk_token = "<|unk|>" if "<|unk|>" in special_tokens else None
    logger.info(
        "Initializing Byte-level BPE tokenizer (%s): target vocab size=%d, special_tokens=%s, unk=%s",
        name,
        vocab_size,
        special_tokens,
        unk_token,
    )

    raw_tokenizer = Tokenizer(BPE(unk_token=unk_token))
    raw_tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )

    start_time = time.perf_counter()
    raw_tokenizer.train_from_iterator(text_iterator, trainer=trainer)
    train_duration = time.perf_counter() - start_time

    wrapper = HuggingFaceBPEWrapper(raw_tokenizer, name=name)
    metrics = {
        "name": name,
        "target_vocab_size": vocab_size,
        "actual_vocab_size": wrapper.vocab_size,
        "training_duration_seconds": round(train_duration, 3),
        "fingerprint": wrapper.fingerprint,
        "special_tokens": {
            "eos_id": wrapper.eos_token_id,
            "unk_id": wrapper.unk_token_id,
            "bos_id": wrapper.bos_token_id,
            "pad_id": wrapper.pad_token_id,
        },
    }
    logger.info(
        "Trained %s: vocab_size=%d (fingerprint=%s) in %.2fs",
        name,
        wrapper.vocab_size,
        wrapper.fingerprint,
        train_duration,
    )
    return wrapper, metrics
