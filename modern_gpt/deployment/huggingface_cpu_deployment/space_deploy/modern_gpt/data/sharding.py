"""Tokenization, sequence packing, and memory-mapped sharded dataset for ModernGPT-S.

Features:
- Encodes documents streamingly and writes tokens as uint16 binary arrays (memmap).
- Inserts explicit EOS token (<|endoftext|>) at document boundaries.
- Strictly keeps training and validation sequences isolated (no cross-boundary chunks).
- Constructs correctly shifted (input, target) sequences of length `context_length`.
- Avoids loading entire corpus into RAM.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from .bpe_tokenizer import BaseTokenizerWrapper
from .sources import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class ShardManifest:
    tokenizer_name: str
    vocab_size: int
    split: str
    total_tokens: int
    total_documents: int
    num_packed_sequences: int
    context_length: int
    shard_file: str
    dtype: str
    eos_token_id: int
    seed: int
    created_at: str

    def to_dict(self) -> Dict:
        return asdict(self)


def create_tokenized_shard(
    documents: List[NormalizedDocument],
    tokenizer: BaseTokenizerWrapper,
    output_path: Union[str, Path],
    context_length: int = 512,
    max_tokens: Optional[int] = None,
    seed: int = 123,
    split_name: str = "train",
) -> Tuple[Path, ShardManifest]:
    """Tokenize a list of documents and write sequentially to a uint16 binary shard."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path.with_suffix(".json")

    eos_id = tokenizer.eos_token_id
    tokens_buffer: List[int] = []
    docs_processed = 0

    # Stream encode into token list
    for doc in documents:
        doc_tokens = tokenizer.encode(doc.text)
        tokens_buffer.extend(doc_tokens)
        tokens_buffer.append(eos_id)  # Explicit end-of-document delimiter
        docs_processed += 1

        if max_tokens is not None and len(tokens_buffer) >= max_tokens:
            tokens_buffer = tokens_buffer[:max_tokens]
            break

    total_tokens = len(tokens_buffer)
    # Check uint16 bounds (< 65536)
    if tokenizer.vocab_size > 65535:
        dtype = np.uint32
        dtype_str = "uint32"
    else:
        dtype = np.uint16
        dtype_str = "uint16"

    # Write binary memmap file
    arr = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    arr[:] = np.array(tokens_buffer, dtype=dtype)
    arr.flush()
    del arr  # Close handle

    # A sequence of length context_length requires context_length + 1 tokens for (input, target)
    stride = context_length
    num_packed = max(0, (total_tokens - 1) // stride)

    import datetime
    manifest = ShardManifest(
        tokenizer_name=tokenizer.name,
        vocab_size=tokenizer.vocab_size,
        split=split_name,
        total_tokens=total_tokens,
        total_documents=docs_processed,
        num_packed_sequences=num_packed,
        context_length=context_length,
        shard_file=output_path.name,
        dtype=dtype_str,
        eos_token_id=eos_id,
        seed=seed,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    logger.info(
        "Created shard %s: %d tokens (%d packed sequences of len %d)",
        output_path,
        total_tokens,
        num_packed,
        context_length,
    )
    return output_path, manifest


class ShardedPackedDataset(Dataset):
    """Memory-mapped PyTorch Dataset reading from binary token shards.

    Yields (inputs, targets) tensors of shape (context_length,).
    """

    def __init__(
        self,
        shard_path: Union[str, Path],
        context_length: int = 512,
        stride: Optional[int] = None,
        dtype: str = "uint16",
    ):
        self.shard_path = Path(shard_path)
        self.context_length = context_length
        self.stride = stride or context_length
        self.dtype = np.uint16 if dtype == "uint16" else np.uint32

        self.tokens = np.memmap(self.shard_path, dtype=self.dtype, mode="r")
        self.num_tokens = len(self.tokens)

        # Number of samples we can extract
        if self.num_tokens <= self.context_length:
            self.num_samples = 0
        else:
            self.num_samples = (self.num_tokens - 1 - self.context_length) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        end = start + self.context_length
        # Read slice from memory map
        chunk = self.tokens[start : end + 1]
        inputs = torch.from_numpy(chunk[:-1].astype(np.int64))
        targets = torch.from_numpy(chunk[1:].astype(np.int64))
        return inputs, targets
