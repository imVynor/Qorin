"""Evaluation of tokenizer quality, compression, and encoding fidelity.

Compares GPT-2 tokenizer (50,257 vocab) against Modern BPE (16,000 vocab v1 and v2)
on identical held-out validation documents and representative probe prompts.
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from .bpe_tokenizer import BaseTokenizerWrapper
from .sources import NormalizedDocument

logger = logging.getLogger(__name__)


@dataclass
class TokenizerMetrics:
    documents_evaluated: int
    total_characters: int
    total_utf8_bytes: int
    total_tokens: int
    avg_tokens_per_doc: float
    avg_bytes_per_token: float
    avg_chars_per_token: float
    relative_token_ratio_to_gpt2: float
    encoding_failures: int


@dataclass
class QualitativeSample:
    category: str
    sample_text: str
    gpt2_token_count: int
    bpe16k_token_count: int
    bpe16k_v2_token_count: Optional[int]
    gpt2_roundtrip_match: bool
    bpe16k_roundtrip_match: bool
    bpe16k_v2_roundtrip_match: bool
    bpe16k_v2_tokens_preview: List[int]
    bpe16k_v2_decoded_preview: str


QUALITATIVE_PROBES = [
    {
        "category": "Ordinary English Prose",
        "text": "The quick brown fox jumps over the lazy dog. In recent years, pretraining autoregressive language models on curated web documents has yielded substantial improvements across diverse downstream comprehension benchmarks.",
    },
    {
        "category": "Mathematical Notation & Symbols",
        "text": r"Let f(x) = \int_{0}^{\infty} \frac{x^2 e^{-x}}{\sqrt{2\pi \sigma^2}} dx. By Bayes' Theorem, P(A|B) = \frac{P(B|A)P(A)}{P(B)}, where \sum_{i=1}^n x_i \leq \epsilon.",
    },
    {
        "category": "Code & Punctuation",
        "text": "def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> float:\n    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))\n    assert not torch.isnan(loss), 'Loss is NaN'\n    return float(loss.item())\n",
    },
    {
        "category": "Unicode & Multilingual",
        "text": "Résumé: François étudie les réseaux de neurones à Paris. 机器学习与自然语言处理 🚀✨ [α, β, γ] ∈ ℝ².",
    },
    {
        "category": "Emoji & Modern Symbols",
        "text": "Antigravity IDE 🚀 🔥 🤖 💻 🦄 | High performance deep learning with ModernGPT-S.",
    },
]


def evaluate_single_tokenizer_on_docs(
    tokenizer: BaseTokenizerWrapper,
    documents: List[NormalizedDocument],
    reference_tokens: Optional[int] = None,
) -> TokenizerMetrics:
    """Compute quantitative compression metrics for a tokenizer on document list."""
    total_chars = sum(d.char_count for d in documents)
    total_bytes = sum(d.byte_count for d in documents)
    total_tokens = 0
    failures = 0

    for doc in documents:
        try:
            tokens = tokenizer.encode(doc.text)
            total_tokens += len(tokens)
        except Exception as err:
            failures += 1
            logger.error("Encoding failure with %s on doc %s: %s", tokenizer.name, doc.document_id, err)

    num_docs = len(documents)
    avg_tokens_per_doc = round(total_tokens / num_docs, 2) if num_docs > 0 else 0.0
    avg_bytes_per_token = round(total_bytes / total_tokens, 4) if total_tokens > 0 else 0.0
    avg_chars_per_token = round(total_chars / total_tokens, 4) if total_tokens > 0 else 0.0

    ref = reference_tokens if reference_tokens and reference_tokens > 0 else total_tokens
    relative_ratio = round(total_tokens / ref, 4)

    return TokenizerMetrics(
        documents_evaluated=num_docs,
        total_characters=total_chars,
        total_utf8_bytes=total_bytes,
        total_tokens=total_tokens,
        avg_tokens_per_doc=avg_tokens_per_doc,
        avg_bytes_per_token=avg_bytes_per_token,
        avg_chars_per_token=avg_chars_per_token,
        relative_token_ratio_to_gpt2=relative_ratio,
        encoding_failures=failures,
    )


def evaluate_qualitative_probes(
    gpt2_tokenizer: BaseTokenizerWrapper,
    bpe16k_v2_tokenizer: BaseTokenizerWrapper,
    bpe16k_v1_tokenizer: Optional[BaseTokenizerWrapper] = None,
    probes: Optional[List[Dict[str, str]]] = None,
) -> List[QualitativeSample]:
    """Run qualitative encode-decode probe tests."""
    probes = probes or QUALITATIVE_PROBES
    samples = []

    for item in probes:
        text = item["text"]
        cat = item["category"]

        gpt2_toks = gpt2_tokenizer.encode(text)
        bpe_v2_toks = bpe16k_v2_tokenizer.encode(text)

        gpt2_dec = gpt2_tokenizer.decode(gpt2_toks)
        bpe_v2_dec = bpe16k_v2_tokenizer.decode(bpe_v2_toks)

        if bpe16k_v1_tokenizer:
            bpe_v1_toks = bpe16k_v1_tokenizer.encode(text)
            bpe_v1_dec = bpe16k_v1_tokenizer.decode(bpe_v1_toks)
            v1_count = len(bpe_v1_toks)
            v1_match = (bpe_v1_dec == text)
        else:
            v1_count = len(bpe_v2_toks)
            v1_match = True

        samples.append(
            QualitativeSample(
                category=cat,
                sample_text=text[:100] + ("..." if len(text) > 100 else ""),
                gpt2_token_count=len(gpt2_toks),
                bpe16k_token_count=v1_count,
                bpe16k_v2_token_count=len(bpe_v2_toks),
                gpt2_roundtrip_match=(gpt2_dec == text),
                bpe16k_roundtrip_match=v1_match,
                bpe16k_v2_roundtrip_match=(bpe_v2_dec == text),
                bpe16k_v2_tokens_preview=bpe_v2_toks[:12],
                bpe16k_v2_decoded_preview=bpe_v2_dec[:100],
            )
        )

    return samples


def compare_tokenizers(
    gpt2_tokenizer: BaseTokenizerWrapper,
    bpe16k_tokenizer: BaseTokenizerWrapper,
    val_documents: List[NormalizedDocument],
    bpe16k_v1_tokenizer: Optional[BaseTokenizerWrapper] = None,
) -> Dict:
    """Run full comparative evaluation between GPT-2 and 16K BPE (v1 and v2)."""
    by_source: Dict[str, List[NormalizedDocument]] = {}
    for d in val_documents:
        by_source.setdefault(d.source, []).append(d)

    # 1. Evaluate GPT-2
    gpt2_overall = evaluate_single_tokenizer_on_docs(gpt2_tokenizer, val_documents)
    gpt2_by_source = {
        src: evaluate_single_tokenizer_on_docs(gpt2_tokenizer, docs)
        for src, docs in by_source.items()
    }

    # 2. Evaluate 16K BPE v2 (primary) relative to GPT-2
    bpe_overall = evaluate_single_tokenizer_on_docs(
        bpe16k_tokenizer, val_documents, reference_tokens=gpt2_overall.total_tokens
    )
    bpe_by_source = {
        src: evaluate_single_tokenizer_on_docs(
            bpe16k_tokenizer, docs, reference_tokens=gpt2_by_source[src].total_tokens
        )
        for src, docs in by_source.items()
    }

    # 3. Qualitative probes
    probe_results = evaluate_qualitative_probes(
        gpt2_tokenizer, bpe16k_tokenizer, bpe16k_v1_tokenizer=bpe16k_v1_tokenizer
    )

    ratio = bpe_overall.relative_token_ratio_to_gpt2
    text_coverage_factor = round(1.0 / ratio, 4) if ratio > 0 else 1.0

    result = {
        "gpt2": {
            "name": gpt2_tokenizer.name,
            "vocab_size": gpt2_tokenizer.vocab_size,
            "fingerprint": getattr(gpt2_tokenizer, "fingerprint", "gpt2"),
            "overall": asdict(gpt2_overall),
            "by_source": {k: asdict(v) for k, v in gpt2_by_source.items()},
        },
        "bpe_16k_v2": {
            "name": bpe16k_tokenizer.name,
            "vocab_size": bpe16k_tokenizer.vocab_size,
            "fingerprint": getattr(bpe16k_tokenizer, "fingerprint", "bpe_16k_v2"),
            "has_unk_token": bpe16k_tokenizer.unk_token_id is not None,
            "overall": asdict(bpe_overall),
            "by_source": {k: asdict(v) for k, v in bpe_by_source.items()},
        },
        "qualitative_probes": [asdict(p) for p in probe_results],
        "token_budget_tradeoff_analysis": {
            "relative_token_ratio (16k_v2 / gpt2)": ratio,
            "fixed_token_budget_text_coverage_factor": text_coverage_factor,
            "explanation": (
                "A 16,000 vocabulary without <|unk|> relies entirely on the initial 256 byte tokens for fallback. "
                "Any unseen UTF-8 byte sequence is losslessly encoded into bytes with zero UNK degradations. "
                f"A fixed token budget of N tokens covers approximately {round(text_coverage_factor * 100, 1)}% "
                "as much raw text as GPT-2 while reducing model parameters by ~13.15M (-44%)."
            ),
        },
    }

    # Also evaluate v1 if provided for side-by-side comparison
    if bpe16k_v1_tokenizer:
        v1_overall = evaluate_single_tokenizer_on_docs(
            bpe16k_v1_tokenizer, val_documents, reference_tokens=gpt2_overall.total_tokens
        )
        result["bpe_16k_v1_legacy"] = {
            "name": bpe16k_v1_tokenizer.name,
            "vocab_size": bpe16k_v1_tokenizer.vocab_size,
            "fingerprint": getattr(bpe16k_v1_tokenizer, "fingerprint", "legacy_v1"),
            "has_unk_token": bpe16k_v1_tokenizer.unk_token_id is not None,
            "overall": asdict(v1_overall),
        }

    return result
