from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Tuple
from backend.retrieval.indexer import DocumentChunk, KnowledgeIndexer


@dataclass
class RetrievedChunkInfo:
    chunk_id: str
    doc_title: str
    doc_source: str
    section: str
    similarity_score: float
    snippet: str
    token_count: int


@dataclass
class RetrievalTrace:
    triggered: bool
    mode: str  # "off" | "always" | "auto"
    decision_reason: str
    query: str
    retrieved_chunks: List[RetrievedChunkInfo] = field(default_factory=list)
    injected_context: str = ""
    retrieval_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "triggered": self.triggered,
            "mode": self.mode,
            "decision_reason": self.decision_reason,
            "query": self.query,
            "retrieved_chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_title": c.doc_title,
                    "doc_source": c.doc_source,
                    "section": c.section,
                    "similarity_score": round(c.similarity_score, 4),
                    "snippet": c.snippet,
                    "token_count": c.token_count,
                }
                for c in self.retrieved_chunks
            ],
            "injected_context": self.injected_context,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 2),
        }


class RetrievalEngine:
    FACTUAL_KEYWORDS = {
        "parameters", "params", "layer", "layers", "head", "heads", "gqa", "mha", "mqa",
        "rope", "rotary", "context", "swiglu", "rmsnorm", "token", "tokens", "pretraining",
        "fineweb", "cosmopedia", "starcoder", "wikipedia", "ultrachat", "dataset", "datasets",
        "sft", "alpaca", "gpu", "rtx", "5060", "vram", "hardware", "duty", "cooldown",
        "temperature", "perplexity", "ppl", "loss", "passkey", "needle", "ast", "mbpp",
        "json", "schema", "500m", "100m", "50m", "16m", "architecture", "spec", "specs",
        "anomalies", "sycophancy", "collapse", "ablation", "checkpoint", "qorin"
    }

    CONVERSATIONAL_PATTERNS = {
        "hi", "hello", "hey", "good morning", "good evening", "how are you",
        "tell me a joke", "write a poem", "sing a song", "bye", "goodbye"
    }

    def __init__(self, indexer: Optional[KnowledgeIndexer] = None):
        self.indexer = indexer or KnowledgeIndexer()

    def should_retrieve_auto(self, query: str) -> Tuple[bool, str]:
        cleaned = query.strip().lower()

        # Pure greeting or small talk check
        if cleaned in self.CONVERSATIONAL_PATTERNS or len(cleaned.split()) <= 2:
            if not any(kw in cleaned for kw in ["500m", "100m", "50m", "16m", "qorin"]):
                return False, "Query classified as conversational/greeting; no external factual retrieval required."

        # Keyword intersection
        words = set(cleaned.split())
        matched_keywords = [kw for kw in self.FACTUAL_KEYWORDS if kw in words or kw in cleaned]

        if matched_keywords:
            return True, f"Automated routing triggered on factual/architectural keywords: {', '.join(matched_keywords[:4])}."

        # Check if query references project documentation or architecture
        tech_context_indicators = {"model", "gpt", "project", "system", "training", "run", "experiment", "weights", "checkpoint"}
        if any(w in cleaned for w in tech_context_indicators):
            return True, "Automated routing triggered on technical architecture inquiry."

        return False, "Query did not match technical knowledge base routing threshold."

    def retrieve(
        self,
        query: str,
        mode: str = "auto",
        top_k: int = 3,
        threshold: float = 0.08,
    ) -> RetrievalTrace:
        start_t = time.perf_counter()

        if mode == "off":
            return RetrievalTrace(
                triggered=False,
                mode="off",
                decision_reason="Retrieval explicitly disabled by user setting.",
                query=query,
                retrieval_latency_ms=0.0,
            )

        if mode == "always":
            triggered = True
            reason = "Retrieval forced by 'always' policy setting."
        elif mode == "auto":
            triggered, reason = self.should_retrieve_auto(query)
        else:
            triggered = False
            reason = f"Unknown retrieval mode '{mode}'; defaulting to off."

        if not triggered:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            return RetrievalTrace(
                triggered=False,
                mode=mode,
                decision_reason=reason,
                query=query,
                retrieval_latency_ms=elapsed,
            )

        # Execute retrieval against index
        raw_matches = self.indexer.query(query, top_k=top_k, threshold=threshold)
        elapsed = (time.perf_counter() - start_t) * 1000.0

        if not raw_matches:
            return RetrievalTrace(
                triggered=False,
                mode=mode,
                decision_reason=f"{reason} (0 internal knowledge chunks exceeded similarity threshold {threshold}; external tools enabled).",
                query=query,
                retrieved_chunks=[],
                injected_context="",
                retrieval_latency_ms=elapsed,
            )

        retrieved_info: List[RetrievedChunkInfo] = []
        context_parts: List[str] = []

        for idx, (chunk, score) in enumerate(raw_matches, 1):
            info = RetrievedChunkInfo(
                chunk_id=chunk.chunk_id,
                doc_title=chunk.doc_title,
                doc_source=chunk.doc_source,
                section=chunk.section,
                similarity_score=score,
                snippet=chunk.content[:220] + ("..." if len(chunk.content) > 220 else ""),
                token_count=chunk.token_count,
            )
            retrieved_info.append(info)
            context_parts.append(
                f"[Source {idx}: {chunk.doc_title} - {chunk.section}]\n{chunk.content}"
            )

        injected = "\n\n".join(context_parts)

        return RetrievalTrace(
            triggered=True,
            mode=mode,
            decision_reason=reason,
            query=query,
            retrieved_chunks=retrieved_info,
            injected_context=injected,
            retrieval_latency_ms=elapsed,
        )

    def inject_context_into_messages(
        self,
        messages: List[Dict[str, str]],
        trace: RetrievalTrace,
    ) -> List[Dict[str, str]]:
        if not trace.triggered or not trace.injected_context:
            return messages

        augmented_messages: List[Dict[str, str]] = []
        system_prefix = (
            "You are QORIN, an independent small language model. "
            "Use the following authoritative verified project documentation to ground your answer:\n\n"
            f"{trace.injected_context}\n\n"
            "If the question cannot be answered from the provided context, answer honestly based on your knowledge."
        )

        has_system = False
        for msg in messages:
            if msg.get("role") == "system":
                augmented_messages.append({
                    "role": "system",
                    "content": f"{system_prefix}\n\nAdditional Instructions:\n{msg.get('content', '')}",
                })
                has_system = True
            else:
                augmented_messages.append(msg)

        if not has_system:
            augmented_messages.insert(0, {
                "role": "system",
                "content": system_prefix,
            })

        return augmented_messages

