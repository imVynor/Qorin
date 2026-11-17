"""
QORIN Evidence Representation System
Structured, immutable evidence records and bundles for grounded retrieval and multi-turn synthesis.
Includes entity reference extraction, passage relevance scoring, and citation tracking.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import hashlib
import json
import re
import time
from typing import Any, Dict, List, Optional, Set


def extract_entity_references(text: str) -> List[str]:
    """Extracts key factual entities, proper nouns, years, and metrics from text."""
    if not text:
        return []
    entities: List[str] = []

    # 1. Years and date ranges (e.g., 2017-2021, 2024, 1946)
    years = re.findall(r"\b(?:18|19|20)\d{2}(?:[–-](?:18|19|20)?\d{2})?\b", text)
    entities.extend(years)

    # 2. Ordinal designations (e.g., 45th, 47th, first, second)
    ordinals = re.findall(r"\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth)\s+(?:president|term|edition|round|place)?\b", text, re.IGNORECASE)
    entities.extend([o.strip() for o in ordinals])

    # 3. Capitalized multi-word proper entities (e.g., "Donald Trump", "Republican Party", "United States")
    proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    entities.extend(proper_nouns)

    # 4. Standalone political/organizational terms
    orgs = re.findall(r"\b(?:Republican|Democrat|Democratic|GOP|Parliament|Congress|Senate|Supreme Court|White House|FIFA)\b", text, re.IGNORECASE)
    entities.extend(orgs)

    # Clean and deduplicate while preserving order
    seen: Set[str] = set()
    cleaned_entities: List[str] = []
    for e in entities:
        norm = e.strip()
        if len(norm) > 1 and norm.lower() not in seen:
            seen.add(norm.lower())
            cleaned_entities.append(norm)

    return cleaned_entities[:10]


@dataclass
class EvidenceRecord:
    evidence_id: str  # e.g. "SOURCE_1"
    source_title: str
    source_url: str
    source_snippet: str
    retrieval_timestamp: float = field(default_factory=time.time)
    search_query: str = ""
    provider: str = "unknown"
    is_cached: bool = False
    passage: str = ""
    passage_relevance: float = 1.0
    relevant_entity_references: List[str] = field(default_factory=list)
    raw_payload: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.passage:
            self.passage = self.source_snippet
        if not self.relevant_entity_references:
            self.relevant_entity_references = extract_entity_references(f"{self.source_title} {self.source_snippet}")

    @property
    def source_id(self) -> str:
        return self.evidence_id

    @property
    def title(self) -> str:
        return self.source_title

    @property
    def url(self) -> str:
        return self.source_url

    @property
    def snippet(self) -> str:
        return self.source_snippet

    @property
    def retrieved_at(self) -> str:
        dt = datetime.fromtimestamp(self.retrieval_timestamp, tz=timezone.utc)
        return dt.isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_id": self.evidence_id,
            "evidence_id": self.evidence_id,
            "title": self.source_title,
            "source_title": self.source_title,
            "url": self.source_url,
            "source_url": self.source_url,
            "snippet": self.source_snippet,
            "source_snippet": self.source_snippet,
            "passage": self.passage,
            "retrieved_at": self.retrieved_at,
            "retrieval_timestamp": self.retrieval_timestamp,
            "passage_relevance": round(self.passage_relevance, 2),
            "relevant_entity_references": self.relevant_entity_references,
            "search_query": self.search_query,
            "provider": self.provider,
            "is_cached": self.is_cached,
            "raw_payload": self.raw_payload or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceRecord":
        sid = data.get("source_id") or data.get("evidence_id", "SOURCE_1")
        title = data.get("title") or data.get("source_title", "Untitled Source")
        url = data.get("url") or data.get("source_url", "")
        snip = data.get("snippet") or data.get("source_snippet", "")
        passage = data.get("passage") or snip
        relevance = float(data.get("passage_relevance", 1.0))
        entities = data.get("relevant_entity_references") or extract_entity_references(f"{title} {snip}")
        ts = data.get("retrieval_timestamp", time.time())

        return cls(
            evidence_id=sid,
            source_title=title,
            source_url=url,
            source_snippet=snip,
            passage=passage,
            retrieval_timestamp=ts,
            search_query=data.get("search_query", ""),
            provider=data.get("provider", "unknown"),
            is_cached=data.get("is_cached", False),
            passage_relevance=relevance,
            relevant_entity_references=entities,
            raw_payload=data.get("raw_payload", {})
        )

    def format_citation_header(self) -> str:
        return f"[{self.evidence_id}]: {self.source_title} ({self.source_url})"


@dataclass
class EvidenceBundle:
    query: str
    records: List[EvidenceRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    provider_name: str = "unknown"
    error: Optional[str] = None

    @property
    def has_evidence(self) -> bool:
        return len(self.records) > 0 and self.error is None

    def get_source_ids(self) -> List[str]:
        return [r.evidence_id for r in self.records]

    def get_record_by_id(self, source_id: str) -> Optional[EvidenceRecord]:
        sid_norm = source_id.strip("[]").upper()
        for r in self.records:
            if r.evidence_id.upper() == sid_norm:
                return r
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "latency_ms": round(self.latency_ms, 2),
            "provider_name": self.provider_name,
            "error": self.error,
            "records": [r.to_dict() for r in self.records]
        }

    def format_for_model(self) -> str:
        """Renders evidence records with explicit entity tags and source IDs for grounded synthesis."""
        if self.error:
            return f"Error executing web research: {self.error}"
        if not self.records:
            return f"No relevant evidence found for query: '{self.query}'."

        lines = [f"Retrieved {len(self.records)} authoritative sources for query '{self.query}':\n"]
        for r in self.records:
            lines.append(f"[{r.evidence_id}] {r.source_title}")
            lines.append(f"URL: {r.source_url}")
            if r.relevant_entity_references:
                lines.append(f"Key Entities: {', '.join(r.relevant_entity_references)}")
            lines.append(f"Evidence Passage: {r.source_snippet.strip()}")
            lines.append("")  # Blank separator
        return "\n".join(lines).strip()
