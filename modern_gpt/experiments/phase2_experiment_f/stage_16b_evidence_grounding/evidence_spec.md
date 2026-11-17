# Structured Evidence Representation Specification

## Objective

Define the canonical, immutable data structure and protocol for passing non-parametric retrieved evidence through QORIN's multi-turn tool observation turns.

---

## 1. Schema Definition

Located in: `backend/tools/evidence.py`

### 1.1 `EvidenceRecord`

Each retrieved passage is encapsulated as an immutable `EvidenceRecord`:

```python
@dataclass
class EvidenceRecord:
    evidence_id: str                      # e.g. "SOURCE_1", "SOURCE_2"
    source_title: str                     # Webpage title or document heading
    source_url: str                       # Canonical destination URL
    source_snippet: str                   # Clean snippet text
    retrieval_timestamp: float            # Epoch timestamp
    search_query: str = ""                # Query that yielded this record
    provider: str = "duckduckgo_lite"     # Provider name
    is_cached: bool = False               # Cache hit flag
    passage: str = ""                     # Extended passage context
    passage_relevance: float = 1.0        # Computed relevance score [0.0 - 1.0]
    relevant_entity_references: List[str] # Extracted proper nouns, years, ordinals
    raw_payload: Optional[Dict[str, Any]] # Original provider response payload
```

#### Properties:
- `source_id`: Canonical alias for `evidence_id`.
- `title`: Canonical alias for `source_title`.
- `url`: Canonical destination URL.
- `snippet`: Clean source snippet.
- `retrieved_at`: ISO 8601 UTC timestamp string (e.g. `2026-11-17T11:59:50.625446+00:00`).

### 1.2 `EvidenceBundle`

Collections of evidence records returned from queries are packaged into an `EvidenceBundle`:

```python
@dataclass
class EvidenceBundle:
    query: str
    records: List[EvidenceRecord] = field(default_factory=list)
    latency_ms: float = 0.0
    provider_name: str = "unknown"
    error: Optional[str] = None
```

---

## 2. Model Prompt Representation

When an `EvidenceBundle` is serialized into the tool observation turn (`format_for_model()`), it emits a standardized, unambiguous representation:

```text
Retrieved 3 authoritative sources for query 'who is donald trump':

[SOURCE_1] Donald Trump - Wikipedia
URL: https://en.wikipedia.org/wiki/Donald_Trump
Key Entities: 1946, 2017, 2021, 47th president, 45th president, Donald Trump, United States, Republican Party
Evidence Passage: Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021.

[SOURCE_2] Robert Trump - Wikipedia
URL: https://en.wikipedia.org/wiki/Robert_Trump
Key Entities: 1948, 2020, 1999, 45th, 47th president, Robert Trump, Donald Trump, United States
Evidence Passage: Robert Stewart Trump (August 26, 1948 - August 15, 2020) was an American businessman and investor. He was the younger brother of Donald Trump...
```

---

## 3. Entity Reference Extraction Algorithm

Implemented in `extract_entity_references(text: str)`:
1. **Years and Date Ranges**: `\b(?:18|19|20)\d{2}(?:[–-](?:18|19|20)?\d{2})?\b`
2. **Ordinals & Presidencies**: `\b(?:\d+(?:st|nd|rd|th)|first|second|third|fourth)\s+(?:president|term|edition|round|place)?\b`
3. **Proper Nouns**: `\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b` (strictly case-sensitive)
4. **Political & Public Organizations**: `Republican`, `Democrat`, `Democratic`, `GOP`, `Senate`, `Congress`, `FIFA`.
