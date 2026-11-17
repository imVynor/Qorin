"""
QORIN Retrieval Provider Abstraction & Persistent Cache Infrastructure
Supports pluggable providers (DuckDuckGo, Mock Fixtures, etc.), SQLite persistent caching,
bounded retries with exponential backoff, and strict structured evidence emission.
"""

from abc import ABC, abstractmethod
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup

from backend.tools.evidence import EvidenceRecord, EvidenceBundle


class BaseRetrievalProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass

    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> EvidenceBundle:
        pass


class SQLiteCacheManager:
    """Persistent SQLite store for deterministic retrieval caching."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = Path(__file__).resolve().parent / "retrieval_cache.sqlite"
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=10.0)

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS retrieval_cache (
                    cache_key TEXT PRIMARY KEY,
                    normalized_query TEXT NOT NULL,
                    provider_name TEXT NOT NULL,
                    bundle_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    @staticmethod
    def normalize_query(query: str) -> str:
        q = query.strip().lower()
        q = re.sub(r"\s+", " ", q)
        return q

    def compute_key(self, query: str, provider_name: str) -> str:
        norm_q = self.normalize_query(query)
        payload = f"{provider_name}:{norm_q}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, query: str, provider_name: str) -> Optional[EvidenceBundle]:
        cache_key = self.compute_key(query, provider_name)
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT bundle_json, hit_count FROM retrieval_cache WHERE cache_key = ?",
                (cache_key,)
            )
            row = cur.fetchone()
            if row:
                bundle_json, hits = row
                conn.execute(
                    "UPDATE retrieval_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (cache_key,)
                )
                conn.commit()
                data = json.loads(bundle_json)
                records = [EvidenceRecord.from_dict(r) for r in data.get("records", [])]
                for r in records:
                    r.is_cached = True
                return EvidenceBundle(
                    query=query,
                    records=records,
                    latency_ms=0.5,
                    provider_name=provider_name,
                    error=data.get("error")
                )
        return None

    def put(self, query: str, provider_name: str, bundle: EvidenceBundle):
        cache_key = self.compute_key(query, provider_name)
        norm_q = self.normalize_query(query)
        bundle_data = bundle.to_dict()
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO retrieval_cache 
                (cache_key, normalized_query, provider_name, bundle_json, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (cache_key, norm_q, provider_name, json.dumps(bundle_data), time.time()))
            conn.commit()

    def clear(self):
        with self._get_connection() as conn:
            conn.execute("DELETE FROM retrieval_cache")
            conn.commit()


class MockFixtureProvider(BaseRetrievalProvider):
    """Deterministic fixture-based provider for reproducible testing and benchmark replay."""

    def __init__(self, fixtures: Optional[Dict[str, List[Dict[str, str]]]] = None):
        self.fixtures = fixtures or {}

    @property
    def provider_name(self) -> str:
        return "mock_fixture"

    def register_fixture(self, query: str, sources: List[Dict[str, str]]):
        norm_q = SQLiteCacheManager.normalize_query(query)
        self.fixtures[norm_q] = sources

    def search(self, query: str, max_results: int = 3) -> EvidenceBundle:
        norm_q = SQLiteCacheManager.normalize_query(query)
        sources = self.fixtures.get(norm_q)
        
        # Check partial match if exact match not found
        if sources is None:
            for k, v in self.fixtures.items():
                if k in norm_q or norm_q in k:
                    sources = v
                    break

        if sources is None:
            return EvidenceBundle(
                query=query,
                records=[],
                latency_ms=1.0,
                provider_name=self.provider_name,
                error=f"No fixture found for query: '{query}'"
            )

        records = []
        for idx, src in enumerate(sources[:max_results], 1):
            records.append(EvidenceRecord(
                evidence_id=f"SOURCE_{idx}",
                source_title=src.get("title", f"Source {idx}"),
                source_url=src.get("url", f"https://example.com/source_{idx}"),
                source_snippet=src.get("snippet", ""),
                retrieval_timestamp=time.time(),
                search_query=query,
                provider=self.provider_name,
                is_cached=False
            ))

        return EvidenceBundle(
            query=query,
            records=records,
            latency_ms=2.0,
            provider_name=self.provider_name,
            error=None
        )


class DuckDuckGoProvider(BaseRetrievalProvider):
    """Live DuckDuckGo HTML scraping provider with bounded retries and backoff."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

    def __init__(self, timeout_seconds: float = 8.0, max_retries: int = 2):
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @property
    def provider_name(self) -> str:
        return "duckduckgo_lite"

    def search(self, query: str, max_results: int = 3) -> EvidenceBundle:
        t0 = time.perf_counter()
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html, "html.parser")

                    results = soup.select(".result")
                    records: List[EvidenceRecord] = []
                    for idx, r in enumerate(results[:max_results], 1):
                        title_elem = r.select_one(".result__title .result__a")
                        snippet_elem = r.select_one(".result__snippet")

                        title = title_elem.get_text(separator=" ", strip=True) if title_elem else "Untitled"
                        snippet = snippet_elem.get_text(separator=" ", strip=True) if snippet_elem else ""
                        title = re.sub(r"\s+", " ", title).strip()
                        snippet = re.sub(r"\s+", " ", snippet).strip()

                        # Extract clean destination URL
                        raw_href = title_elem.get("href", "") if title_elem else ""
                        actual_url = raw_href
                        if "uddg=" in raw_href:
                            parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                            actual_url = parsed_qs.get("uddg", [raw_href])[0]

                        records.append(EvidenceRecord(
                            evidence_id=f"SOURCE_{idx}",
                            source_title=title,
                            source_url=actual_url,
                            source_snippet=snippet,
                            retrieval_timestamp=time.time(),
                            search_query=query,
                            provider=self.provider_name,
                            is_cached=False
                        ))

                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    if not records:
                        return EvidenceBundle(
                            query=query,
                            records=[],
                            latency_ms=latency_ms,
                            provider_name=self.provider_name,
                            error="Search returned zero results."
                        )

                    return EvidenceBundle(
                        query=query,
                        records=records,
                        latency_ms=latency_ms,
                        provider_name=self.provider_name,
                        error=None
                    )
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    # Exponential backoff
                    time.sleep(1.0 * (2 ** attempt))

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return EvidenceBundle(
            query=query,
            records=[],
            latency_ms=latency_ms,
            provider_name=self.provider_name,
            error=f"Retrieval failed after {self.max_retries + 1} attempts: {last_error}"
        )


class UnifiedWebResearcher:
    """High-level retrieval interface decoupling caller from provider and caching."""

    def __init__(
        self,
        provider: Optional[BaseRetrievalProvider] = None,
        cache_manager: Optional[SQLiteCacheManager] = None,
        enable_cache: bool = True
    ):
        self.provider = provider or DuckDuckGoProvider()
        self.cache_manager = cache_manager or SQLiteCacheManager()
        self.enable_cache = enable_cache

    def set_provider(self, provider: BaseRetrievalProvider):
        self.provider = provider

    def search(self, query: str, max_results: int = 3, force_live: bool = False) -> EvidenceBundle:
        # Check cache
        if self.enable_cache and not force_live:
            cached_bundle = self.cache_manager.get(query, self.provider.provider_name)
            if cached_bundle is not None:
                return cached_bundle

        # Execute live query
        bundle = self.provider.search(query, max_results=max_results)

        # Cache valid evidence
        if self.enable_cache and bundle.has_evidence:
            self.cache_manager.put(query, self.provider.provider_name, bundle)

        return bundle
