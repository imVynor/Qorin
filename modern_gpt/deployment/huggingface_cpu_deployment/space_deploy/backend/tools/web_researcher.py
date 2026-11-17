import json
import re
import time
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request
from bs4 import BeautifulSoup


class WebResearchResult:
    def __init__(
        self,
        query: str,
        sources: List[Dict[str, str]],
        summary: str,
        latency_ms: float,
        error: Optional[str] = None,
    ):
        self.query = query
        self.sources = sources  # List of {title, url, snippet}
        self.summary = summary
        self.latency_ms = latency_ms
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "sources": self.sources,
            "summary": self.summary,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
        }


class WebResearcher:
    """Executes live web searches and extracts clean text summaries with authentic citations."""

    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    TIMEOUT_SECONDS = 4.0

    def search_duckduckgo_lite(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        encoded = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded}"

        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        req = urllib.request.Request(url, headers=headers)
        sources: List[Dict[str, str]] = []

        try:
            with urllib.request.urlopen(req, timeout=self.TIMEOUT_SECONDS) as resp:
                html = resp.read().decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html, "html.parser")

                results = soup.select(".result")
                for r in results[:max_results]:
                    title_elem = r.select_one(".result__title .result__a")
                    snippet_elem = r.select_one(".result__snippet")
                    url_elem = r.select_one(".result__url")

                    if title_elem and title_elem.text:
                        title = title_elem.text.strip()
                        raw_href = title_elem.get("href", "")
                        # Parse actual target from DDG redirect link
                        parsed_url = raw_href
                        if "uddg=" in raw_href:
                            try:
                                parsed_url = urllib.parse.unquote(raw_href.split("uddg=")[1].split("&")[0])
                            except Exception:
                                pass
                        snippet = snippet_elem.text.strip() if snippet_elem else ""

                        sources.append({
                            "title": title,
                            "url": parsed_url,
                            "snippet": snippet,
                        })

        except Exception:
            pass

        # If DDG HTML was empty or blocked, query Wikipedia API directly
        if not sources:
            try:
                wiki_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={encoded}&limit={max_results}&namespace=0&format=json"
                wreq = urllib.request.Request(wiki_url, headers=headers)
                with urllib.request.urlopen(wreq, timeout=self.TIMEOUT_SECONDS) as wresp:
                    wdata = json.loads(wresp.read().decode("utf-8"))
                    if len(wdata) >= 4:
                        titles = wdata[1]
                        snippets = wdata[2]
                        urls = wdata[3]
                        for t, s, u in zip(titles, snippets, urls):
                            sources.append({
                                "title": f"{t} (Wikipedia)",
                                "url": u,
                                "snippet": s or f"Encyclopedic entry for {t}.",
                            })
            except Exception:
                pass

        return sources

    def research(self, query: str, max_sources: int = 3) -> WebResearchResult:
        start_time = time.perf_counter()
        sources = self.search_duckduckgo_lite(query, max_results=max_sources)
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        if not sources:
            return WebResearchResult(
                query=query,
                sources=[],
                summary="No live web sources could be retrieved for this query.",
                latency_ms=latency_ms,
                error="Search returned 0 results or request timed out.",
            )

        summary_lines = []
        for idx, s in enumerate(sources, 1):
            summary_lines.append(f"[{idx}] {s['title']} ({s['url']}):\n{s['snippet']}")

        summary = "\n\n".join(summary_lines)
        return WebResearchResult(
            query=query,
            sources=sources,
            summary=summary,
            latency_ms=latency_ms,
        )
