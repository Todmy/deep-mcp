"""Web search providers — Tavily, Brave, SearXNG behind SearchProvider protocol."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from rlm_research.config import SearchConfig

log = logging.getLogger(__name__)


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str | None = None


@runtime_checkable
class SearchProvider(Protocol):
    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]: ...


class TavilySearch:
    """Tavily search API — free tier 1000 req/month."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=30)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
                "include_raw_content": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
                content=r.get("raw_content"),
            )
            for r in data.get("results", [])
        ]

    async def close(self) -> None:
        await self._client.aclose()


class BraveSearch:
    """Brave Search API — free tier 2000 req/month."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=30)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self._api_key},
            params={"q": query, "count": max_results},
        )
        resp.raise_for_status()
        data = resp.json()

        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("description", ""),
            )
            for r in data.get("web", {}).get("results", [])
        ]

    async def close(self) -> None:
        await self._client.aclose()


class SearXNGSearch:
    """SearXNG self-hosted search — no API key needed."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self._base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=30)

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        resp = await self._client.get(
            f"{self._base_url}/search",
            params={"q": query, "format": "json", "pageno": 1},
        )
        resp.raise_for_status()
        data = resp.json()

        results = data.get("results", [])[:max_results]
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            )
            for r in results
        ]

    async def close(self) -> None:
        await self._client.aclose()


# Shared client for fetch_url — created lazily, reused across calls
_fetch_client: httpx.AsyncClient | None = None


async def _get_fetch_client() -> httpx.AsyncClient:
    global _fetch_client
    if _fetch_client is None:
        _fetch_client = httpx.AsyncClient(
            timeout=30,
            follow_redirects=True,
            headers={"User-Agent": "rlm-research/0.1 (research bot)"},
        )
    return _fetch_client


async def fetch_url(url: str) -> str:
    """Fetch a single URL and extract text content."""
    client = await _get_fetch_client()
    resp = await client.get(url)
    resp.raise_for_status()
    html = resp.text

    try:
        import trafilatura
        extracted = trafilatura.extract(html)
        if extracted:
            return extracted
    except ImportError:
        pass

    # Fallback: naive tag stripping
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text).strip()


async def close_search_provider(provider: SearchProvider | None) -> None:
    """Close a search provider's HTTP client if it has one."""
    if provider and hasattr(provider, "close"):
        await provider.close()


def create_search_provider(config: SearchConfig) -> SearchProvider | None:
    """Factory: create search provider from config."""
    provider = config.provider.lower()

    if provider == "tavily":
        if not config.api_key:
            log.warning("Tavily search requires api_key — search disabled")
            return None
        return TavilySearch(api_key=config.api_key)

    elif provider == "brave":
        if not config.api_key:
            log.warning("Brave search requires api_key — search disabled")
            return None
        return BraveSearch(api_key=config.api_key)

    elif provider == "searxng":
        return SearXNGSearch(base_url=config.base_url)

    else:
        log.warning("Unknown search provider: %s — search disabled", provider)
        return None
