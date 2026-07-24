"""
Async web search integration (Tavily) with retry/backoff and normalization
into the ResearchState's SearchResult schema.

Uses the raw Tavily REST endpoint via httpx rather than a provider SDK so the
request/response shape is explicit and easy to swap for another provider
(e.g. Serper) by editing this module only.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import List

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.graph.state import SearchResult

logger = logging.getLogger("deep_research.search")

TAVILY_API_URL = "https://api.tavily.com/search"


class SearchProviderError(RuntimeError):
    """Raised when the search provider fails after all retry attempts."""


def _tavily_api_key() -> str:
    return os.getenv("TAVILY_API_KEY", "")


def _search_timeout_seconds() -> float:
    return float(os.getenv("SEARCH_TIMEOUT_SECONDS", "30"))


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type((httpx.HTTPError, SearchProviderError)),
)
async def _tavily_request(client: httpx.AsyncClient, query: str, max_results: int) -> dict:
    api_key = _tavily_api_key()
    if not api_key:
        raise SearchProviderError(
            "TAVILY_API_KEY is not set. Add it to your .env file before running searches."
        )
    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
        "max_results": max_results,
    }
    response = await client.post(TAVILY_API_URL, json=payload, timeout=_search_timeout_seconds())
    if response.status_code >= 500:
        # Transient server-side failure -> let tenacity retry with backoff.
        raise SearchProviderError(f"Tavily server error: {response.status_code}")
    if response.status_code == 429:
        raise SearchProviderError("Tavily rate limit hit (429).")
    response.raise_for_status()
    return response.json()


async def tavily_search(query: str, max_results: int = 5) -> List[SearchResult]:
    """Run a single Tavily search query and normalize the results.

    Failures (missing API key, exhausted retries, malformed response) are
    logged and degrade to an empty result list rather than raising, so that a
    single bad query cannot crash a research run that fanned out several
    concurrent searches.
    """
    async with httpx.AsyncClient() as client:
        try:
            data = await _tavily_request(client, query, max_results)
        except Exception as exc:  # noqa: BLE001 - intentionally broad: degrade gracefully
            logger.warning("Tavily search failed for query %r: %s", query, exc)
            return []

    results: List[SearchResult] = []
    for item in data.get("results", []):
        url = item.get("url", "")
        if not url:
            continue
        results.append(
            SearchResult(
                query=query,
                url=url,
                title=item.get("title") or url,
                content=item.get("content", ""),
                score=float(item.get("score", 0.0) or 0.0),
                published_date=item.get("published_date"),
                source="tavily",
            )
        )
    return results


async def run_searches_concurrently(
    queries: List[str], max_results_per_query: int = 5
) -> List[SearchResult]:
    """Fan out multiple search queries concurrently and flatten the results."""
    if not queries:
        return []
    tasks = [tavily_search(q, max_results_per_query) for q in queries]
    nested_results = await asyncio.gather(*tasks)
    flattened: List[SearchResult] = []
    for result_set in nested_results:
        flattened.extend(result_set)
    return flattened
