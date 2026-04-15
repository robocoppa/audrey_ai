"""
Audrey — web search detection and backends.

Regex-based detection of time-sensitive / factual queries, SearXNG and Brave
search backends, and result formatting for injection into the conversation.
"""

import logging
import re
from typing import Any

import aiohttp

import state
from config import (
    BRAVE_API_KEY,
    BRAVE_MAX_RESULTS,
    SEARCH_BACKEND,
    SEARXNG_MAX_RESULTS,
    SEARXNG_URL,
)

logger = logging.getLogger("audrey.search")


# ── Search-trigger patterns ──────────────────────────────────────────────────

_SP = [
    re.compile(
        r"\b(today|tonight|yesterday|this week|this month|right now|currently|latest|recent)\b",
        re.I,
    ),
    re.compile(r"\b(who is|who are)\b.{0,30}\b(current|president|ceo|mayor)\b", re.I),
    re.compile(
        r"\b(what is the|what's the)\b.{0,20}\b(price|score|weather|rate|status)\b",
        re.I,
    ),
    re.compile(r"\b(when is|when does|when will)\b", re.I),
    re.compile(r"\b(news|headline|update|announcement|released|launched)\b", re.I),
    re.compile(r"\b(stock|market|crypto|bitcoin)\b.{0,20}\b(price|value|worth)\b", re.I),
    re.compile(r"\b(score|won|lost|beat|defeated|playoff|standings)\b", re.I),
    re.compile(r"\b(election|voted|poll|ballot)\b", re.I),
    re.compile(r"\b20(2[5-9]|[3-9]\d)\b"),
    re.compile(r"\b(search|look up|google|find out)\b", re.I),
]

_FP = [
    re.compile(
        r"\b(how (much|many|old|tall|far|long))\b.{0,30}\b(is|are|was|were)\b", re.I
    ),
    re.compile(r"\b(population|capital|founded|born|died)\b.{0,20}\b(of|in|on)\b", re.I),
]


def needs_web_search(text: str) -> bool:
    if SEARCH_BACKEND == "brave" and not BRAVE_API_KEY:
        return False
    for p in _SP:
        if p.search(text):
            return True
    for p in _FP:
        if p.search(text):
            return True
    return False


def extract_search_query(text: str) -> str:
    q = re.sub(
        r"^(hey|hi|hello|please|can you|could you|tell me|search for|look up|find|what is|what's)\s+",
        "",
        text.strip(),
        flags=re.I,
    )
    if len(q) > 80:
        q = q[:80].rsplit(" ", 1)[0]
    return q.strip() or text[:80].strip()


# ── SearXNG backend ──────────────────────────────────────────────────────────

async def searxng_search(query: str) -> list[dict[str, str]]:
    try:
        async with state.ext_session.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json", "safesearch": "0"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
        return [
            {
                "title": i.get("title", ""),
                "url": i.get("url", ""),
                "snippet": i.get("content", ""),
            }
            for i in data.get("results", [])[:SEARXNG_MAX_RESULTS]
        ]
    except Exception as e:
        logger.warning("SearXNG: %s", e)
        return []


# ── Brave backend ────────────────────────────────────────────────────────────

async def brave_search(query: str) -> list[dict[str, str]]:
    if not BRAVE_API_KEY:
        return []
    try:
        async with state.ext_session.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            params={"q": query, "count": BRAVE_MAX_RESULTS},
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status != 200:
                return []
            data = await r.json()
        return [
            {
                "title": i.get("title", ""),
                "url": i.get("url", ""),
                "snippet": i.get("description", ""),
            }
            for i in data.get("web", {}).get("results", [])[:BRAVE_MAX_RESULTS]
        ]
    except Exception as e:
        logger.warning("Brave: %s", e)
        return []


# ── Dispatch ─────────────────────────────────────────────────────────────────

async def web_search(query: str) -> list[dict[str, str]]:
    if SEARCH_BACKEND == "searxng":
        return await searxng_search(query)
    return await brave_search(query)


def format_search_results(results: list[dict[str, str]]) -> str:
    if not results:
        return ""
    parts = ["Here are recent web search results:\n"]
    for i, r in enumerate(results, 1):
        parts.append(
            f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['url']}\n"
        )
    parts.append(
        "\nUse these results for an accurate answer. Cite sources when relevant."
    )
    return "\n".join(parts)
