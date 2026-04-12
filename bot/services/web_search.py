"""
Web search service with multiple sources, deduplication, and ranking.
Uses DuckDuckGo as the primary search engine (no API key required).
"""

import httpx
import re
from typing import Optional
from bot.utils.logging import get_logger

log = get_logger("services.web_search")


async def search_text(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for text results.
    Returns list of {title, url, snippet}.
    """
    results = []

    # Try DuckDuckGo HTML search
    try:
        results = await _ddg_text_search(query, max_results)
    except Exception as e:
        log.warning(f"DuckDuckGo text search failed: {e}")

    # Deduplicate by URL
    seen_urls = set()
    unique = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique.append(r)

    return unique[:max_results]


async def search_images(query: str, max_results: int = 5, safesearch: str = "off") -> list[str]:
    """
    Search for images. Returns list of image URLs.
    """
    urls = []

    # Try duckduckgo-search library first
    try:
        urls = await _ddg_images_library(query, max_results, safesearch)
    except Exception as e:
        log.warning(f"DDG images library failed: {e}")

    # Fallback to manual DDG image search
    if not urls:
        try:
            urls = await _ddg_images_manual(query, max_results, safesearch)
        except Exception as e:
            log.warning(f"DDG images manual failed: {e}")

    return urls[:max_results]


async def fetch_page_text(url: str, max_chars: int = 5000) -> str:
    """Fetch a web page and extract text content."""
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            html = resp.text

            # Simple HTML to text
            text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text[:max_chars]
    except Exception as e:
        log.warning(f"Failed to fetch {url}: {e}")
        return ""


# ── DuckDuckGo implementations ──────────────────────────────

async def _ddg_text_search(query: str, max_results: int) -> list[dict]:
    """DuckDuckGo text search via HTML scraping."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        resp.raise_for_status()
        html = resp.text

    results = []
    # Parse result blocks
    blocks = re.findall(
        r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        html, re.DOTALL
    )
    for url, title, snippet in blocks[:max_results]:
        title = re.sub(r'<[^>]+>', '', title).strip()
        snippet = re.sub(r'<[^>]+>', '', snippet).strip()
        # Decode DDG redirect URL
        if "uddg=" in url:
            import urllib.parse
            match = re.search(r'uddg=([^&]+)', url)
            if match:
                url = urllib.parse.unquote(match.group(1))
        results.append({"title": title, "url": url, "snippet": snippet})

    return results


async def _ddg_images_library(query: str, max_results: int, safesearch: str) -> list[str]:
    """Use duckduckgo-search library for image search."""
    try:
        from duckduckgo_search import DDGS
        import asyncio

        def _search():
            with DDGS() as ddgs:
                results = list(ddgs.images(query, max_results=max_results, safesearch=safesearch))
                return [r["image"] for r in results if r.get("image")]

        return await asyncio.get_running_loop().run_in_executor(None, _search)
    except ImportError:
        log.info("duckduckgo-search not installed, using manual method")
        raise


async def _ddg_images_manual(query: str, max_results: int, safesearch: str) -> list[str]:
    """Manual DuckDuckGo image search via API endpoint."""
    p_val = "-1" if safesearch == "off" else ("1" if safesearch == "strict" else "-1")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # Get vqd token
        resp = await client.get(
            "https://duckduckgo.com/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
        )
        vqd_match = re.search(r'vqd=["\']([^"\']+)', resp.text)
        if not vqd_match:
            return []
        vqd = vqd_match.group(1)

        # Search images
        resp = await client.get(
            "https://duckduckgo.com/i.js",
            params={
                "l": "ru-ru",
                "o": "json",
                "q": query,
                "vqd": vqd,
                "f": ",,,",
                "p": p_val,
            },
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": "https://duckduckgo.com/",
            },
        )
        data = resp.json()
        return [r["image"] for r in data.get("results", [])[:max_results] if r.get("image")]
