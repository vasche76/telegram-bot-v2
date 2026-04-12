"""
Photo search service with smart routing between Pexels (stock) and DuckDuckGo (web).
GPT decides the best source based on the query.
"""

import httpx
import random
from typing import Optional

from bot.config import PEXELS_API_KEY
from bot.services.ai import structured_extraction
from bot.services.web_search import search_images
from bot.utils.logging import get_logger

log = get_logger("services.photos")


async def smart_photo_search(
    query: str,
    count: int = 3,
    verify: bool = True,
) -> tuple[list[str], str]:
    """
    Smart photo search: GPT decides source, translates query, returns URLs.
    If verify=True, uses GPT Vision to filter out irrelevant photos.
    Returns (urls, search_query_used).
    """
    import asyncio

    # Ask GPT to analyze the query and decide source
    routing = await structured_extraction(
        prompt=f"""Пользователь хочет фото: "{query}"

Определи:
1. search_query — точный поисковый запрос на АНГЛИЙСКОМ для поиска изображений.
   Примеры перевода:
   - "большая щука" → "big pike fish catch"
   - "Марго Робби" → "Margot Robbie actress"
   - "рыболовные катушки" → "fishing reels"
   - "красивые пейзажи" → "beautiful landscape nature"
   - "котики" → "cute cats"
   - "карпы" → "carp fish"

2. source — где лучше искать:
   - "pexels" — для природы, пейзажей, животных, еды, абстрактных тем (стоковые фото)
   - "web" — для знаменитостей, конкретных людей, брендов, товаров, техники, снастей, конкретных вещей

3. verification_topic — краткое описание на русском, что ДОЛЖНО быть на фото (для проверки).
   Например: "рыба карп", "актриса Марго Робби", "рыболовная катушка"

Ответь JSON: {{"search_query": "...", "source": "pexels" или "web", "verification_topic": "..."}}""",
        system="Ты помощник для поиска изображений. Отвечай только JSON.",
    )

    search_query = routing.get("search_query", query)
    source = routing.get("source", "web")
    verification_topic = routing.get("verification_topic", query)
    log.info(f"Photo search: '{query}' → '{search_query}' via {source}, verify_topic='{verification_topic}'")

    # Fetch extra photos for verification buffer (2x requested)
    fetch_count = count * 3 if verify else count
    urls = []

    # Try primary source
    if source == "pexels" and PEXELS_API_KEY:
        urls = await search_pexels(search_query, fetch_count)
    else:
        urls = await search_images(search_query, fetch_count, safesearch="off")

    # Fallback to other source
    if len(urls) < fetch_count:
        log.info(f"Primary source returned {len(urls)} results, trying fallback...")
        if source == "pexels":
            fallback_urls = await search_images(search_query, fetch_count - len(urls), safesearch="off")
        else:
            fallback_urls = await search_pexels(search_query, fetch_count - len(urls)) if PEXELS_API_KEY else []
        urls.extend(fallback_urls)

    if not verify or not urls:
        return urls[:count], search_query

    # Verify photos with GPT Vision (parallel, max 5 at a time)
    log.info(f"Verifying {len(urls)} photos for topic: '{verification_topic}'")
    verified = []

    # Check in batches to avoid overwhelming API
    for i in range(0, len(urls), 5):
        batch = urls[i:i + 5]
        tasks = [verify_photo_relevance(url, verification_topic) for url in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, result in zip(batch, results):
            if isinstance(result, Exception):
                verified.append(url)  # On error, include
            elif result:
                verified.append(url)
            else:
                log.info(f"Rejected irrelevant photo: {url[:80]}")

        if len(verified) >= count:
            break

    log.info(f"Verification complete: {len(verified)}/{len(urls)} photos passed")
    return verified[:count], search_query


async def search_pexels(query: str, count: int = 5, page: int = 0) -> list[str]:
    """Search Pexels for photos."""
    if not PEXELS_API_KEY:
        return []

    if page == 0:
        page = random.randint(1, 3)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://api.pexels.com/v1/search",
                params={"query": query, "per_page": min(count * 2, 15), "page": page},
                headers={"Authorization": PEXELS_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()
            photos = data.get("photos", [])
            urls = []
            for p in photos:
                url = p.get("src", {}).get("large2x") or p.get("src", {}).get("original")
                if url:
                    urls.append(url)
            return urls[:count]
    except Exception as e:
        log.warning(f"Pexels search failed: {e}")
        return []


async def verify_photo_relevance(url: str, expected_topic: str) -> bool:
    """
    Use GPT Vision to check if a photo matches the expected topic.
    Returns True if relevant, False otherwise.
    """
    from bot.services.ai import vision_structured
    try:
        result = await vision_structured(
            image_url=url,
            prompt=f"""Посмотри на это изображение. Ожидаемая тема: \"{expected_topic}\".

Соответствует ли изображение теме? Ответь JSON:
{{"relevant": true/false, "description": "краткое описание что на фото"}}

Будь строгим: если на фото явно не то, что запрошено — ставь false.
Например: если запрошены "карпы", а на фото крокодил — false.""",
            system="Ты проверяешь релевантность изображений. Отвечай только JSON.",
        )
        relevant = result.get("relevant", True)
        desc = result.get("description", "")
        log.info(f"Photo verification: '{expected_topic}' → relevant={relevant}, desc='{desc}'")
        return relevant
    except Exception as e:
        log.warning(f"Photo verification failed: {e}")
        return True  # On error, assume relevant (don't block)


async def download_image(url: str) -> Optional[bytes]:
    """Download an image and return bytes."""
    try:
        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "image" in content_type or url.endswith((".jpg", ".jpeg", ".png", ".webp")):
                return resp.content
            return None
    except Exception as e:
        log.warning(f"Failed to download image {url}: {e}")
        return None
