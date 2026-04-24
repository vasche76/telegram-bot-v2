"""
OpenAI API service layer.
Uses structured outputs (JSON mode) for all extraction tasks.
"""

import asyncio
import json
import httpx
from typing import Optional, Any

from bot.config import OPENAI_API_KEY, OPENAI_CHAT_MODEL, OPENAI_VISION_MODEL
from bot.utils.logging import get_logger

log = get_logger("services.ai")

_client: Optional[httpx.AsyncClient] = None

_MAX_RETRIES = 3
_RETRYABLE_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
    return _client


async def chat_completion(
    messages: list[dict],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000,
    json_mode: bool = False,
) -> str:
    """Send a chat completion request and return the response text."""
    client = _get_client()
    body: dict[str, Any] = {
        "model": model or OPENAI_CHAT_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = await client.post("/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices")
            if not choices:
                log.error(f"OpenAI returned no choices: {data}")
                raise ValueError("OpenAI returned empty choices list")
            content = choices[0].get("message", {}).get("content")
            if content is None:
                log.error(f"OpenAI choice has no content: {choices[0]}")
                raise ValueError("OpenAI message content is None")
            return content
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt < _MAX_RETRIES:
                delay = 2 ** (attempt - 1)
                log.warning(
                    f"OpenAI 429 rate-limited (attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                continue
            log.error(f"OpenAI API error: {e.response.status_code} - {e.response.text[:200]}")
            raise
        except _RETRYABLE_NETWORK_ERRORS as e:
            if attempt < _MAX_RETRIES:
                delay = 2 ** (attempt - 1)
                log.warning(
                    f"OpenAI request failed ({type(e).__name__}, attempt {attempt}/{_MAX_RETRIES}), "
                    f"retrying in {delay}s"
                )
                await asyncio.sleep(delay)
                continue
            log.error(f"OpenAI request failed: {e}")
            raise
        except Exception as e:
            log.error(f"OpenAI request failed: {e}")
            raise


async def structured_extraction(
    prompt: str,
    system: str = "You are a helpful assistant. Always respond with valid JSON.",
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> dict:
    """Extract structured data using JSON mode."""
    text = await chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=temperature,
        json_mode=True,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.error(f"Failed to parse JSON from AI response: {text[:200]}")
        return {}


async def vision_analyze(
    image_url: str,
    prompt: str,
    model: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: int = 1500,
) -> str:
    """Analyze an image using the vision model."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        }
    ]
    return await chat_completion(
        messages=messages,
        model=model or OPENAI_VISION_MODEL,
        max_tokens=max_tokens,
        json_mode=json_mode,
    )


async def vision_structured(
    image_url: str,
    prompt: str,
    system: str = "You are a helpful assistant. Always respond with valid JSON.",
    model: Optional[str] = None,
) -> dict:
    """Analyze an image and return structured JSON."""
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            ],
        },
    ]
    text = await chat_completion(
        messages=messages,
        model=model or OPENAI_VISION_MODEL,
        json_mode=True,
        temperature=0.3,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log.error(f"Failed to parse vision JSON: {text[:200]}")
        return {}


async def transcribe_audio(file_input) -> str:
    """Transcribe audio using Whisper API.
    
    Args:
        file_input: Either a file path (str) or raw bytes/bytearray.
    """
    from bot.config import OPENAI_WHISPER_MODEL
    import io
    client = _get_client()

    if isinstance(file_input, (bytes, bytearray)):
        file_obj = io.BytesIO(file_input)
        filename = "voice.ogg"
        resp = await client.post(
            "/audio/transcriptions",
            data={"model": OPENAI_WHISPER_MODEL, "language": "ru"},
            files={"file": (filename, file_obj)},
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=120.0,
        )
        resp.raise_for_status()
        return resp.json().get("text", "")
    else:
        filename = str(file_input).split("/")[-1]
        with open(file_input, "rb") as file_obj:
            resp = await client.post(
                "/audio/transcriptions",
                data={"model": OPENAI_WHISPER_MODEL, "language": "ru"},
                files={"file": (filename, file_obj)},
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json().get("text", "")


async def close():
    """Close the HTTP client."""
    global _client
    if _client:
        await _client.aclose()
        _client = None
