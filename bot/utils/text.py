"""
Text sanitization utilities for user-supplied content.
"""

import re

# Control characters to strip: 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F, 0x7F
# Preserved: \t (0x09), \n (0x0A), \r (0x0D) — valid whitespace in captions
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_caption(text: str | None, max_len: int = 500) -> str:
    """Strip ASCII control characters, cap length, trim whitespace.

    Treats input as untrusted user text. Does NOT strip non-ASCII (Russian OK).
    Returns "" for None or empty input.
    """
    if not text:
        return ""
    text = _CONTROL_CHAR_RE.sub("", text)
    text = text.strip()
    return text[:max_len]
