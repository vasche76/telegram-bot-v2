"""
Centralized configuration loaded from environment variables.
All secrets and tunables live here — never hardcode them in business logic.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── Required ─────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.environ.get("TELEGRAM_BOT_TOKEN", "") or os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_TOKEN = TELEGRAM_BOT_TOKEN  # alias for convenience
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# ── Optional integrations ────────────────────────────────────
PEXELS_API_KEY: str = os.environ.get("PEXELS_API_KEY", "")

# ── Database ─────────────────────────────────────────────────
DATABASE_PATH: str = os.environ.get("DATABASE_PATH", str(_PROJECT_ROOT / "data" / "bot.db"))

# ── Logging ──────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")

# ── Scheduler intervals (seconds) ───────────────────────────
WEATHER_INTERVAL: int = int(os.environ.get("WEATHER_INTERVAL", "43200"))
PHOTO_CHECK_INTERVAL: int = int(os.environ.get("PHOTO_CHECK_INTERVAL", "60"))

# ── OpenAI models ────────────────────────────────────────────
OPENAI_CHAT_MODEL: str = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_VISION_MODEL: str = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o-mini")
OPENAI_WHISPER_MODEL: str = os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1")

# # ── RAG settings ─────────────────────────────────────────
RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "15"))

# ── Fish Vision limits ────────────────────────────────
MAX_FISH_COUNT: int = int(os.environ.get("MAX_FISH_COUNT", "20"))

# ── Admin ─────────────────────────────────────────────
# Comma-separated list of Telegram user IDs that can use /status
ADMIN_USER_IDS: list[int] = [
    int(x.strip()) for x in os.environ.get("ADMIN_USER_IDS", "").split(",")
    if x.strip().isdigit()
]

# ── Validation ───────────────────────────────────────────────
def validate() -> list[str]:
    """Return list of missing required config keys."""
    missing = []
    if not TELEGRAM_BOT_TOKEN:
        missing.append("TELEGRAM_BOT_TOKEN or TELEGRAM_TOKEN")
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    return missing
