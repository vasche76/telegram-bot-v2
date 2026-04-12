"""
Scheduled tasks: auto weather, photo delivery.
Uses APScheduler with async jobs.
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from bot.utils.logging import get_logger

log = get_logger("scheduler")

_scheduler: AsyncIOScheduler | None = None
_bot = None

# Auto-weather send time (UTC). Set via WEATHER_SEND_HOUR / WEATHER_SEND_MINUTE env vars.
# Default: 07:00 UTC = 10:00 Moscow time (MSK = UTC+3).
import os as _os
_WEATHER_HOUR = int(_os.environ.get("WEATHER_SEND_HOUR", "7"))
_WEATHER_MINUTE = int(_os.environ.get("WEATHER_SEND_MINUTE", "0"))


def init_scheduler(bot) -> AsyncIOScheduler:
    """Initialize the scheduler with all jobs."""
    global _scheduler, _bot
    _bot = bot

    _scheduler = AsyncIOScheduler(timezone="UTC")

    # Auto weather job — daily at a fixed UTC time (not 24h after startup)
    _scheduler.add_job(
        _auto_weather_job,
        trigger=CronTrigger(hour=_WEATHER_HOUR, minute=_WEATHER_MINUTE, timezone="UTC"),
        id="auto_weather",
        name="auto_weather",
        replace_existing=True,
    )

    # Photo schedule checker — every 5 minutes
    _scheduler.add_job(
        _photo_schedule_job,
        trigger=IntervalTrigger(minutes=5),
        id="photo_schedule",
        name="photo_schedule",
        replace_existing=True,
    )

    log.info("Scheduler initialized with 2 jobs")
    return _scheduler


async def _auto_weather_job() -> None:
    """Send auto weather to configured chats."""
    from bot.storage.database import fetch_all
    from bot.services.weather import get_weather, format_weather_message

    try:
        configs = await fetch_all(
            "SELECT * FROM weather_subs WHERE enabled = 1"
        )

        for cfg in configs:
            chat_id = cfg["chat_id"]
            lat = cfg.get("latitude")
            lon = cfg.get("longitude")

            if not lat or not lon:
                continue

            try:
                weather = await get_weather(lat, lon, f"Авто-прогноз (чат {chat_id})")
                text = format_weather_message(weather, include_fishing=True)
                await _bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
                log.info(f"Auto weather sent to {chat_id}")
            except Exception as e:
                log.error(f"Auto weather failed for {chat_id}: {e}")

    except Exception as e:
        log.error(f"Auto weather job error: {e}")


async def _photo_schedule_job() -> None:
    """Check and send scheduled photos."""
    from bot.handlers.photos import send_scheduled_photos

    try:
        await send_scheduled_photos(_bot)
    except Exception as e:
        log.error(f"Photo schedule job error: {e}")
