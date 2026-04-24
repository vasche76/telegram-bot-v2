"""
Telegram Bot v2 — Main entry point with self-healing.

CHANGELOG vs original:
- get_updates_read_timeout=20s (was 30) — prevents event-loop freeze on Mac sleep/VPN toggle.
- Separate network vs logic error counters — VPN/sleep events do NOT trigger restart until
  MAX_NETWORK_ERRORS=50 (was 20 combined), only sustained connectivity failure restarts bot.
- Loop-aliveness probe: watchdog sends call_soon_threadsafe to detect frozen event loop
  independently of the asyncio heartbeat coroutine.
- Heartbeat interval lowered to 30s (was 60s) for faster freeze detection.
- Frozen loop threshold 300s (was 900s) — faster recovery.
- Event loop properly closed after each run to prevent resource leaks.
- LOG_LEVEL from config (was hardcoded "INFO").

Usage: python3 main.py
"""

import sys
print(f"[1/6] Python {sys.version.split()[0]}", flush=True)

import asyncio
import os
import signal
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Global stop event — signals all watchdog threads to exit cleanly on restart
_watchdog_stop = threading.Event()
print("[2/6] stdlib OK", flush=True)

from telegram import Update
from telegram.constants import ChatType
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    filters,
)
from telegram.error import (
    NetworkError,
    TimedOut,
    RetryAfter,
    Conflict,
    InvalidToken,
    TelegramError,
)
print("[3/6] telegram OK", flush=True)

from bot.config import TELEGRAM_TOKEN, LOG_LEVEL
print("[4/6] config OK", flush=True)

from bot.storage.database import init_db
from bot.storage.messages import save_message
from bot.handlers.messages import (
    handle_text_message,
    set_bot_username,
    set_bot_id,
    _is_mention,
    _bot_username,
)
from bot.handlers.vision import handle_photo_message
from bot.handlers.weather import handle_location, handle_weather_query
from bot.handlers.help import send_help
from bot.handlers.status import (
    handle_status,
    handle_status_callback,
    set_health_refs,
    set_clear_errors_fn,
)
from bot.services.ai import transcribe_audio
from bot.services.intent import detect_intent
from bot.services.response import generate_response
from bot.scheduler import init_scheduler
from bot.utils.logging import get_logger, setup_logging
print("[5/6] bot modules OK", flush=True)

setup_logging(LOG_LEVEL or "INFO")
log = get_logger("main")
print("[6/6] Ready!", flush=True)


# ─── Health state ─────────────────────────────────────────────────
@dataclass
class _HealthState:
    last_update: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    last_loop_probe_response: float = field(default_factory=time.time)
    # Separate counters: network errors vs logic/programming errors
    consecutive_network_errors: int = 0
    consecutive_logic_errors: int = 0
    restart_count: int = 0
    conflict_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def mark_activity(self) -> None:
        with self._lock:
            self.last_update = time.time()
            self.last_heartbeat = time.time()
            self.consecutive_network_errors = 0
            self.consecutive_logic_errors = 0

    def mark_heartbeat(self) -> None:
        with self._lock:
            self.last_heartbeat = time.time()

    def mark_loop_alive(self) -> None:
        """Called via call_soon_threadsafe from watchdog — proves loop is responsive."""
        with self._lock:
            self.last_loop_probe_response = time.time()

    def mark_network_error(self) -> None:
        with self._lock:
            self.consecutive_network_errors += 1

    def mark_logic_error(self) -> None:
        with self._lock:
            self.consecutive_logic_errors += 1

    def clear_errors(self) -> None:
        with self._lock:
            self.consecutive_network_errors = 0
            self.consecutive_logic_errors = 0
        log.info("Error counters cleared via /status")

    @property
    def consecutive_errors(self) -> int:
        """Legacy compat for /status display."""
        return self.consecutive_logic_errors


health = _HealthState()

# --- Thresholds ---
MAX_LOGIC_ERRORS = 15       # restart on cascading programming errors
MAX_NETWORK_ERRORS = 50     # restart only on very long sustained connectivity failure
HEALTH_CHECK_INTERVAL = 60  # watchdog checks every 60s
HEARTBEAT_INTERVAL = 30     # asyncio heartbeat every 30s
LOOP_PROBE_TIMEOUT = 300    # consider event loop frozen after 5 minutes of no response


# ─── Global error handler ─────────────────────────────────────────
async def error_handler(update: object, context) -> None:
    err = context.error

    # Network / connectivity errors — NOT counted toward logic error limit
    if isinstance(err, (NetworkError, TimedOut)):
        health.mark_network_error()
        log.warning(
            f"Network error (net={health.consecutive_network_errors}, "
            f"logic={health.consecutive_logic_errors}): "
            f"{err.__class__.__name__}: {err}"
        )
        if health.consecutive_network_errors >= MAX_NETWORK_ERRORS:
            log.error(f"{MAX_NETWORK_ERRORS} consecutive network errors! Restarting.")
            os._exit(2)
        return

    if isinstance(err, RetryAfter):
        wait = err.retry_after
        log.warning(f"Rate limited. Waiting {wait}s...")
        await asyncio.sleep(wait)
        return

    if isinstance(err, Conflict):
        health.conflict_count += 1
        if health.conflict_count <= 5:
            wait = health.conflict_count * 15
            log.warning(
                f"CONFLICT #{health.conflict_count}: another instance. Waiting {wait}s..."
            )
            print(
                f"⚠️ CONFLICT: другой экземпляр. Жду {wait}с... "
                f"(попытка {health.conflict_count}/5)",
                flush=True,
            )
            await asyncio.sleep(wait)
            health.conflict_count = max(0, health.conflict_count - 1)
            return
        else:
            log.error("CONFLICT: 5 retries exhausted. Restarting in 60s...")
            health.conflict_count = 0
            await asyncio.sleep(60)
            os._exit(2)

    if isinstance(err, InvalidToken):
        log.error("INVALID TOKEN! Check TELEGRAM_BOT_TOKEN in .env file. Stopping (no auto-restart).")
        print("❌ INVALID TOKEN — bot stopped. Fix .env and restart manually.", flush=True)
        # Exit code 0 = successful exit → launchd KeepAlive(SuccessfulExit:false) will NOT restart.
        # This prevents an infinite restart loop hammering the Telegram API with a bad token.
        os._exit(0)

    if isinstance(err, TelegramError):
        health.mark_logic_error()
        log.error(
            f"Telegram error (logic={health.consecutive_logic_errors}): {err}",
            exc_info=True,
        )
    else:
        health.mark_logic_error()
        log.error(
            f"Unexpected error (logic={health.consecutive_logic_errors}): {err}",
            exc_info=True,
        )

    if health.consecutive_logic_errors >= MAX_LOGIC_ERRORS:
        log.error(f"{MAX_LOGIC_ERRORS} consecutive logic errors! Restarting.")
        os._exit(2)


# ─── Post-init ─────────────────────────────────────────────────────
async def post_init(app) -> None:
    await init_db()
    log.info("Database initialized (WAL checkpointed)")

    me = await app.bot.get_me()
    set_bot_username(me.username)
    set_bot_id(me.id)
    log.info(f"Bot identity confirmed: @{me.username} (id={me.id})")

    set_health_refs(
        get_last_update=lambda: health.last_update,
        get_errors=lambda: health.consecutive_logic_errors,
        get_restarts=lambda: health.restart_count,
    )
    set_clear_errors_fn(health.clear_errors)

    scheduler = init_scheduler(app.bot)
    scheduler.start()
    log.info("Scheduler started")

    asyncio.create_task(_heartbeat_loop())
    log.info(f"Heartbeat started (interval={HEARTBEAT_INTERVAL}s)")

    log.info("All systems ready!")


async def _heartbeat_loop() -> None:
    """Periodic asyncio heartbeat — marks event loop as alive."""
    while True:
        health.mark_heartbeat()
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _save_bot_reply(chat_id: int, text: str) -> None:
    await save_message(
        chat_id=chat_id,
        user_id=0,
        username=_bot_username or "bot",
        message_text=f"[бот] {text[:2000]}",
        message_type="bot_response",
    )


# ─── Voice handler ─────────────────────────────────────────────────
async def handle_voice(update: Update, context) -> None:
    msg = update.message
    if not msg or (not msg.voice and not msg.audio):
        return

    health.mark_activity()

    await save_message(
        chat_id=msg.chat_id,
        user_id=msg.from_user.id if msg.from_user else 0,
        username=msg.from_user.username or "" if msg.from_user else "",
        message_text="[голосовое сообщение]",
        message_id=msg.message_id,
        message_type="voice",
    )

    is_private = msg.chat.type == ChatType.PRIVATE
    if not is_private:
        is_reply_to_bot = (
            msg.reply_to_message
            and msg.reply_to_message.from_user
            and msg.reply_to_message.from_user.is_bot
            and _bot_username
            and msg.reply_to_message.from_user.username
            and msg.reply_to_message.from_user.username.lower() == _bot_username
        )
        is_mentioned = msg.caption and _is_mention(msg.caption, msg.caption_entities)
        if not (is_reply_to_bot or is_mentioned):
            return

    try:
        voice = msg.voice or msg.audio
        file = await context.bot.get_file(voice.file_id)
        file_bytes = await file.download_as_bytearray()
        text = await transcribe_audio(file_bytes)
        if not text:
            return

        await msg.reply_text(f"🎤 <i>{text}</i>", parse_mode="HTML")
        await save_message(
            chat_id=msg.chat_id,
            user_id=msg.from_user.id if msg.from_user else 0,
            username=msg.from_user.username or "" if msg.from_user else "",
            message_text=f"[голос→текст] {text}",
            message_type="voice_transcription",
        )

        intent_data = await detect_intent(text, has_photo=False)
        intent = intent_data.get("intent", "general")
        user_name = (
            msg.from_user.first_name or msg.from_user.username or ""
            if msg.from_user else ""
        )

        if intent == "weather":
            await handle_weather_query(update, context, text, intent_data)
        elif intent == "help":
            await send_help(update, context)
        else:
            response = await generate_response(
                query=text,
                chat_id=msg.chat_id,
                user_name=user_name,
                intent_data=intent_data,
            )
            try:
                await msg.reply_text(response, parse_mode="HTML")
            except Exception:
                await msg.reply_text(response)
            await _save_bot_reply(msg.chat_id, response)

    except (NetworkError, TimedOut) as e:
        log.warning(f"Network error in voice handler: {e}")
    except Exception as e:
        log.error(f"Voice handler failed: {e}", exc_info=True)


# ─── Wrapped handlers ─────────────────────────────────────────────
async def handle_text_with_tracking(update: Update, context) -> None:
    health.mark_activity()
    await handle_text_message(update, context)


async def handle_photo_with_tracking(update: Update, context) -> None:
    health.mark_activity()
    await handle_photo_message(update, context)


async def handle_location_with_tracking(update: Update, context) -> None:
    health.mark_activity()
    await handle_location(update, context)


# ─── Watchdog ──────────────────────────────────────────────────────
def _watchdog_thread(loop: asyncio.AbstractEventLoop, stop_event: threading.Event) -> None:
    """
    Background thread monitoring health.
    - Probes event loop aliveness via call_soon_threadsafe
    - Restarts if loop is frozen OR too many logic errors
    - Does NOT restart on network errors alone (VPN/sleep tolerance)
    - Exits cleanly when stop_event is set (on in-process restart)
    """
    log.info(
        f"Watchdog started (interval={HEALTH_CHECK_INTERVAL}s, "
        f"loop_freeze_threshold={LOOP_PROBE_TIMEOUT}s)"
    )

    while not stop_event.is_set():
        # Use wait() instead of sleep() so we wake up fast when stop_event fires
        stop_event.wait(timeout=HEALTH_CHECK_INTERVAL)
        if stop_event.is_set():
            log.info("Watchdog: stop requested, exiting")
            return
        now = time.time()
        now_str = datetime.now().strftime("%H:%M:%S")

        # Probe event loop — non-blocking, just schedule a callback
        try:
            loop.call_soon_threadsafe(health.mark_loop_alive)
        except RuntimeError:
            pass  # loop is closed, process is shutting down

        # Wait a bit for the probe to be processed
        stop_event.wait(2)

        heartbeat_age = now - health.last_heartbeat
        probe_age = now - health.last_loop_probe_response

        # Check 1: Too many logic errors
        if health.consecutive_logic_errors >= MAX_LOGIC_ERRORS:
            log.error(
                f"[{now_str}] Watchdog: {health.consecutive_logic_errors} "
                "consecutive logic errors! Restarting."
            )
            print(
                f"🐕 Watchdog: {health.consecutive_logic_errors} логических ошибок. "
                "Перезапуск...",
                flush=True,
            )
            os._exit(2)

        # Check 2: Event loop frozen (both heartbeat AND probe stale)
        if heartbeat_age > LOOP_PROBE_TIMEOUT and probe_age > LOOP_PROBE_TIMEOUT:
            log.warning(
                f"[{now_str}] Watchdog: event loop frozen "
                f"(heartbeat {int(heartbeat_age)}s, probe {int(probe_age)}s, "
                f"threshold {LOOP_PROBE_TIMEOUT}s). Restarting."
            )
            print(
                f"🐕 Watchdog: event loop завис ({int(heartbeat_age)}с). Перезапуск...",
                flush=True,
            )
            os._exit(2)

        net_e = health.consecutive_network_errors
        log_e = health.consecutive_logic_errors
        status = "OK" if net_e == 0 and log_e == 0 else f"net={net_e} logic={log_e}"
        log.info(
            f"[{now_str}] Watchdog: {status}, "
            f"heartbeat {int(heartbeat_age)}s ago, "
            f"probe {int(probe_age)}s ago"
        )


# ─── Main ──────────────────────────────────────────────────────────
def main() -> None:
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN not set. Check .env file.", flush=True)
        sys.exit(1)

    log.info("=" * 50)
    log.info("Starting Telegram Bot v2...")
    log.info(
        "Features: Chat History + RAG + Web Search + Weather + Photos + "
        "Fish Analysis + Face Registry + Expenses + Voice"
    )
    log.info("=" * 50)

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        # Regular API call timeouts (15s each)
        .connect_timeout(15)
        .read_timeout(15)
        .write_timeout(15)
        .pool_timeout(10)
        # getUpdates long-poll timeouts
        # Server holds connection ~10s; 20s read_timeout gives 10s buffer
        # This prevents the event loop from hanging >20s on a dead TCP connection
        .get_updates_connect_timeout(15)
        .get_updates_read_timeout(20)
        .get_updates_write_timeout(10)
        .get_updates_pool_timeout(10)
        .build()
    )

    app.add_error_handler(error_handler)
    log.info("Error handler registered")

    app.add_handler(CommandHandler("start", send_help))
    app.add_handler(CommandHandler("help", send_help))
    app.add_handler(CommandHandler("status", handle_status))
    app.add_handler(CallbackQueryHandler(handle_status_callback, pattern="^bot_"))
    app.add_handler(MessageHandler(filters.LOCATION, handle_location_with_tracking))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo_with_tracking))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_with_tracking)
    )

    log.info("Handlers registered. Starting polling...")
    print("🚀 Бот запускается...", flush=True)

    # Start watchdog with reference to the current event loop
    try:
        loop = asyncio.get_event_loop()
        watchdog = threading.Thread(
            target=_watchdog_thread,
            args=(loop, _watchdog_stop),
            daemon=True,
            name="watchdog",
        )
        watchdog.start()
        log.info("Watchdog thread started")
    except Exception as e:
        log.warning(f"Could not start watchdog with loop ref: {e}")

    app.run_polling(
        drop_pending_updates=True,
        allowed_updates=["message", "callback_query"],
    )


# ─── Entry point with auto-restart ────────────────────────────────
if __name__ == "__main__":
    print("🤖 Telegram Bot v2 — Запуск...", flush=True)

    while True:
        try:
            # Reset watchdog stop event for this run
            _watchdog_stop.clear()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            health.mark_activity()
            main()
            log.info("Bot stopped normally.")
            break

        except KeyboardInterrupt:
            print("\n👋 Бот остановлен (Ctrl+C)", flush=True)
            break

        except SystemExit as e:
            if e.code == 2:
                health.restart_count += 1
                wait = min(10, 3 * health.restart_count)
                log.warning(
                    f"Auto-restart #{health.restart_count} in {wait}s "
                    "(watchdog triggered)"
                )
                print(
                    f"🔄 Авто-перезапуск #{health.restart_count} через {wait}с...",
                    flush=True,
                )
                _watchdog_stop.set()  # Signal old watchdog to exit
                time.sleep(wait)
                health.consecutive_network_errors = 0
                health.consecutive_logic_errors = 0
                continue
            else:
                break

        except Exception as e:
            health.restart_count += 1
            wait = min(30, 5 * health.restart_count)
            log.error(f"Bot crashed (#{health.restart_count}): {e}", exc_info=True)
            print(f"💥 Бот упал: {e}", flush=True)
            print(f"🔄 Перезапуск через {wait}с...", flush=True)
            _watchdog_stop.set()  # Signal old watchdog to exit
            time.sleep(wait)
            health.consecutive_network_errors = 0
            health.consecutive_logic_errors = 0

        finally:
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.close()
            except Exception:
                pass

    print("Бот завершил работу.", flush=True)
