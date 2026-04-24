"""
/status command — remote bot diagnostics and control from Telegram.

Features:
- System health check (uptime, errors, memory, CPU)
- Database status (message count, DB size)
- Interactive buttons: restart, clear errors, run diagnostics
- Admin-only access (or open to all in private chats if no admins configured)
"""

import os
import sys
import time
import asyncio
import platform
from datetime import datetime, timedelta

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from bot.config import ADMIN_USER_IDS, DATABASE_PATH, OPENAI_CHAT_MODEL, OPENAI_VISION_MODEL
from bot.utils.logging import get_logger

log = get_logger("handlers.status")

# ── Boot time (set once at import) ──
_boot_time: float = time.time()

# ── Reference to health tracking vars from main.py ──
# These will be set by main.py via set_health_refs()
_get_last_update_time = None
_get_consecutive_errors = None
_get_restart_count = None


def set_health_refs(get_last_update, get_errors, get_restarts):
    """Set references to health tracking functions from main.py."""
    global _get_last_update_time, _get_consecutive_errors, _get_restart_count
    _get_last_update_time = get_last_update
    _get_consecutive_errors = get_errors
    _get_restart_count = get_restarts


def _is_admin(user_id: int) -> bool:
    """Check if user is an admin. Empty ADMIN_USER_IDS denies all."""
    if not ADMIN_USER_IDS:
        return False  # No admins configured → deny all (fail-closed)
    return user_id in ADMIN_USER_IDS


if not ADMIN_USER_IDS:
    log.warning(
        "ADMIN_USER_IDS is empty — /status command is disabled for all users. "
        "Set ADMIN_USER_IDS in .env to enable it."
    )


def _format_uptime(seconds: float) -> str:
    """Format seconds into human-readable uptime."""
    td = timedelta(seconds=int(seconds))
    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts = []
    if days > 0:
        parts.append(f"{days}д")
    if hours > 0:
        parts.append(f"{hours}ч")
    if minutes > 0:
        parts.append(f"{minutes}м")
    parts.append(f"{secs}с")
    return " ".join(parts)


def _get_db_size() -> str:
    """Get database file size."""
    try:
        db_path = DATABASE_PATH
        if os.path.exists(db_path):
            size = os.path.getsize(db_path)
            if size < 1024:
                return f"{size} B"
            elif size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            else:
                return f"{size / (1024 * 1024):.1f} MB"
        return "не найдена"
    except Exception:
        return "ошибка"


def _get_memory_usage() -> str:
    """Get current process memory usage."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        if platform.system() == "Darwin":
            mb = usage.ru_maxrss / (1024 * 1024)
        else:
            mb = usage.ru_maxrss / 1024
        return f"{mb:.1f} MB"
    except Exception:
        return "н/д"


async def _get_message_count() -> str:
    """Get total messages in database."""
    try:
        from bot.storage.messages import get_message_count
        # get_message_count requires chat_id, so we count all
        from bot.storage.database import fetch_scalar
        count = await fetch_scalar("SELECT COUNT(*) FROM messages")
        return str(count or 0)
    except Exception:
        return "ошибка"


async def handle_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command — show bot diagnostics with action buttons."""
    msg = update.message
    if not msg:
        return

    user_id = msg.from_user.id if msg.from_user else 0

    if not _is_admin(user_id):
        await msg.reply_text("🔒 Эта команда доступна только администраторам.")
        return

    # Gather diagnostics
    now = time.time()
    uptime = now - _boot_time

    last_activity = _get_last_update_time() if _get_last_update_time else now
    idle_time = now - last_activity
    errors = _get_consecutive_errors() if _get_consecutive_errors else 0
    restarts = _get_restart_count() if _get_restart_count else 0

    # Determine status
    if errors >= 10:
        status_emoji = "🔴"
        status_text = "КРИТИЧНО"
        status_desc = f"Много ошибок подряд ({errors}). Бот может работать нестабильно."
    elif errors >= 3:
        status_emoji = "🟡"
        status_text = "ДЕГРАДАЦИЯ"
        status_desc = f"Есть ошибки ({errors}). Бот работает, но с проблемами."
    elif idle_time > 300:
        status_emoji = "🟡"
        status_text = "ПРОСТОЙ"
        status_desc = f"Нет активности {_format_uptime(idle_time)}. Возможно, нет входящих сообщений."
    else:
        status_emoji = "🟢"
        status_text = "РАБОТАЕТ"
        status_desc = "Всё в порядке. Бот активен и отвечает."

    # Get DB stats
    db_size = _get_db_size()
    msg_count = await _get_message_count()
    memory = _get_memory_usage()

    # Format message
    lines = [
        f"{status_emoji} <b>Статус бота: {status_text}</b>",
        f"📝 {status_desc}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"⏱ <b>Аптайм:</b> {_format_uptime(uptime)}",
        f"🕐 <b>Последняя активность:</b> {_format_uptime(idle_time)} назад",
        f"❌ <b>Ошибки подряд:</b> {errors}",
        f"🔄 <b>Перезапусков:</b> {restarts}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💾 <b>Память:</b> {memory}",
        f"🗄 <b>БД размер:</b> {db_size}",
        f"📨 <b>Сообщений в БД:</b> {msg_count}",
        f"🤖 <b>Модель:</b> {OPENAI_CHAT_MODEL}",
        f"👁 <b>Vision:</b> {OPENAI_VISION_MODEL}",
        f"🐍 <b>Python:</b> {platform.python_version()}",
        f"💻 <b>Система:</b> {platform.system()} {platform.machine()}",
        "",
        "━━━━━━━━━━━━━━━━━━━━",
        "⬇️ <b>Выберите действие:</b>",
    ]

    # Build inline keyboard with actions
    keyboard = []

    if errors > 0 or idle_time > 300:
        keyboard.append([
            InlineKeyboardButton("🔄 Перезагрузить СЕЙЧАС", callback_data="bot_restart_now"),
        ])
        keyboard.append([
            InlineKeyboardButton("⏳ Подождать 30с", callback_data="bot_wait_30"),
            InlineKeyboardButton("⏳ Подождать 60с", callback_data="bot_wait_60"),
        ])

    keyboard.append([
        InlineKeyboardButton("🧹 Сбросить ошибки", callback_data="bot_clear_errors"),
        InlineKeyboardButton("🔍 Диагностика", callback_data="bot_diagnostics"),
    ])
    keyboard.append([
        InlineKeyboardButton("📊 Обновить статус", callback_data="bot_refresh_status"),
    ])

    if errors >= 5:
        keyboard.append([
            InlineKeyboardButton("⚠️ Принудительный рестарт", callback_data="bot_force_restart"),
        ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    await msg.reply_text(
        "\n".join(lines),
        parse_mode=ParseMode.HTML,
        reply_markup=reply_markup,
    )


async def handle_status_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button presses from /status message."""
    query = update.callback_query
    if not query:
        return

    user_id = query.from_user.id if query.from_user else 0
    if not _is_admin(user_id):
        await query.answer("🔒 Только для администраторов", show_alert=True)
        return

    data = query.data
    await query.answer()  # Acknowledge the button press

    if data == "bot_restart_now":
        await query.edit_message_text(
            "🔄 <b>Перезагрузка...</b>\n\n"
            "Бот перезапустится через 3 секунды.\n"
            "Отправьте /status через 10-15 секунд для проверки.",
            parse_mode=ParseMode.HTML,
        )
        log.warning("🔄 Admin requested restart via /status")
        # Give time for the message to be sent
        await asyncio.sleep(2)
        os._exit(2)  # Exit code 2 = restart requested

    elif data == "bot_force_restart":
        await query.edit_message_text(
            "⚠️ <b>Принудительный рестарт!</b>\n\n"
            "Бот будет остановлен и перезапущен немедленно.\n"
            "Отправьте /status через 15-20 секунд.",
            parse_mode=ParseMode.HTML,
        )
        log.error("⚠️ Admin requested FORCE restart via /status")
        await asyncio.sleep(1)
        os._exit(2)

    elif data == "bot_wait_30":
        await query.edit_message_text(
            "⏳ <b>Ожидание 30 секунд...</b>\n\n"
            "Бот продолжает работать. Проверю состояние через 30с.",
            parse_mode=ParseMode.HTML,
        )
        await asyncio.sleep(30)
        # After waiting, send a fresh status
        now = time.time()
        errors = _get_consecutive_errors() if _get_consecutive_errors else 0
        idle = now - (_get_last_update_time() if _get_last_update_time else now)

        if errors == 0 and idle < 60:
            status = "🟢 После ожидания: бот работает нормально!"
        elif errors > 0:
            status = f"🟡 После ожидания: всё ещё {errors} ошибок. Рекомендую перезагрузить."
        else:
            status = f"🟡 После ожидания: простой {_format_uptime(idle)}."

        keyboard = [[
            InlineKeyboardButton("🔄 Перезагрузить", callback_data="bot_restart_now"),
            InlineKeyboardButton("📊 Обновить", callback_data="bot_refresh_status"),
        ]]
        await query.edit_message_text(
            status,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    elif data == "bot_wait_60":
        await query.edit_message_text(
            "⏳ <b>Ожидание 60 секунд...</b>\n\n"
            "Бот продолжает работать. Проверю состояние через 1 минуту.",
            parse_mode=ParseMode.HTML,
        )
        await asyncio.sleep(60)
        now = time.time()
        errors = _get_consecutive_errors() if _get_consecutive_errors else 0
        idle = now - (_get_last_update_time() if _get_last_update_time else now)

        if errors == 0 and idle < 60:
            status = "🟢 После ожидания: бот работает нормально!"
        elif errors > 0:
            status = f"🟡 После ожидания: всё ещё {errors} ошибок. Рекомендую перезагрузить."
        else:
            status = f"🟡 После ожидания: простой {_format_uptime(idle)}."

        keyboard = [[
            InlineKeyboardButton("🔄 Перезагрузить", callback_data="bot_restart_now"),
            InlineKeyboardButton("📊 Обновить", callback_data="bot_refresh_status"),
        ]]
        await query.edit_message_text(
            status,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    elif data == "bot_clear_errors":
        # Reset error counter
        from bot.handlers.status import _clear_errors_callback
        _clear_errors_callback()
        await query.edit_message_text(
            "🧹 <b>Счётчик ошибок сброшен!</b>\n\n"
            "Ошибки обнулены. Бот продолжает работать.\n"
            "Отправьте /status для проверки.",
            parse_mode=ParseMode.HTML,
        )

    elif data == "bot_diagnostics":
        await query.edit_message_text(
            "🔍 <b>Запуск диагностики...</b>",
            parse_mode=ParseMode.HTML,
        )
        diag = await _run_diagnostics()
        keyboard = [[
            InlineKeyboardButton("🔄 Перезагрузить", callback_data="bot_restart_now"),
            InlineKeyboardButton("📊 Обновить статус", callback_data="bot_refresh_status"),
        ]]
        await query.edit_message_text(
            diag,
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )

    elif data == "bot_refresh_status":
        # Re-run the full status check
        # We need to rebuild the status message
        now = time.time()
        uptime = now - _boot_time
        last_activity = _get_last_update_time() if _get_last_update_time else now
        idle_time = now - last_activity
        errors = _get_consecutive_errors() if _get_consecutive_errors else 0
        restarts = _get_restart_count() if _get_restart_count else 0

        if errors >= 10:
            status_emoji = "🔴"
            status_text = "КРИТИЧНО"
        elif errors >= 3:
            status_emoji = "🟡"
            status_text = "ДЕГРАДАЦИЯ"
        elif idle_time > 300:
            status_emoji = "🟡"
            status_text = "ПРОСТОЙ"
        else:
            status_emoji = "🟢"
            status_text = "РАБОТАЕТ"

        db_size = _get_db_size()
        msg_count = await _get_message_count()
        memory = _get_memory_usage()

        lines = [
            f"{status_emoji} <b>Статус: {status_text}</b> (обновлено {datetime.now().strftime('%H:%M:%S')})",
            "",
            f"⏱ Аптайм: {_format_uptime(uptime)}",
            f"🕐 Простой: {_format_uptime(idle_time)}",
            f"❌ Ошибки: {errors} | 🔄 Рестарты: {restarts}",
            f"💾 RAM: {memory} | 🗄 БД: {db_size} ({msg_count} сообщ.)",
        ]

        keyboard = []
        if errors > 0 or idle_time > 300:
            keyboard.append([
                InlineKeyboardButton("🔄 Перезагрузить", callback_data="bot_restart_now"),
            ])
        keyboard.append([
            InlineKeyboardButton("🧹 Сбросить ошибки", callback_data="bot_clear_errors"),
            InlineKeyboardButton("🔍 Диагностика", callback_data="bot_diagnostics"),
        ])
        keyboard.append([
            InlineKeyboardButton("📊 Обновить", callback_data="bot_refresh_status"),
        ])

        await query.edit_message_text(
            "\n".join(lines),
            parse_mode=ParseMode.HTML,
            reply_markup=InlineKeyboardMarkup(keyboard),
        )


# ── Error clearing callback (called from main.py) ──
_clear_errors_fn = None


def set_clear_errors_fn(fn):
    """Set the function to clear errors from main.py."""
    global _clear_errors_fn
    _clear_errors_fn = fn


def _clear_errors_callback():
    """Clear the error counter."""
    if _clear_errors_fn:
        _clear_errors_fn()


async def _run_diagnostics() -> str:
    """Run a comprehensive diagnostics check."""
    results = []
    results.append("🔍 <b>Диагностика бота</b>\n")

    # 1. Database check
    try:
        from bot.storage.database import fetch_scalar
        count = await fetch_scalar("SELECT COUNT(*) FROM messages")
        results.append(f"✅ БД: доступна ({count} сообщений)")
    except Exception as e:
        results.append(f"❌ БД: ошибка — {e}")

    # 2. OpenAI API check
    try:
        from bot.services.ai import chat_completion
        test_resp = await asyncio.wait_for(
            chat_completion(
                messages=[{"role": "user", "content": "Ответь одним словом: работает"}],
                max_tokens=10,
            ),
            timeout=15,
        )
        results.append(f"✅ OpenAI API: работает (ответ: {test_resp[:30]})")
    except asyncio.TimeoutError:
        results.append("❌ OpenAI API: таймаут (>15с)")
    except Exception as e:
        results.append(f"❌ OpenAI API: ошибка — {str(e)[:100]}")

    # 3. Telegram API check
    results.append("✅ Telegram API: работает (раз вы видите это сообщение)")

    # 4. Memory check
    memory = _get_memory_usage()
    results.append(f"💾 Память: {memory}")

    # 5. Disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (1024 ** 3)
        results.append(f"💿 Диск: {free_gb:.1f} GB свободно")
    except Exception:
        results.append("💿 Диск: н/д")

    # 6. Python version
    results.append(f"🐍 Python: {platform.python_version()}")

    # 7. Uptime
    uptime = time.time() - _boot_time
    results.append(f"⏱ Аптайм: {_format_uptime(uptime)}")

    results.append("\n━━━━━━━━━━━━━━━━━━━━")

    # Summary
    error_count = sum(1 for r in results if r.startswith("❌"))
    if error_count == 0:
        results.append("✅ <b>Все проверки пройдены!</b>")
    else:
        results.append(f"⚠️ <b>Обнаружено проблем: {error_count}</b>")

    return "\n".join(results)
