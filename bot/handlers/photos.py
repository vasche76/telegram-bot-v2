"""
Photo request handler: instant photos and scheduled delivery.
"""

import io
from telegram import Update, InputMediaPhoto
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from bot.services.photos import smart_photo_search, download_image
from bot.services.ai import structured_extraction
from bot.storage.database import execute, fetch_all
from bot.utils.logging import get_logger

log = get_logger("handlers.photos")


async def handle_photo_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    intent_data: dict,
) -> None:
    """Handle instant photo requests with smart search and target user tagging."""
    msg = update.message
    chat_id = msg.chat_id

    # Extract target user and search query from intent
    target_user = intent_data.get("target_user")
    search_query = intent_data.get("search_query") or query
    count = intent_data.get("count") or 3
    count = min(max(1, int(count)), 10)  # Clamp 1-10

    # If intent didn't extract well, use GPT
    if not search_query or search_query == query:
        extraction = await structured_extraction(
            prompt=f"""Из запроса на фото извлеки:
1. search_topic — ЧТО искать (тема фото, без имени адресата)
2. target_person — КОМУ отправить (имя человека, если указано), или null
3. count — сколько фото (число), по умолчанию 3

Запрос: "{query}"

Примеры:
- "пришли Васе 5 фото котиков" → search_topic="котики", target_person="Вася", count=5
- "покажи фото Марго Робби" → search_topic="Марго Робби", target_person=null, count=3
- "скинь красивые пейзажи" → search_topic="красивые пейзажи", target_person=null, count=3

JSON: {{"search_topic": "...", "target_person": "..." или null, "count": число}}""",
            system="Извлеки данные из запроса на фото. Отвечай JSON.",
        )
        search_query = extraction.get("search_topic", query)
        target_user = extraction.get("target_person") or target_user
        count = extraction.get("count") or count
        count = min(max(1, int(count)), 10)

    log.info(f"Photo request: query='{search_query}', target='{target_user}', count={count}")

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")

    # Search photos
    urls, used_query = await smart_photo_search(search_query, count)

    if not urls:
        await msg.reply_text(f"😔 Не удалось найти фото по запросу «{search_query}».")
        return

    # Send photos
    sent_count = 0
    media_group = []

    for i, url in enumerate(urls):
        try:
            # Try sending by URL first
            if len(urls) > 1:
                media_group.append(InputMediaPhoto(media=url))
            else:
                await msg.reply_photo(photo=url)
                sent_count = 1
        except Exception:
            # Download and send as bytes
            try:
                img_bytes = await download_image(url)
                if img_bytes:
                    if len(urls) > 1:
                        media_group.append(InputMediaPhoto(media=io.BytesIO(img_bytes)))
                    else:
                        await msg.reply_photo(photo=io.BytesIO(img_bytes))
                        sent_count = 1
            except Exception as e:
                log.warning(f"Failed to send photo {url}: {e}")

    # Send media group
    if media_group:
        try:
            await context.bot.send_media_group(chat_id=chat_id, media=media_group)
            sent_count = len(media_group)
        except Exception as e:
            log.error(f"Failed to send media group: {e}")
            # Try one by one
            for media in media_group:
                try:
                    await msg.reply_photo(photo=media.media)
                    sent_count += 1
                except Exception:
                    pass

    # Tag target user
    if sent_count > 0 and target_user:
        await msg.reply_text(
            f"✅ {target_user}, тебе {sent_count} фото! ⬆️",
            parse_mode=ParseMode.HTML,
        )
    elif sent_count == 0:
        await msg.reply_text(f"😔 Не удалось отправить фото по запросу «{search_query}».")


async def handle_photo_schedule(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    intent_data: dict,
) -> None:
    """Handle photo schedule setup."""
    msg = update.message
    chat_id = msg.chat_id
    username = msg.from_user.username or "" if msg.from_user else ""

    # Extract schedule parameters
    schedule = await structured_extraction(
        prompt=f"""Из запроса извлеки параметры расписания отправки фото:

Запрос: "{query}"

Извлеки:
- target_user: кому отправлять (имя)
- send_time: время отправки (формат HH:MM, 24ч)
- photo_count: количество фото за раз (по умолчанию 5)
- search_query: что искать (тема фото)

JSON: {{"target_user": "...", "send_time": "HH:MM", "photo_count": число, "search_query": "..."}}""",
        system="Извлеки параметры расписания. Отвечай JSON.",
    )

    target = schedule.get("target_user", "")
    send_time = schedule.get("send_time", "09:00")
    photo_count = schedule.get("photo_count", 5)
    search_q = schedule.get("search_query", "beautiful nature landscape")

    if not target:
        await msg.reply_text(
            "📅 Для настройки расписания укажите:\n"
            "• Кому отправлять\n"
            "• Время (например, 22:30)\n"
            "• Тему фото\n\n"
            "Пример: <code>@бот присылай Васе в 22:30 фото природы</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Save schedule
    await execute(
        """INSERT INTO photo_schedules (chat_id, target_user, send_time, photo_count, search_query, created_by)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chat_id, target, send_time, photo_count, search_q, username),
    )

    await msg.reply_text(
        f"✅ Расписание создано!\n\n"
        f"👤 Для: {target}\n"
        f"⏰ Время: {send_time}\n"
        f"📷 Фото: {photo_count} шт.\n"
        f"🔍 Тема: {search_q}\n\n"
        f"Используйте /photo_off чтобы отключить.",
        parse_mode=ParseMode.HTML,
    )


async def send_scheduled_photos(bot) -> None:
    """Job: send photos according to schedules.

    FIX: Original code used exact HH:MM match, but the job runs every 5 minutes
    so schedules set to an off-minute (e.g., 09:03) would never fire.
    Now fires if we're within 5 minutes of the scheduled time (and not already
    sent today).
    """
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    now_minutes = now.hour * 60 + now.minute  # total minutes from midnight

    schedules = await fetch_all(
        "SELECT * FROM photo_schedules WHERE enabled = 1"
    )

    for sched in schedules:
        # Parse scheduled time
        try:
            h, m = map(int, sched["send_time"].split(":"))
            sched_minutes = h * 60 + m
        except Exception:
            continue

        # Fire if within 5-minute window and not already sent today
        if abs(now_minutes - sched_minutes) > 5:
            continue
        if sched.get("last_sent_date") == today:
            continue

        chat_id = sched["chat_id"]
        search_query = sched.get("search_query", "beautiful nature")
        count = sched.get("photo_count", 5)
        target = sched.get("target_user", "")

        try:
            urls, _ = await smart_photo_search(search_query, count)
            if urls:
                media_group = [InputMediaPhoto(media=url) for url in urls]
                await bot.send_media_group(chat_id=chat_id, media=media_group)
                if target:
                    await bot.send_message(chat_id=chat_id, text=f"📷 {target}, вот твои фото на сегодня! ⬆️")

            await execute(
                "UPDATE photo_schedules SET last_sent_date = ? WHERE id = ?",
                (today, sched["id"]),
            )
            log.info(f"Sent scheduled photos to chat {chat_id} for {target}")
        except Exception as e:
            log.error(f"Failed to send scheduled photos: {e}")
