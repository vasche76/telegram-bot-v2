"""
Main message handler: silent history recording + @mention response routing.
All text messages pass through here.
"""

import re
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ChatType, ParseMode
from telegram.error import NetworkError, TimedOut, RetryAfter

from bot.storage.messages import save_message
from bot.storage.users import upsert_user
from bot.services.intent import detect_intent
from bot.services.response import generate_response
from bot.services.response import set_response_bot_username
from bot.utils.logging import get_logger, set_correlation_id

log = get_logger("handlers.messages")

# Bot username (set during startup)
_bot_username: str = ""


def set_bot_username(username: str) -> None:
    global _bot_username
    _bot_username = username.lower()
    # Also set in response module for conversation history matching
    set_response_bot_username(username.lower())


def _is_mention(text: str, entities: list) -> bool:
    """Check if the bot is mentioned in the message."""
    if not _bot_username:
        return False

    # Check @mention entities
    for entity in (entities or []):
        if entity.type == "mention":
            mention = text[entity.offset:entity.offset + entity.length].lower()
            if _bot_username in mention:
                return True
        elif entity.type == "text_mention":
            return True

    # Check text for bot username
    if f"@{_bot_username}" in text.lower():
        return True

    return False


def _strip_mention(text: str) -> str:
    """Remove bot @mention from text.

    FIX: Original regex r'@\w+bot\b' only stripped mentions ending in 'bot'.
    Now strips any @word mention (handles all bot usernames like @MyHelper_Bot,
    @Vassiliy_Chekulaev_bot, etc.), then falls back to keeping original text
    if the result is empty.
    """
    if _bot_username:
        # Strip the known bot username (most reliable)
        cleaned = re.sub(
            rf'@{re.escape(_bot_username)}\b', '', text, flags=re.IGNORECASE
        ).strip()
        if cleaned:
            return cleaned
    # Fallback: strip any @word mention
    cleaned = re.sub(r'@\w+', '', text, flags=re.IGNORECASE).strip()
    return cleaned if cleaned else text


async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all incoming text messages."""
    if not update.message or not update.message.text:
        return

    msg = update.message
    set_correlation_id(update.update_id)

    user = msg.from_user
    chat_id = msg.chat_id
    text = msg.text
    user_id = user.id if user else 0
    username = user.username or "" if user else ""
    first_name = user.first_name or "" if user else ""
    last_name = user.last_name or "" if user else ""

    # Step 1: Always save message to history (silent recording)
    await save_message(
        chat_id=chat_id,
        user_id=user_id,
        username=username,
        message_text=text,
        message_id=msg.message_id,
        message_type="text",
        reply_to=msg.reply_to_message.message_id if msg.reply_to_message else None,
    )

    # Step 2: Update user profile
    await upsert_user(
        user_id=user_id,
        chat_id=chat_id,
        username=username,
        first_name=first_name,
        last_name=last_name,
    )

    # Step 3: Skip stale messages (older than 120s) to avoid replaying on restart
    import time as _time
    msg_age = _time.time() - msg.date.timestamp()
    if msg_age > 120:
        log.debug(f"Skipping stale message ({msg_age:.0f}s old) from {username}")
        return

    # Step 4: Check if bot is mentioned (or private chat)
    is_private = msg.chat.type == ChatType.PRIVATE
    is_mentioned = _is_mention(text, msg.entities)
    is_reply_to_bot = (
        msg.reply_to_message
        and msg.reply_to_message.from_user
        and msg.reply_to_message.from_user.is_bot
        and _bot_username
        and msg.reply_to_message.from_user.username
        and msg.reply_to_message.from_user.username.lower() == _bot_username
    )

    if not (is_private or is_mentioned or is_reply_to_bot):
        return  # Silent recording only

    # Step 5: Process the mention
    query = _strip_mention(text) if is_mentioned else text
    log.info(f"Processing query from {username} in {chat_id}: {query[:80]}")

    # Step 5a: Check if this is a reply to a photo message — route to vision
    replied = msg.reply_to_message
    if replied and replied.photo:
        log.info(f"Reply-to-photo detected, routing to vision analysis")
        try:
            from bot.services.ai import vision_analyze
            photo = replied.photo[-1]
            file = await context.bot.get_file(photo.file_id)
            image_url = file.file_path
            prompt = f"Пользователь спрашивает про это фото: {query}" if query else "Подробно опиши что изображено на этом фото."
            await context.bot.send_chat_action(chat_id=chat_id, action="typing")
            result = await vision_analyze(image_url=image_url, prompt=prompt)
            await msg.reply_text(result)
            # Save bot response to history
            await save_message(
                chat_id=chat_id, user_id=0,
                username=_bot_username or "bot",
                message_text=f"[бот анализ фото] {result[:2000]}",
                message_type="bot_response",
            )
            return
        except Exception as e:
            log.error(f"Reply-to-photo analysis failed: {e}", exc_info=True)
            # Fall through to normal text processing

    # Step 6: Detect intent
    intent_data = await detect_intent(query, has_photo=False)
    intent = intent_data.get("intent", "general")

    # Step 7: Route to handler
    try:
        if intent == "weather":
            from bot.handlers.weather import handle_weather_query
            await handle_weather_query(update, context, query, intent_data)

        elif intent == "photo":
            from bot.handlers.photos import handle_photo_request
            await handle_photo_request(update, context, query, intent_data)

        elif intent == "photo_schedule":
            from bot.handlers.photos import handle_photo_schedule
            await handle_photo_schedule(update, context, query, intent_data)

        elif intent in ("expense_start", "expense_add", "expense_close", "expense_status"):
            from bot.handlers.expenses import handle_expense_command
            await handle_expense_command(update, context, query, intent_data)

        elif intent == "catch_stats":
            from bot.handlers.vision import handle_catch_stats
            await handle_catch_stats(update, context, query)

        elif intent == "search_history":
            from bot.handlers.search import handle_history_search
            await handle_history_search(update, context, query)

        elif intent in ("web_search", "general"):
            # Both use the RAG pipeline
            response = await generate_response(
                query=query,
                chat_id=chat_id,
                user_name=first_name or username,
                intent_data=intent_data,
            )
            await _send_response(msg, response)

        elif intent == "help":
            from bot.handlers.help import send_help
            await send_help(update, context)

        else:
            response = await generate_response(
                query=query,
                chat_id=chat_id,
                user_name=first_name or username,
                intent_data=intent_data,
            )
            await _send_response(msg, response)

    except (NetworkError, TimedOut, RetryAfter) as e:
        log.warning(f"Network error for intent '{intent}': {e}")
        # Don't try to reply — network is down, just log it
    except Exception as e:
        log.error(f"Handler error for intent '{intent}': {e}", exc_info=True)
        try:
            await msg.reply_text(
                "⚠️ Произошла ошибка при обработке запроса. Попробуйте ещё раз.",
                parse_mode=ParseMode.HTML,
            )
        except (NetworkError, TimedOut):
            log.warning("Could not send error message — network issue")


async def _send_response(msg, text: str) -> None:
    """Send a response, splitting if too long. Also saves bot reply to history."""
    MAX_LEN = 4000
    try:
        if len(text) <= MAX_LEN:
            try:
                await msg.reply_text(text, parse_mode=ParseMode.HTML)
            except (NetworkError, TimedOut):
                raise
            except Exception:
                await msg.reply_text(text)
        else:
            chunks = [text[i:i + MAX_LEN] for i in range(0, len(text), MAX_LEN)]
            for chunk in chunks:
                try:
                    await msg.reply_text(chunk, parse_mode=ParseMode.HTML)
                except (NetworkError, TimedOut):
                    raise
                except Exception:
                    await msg.reply_text(chunk)

        # Save bot's response to history for conversation memory
        try:
            await save_message(
                chat_id=msg.chat_id,
                user_id=0,
                username=_bot_username or "bot",
                message_text=f"[бот] {text[:2000]}",
                message_type="bot_response",
            )
        except Exception as e:
            log.warning(f"Failed to save bot response: {e}")

    except (NetworkError, TimedOut) as e:
        log.warning(f"Network error sending response: {e}")
