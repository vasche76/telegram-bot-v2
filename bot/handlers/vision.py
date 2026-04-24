"""
Vision handlers: fish recognition, face registry, receipt parsing.

FISH RECOGNITION: uses the dedicated two-stage FishVisionPipeline.
  Stage A (detector) — filters lures / parts / fry / no_fish.
  Stage B (classifier) — species ID with confidence threshold.
  Bad detections NEVER enter catch statistics.

All other vision tasks (face, receipt) use GPT-4o-mini structured outputs.
"""

import json
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from bot.services.ai import vision_structured, vision_analyze
from bot.storage.catches import save_catch, get_chat_leaderboard
from bot.storage.database import execute, fetch_all, fetch_one
from bot.storage.users import get_display_name
from bot.storage.expenses import add_expense, get_active_session, create_session, is_receipt_already_added
from bot.fish_vision.pipeline import analyze_fish_photo
from bot.utils.logging import get_logger

log = get_logger("handlers.vision")


# ── Fish Analysis (two-stage pipeline) ─────────────────────

async def handle_fish_photo(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    caption: str = "",
) -> None:
    """
    Analyze a fish photo using the dedicated two-stage FishVisionPipeline.

    Stage A: filter (whole_fish / lure / fish_part / fry / no_fish)
    Stage B: species classification (pike / taimen / grayling / whitefish / perch / unknown)

    Rules:
    - Lures, fish parts, fry → rejected, NOT saved to statistics.
    - Uncertain detections (low confidence) → NOT saved to statistics.
    - Only confirmed whole fish at sufficient confidence → saved.
    - All submissions (including rejections) are stored for audit trail.
    """
    msg = update.message
    chat_id = msg.chat_id
    user_id = msg.from_user.id if msg.from_user else 0

    if not msg.photo:
        return

    # Download the largest version of the photo
    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_url = file.file_path  # Telegram CDN URL

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # ── Two-stage pipeline ──────────────────────────────────
    result = await analyze_fish_photo(image_url=image_url, caption=caption)

    # ── Determine angler name ───────────────────────────────
    # Priority: name seen in photo → name in caption (already in caption) → user profile
    person = (
        result.person_name_in_photo
        or await get_display_name(user_id, chat_id)
    )

    # ── Always save the attempt (for audit) ────────────────
    # is_valid_catch controls whether it enters the leaderboard
    confidence_label = result.confidence_label if result.is_valid_catch else "rejected"
    await save_catch(
        chat_id=chat_id,
        user_id=user_id,
        person_name=person,
        fish_species=result.species_ru,
        fish_count=result.fish_count,
        weight_kg=result.weight_kg_estimate,
        length_cm=result.length_cm_estimate,
        confidence=confidence_label,
        photo_file_id=photo.file_id,
        analysis_text=result.classification_reasoning or result.detection_reasoning,
        # Fish-vision pipeline fields
        object_type=result.object_type,
        species_confidence=result.species_confidence,
        is_valid_catch=result.is_valid_catch,
        rejection_reason=result.rejection_reason,
    )

    # ── Rejected: not a valid catch ─────────────────────────
    if not result.is_valid_catch:
        log.info(
            f"Fish photo rejected for {person}: "
            f"type={result.object_type}, reason={result.rejection_reason}"
        )
        response_text = result.rejection_message or (
            "❓ Не удалось надёжно определить рыбу на фото. "
            "Попробуйте более чёткое фото."
        )
        await msg.reply_text(response_text, parse_mode=ParseMode.HTML)
        await _save_bot_response(chat_id, f"[бот отклонил фото рыбы] {result.rejection_reason}")
        return

    # ── Valid catch: format response ────────────────────────
    lines = ["🐟 <b>Улов записан!</b>\n"]
    lines.append(f"👤 Рыбак: <b>{person}</b>")
    lines.append(f"🐠 Вид: <b>{result.species_ru}</b>")

    if result.fish_count > 1:
        lines.append(f"🔢 Количество: {result.fish_count} шт.")

    if result.weight_kg_estimate:
        lines.append(f"⚖️ Вес: ~{result.weight_kg_estimate} кг (оценка)")
    if result.length_cm_estimate:
        lines.append(f"📏 Длина: ~{result.length_cm_estimate} см (оценка)")

    lines.append(f"📊 Уверенность: {result.confidence_label}")

    # Show which features led to the species ID
    if result.distinguishing_features and result.species_key != "unknown_fish":
        lines.append(f"\n🔍 {result.distinguishing_features}")

    if result.species_key == "unknown_fish":
        lines.append(
            "\n⚠️ Вид точно не определён — записано как «рыба».\n"
            "Если знаете вид — напишите в подписи к следующему фото."
        )

    response_text = "\n".join(lines)
    await msg.reply_text(response_text, parse_mode=ParseMode.HTML)
    await _save_bot_response(chat_id, f"[бот анализ рыбы] {response_text}")


async def handle_catch_stats(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str = "",
) -> None:
    """Show fishing statistics and leaderboard (valid catches only)."""
    msg = update.message
    chat_id = msg.chat_id

    from bot.storage.catches import get_catch_stats_for_chat
    leaderboard = await get_chat_leaderboard(chat_id)
    stats = await get_catch_stats_for_chat(chat_id)

    if not leaderboard:
        hint = ""
        if stats["lures_caught"] > 0:
            hint = f"\n\n⚠️ Было отклонено {stats['lures_caught']} фото приманок."
        elif stats["rejected_total"] > 0:
            hint = f"\n\n⚠️ Было отклонено {stats['rejected_total']} некорректных фото."
        await msg.reply_text(
            "🎣 Пока нет подтверждённых уловов. Отправьте фото рыбы, чтобы начать!" + hint
        )
        return

    lines = ["🏆 <b>Рейтинг рыбаков</b> (подтверждённые уловы)\n"]
    medals = ["🥇", "🥈", "🥉"]

    for i, row in enumerate(leaderboard):
        medal = medals[i] if i < 3 else f"{i+1}."
        name = row["person_name"]
        total = row["total_fish"] or row["total_catches"]
        weight = row["total_weight_kg"]
        biggest = row["biggest_catch_kg"]
        species = row["species_list"] or ""

        line = f"{medal} <b>{name}</b>: {total} рыб"
        if weight:
            line += f", {weight:.1f} кг всего"
        if biggest:
            line += f" (рекорд {biggest:.1f} кг)"
        if species:
            line += f"\n   Виды: {species}"
        lines.append(line)

    # Show rejection stats if any
    if stats["rejected_total"] > 0:
        lines.append(
            f"\n<i>ℹ️ Отклонено некорректных фото: {stats['rejected_total']} "
            f"(не попали в рейтинг)</i>"
        )

    await msg.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


# ── Face Registry ────────────────────────────────────────────

async def handle_face_register(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    caption: str = "",
) -> None:
    """Register a face from a photo."""
    msg = update.message
    chat_id = msg.chat_id
    user_id = msg.from_user.id if msg.from_user else 0

    if not msg.photo:
        return

    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_url = file.file_path

    # Extract person name from caption
    analysis = await vision_structured(
        image_url=image_url,
        prompt=f"""На фото человек. Подпись: "{caption}"

Определи:
1. Имя человека (из подписи)
2. Описание внешности (цвет волос, возраст, особенности)

JSON: {{"person_name": "...", "face_description": "описание внешности"}}""",
        system="Ты определяешь людей на фото. Отвечай JSON.",
    )

    person_name = analysis.get("person_name", "").strip()
    if not person_name:
        await msg.reply_text("❓ Укажите имя человека. Пример: <code>@бот запомни, это Вася</code>",
                             parse_mode=ParseMode.HTML)
        return

    face_desc = analysis.get("face_description", "")

    await execute(
        """INSERT INTO face_registry (chat_id, person_name, user_id, photo_file_id, face_description, registered_by_user_id)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (chat_id, person_name, None, photo.file_id, face_desc, user_id),
    )

    await msg.reply_text(
        f"✅ Запомнил! <b>{person_name}</b>\n📝 {face_desc}",
        parse_mode=ParseMode.HTML,
    )


async def handle_face_identify(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Try to identify a person in a photo using face registry."""
    msg = update.message
    chat_id = msg.chat_id

    if not msg.photo:
        return

    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_url = file.file_path

    # Get registered faces
    faces = await fetch_all(
        "SELECT person_name, face_description FROM face_registry WHERE chat_id = ?",
        (chat_id,),
    )

    if not faces:
        await msg.reply_text("📸 В базе лиц пока никого нет. Используйте: <code>@бот запомни, это Вася</code>",
                             parse_mode=ParseMode.HTML)
        return

    faces_list = "\n".join(f"- {f['person_name']}: {f['face_description']}" for f in faces)

    result = await vision_analyze(
        image_url=image_url,
        prompt=f"""Сравни человека на фото с базой известных людей:

{faces_list}

Кто это может быть? Если не уверен — скажи честно.""",
    )

    await msg.reply_text(result, parse_mode=ParseMode.HTML if "<" in result else None)


# ── Receipt Parsing ──────────────────────────────────────────

async def handle_receipt(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    caption: str = "",
) -> None:
    """Parse a receipt/check photo and add to expenses."""
    msg = update.message
    chat_id = msg.chat_id
    user_id = msg.from_user.id if msg.from_user else 0
    username = msg.from_user.username or "" if msg.from_user else ""

    if not msg.photo:
        return

    photo = msg.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_url = file.file_path

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # Parse receipt
    receipt = await vision_structured(
        image_url=image_url,
        prompt=f"""Распознай чек/квитанцию на фото.

Подпись: "{caption}"

Извлеки:
- merchant: название магазина/заведения
- date: дата покупки (YYYY-MM-DD)
- items: список товаров [{{"name": "...", "price": число}}]
- total: итого
- currency: валюта (RUB, USD, EUR...)

JSON: {{
  "merchant": "...",
  "date": "YYYY-MM-DD",
  "items": [...],
  "total": число,
  "currency": "RUB",
  "is_receipt": true/false
}}""",
        system="Ты — OCR система для чеков. Извлекай данные максимально точно. Отвечай JSON.",
    )

    if not receipt.get("is_receipt", False):
        await msg.reply_text("🧾 Не удалось распознать чек на этом фото.")
        return

    # Format response
    merchant = receipt.get("merchant", "Неизвестно")
    try:
        total = float(receipt.get("total", 0) or 0)
        if not (0 <= total <= 1_000_000):
            raise ValueError(f"total out of range: {total}")
    except (TypeError, ValueError):
        await msg.reply_text("🧾 Не удалось распознать сумму чека. Попробуйте ещё раз.")
        return
    currency = receipt.get("currency", "RUB")
    items = receipt.get("items", [])
    date = receipt.get("date", "")

    lines = [f"🧾 <b>Чек распознан!</b>\n"]
    lines.append(f"🏪 {merchant}")
    if date:
        lines.append(f"📅 {date}")
    if items:
        lines.append("\n📋 <b>Товары:</b>")
        for item in items[:15]:
            lines.append(f"  • {item.get('name', '?')} — {item.get('price', '?')} {currency}")
    lines.append(f"\n💰 <b>Итого: {total} {currency}</b>")

    # Check if there's an active expense session
    session = await get_active_session(chat_id)
    if session:
        display_name = await get_display_name(user_id, chat_id)
        session_id = session["session_id"]

        # Check for duplicate BEFORE inserting — clean, no timing hacks
        already_added = await is_receipt_already_added(chat_id, session_id, photo.file_id)
        if already_added:
            lines.append(f"\n⚠️ Этот чек уже был добавлен в сессию «{session_id}»")
        else:
            await add_expense(
                chat_id=chat_id,
                session_id=session_id,
                paid_by_user_id=user_id,
                paid_by_name=display_name,
                amount=total,
                description=f"Чек: {merchant}",
                merchant=merchant,
                receipt_date=date,
                currency=currency,
                photo_file_id=photo.file_id,
            )
            lines.append(f"\n✅ Добавлено в расходы сессии «{session_id}»")
            lines.append(f"👤 Оплатил: {display_name}")

    response_text = "\n".join(lines)
    await msg.reply_text(response_text, parse_mode=ParseMode.HTML)

    # Save bot's response to history
    await _save_bot_response(chat_id, f"[бот чек] {response_text}")


async def _save_bot_response(chat_id: int, text: str) -> None:
    """Save bot's own response to message history for conversation memory."""
    try:
        from bot.storage.messages import save_message
        from bot.handlers.messages import _bot_username
        await save_message(
            chat_id=chat_id,
            user_id=0,
            username=_bot_username or "bot",
            message_text=text,
            message_type="bot_response",
        )
    except Exception as e:
        log.warning(f"Failed to save bot response to history: {e}")


# ── Photo message router ────────────────────────────────────

async def handle_photo_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Route photo messages based on caption intent."""
    msg = update.message
    if not msg.photo:
        return

    caption = msg.caption or ""
    chat_id = msg.chat_id

    # Save to history
    from bot.storage.messages import save_message
    await save_message(
        chat_id=chat_id,
        user_id=msg.from_user.id if msg.from_user else 0,
        username=msg.from_user.username or "" if msg.from_user else "",
        message_text=f"[фото] {caption}" if caption else "[фото]",
        message_id=msg.message_id,
        message_type="photo",
    )

    # Check if bot is mentioned in caption (or private chat)
    from bot.handlers.messages import _is_mention, _strip_mention
    from telegram.constants import ChatType
    is_private = msg.chat.type == ChatType.PRIVATE

    if is_private:
        # In private chats, always process photos
        clean_caption = _strip_mention(caption) if caption else ""
    elif not caption or not _is_mention(caption, msg.caption_entities):
        return  # Just record, don't process in groups without @mention
    else:
        clean_caption = _strip_mention(caption)

    # Detect intent for photo
    from bot.services.intent import detect_intent
    intent_data = await detect_intent(clean_caption, has_photo=True)
    intent = intent_data.get("intent", "general")

    if intent == "fish_analyze":
        await handle_fish_photo(update, context, clean_caption)
    elif intent == "face_register":
        await handle_face_register(update, context, clean_caption)
    elif intent == "face_identify":
        await handle_face_identify(update, context)
    elif intent == "receipt":
        await handle_receipt(update, context, clean_caption)
    else:
        # Default: analyze the photo
        photo = msg.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        prompt = f"Опиши что на фото. Контекст: {clean_caption}" if clean_caption else "Подробно опиши что изображено на этом фото. Если видишь животных — определи породу/вид. Если видишь людей — опиши. Если видишь текст — прочитай."
        result = await vision_analyze(
            image_url=file.file_path,
            prompt=prompt,
        )
        await msg.reply_text(result)

        # Save bot's response to history so it's available for follow-up questions
        await _save_bot_response(chat_id, f"[бот анализ фото] {result}")
