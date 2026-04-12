"""
Expense tracking handler: start/add/close sessions, calculate debts.
"""

import uuid
from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from bot.services.ai import structured_extraction
from bot.storage.expenses import (
    create_session, get_active_session, close_session,
    add_expense, get_session_expenses, calculate_debts,
    set_session_participants, get_session_participants,
)
from bot.storage.users import get_display_name
from bot.utils.logging import get_logger

log = get_logger("handlers.expenses")


async def handle_expense_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    intent_data: dict,
) -> None:
    """Route expense commands."""
    intent = intent_data.get("intent", "")

    if intent == "expense_start":
        await _start_session(update, context, query)
    elif intent == "expense_add":
        await _add_expense(update, context, query)
    elif intent == "expense_close":
        await _close_session(update, context)
    elif intent == "expense_status":
        await _show_status(update, context)


async def _start_session(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """Start a new expense tracking session.

    Accepts an optional participant list in the query so that split_among=NULL
    is resolved correctly from the start (not just from payers seen so far).
    Example: "@бот начинаем считать расходы: Вася, Петя, Катя"
    """
    msg = update.message
    chat_id = msg.chat_id
    user_id = msg.from_user.id if msg.from_user else 0

    existing = await get_active_session(chat_id)
    if existing:
        await msg.reply_text(
            f"⚠️ Уже есть активная сессия: <b>{existing['session_id']}</b>\n"
            f"Закройте её: <code>@бот закрой расходы</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Extract session name and optional participants
    extraction = await structured_extraction(
        prompt=f"""Из запроса на создание сессии расходов извлеки:
1. session_name — название поездки/события (если есть), иначе null
2. participants — список имён участников через запятую (если перечислены), иначе []

Запрос: "{query}"

Примеры:
- "начинаем считать расходы поездка на море Вася Петя Катя" → session_name="поездка на море", participants=["Вася","Петя","Катя"]
- "начинаем считать расходы" → session_name=null, participants=[]
- "новая сессия Дача" → session_name="Дача", participants=[]

JSON: {{"session_name": "..." или null, "participants": [...]}}""",
        system="Извлеки данные о сессии расходов. Отвечай JSON.",
    )

    session_name = extraction.get("session_name") or f"trip-{uuid.uuid4().hex[:6]}"
    participants = extraction.get("participants") or []

    # Always add the creator's display name to participants
    from bot.storage.users import get_display_name
    creator_name = await get_display_name(user_id, chat_id)
    if creator_name and creator_name not in participants:
        participants.insert(0, creator_name)

    await create_session(chat_id, session_name, participants if participants else None)

    parts_line = (
        f"👥 Участники: {', '.join(participants)}\n" if participants else ""
    )

    await msg.reply_text(
        f"✅ <b>Сессия расходов создана!</b>\n\n"
        f"📋 ID: <code>{session_name}</code>\n"
        f"{parts_line}\n"
        f"Добавляйте расходы:\n"
        f"• <code>@бот я заплатил 5000 за продукты</code>\n"
        f"• <code>@бот Вася заплатил 3000 за бензин</code>\n"
        f"• Отправьте фото чека с @упоминанием\n\n"
        f"Закрыть: <code>@бот закрой расходы</code>",
        parse_mode=ParseMode.HTML,
    )


async def _add_expense(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """Add an expense to the active session."""
    msg = update.message
    chat_id = msg.chat_id
    user_id = msg.from_user.id if msg.from_user else 0

    session = await get_active_session(chat_id)
    if not session:
        await msg.reply_text(
            "❌ Нет активной сессии расходов.\n"
            "Начните: <code>@бот начинаем считать расходы</code>",
            parse_mode=ParseMode.HTML,
        )
        return

    # Extract expense data
    expense_data = await structured_extraction(
        prompt=f"""Извлеки данные о расходе из сообщения:

"{query}"

JSON: {{
  "paid_by": "кто заплатил (имя)",
  "amount": число (сумма),
  "description": "за что",
  "currency": "RUB/USD/EUR",
  "split_among": ["имя1", "имя2"] или null (если на всех)
}}

Если "я заплатил" — paid_by = "автор сообщения".""",
        system="Извлеки данные о расходе. Отвечай JSON.",
    )

    amount = expense_data.get("amount", 0)
    if not amount or amount <= 0:
        await msg.reply_text("❌ Не удалось определить сумму. Пример: <code>@бот я заплатил 5000 за продукты</code>",
                             parse_mode=ParseMode.HTML)
        return

    paid_by = expense_data.get("paid_by", "")
    if paid_by in ("автор сообщения", "я", ""):
        paid_by = await get_display_name(user_id, chat_id)

    description = expense_data.get("description", "")
    currency = expense_data.get("currency", "RUB")
    split = expense_data.get("split_among")

    await add_expense(
        chat_id=chat_id,
        session_id=session["session_id"],
        paid_by_user_id=user_id,
        paid_by_name=paid_by,
        amount=amount,
        description=description,
        currency=currency,
        split_among=split,
    )

    await msg.reply_text(
        f"✅ <b>Расход записан!</b>\n\n"
        f"👤 {paid_by} заплатил {amount} {currency}\n"
        f"📝 {description}",
        parse_mode=ParseMode.HTML,
    )


async def _close_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Close the active session and show final settlements."""
    msg = update.message
    chat_id = msg.chat_id

    session = await get_active_session(chat_id)
    if not session:
        await msg.reply_text("❌ Нет активной сессии расходов.")
        return

    debts = await calculate_debts(chat_id, session["session_id"])
    await close_session(chat_id, session["session_id"])

    lines = [f"📊 <b>Итоги: {session['session_id']}</b>\n"]
    lines.append(f"💰 Всего потрачено: {debts['total']} руб.\n")

    if debts["settlements"]:
        lines.append("<b>💸 Кто кому должен:</b>")
        for s in debts["settlements"]:
            lines.append(f"  {s['from']} → {s['to']}: <b>{s['amount']} руб.</b>")
    else:
        lines.append("✅ Все расчёты равны!")

    lines.append(f"\n✅ Сессия закрыта.")

    await msg.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def _show_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show current expense status."""
    msg = update.message
    chat_id = msg.chat_id

    session = await get_active_session(chat_id)
    if not session:
        await msg.reply_text("❌ Нет активной сессии расходов.")
        return

    expenses = await get_session_expenses(chat_id, session["session_id"])
    debts = await calculate_debts(chat_id, session["session_id"])

    lines = [f"📊 <b>Текущие расходы: {session['session_id']}</b>\n"]
    lines.append(f"💰 Всего: {debts['total']} руб.\n")

    if expenses:
        lines.append("<b>📋 Расходы:</b>")
        for e in expenses[-10:]:
            lines.append(f"  • {e['paid_by_name']}: {e['amount']} руб. — {e['description']}")

    if debts["settlements"]:
        lines.append("\n<b>💸 Текущие долги:</b>")
        for s in debts["settlements"]:
            lines.append(f"  {s['from']} → {s['to']}: {s['amount']} руб.")

    await msg.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)
