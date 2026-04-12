"""
Chat history search handler.
"""

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from bot.storage.messages import search_messages_fts
from bot.utils.logging import get_logger

log = get_logger("handlers.search")


async def handle_history_search(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
) -> None:
    """Search chat history using FTS5."""
    msg = update.message
    chat_id = msg.chat_id

    # Clean query from search keywords
    search_terms = query.lower()
    for word in ["найди", "поищи", "искать", "в чате", "в истории", "кто говорил",
                 "когда обсуждали", "кто писал", "найди в чате"]:
        search_terms = search_terms.replace(word, "")
    search_terms = search_terms.strip()

    if not search_terms or len(search_terms) < 2:
        await msg.reply_text("🔍 Укажите, что искать. Например: <code>@бот найди в чате про рыбалку</code>",
                             parse_mode=ParseMode.HTML)
        return

    results = await search_messages_fts(chat_id, search_terms, limit=10)

    if not results:
        await msg.reply_text(f"🔍 По запросу «{search_terms}» ничего не найдено в истории чата.")
        return

    # Format results
    lines = [f"🔍 <b>Результаты поиска: «{search_terms}»</b>\n"]
    for i, r in enumerate(results, 1):
        username = r.get("username", "?")
        text = r.get("message_text", "")[:150]
        date = r.get("created_at", "")[:16]
        lines.append(f"{i}. <b>{username}</b> ({date}):\n   {text}\n")

    response = "\n".join(lines)
    if len(response) > 4000:
        response = response[:4000] + "\n\n... (показаны первые результаты)"

    try:
        await msg.reply_text(response, parse_mode=ParseMode.HTML)
    except Exception:
        await msg.reply_text(response)
