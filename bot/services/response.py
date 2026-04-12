"""
Response generation with RAG (chat history search + web search)
and proper conversation memory.

The bot now builds a full messages array from recent chat history,
so GPT sees the conversation flow (including bot's own replies and photo analyses).
"""

from typing import Optional
from bot.services.ai import chat_completion, structured_extraction
from bot.services.web_search import search_text, fetch_page_text
from bot.services.intent import needs_web_search
from bot.storage.messages import search_messages_fts, get_recent_messages
from bot.utils.logging import get_logger

log = get_logger("services.response")

# Bot username — set from main.py during startup
_bot_username: str = ""


def set_response_bot_username(username: str) -> None:
    """Set the bot username for identifying bot messages in history."""
    global _bot_username
    _bot_username = username.lower()


async def generate_response(
    query: str,
    chat_id: int,
    user_name: str,
    intent_data: Optional[dict] = None,
) -> str:
    """
    Generate a response using RAG pipeline with full conversation memory:
    1. Build conversation history from recent messages (user + bot)
    2. Search chat history for relevant older messages (FTS5)
    3. Optionally search web
    4. Generate response with full context
    """

    # ── Step 1: Build conversation messages from recent history ──
    # Get last 50 messages to build a proper conversation
    recent = await get_recent_messages(chat_id, limit=50)
    
    # Build OpenAI-style messages array from chat history
    conversation_messages = _build_conversation(recent)

    # ── Step 2: Search chat history for relevant older context ──
    rag_context_parts = []
    history_results = await search_messages_fts(chat_id, query, limit=10)
    if history_results:
        # Filter out messages already in recent (avoid duplication)
        recent_texts = {r['message_text'][:100] for r in (recent or [])}
        unique_history = [
            r for r in history_results
            if r['message_text'][:100] not in recent_texts
        ]
        if unique_history:
            history_text = "\n".join(
                f"[{r['created_at']}] {r['username']}: {r['message_text'][:200]}"
                for r in unique_history[:8]
            )
            rag_context_parts.append(f"=== Релевантные сообщения из старой истории чата ===\n{history_text}")
            log.info(f"Found {len(unique_history)} unique history results for '{query[:40]}'")

    # ── Step 3: Web search if needed ──
    web_context = ""
    if await needs_web_search(query) or (intent_data and intent_data.get("intent") == "web_search"):
        log.info(f"Performing web search for: {query[:60]}")
        web_results = await search_text(query, max_results=5)
        if web_results:
            web_parts = []
            for r in web_results[:3]:
                snippet = r.get("snippet", "")
                if not snippet or len(snippet) < 100:
                    page_text = await fetch_page_text(r["url"], max_chars=3000)
                    if page_text:
                        snippet = page_text[:1000]
                web_parts.append(f"[{r['title']}]({r['url']})\n{snippet}")

            web_context = "\n\n".join(web_parts)
            rag_context_parts.append(f"=== Результаты из интернета ===\n{web_context}")
            log.info(f"Found {len(web_results)} web results")

    # ── Step 4: Build the final messages array ──
    system_prompt = """Ты — умный ассистент в Telegram-чате. Твои правила:

1. ПАМЯТЬ: Ты помнишь ВСЮ историю диалога. Если пользователь ссылается на предыдущие сообщения, фото, или ответы — используй эту информацию.
2. ФОТО: Если в истории есть твой анализ фото — ты ПОМНИШЬ что было на фото. Когда спрашивают про "то фото" или "картинку которую я прислал" — используй свой предыдущий ответ.
3. ИСТОЧНИКИ: Сначала ищи ответ в истории диалога, потом в дополнительном контексте, потом из общих знаний.
4. ЦИТИРОВАНИЕ: Если информация из истории чата — укажи кто и когда это говорил. Если из интернета — дай ссылку.
5. ЧЕСТНОСТЬ: Если не уверен — скажи об этом. Не выдумывай факты.
6. СТИЛЬ: Отвечай по-русски, дружелюбно, но информативно. Используй эмодзи умеренно.
7. КРАТКОСТЬ: Не пиши длинные простыни. Отвечай по существу.
8. КОНТЕКСТ: Учитывай контекст чата — это может быть личный диалог или группа друзей."""

    messages = [{"role": "system", "content": system_prompt}]

    # Add RAG context as a system message if available
    if rag_context_parts:
        rag_text = "\n\n".join(rag_context_parts)
        messages.append({
            "role": "system",
            "content": f"Дополнительный контекст (старая история и веб-поиск):\n{rag_text}"
        })

    # Add conversation history (user and assistant messages)
    messages.extend(conversation_messages)

    # Add the current user query
    messages.append({"role": "user", "content": f"{user_name}: {query}"})

    # Safety: trim messages if total estimated tokens would exceed model limit.
    # gpt-4o-mini context = 128k tokens; rough estimate: 4 chars ≈ 1 token.
    # Keep system + RAG + last N conversation turns if total is too long.
    _total_chars = sum(len(m.get("content", "")) for m in messages)
    if _total_chars > 300_000:  # ~75k tokens — leave room for 2k response
        log.warning(f"Message context too large ({_total_chars} chars), trimming history")
        # Keep system messages (first 1-2) + last user query, trim middle
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_query_msg = messages[-1]
        conv_msgs = [m for m in messages if m["role"] != "system"][:-1]
        # Keep only last 10 conversation turns
        conv_msgs = conv_msgs[-10:]
        messages = system_msgs + conv_msgs + [user_query_msg]

    log.info(f"Sending {len(messages)} messages to GPT ({len(conversation_messages)} from history)")

    # ── Step 5: Generate response ──
    response = await chat_completion(
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
    )

    # ── Step 6: Self-verification (lightweight, only for long responses with sources) ──
    if len(response) > 200 and (web_context or history_results):
        response = await _self_verify(query, response, rag_context_parts)

    return response


def _build_conversation(recent_messages: list[dict]) -> list[dict]:
    """
    Convert recent DB messages into OpenAI messages array.
    Bot messages become "assistant", all others become "user".
    Groups consecutive messages from the same role.
    """
    if not recent_messages:
        return []

    # recent_messages are DESC order, reverse to chronological
    messages_asc = list(reversed(recent_messages))

    openai_messages = []
    
    for msg in messages_asc:
        username = (msg.get("username") or "").lower()
        text = msg.get("message_text", "")
        msg_type = msg.get("message_type", "text")
        
        if not text or text.strip() == "":
            continue

        # Determine if this is a bot message
        is_bot_msg = False
        if _bot_username and _bot_username in username:
            is_bot_msg = True
        # Also check for common bot markers
        if username in ("bot", "assistant") or text.startswith("[бот]"):
            is_bot_msg = True

        if is_bot_msg:
            role = "assistant"
            content = text
        else:
            role = "user"
            display_name = msg.get("username") or "user"
            if msg_type == "photo":
                content = f"{display_name}: {text}"
            else:
                content = f"{display_name}: {text}"

        # Merge consecutive messages with the same role
        if openai_messages and openai_messages[-1]["role"] == role:
            openai_messages[-1]["content"] += f"\n{content}"
        else:
            openai_messages.append({"role": role, "content": content})

    # Limit to last ~30 messages to stay within token limits
    # but keep enough for good context
    if len(openai_messages) > 30:
        openai_messages = openai_messages[-30:]

    return openai_messages


async def _self_verify(query: str, response: str, context_parts: list[str]) -> str:
    """
    Self-verify the response quality.
    Check for hallucinations and add uncertainty markers.
    """
    try:
        verification = await structured_extraction(
            prompt=f"""Проверь качество ответа бота.

Вопрос: {query}
Ответ бота: {response}
Доступный контекст: {' '.join(context_parts)[:2000]}

Оцени:
1. Есть ли в ответе утверждения, которые НЕ подтверждаются контекстом? (hallucination)
2. Ответ по существу вопроса?
3. Нужно ли добавить оговорку о неуверенности?

JSON: {{"is_good": true/false, "needs_disclaimer": true/false, "disclaimer": "..." или null}}""",
            temperature=0.1,
        )

        if verification.get("needs_disclaimer") and verification.get("disclaimer"):
            response += f"\n\n⚠️ {verification['disclaimer']}"

    except Exception as e:
        log.warning(f"Self-verification failed: {e}")

    return response


async def generate_simple_response(
    query: str,
    chat_history: str = "",
    system: str = "",
) -> str:
    """Generate a simple response without RAG pipeline."""
    if not system:
        system = "Ты — дружелюбный ассистент в Telegram-чате. Отвечай по-русски, кратко и по делу."

    messages = [{"role": "system", "content": system}]
    if chat_history:
        messages.append({"role": "user", "content": f"Контекст чата:\n{chat_history}"})
    messages.append({"role": "user", "content": query})

    return await chat_completion(messages=messages, temperature=0.7)
