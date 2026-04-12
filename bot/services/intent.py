"""
Intent detection using GPT structured outputs.
Replaces regex-based intent parsing with reliable JSON extraction.
"""

from bot.services.ai import structured_extraction
from bot.utils.logging import get_logger

log = get_logger("services.intent")

# Intent types
INTENTS = [
    "weather",          # Weather query (with or without location)
    "photo",            # Send photos
    "photo_schedule",   # Schedule recurring photos
    "fish_analyze",     # Analyze fish in photo
    "face_register",    # Register a face
    "face_identify",    # Identify a person in photo
    "receipt",          # Parse a receipt/check
    "expense_start",    # Start expense tracking
    "expense_add",      # Add an expense
    "expense_close",    # Close expense session
    "expense_status",   # Show expense status
    "catch_stats",      # Show fishing stats
    "search_history",   # Search chat history
    "web_search",       # Search the internet
    "help",             # Show help
    "general",          # General conversation / question
]


async def detect_intent(text: str, has_photo: bool = False) -> dict:
    """
    Detect user intent from message text.
    Returns structured intent data.
    """
    photo_context = ""
    if has_photo:
        photo_context = """
К сообщению приложено фото. Учти это при определении интента:
- Если текст про рыбу/улов/рыбалку + фото → fish_analyze
- Если текст "запомни лицо/это X" + фото → face_register
- Если текст "кто это" + фото → face_identify
- Если текст про чек/счёт/квитанцию + фото → receipt
"""

    result = await structured_extraction(
        prompt=f"""Определи интент пользователя из сообщения в Telegram-чате.

Сообщение: "{text}"
{photo_context}

Возможные интенты:
- weather — запрос погоды (погода, температура, дождь, ветер, клёв, прогноз, восход, закат, снег, осадки, гроза, зонт). ВАЖНО: уточняющие вопросы про погоду ("а дождь будет?", "снег ожидается?") тоже weather, даже без названия места
- photo — отправить фото/картинки (пришли фото, покажи картинку, скинь фотки)
- photo_schedule — настроить расписание отправки фото (присылай каждый день, по расписанию)
- fish_analyze — анализ рыбы на фото (что за рыба, сколько весит)
- face_register — запомнить лицо (запомни, это Вася)
- face_identify — узнать кто на фото (кто это, кто на фото)
- receipt — распознать чек/счёт (чек, квитанция, счёт)
- expense_start — начать учёт расходов (начинаем считать, новая поездка)
- expense_add — добавить расход (я заплатил, потратил)
- expense_close — закрыть учёт (закрыть, подвести итоги)
- expense_status — показать расходы (кто сколько должен, баланс)
- catch_stats — статистика уловов (статистика, рейтинг, кто больше поймал)
- search_history — поиск по истории чата (кто говорил, когда обсуждали, найди в чате)
- web_search — поиск в интернете (найди в интернете, загугли, что такое, обзор, рейтинг)
- help — помощь (помощь, что умеешь, команды)
- general — общий вопрос/разговор (всё остальное)

Также извлеки:
- location: название места (если есть, для погоды)
- target_user: имя адресата (если фото/действие для конкретного человека)
- search_query: что именно искать (для фото/поиска)
- count: количество (фото, результатов)

Ответь JSON:
{{
  "intent": "...",
  "confidence": 0.0-1.0,
  "location": "..." или null,
  "target_user": "..." или null,
  "search_query": "..." или null,
  "count": число или null,
  "reasoning": "краткое пояснение"
}}""",
        system="Ты система определения интентов для Telegram-бота. Отвечай только JSON.",
        temperature=0.2,
    )

    intent = result.get("intent", "general")
    if intent not in INTENTS:
        intent = "general"
    result["intent"] = intent

    log.info(f"Intent: {intent} (conf={result.get('confidence', 0)}) for: {text[:60]}...")
    return result


async def needs_web_search(text: str) -> bool:
    """Quick check if a message needs web search."""
    lower = text.lower()
    web_keywords = [
        "найди в интернете", "поищи в сети", "загугли", "погугли",
        "в интернете", "в инете", "из интернета",
        "найди информацию", "всю информацию", "всё о",
        "расскажи всё о", "что такое", "кто такой", "кто такая",
        "обзор", "рейтинг", "топ ", "лучшие", "отзывы",
        "где купить", "сколько стоит", "цена",
        "новости", "последние новости",
    ]
    return any(kw in lower for kw in web_keywords)
