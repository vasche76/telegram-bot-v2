"""
Weather handler: by location name, GPS coordinates, or web search.
"""

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode

from bot.services.weather import geocode, get_weather, format_weather_message
from bot.services.web_search import search_text, fetch_page_text
from bot.services.ai import chat_completion
from bot.storage.database import fetch_one, execute
from bot.storage.messages import save_message
from bot.utils.logging import get_logger

log = get_logger("handlers.weather")

# Cache of last queried weather location per chat
_last_weather_location: dict[int, dict] = {}


async def handle_weather_query(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    query: str,
    intent_data: dict,
) -> None:
    """Handle weather requests with smart location detection."""
    msg = update.message
    chat_id = msg.chat_id
    location_name = intent_data.get("location")

    # If no location in query, check cache of last queried location for this chat
    if not location_name and chat_id in _last_weather_location:
        cached = _last_weather_location[chat_id]
        log.info(f"Using cached weather location for chat {chat_id}: {cached['name']}")
        weather = await get_weather(cached["latitude"], cached["longitude"], cached["name"])
        text = format_weather_message(weather, include_fishing=True)
        # Add specific answer about rain if user asked about it
        if _asks_about_rain(query):
            rain_answer = _answer_rain_question(weather)
            text = rain_answer + "\n\n" + text
        await msg.reply_text(text, parse_mode=ParseMode.HTML)
        await _save_bot_weather_response(chat_id, text)
        return

    if location_name:
        # Location name found in query — geocode it
        log.info(f"Weather for location: {location_name}")
        geo = await geocode(location_name)
        if geo:
            # Cache this location for follow-up questions
            _last_weather_location[chat_id] = {
                "name": geo["name"],
                "latitude": geo["latitude"],
                "longitude": geo["longitude"],
            }
            weather = await get_weather(geo["latitude"], geo["longitude"], geo["name"])
            text = format_weather_message(weather, include_fishing=True)
            # Add specific answer about rain if user asked about it
            if _asks_about_rain(query):
                rain_answer = _answer_rain_question(weather)
                text = rain_answer + "\n\n" + text
            await msg.reply_text(text, parse_mode=ParseMode.HTML)
            await _save_bot_weather_response(chat_id, text)
            return
        else:
            # Geocoding failed — try web search
            log.info(f"Geocoding failed for '{location_name}', trying web search")
            await _weather_via_web(msg, location_name)
            return

    # No location in query — check saved GPS
    user_id = msg.from_user.id if msg.from_user else 0
    saved_loc = await fetch_one(
        "SELECT latitude, longitude FROM locations WHERE chat_id = ? AND user_id = ?",
        (chat_id, user_id),
    )
    if not saved_loc:
        # Check chat-level location
        saved_loc = await fetch_one(
            "SELECT latitude, longitude FROM locations WHERE chat_id = ? ORDER BY updated_at DESC LIMIT 1",
            (chat_id,),
        )

    if saved_loc:
        weather = await get_weather(saved_loc["latitude"], saved_loc["longitude"], "Сохранённая локация")
        text = format_weather_message(weather, include_fishing=True)
        await msg.reply_text(text, parse_mode=ParseMode.HTML)
        await _save_bot_weather_response(chat_id, text)
        return

    # No location at all — ask user
    await msg.reply_text(
        "🌍 Укажите место для прогноза погоды. Примеры:\n\n"
        "• <code>@бот погода в Москве</code>\n"
        "• <code>@бот погода Териберка</code>\n"
        "• <code>@бот прогноз Енисей</code>\n\n"
        "Или отправьте геолокацию 📍 в чат.",
        parse_mode=ParseMode.HTML,
    )


async def _weather_via_web(msg, location_name: str) -> None:
    """Get weather via web search when geocoding fails."""
    search_query = f"погода {location_name} сейчас прогноз на 3 дня"
    results = await search_text(search_query, max_results=3)

    if not results:
        await msg.reply_text(f"❌ Не удалось найти погоду для «{location_name}».")
        return

    # Fetch content from top results
    web_text = ""
    for r in results[:2]:
        page = await fetch_page_text(r["url"], max_chars=2000)
        if page:
            web_text += f"\n{page}\n"

    if not web_text:
        web_text = "\n".join(r.get("snippet", "") for r in results)

    # Format with GPT
    response = await chat_completion(
        messages=[
            {
                "role": "system",
                "content": """Ты — погодный бот. Из данных веб-поиска составь красивый прогноз погоды.
Формат:
🌍 Погода: [место]
🌡 Температура, ощущается как
💨 Ветер
💧 Влажность
☁️ Облачность
📅 Прогноз на 3 дня
🌅 Восход/закат
🎣 Рекомендация для рыбалки (оценка клёва 1-10)

Используй HTML-теги <b> для выделения. Если данных мало — честно скажи."""
            },
            {"role": "user", "content": f"Данные из интернета о погоде в {location_name}:\n{web_text[:3000]}"},
        ],
        temperature=0.5,
        max_tokens=1500,
    )

    try:
        await msg.reply_text(response, parse_mode=ParseMode.HTML)
    except Exception:
        await msg.reply_text(response)


async def handle_location(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle when user sends GPS location."""
    msg = update.message
    if not msg.location:
        return

    chat_id = msg.chat_id
    user_id = msg.from_user.id if msg.from_user else 0
    username = msg.from_user.username or "" if msg.from_user else ""
    lat = msg.location.latitude
    lon = msg.location.longitude

    # Save location
    await execute(
        """INSERT INTO locations (chat_id, user_id, username, latitude, longitude, updated_at)
           VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(chat_id, user_id) DO UPDATE SET
             latitude = excluded.latitude,
             longitude = excluded.longitude,
             updated_at = CURRENT_TIMESTAMP""",
        (chat_id, user_id, username, lat, lon),
    )

    # Cache this location for follow-up questions
    _last_weather_location[chat_id] = {
        "name": f"📍 {lat:.2f}, {lon:.2f}",
        "latitude": lat,
        "longitude": lon,
    }

    # Send weather for this location
    weather = await get_weather(lat, lon, f"📍 {lat:.2f}, {lon:.2f}")
    text = format_weather_message(weather, include_fishing=True)
    await msg.reply_text(text, parse_mode=ParseMode.HTML)
    await _save_bot_weather_response(chat_id, text)


async def _save_bot_weather_response(chat_id: int, text: str) -> None:
    """Save bot's weather response to history."""
    try:
        from bot.handlers.messages import _bot_username
        await save_message(
            chat_id=chat_id,
            user_id=0,
            username=_bot_username or "bot",
            message_text=f"[бот погода] {text[:2000]}",
            message_type="bot_response",
        )
    except Exception as e:
        log.warning(f"Failed to save weather response: {e}")


def _asks_about_rain(query: str) -> bool:
    """Check if user is asking about rain/precipitation."""
    rain_words = [
        "дождь", "дожд", "ливень", "осадки", "осадк",
        "мокр", "зонт", "промокн", "дожди", "дождик",
        "снег", "снеж", "метель", "гроза", "грозы",
    ]
    lower = query.lower()
    return any(w in lower for w in rain_words)


def _answer_rain_question(weather: dict) -> str:
    """Generate a direct answer about rain/precipitation."""
    from bot.services.weather import WMO_CODES

    daily = weather.get("daily", {})
    precip_list = daily.get("precipitation", [])
    codes = daily.get("weather_codes", [])
    day_names = ["Сегодня", "Завтра", "Послезавтра"]

    rain_days = []
    dry_days = []

    for i in range(min(3, len(precip_list))):
        name = day_names[i] if i < len(day_names) else daily.get("dates", [""])[i]
        precip = precip_list[i] if i < len(precip_list) else 0
        code = codes[i] if i < len(codes) else 0
        desc = WMO_CODES.get(code, "")

        # Rain codes: 51-67 (drizzle/rain), 80-82 (showers), 95-99 (thunderstorm)
        is_rainy = code in range(51, 68) or code in range(80, 83) or code in range(95, 100)
        has_precip = precip and precip > 0

        if is_rainy or has_precip:
            rain_days.append(f"<b>{name}</b>: {desc}, осадки {precip}мм")
        else:
            dry_days.append(name)

    if rain_days:
        answer = "🌧 <b>Да, дождь ожидается:</b>\n" + "\n".join(f"  • {d}" for d in rain_days)
        if dry_days:
            answer += f"\n☀️ Без осадков: {', '.join(dry_days)}"
    else:
        answer = "☀️ <b>Нет, дождь не ожидается</b> в ближайшие 3 дня."

    return answer
