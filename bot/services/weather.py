"""
Weather service using Open-Meteo (free, no API key) with geocoding.
"""

import httpx
from typing import Optional
from bot.utils.logging import get_logger

log = get_logger("services.weather")

# Weather code descriptions
WMO_CODES = {
    0: "☀️ Ясно", 1: "🌤 Малооблачно", 2: "⛅ Переменная облачность",
    3: "☁️ Пасмурно", 45: "🌫 Туман", 48: "🌫 Изморозь",
    51: "🌦 Лёгкая морось", 53: "🌦 Морось", 55: "🌧 Сильная морось",
    61: "🌧 Небольшой дождь", 63: "🌧 Дождь", 65: "🌧 Сильный дождь",
    66: "🌨 Ледяной дождь", 67: "🌨 Сильный ледяной дождь",
    71: "🌨 Небольшой снег", 73: "🌨 Снег", 75: "❄️ Сильный снег",
    77: "❄️ Снежная крупа", 80: "🌦 Ливень", 81: "🌧 Сильный ливень",
    82: "⛈ Очень сильный ливень", 85: "🌨 Снегопад", 86: "❄️ Сильный снегопад",
    95: "⛈ Гроза", 96: "⛈ Гроза с градом", 99: "⛈ Сильная гроза с градом",
}


async def geocode(location_name: str) -> Optional[dict]:
    """
    Geocode a location name to coordinates.
    Returns {name, latitude, longitude, country} or None.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location_name, "count": 1, "language": "ru", "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        if not results:
            return None
        r = results[0]
        return {
            "name": r.get("name", location_name),
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            "country": r.get("country", ""),
            "admin1": r.get("admin1", ""),
        }


async def get_weather(
    latitude: float,
    longitude: float,
    location_name: str = "",
) -> dict:
    """
    Get current weather and 3-day forecast.
    Returns structured weather data.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_direction_10m,pressure_msl,cloud_cover",
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,sunrise,sunset",
                "hourly": "temperature_2m,precipitation_probability,weather_code,wind_speed_10m",
                "timezone": "auto",
                "forecast_days": 3,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    current = data.get("current", {})
    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    return {
        "location": location_name,
        "latitude": latitude,
        "longitude": longitude,
        "current": {
            "temperature": current.get("temperature_2m"),
            "feels_like": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "weather_code": current.get("weather_code"),
            "weather_desc": WMO_CODES.get(current.get("weather_code", 0), "Неизвестно"),
            "wind_speed": current.get("wind_speed_10m"),
            "wind_direction": current.get("wind_direction_10m"),
            "pressure": current.get("pressure_msl"),
            "cloud_cover": current.get("cloud_cover"),
        },
        "daily": {
            "dates": daily.get("time", []),
            "weather_codes": daily.get("weather_code", []),
            "temp_max": daily.get("temperature_2m_max", []),
            "temp_min": daily.get("temperature_2m_min", []),
            "precipitation": daily.get("precipitation_sum", []),
            "wind_max": daily.get("wind_speed_10m_max", []),
            "sunrise": daily.get("sunrise", []),
            "sunset": daily.get("sunset", []),
        },
        "hourly": hourly,
    }


def format_weather_message(weather: dict, include_fishing: bool = True) -> str:
    """Format weather data into a readable Telegram message."""
    c = weather["current"]
    d = weather["daily"]
    loc = weather.get("location", "")

    lines = []
    lines.append(f"🌍 <b>Погода: {loc}</b>\n")

    # Current
    lines.append(f"<b>Сейчас:</b> {c['weather_desc']}")
    lines.append(f"🌡 Температура: {c['temperature']}°C (ощущается {c['feels_like']}°C)")
    lines.append(f"💧 Влажность: {c['humidity']}%")
    lines.append(f"💨 Ветер: {c['wind_speed']} км/ч ({_wind_direction_text(c['wind_direction'])})")
    if c.get("pressure"):
        lines.append(f"📊 Давление: {round(c['pressure'] * 0.75006, 1)} мм рт.ст.")
    if c.get("precipitation") and c["precipitation"] > 0:
        lines.append(f"🌧 Осадки: {c['precipitation']} мм")
    lines.append(f"☁️ Облачность: {c.get('cloud_cover', 0)}%")

    # Daily forecast
    lines.append("\n<b>📅 Прогноз на 3 дня:</b>")
    day_names = ["Сегодня", "Завтра", "Послезавтра"]
    for i in range(min(3, len(d.get("dates", [])))):
        code = d["weather_codes"][i] if i < len(d.get("weather_codes", [])) else 0
        desc = WMO_CODES.get(code, "")
        t_max = d["temp_max"][i] if i < len(d.get("temp_max", [])) else "?"
        t_min = d["temp_min"][i] if i < len(d.get("temp_min", [])) else "?"
        precip = d["precipitation"][i] if i < len(d.get("precipitation", [])) else 0
        wind = d["wind_max"][i] if i < len(d.get("wind_max", [])) else "?"

        name = day_names[i] if i < len(day_names) else d["dates"][i]
        line = f"  <b>{name}:</b> {desc}, {t_min}..{t_max}°C"
        if precip and precip > 0:
            line += f", 🌧 осадки {precip}мм"
        else:
            line += ", без осадков"
        line += f", ветер до {wind} км/ч"
        lines.append(line)

    # Sunrise/sunset
    if d.get("sunrise") and d["sunrise"]:
        sunrise = d["sunrise"][0].split("T")[1] if "T" in str(d["sunrise"][0]) else d["sunrise"][0]
        sunset = d["sunset"][0].split("T")[1] if "T" in str(d["sunset"][0]) else d["sunset"][0]
        lines.append(f"\n🌅 Восход: {sunrise} | 🌇 Закат: {sunset}")

    # Fishing recommendation
    if include_fishing:
        lines.append("\n" + _fishing_recommendation(c, d))

    return "\n".join(lines)


def _wind_direction_text(degrees: Optional[float]) -> str:
    """Convert wind direction degrees to text."""
    if degrees is None:
        return "?"
    dirs = ["С", "ССВ", "СВ", "ВСВ", "В", "ВЮВ", "ЮВ", "ЮЮВ",
            "Ю", "ЮЮЗ", "ЮЗ", "ЗЮЗ", "З", "ЗСЗ", "СЗ", "ССЗ"]
    idx = round(degrees / 22.5) % 16
    return dirs[idx]


def _fishing_recommendation(current: dict, daily: dict) -> str:
    """Generate fishing recommendation based on weather."""
    temp = current.get("temperature", 15)
    wind = current.get("wind_speed", 0)
    pressure = current.get("pressure")
    precip = current.get("precipitation", 0)

    score = 5  # Base score out of 10
    tips = []

    # Pressure analysis
    if pressure:
        mmhg = pressure * 0.75006
        if 755 <= mmhg <= 765:
            score += 2
            tips.append("давление стабильное — хорошо для клёва")
        elif mmhg < 750:
            score -= 1
            tips.append("низкое давление — рыба может быть пассивной")
        elif mmhg > 770:
            score -= 1
            tips.append("высокое давление — клёв может быть слабым")

    # Wind analysis
    if wind < 15:
        score += 1
        tips.append("слабый ветер — комфортная рыбалка")
    elif wind > 30:
        score -= 2
        tips.append("сильный ветер — рыбалка затруднена")

    # Temperature
    if 10 <= temp <= 25:
        score += 1
    elif temp < 0:
        tips.append("мороз — одевайтесь теплее")

    # Precipitation
    if precip > 5:
        score -= 1
        tips.append("осадки — возьмите дождевик")

    score = max(1, min(10, score))
    emoji = "🟢" if score >= 7 else "🟡" if score >= 4 else "🔴"

    result = f"🎣 <b>Рыбалка:</b> {emoji} {score}/10"
    if tips:
        result += "\n💡 " + "; ".join(tips)

    return result
