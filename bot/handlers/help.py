"""
Help and command list handler.
"""

from telegram import Update
from telegram.ext import ContextTypes
from telegram.constants import ParseMode


HELP_TEXT = """
🤖 <b>Что я умею</b>

<b>💬 Общение</b>
Упомяните меня @ботом и задайте вопрос — отвечу из истории чата, интернета или общих знаний.

<b>🌤 Погода</b>
• <code>@бот погода в Москве</code>
• <code>@бот прогноз Териберка</code>
• <code>@бот клёв в Мурманске</code>
• Или отправьте 📍 геолокацию

<b>📷 Фото</b>
• <code>@бот пришли фото котиков</code>
• <code>@бот скинь Васе 5 фото щуки</code>
• <code>@бот присылай Пете в 22:30 фото природы</code>

<b>🐟 Рыбалка</b>
• Отправьте фото рыбы с @упоминанием — определю вид, вес, длину
• <code>@бот статистика уловов</code> — рейтинг рыбаков

<b>👤 Лица</b>
• Фото + <code>@бот запомни, это Вася</code>
• Фото + <code>@бот кто это?</code>

<b>🧾 Чеки и расходы</b>
• Фото чека + @упоминание — распознаю и добавлю в расходы
• <code>@бот начинаем считать расходы</code>
• <code>@бот я заплатил 5000 за продукты</code>
• <code>@бот кто сколько должен</code>
• <code>@бот закрой расходы</code>

<b>🔍 Поиск</b>
• <code>@бот найди в чате про рыбалку</code>
• <code>@бот найди в интернете рыболовные катушки</code>

<b>⚙️ Команды</b>
• /help — эта справка
• /weather — погода (нужна геолокация)
• /stats — статистика уловов
• /photo_off — отключить расписание фото
"""


async def send_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help message."""
    await update.message.reply_text(HELP_TEXT.strip(), parse_mode=ParseMode.HTML)
