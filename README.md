# 🤖 Telegram Bot v2 — Модульный рыболовный бот

Production-quality Telegram-бот для группового чата с функциями:
- 💬 Умные ответы на @упоминания (RAG: история чата + интернет + GPT)
- 🌤 Погода по названию места или GPS
- 📷 Поиск и отправка фото (Pexels + DuckDuckGo)
- 🐟 Распознавание рыбы на фото (вид, вес, длина)
- 👤 Реестр лиц (запомни/узнай)
- 🧾 Распознавание чеков и учёт расходов
- 🔍 Поиск по истории чата (FTS5)
- 🎤 Транскрипция голосовых сообщений
- ⏰ Расписание отправки фото

## 📁 Структура проекта

```
telegram-bot-v2/
├── main.py                 ← Точка входа
├── .env                    ← API-ключи (создать из .env.example)
├── requirements.txt        ← Зависимости
├── Dockerfile              ← Docker-образ
├── docker-compose.yml      ← Docker Compose
├── migrate_old_db.py       ← Миграция из старой БД
├── bot/
│   ├── config.py           ← Конфигурация из .env
│   ├── scheduler.py        ← Планировщик задач
│   ├── handlers/           ← Обработчики сообщений
│   │   ├── messages.py     ← Главный роутер (текст)
│   │   ├── weather.py      ← Погода
│   │   ├── photos.py       ← Фото
│   │   ├── vision.py       ← Рыба, лица, чеки
│   │   ├── expenses.py     ← Расходы
│   │   ├── search.py       ← Поиск по истории
│   │   └── help.py         ← Справка
│   ├── services/           ← Бизнес-логика
│   │   ├── ai.py           ← OpenAI (GPT, Vision, Whisper)
│   │   ├── intent.py       ← Определение интента
│   │   ├── response.py     ← RAG-генерация ответов
│   │   ├── weather.py      ← Погодный сервис
│   │   ├── photos.py       ← Поиск фото
│   │   └── web_search.py   ← Веб-поиск
│   ├── storage/            ← Работа с БД
│   │   ├── database.py     ← Async SQLite + миграции + FTS5
│   │   ├── messages.py     ← Хранение сообщений
│   │   ├── users.py        ← Профили пользователей
│   │   ├── catches.py      ← Уловы
│   │   └── expenses.py     ← Расходы
│   └── utils/
│       └── logging.py      ← Логирование
└── data/                   ← БД и данные (создаётся автоматически)
```

## 🚀 Быстрый старт (iMac / macOS)

### 1. Установите зависимости

```bash
cd ~/Desktop/telegram-bot-v2
pip3 install -r requirements.txt
```

### 2. Создайте .env файл

```bash
cp .env.example .env
```

Отредактируйте `.env` — впишите свои API-ключи:

```
TELEGRAM_TOKEN=ваш_токен_бота
OPENAI_API_KEY=ваш_ключ_openai
PEXELS_API_KEY=ваш_ключ_pexels
```

### 3. Мигрируйте данные из старого бота (опционально)

```bash
python3 migrate_old_db.py ~/Desktop/telegram-bot/chat_history.db
```

### 4. Запустите бота

```bash
python3 main.py
```

## 🐳 Docker (для VPS)

```bash
# Создайте .env файл
cp .env.example .env
nano .env  # впишите ключи

# Запустите
docker-compose up -d

# Логи
docker-compose logs -f

# Остановить
docker-compose down
```

## 🔄 Миграция со старого бота

Скрипт `migrate_old_db.py` перенесёт все данные:
- Историю сообщений
- Сохранённые локации
- Записи об уловах
- Расходы
- Реестр лиц

```bash
python3 migrate_old_db.py /путь/к/старому/chat_history.db
```

## 📋 Что изменилось по сравнению с v1

| Аспект | v1 (bot.py) | v2 (модульный) |
|--------|-------------|----------------|
| Архитектура | 1 файл, 2500 строк | 15+ модулей |
| БД | Синхронный sqlite3 | Async aiosqlite + FTS5 |
| Интенты | Regex | GPT Structured Outputs |
| Поиск | Простой LIKE | FTS5 полнотекстовый |
| Фото | Только Pexels | Pexels + DuckDuckGo + fallback |
| Погода | Только GPS | Название места + GPS + веб |
| Логирование | print() | Structured logging |
| Расходы | Базовый | Сессии + чеки + долги |
| Тесты | Нет | Готово к тестированию |
| Docker | Базовый | Compose + volumes |

## Development Workflow

This repository is maintained with Claude Code and the Compound Engineering Plugin.

For complex or risky work, use:
- `/ce-code-review` — audit without changes
- `/ce-brainstorm` — explore options
- `/ce-plan` — prepare implementation plan
- `/ce-work` — execute planned changes
- `/ce-compound` — save reusable project knowledge

Before modifying production-sensitive areas, run a read-only audit first.

See: `docs/DEVELOPMENT_WORKFLOW.md`
