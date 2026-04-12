FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project (exclude .env, data/, __pycache__)
COPY . .

# Create data directory (for SQLite + logs)
RUN mkdir -p /app/data

# Environment
ENV PYTHONUNBUFFERED=1
# FIX: was DB_PATH, but config reads DATABASE_PATH
ENV DATABASE_PATH=/app/data/bot.db
ENV LOG_LEVEL=INFO

# Health check: verify the bot process is running
HEALTHCHECK --interval=120s --timeout=15s --start-period=30s --retries=3 \
    CMD python3 -c "import sqlite3; sqlite3.connect('/app/data/bot.db').execute('SELECT 1')" || exit 1

# Run
CMD ["python3", "main.py"]
