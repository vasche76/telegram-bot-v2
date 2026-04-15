"""
Async SQLite storage layer with WAL mode, FTS5 full-text search,
and automatic schema migrations.

FIXES vs original:
- WAL checkpoint (TRUNCATE) on every connection open to prevent WAL from growing
  unbounded (observed: bot.db-wal was 3.3MB with only 64 rows — never checkpointed).
- _write_lock added to serialise writes — prevents concurrent writes from multiple
  coroutines corrupting the WAL under high message load.
- get_db() now detects stale/closed connections and reconnects automatically.
- Periodic auto-checkpoint every 200 writes to keep WAL small.
"""

import aiosqlite
import asyncio
from pathlib import Path
from typing import Any, Optional

from bot.config import DATABASE_PATH
from bot.utils.logging import get_logger

log = get_logger("storage")

_db: Optional[aiosqlite.Connection] = None
_connect_lock: Optional[asyncio.Lock] = None
_write_lock: Optional[asyncio.Lock] = None
_write_count: int = 0
CHECKPOINT_EVERY = 200  # checkpoint WAL every N writes


def _get_connect_lock() -> asyncio.Lock:
    global _connect_lock
    if _connect_lock is None:
        _connect_lock = asyncio.Lock()
    return _connect_lock


def _get_write_lock() -> asyncio.Lock:
    global _write_lock
    if _write_lock is None:
        _write_lock = asyncio.Lock()
    return _write_lock


async def _is_connection_alive(db: aiosqlite.Connection) -> bool:
    """Quick ping to verify the connection is still usable."""
    try:
        await db.execute("SELECT 1")
        return True
    except Exception:
        return False


async def get_db() -> aiosqlite.Connection:
    """Get or create the singleton database connection (with reconnect on failure)."""
    global _db
    # Fast path: connection exists and is alive
    if _db is not None:
        if await _is_connection_alive(_db):
            return _db
        else:
            log.warning("Database connection lost — reconnecting...")
            try:
                await _db.close()
            except Exception:
                pass
            _db = None

    async with _get_connect_lock():
        if _db is not None and await _is_connection_alive(_db):
            return _db  # Another coroutine already reconnected

        db_path = Path(DATABASE_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _db = await aiosqlite.connect(str(db_path))
        _db.row_factory = aiosqlite.Row
        await _db.execute("PRAGMA journal_mode=WAL")
        await _db.execute("PRAGMA busy_timeout=5000")
        await _db.execute("PRAGMA foreign_keys=ON")
        await _db.execute("PRAGMA synchronous=NORMAL")
        # Checkpoint WAL immediately on connect — merges any stale WAL pages.
        # This fixes the observed 3.3MB WAL with only 64 rows.
        await _db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        log.info(f"Database connected + WAL checkpointed: {db_path}")
    return _db


async def close_db() -> None:
    """Close the database connection."""
    global _db
    if _db:
        await _db.close()
        _db = None
        log.info("Database connection closed")


async def execute(sql: str, params: tuple = ()) -> aiosqlite.Cursor:
    """Execute a single SQL statement (serialised via write lock)."""
    global _write_count
    async with _get_write_lock():
        db = await get_db()
        cursor = await db.execute(sql, params)
        await db.commit()
        _write_count += 1
        if _write_count % CHECKPOINT_EVERY == 0:
            try:
                await db.execute("PRAGMA wal_checkpoint(PASSIVE)")
                log.debug(f"Auto-checkpoint at {_write_count} writes")
            except Exception as e:
                log.warning(f"Auto-checkpoint failed: {e}")
        return cursor


async def execute_many(sql: str, params_list: list[tuple]) -> None:
    """Execute a SQL statement with multiple parameter sets."""
    async with _get_write_lock():
        db = await get_db()
        await db.executemany(sql, params_list)
        await db.commit()


async def fetch_one(sql: str, params: tuple = ()) -> Optional[dict]:
    """Fetch a single row as dict."""
    db = await get_db()
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def fetch_all(sql: str, params: tuple = ()) -> list[dict]:
    """Fetch all rows as list of dicts."""
    db = await get_db()
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def fetch_scalar(sql: str, params: tuple = ()) -> Any:
    """Fetch a single scalar value."""
    db = await get_db()
    cursor = await db.execute(sql, params)
    row = await cursor.fetchone()
    return row[0] if row else None


# ── Schema & Migrations ─────────────────────────────────────

SCHEMA_VERSION = 4  # Increment when adding migrations

MIGRATIONS = {
    1: """
    -- Core tables (v1)
    CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER NOT NULL,
        chat_id INTEGER NOT NULL,
        username TEXT,
        first_name TEXT,
        last_name TEXT,
        display_name TEXT,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (user_id, chat_id)
    );

    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        message_id INTEGER,
        user_id INTEGER,
        username TEXT,
        message_text TEXT,
        message_type TEXT DEFAULT 'text',
        reply_to_message_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
    CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(chat_id, user_id);

    CREATE TABLE IF NOT EXISTS user_aliases (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        alias TEXT NOT NULL,
        canonical_name TEXT NOT NULL,
        created_by TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(chat_id, alias)
    );

    CREATE TABLE IF NOT EXISTS expense_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        session_id TEXT UNIQUE NOT NULL,
        status TEXT DEFAULT 'active',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        closed_at DATETIME
    );

    CREATE TABLE IF NOT EXISTS expenses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        session_id TEXT NOT NULL,
        paid_by_user_id INTEGER,
        paid_by_name TEXT,
        amount REAL NOT NULL,
        currency TEXT DEFAULT 'RUB',
        description TEXT,
        merchant TEXT,
        receipt_date TEXT,
        split_among TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS face_registry (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        person_name TEXT NOT NULL,
        user_id INTEGER,
        photo_file_id TEXT,
        face_description TEXT,
        registered_by_user_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS catches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        user_id INTEGER,
        person_name TEXT,
        fish_species TEXT,
        fish_count INTEGER DEFAULT 1,
        estimated_weight_kg REAL,
        estimated_length_cm REAL,
        confidence TEXT DEFAULT 'medium',
        photo_file_id TEXT,
        analysis_text TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS locations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        user_id INTEGER NOT NULL,
        username TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(chat_id, user_id)
    );

    CREATE TABLE IF NOT EXISTS weather_subs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER UNIQUE NOT NULL,
        latitude REAL,
        longitude REAL,
        enabled INTEGER DEFAULT 1,
        last_sent DATETIME
    );

    CREATE TABLE IF NOT EXISTS photo_schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER NOT NULL,
        target_user TEXT,
        target_user_id INTEGER,
        send_time TEXT NOT NULL,
        photo_count INTEGER DEFAULT 5,
        search_query TEXT,
        enabled INTEGER DEFAULT 1,
        last_sent_date TEXT,
        created_by TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER NOT NULL
    );
    INSERT INTO schema_version (version) VALUES (1);
    """,

    3: """
    -- v3: Add participants tracking to expense_sessions and receipt dedup.
    -- This fixes the "split among who?" bug where NULL split_among was resolved
    -- to list(paid.keys()) — the set of payers so far, not actual participants.
    ALTER TABLE expense_sessions ADD COLUMN participants TEXT;
    ALTER TABLE expenses ADD COLUMN photo_file_id TEXT;
    UPDATE schema_version SET version = 3;
    """,

    4: """
    -- v4: Fish vision pipeline columns.
    -- object_type: what Stage A detected (whole_fish/lure/fish_part/fry/no_fish).
    -- species_confidence: numeric 0.0-1.0 from Stage B (replaces text 'high/medium/low').
    -- is_valid_catch: 1 = reliable, 0 = rejected/uncertain (only 1 enters leaderboard).
    -- rejection_reason: why the catch was rejected (for debugging bad photos).
    -- fish_vision_version: which pipeline version produced this record.
    ALTER TABLE catches ADD COLUMN object_type TEXT DEFAULT 'whole_fish';
    ALTER TABLE catches ADD COLUMN species_confidence REAL DEFAULT 0.7;
    ALTER TABLE catches ADD COLUMN is_valid_catch INTEGER DEFAULT 1;
    ALTER TABLE catches ADD COLUMN rejection_reason TEXT;
    ALTER TABLE catches ADD COLUMN fish_vision_version INTEGER DEFAULT 1;
    -- Backfill: all existing catches are assumed valid (they were recorded pre-pipeline)
    UPDATE catches SET is_valid_catch = 1, object_type = 'whole_fish' WHERE is_valid_catch IS NULL;
    UPDATE schema_version SET version = 4;
    """,

    2: """
    -- FTS5 full-text search index (v2)
    CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
        message_text,
        content='messages',
        content_rowid='id',
        tokenize='unicode61'
    );

    -- Triggers to keep FTS in sync
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, message_text) VALUES (new.id, new.message_text);
    END;

    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, message_text) VALUES('delete', old.id, old.message_text);
    END;

    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, message_text) VALUES('delete', old.id, old.message_text);
        INSERT INTO messages_fts(messages_fts, rowid, message_text) VALUES (new.id, new.message_text);
    END;

    -- Rebuild FTS index from existing messages
    INSERT INTO messages_fts(messages_fts) VALUES('rebuild');

    UPDATE schema_version SET version = 2;
    """,
}


async def run_migrations() -> None:
    """Run pending database migrations."""
    db = await get_db()

    # Check if schema_version table exists
    row = await fetch_one(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
    )

    current_version = 0
    if row:
        v = await fetch_scalar("SELECT version FROM schema_version LIMIT 1")
        current_version = v or 0

    if current_version >= SCHEMA_VERSION:
        log.info(f"Database schema is up to date (v{current_version})")
        return

    for version in range(current_version + 1, SCHEMA_VERSION + 1):
        if version in MIGRATIONS:
            log.info(f"Running migration v{version}...")
            await db.executescript(MIGRATIONS[version])
            log.info(f"Migration v{version} complete")

    await db.commit()
    log.info(f"Database migrated to v{SCHEMA_VERSION}")


async def migrate_from_old_db(old_db_path: str) -> None:
    """Import data from the old bot.py SQLite database."""
    if not Path(old_db_path).exists():
        log.warning(f"Old database not found: {old_db_path}")
        return

    log.info(f"Migrating data from {old_db_path}...")
    old = await aiosqlite.connect(old_db_path)
    old.row_factory = aiosqlite.Row
    db = await get_db()

    try:
        # Migrate messages
        cursor = await old.execute(
            "SELECT chat_id, user_id, username, message_text, timestamp FROM messages ORDER BY id"
        )
        rows = await cursor.fetchall()
        count = 0
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO messages (chat_id, user_id, username, message_text, created_at) VALUES (?,?,?,?,?)",
                (r["chat_id"], r["user_id"], r["username"], r["message_text"], r["timestamp"])
            )
            count += 1
        log.info(f"Migrated {count} messages")

        # Migrate user_aliases
        cursor = await old.execute("SELECT * FROM user_aliases")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO user_aliases (chat_id, alias, canonical_name, created_by, created_at) VALUES (?,?,?,?,?)",
                (r["chat_id"], r["alias"], r["canonical_name"], r["created_by"], r["timestamp"])
            )
        log.info(f"Migrated {len(rows)} aliases")

        # Migrate face_registry
        cursor = await old.execute("SELECT * FROM face_registry")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO face_registry (chat_id, person_name, photo_file_id, face_description, created_at) VALUES (?,?,?,?,?)",
                (r["chat_id"], r["person_name"], r["photo_file_id"], r["face_description"], r["timestamp"])
            )
        log.info(f"Migrated {len(rows)} face records")

        # Migrate catches
        cursor = await old.execute("SELECT * FROM catches")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO catches (chat_id, person_name, fish_species, estimated_weight_kg, estimated_length_cm, photo_file_id, analysis_text, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (r["chat_id"], r["person_name"], r["fish_species"], r["estimated_weight_kg"],
                 r["estimated_length_cm"], r["photo_file_id"], r["analysis_text"], r["timestamp"])
            )
        log.info(f"Migrated {len(rows)} catches")

        # Migrate weather_subs
        cursor = await old.execute("SELECT * FROM weather_subs")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO weather_subs (chat_id, latitude, longitude, enabled, last_sent) VALUES (?,?,?,?,?)",
                (r["chat_id"], r["latitude"], r["longitude"], r["enabled"], r["last_sent"])
            )
        log.info(f"Migrated {len(rows)} weather subscriptions")

        # Migrate photo_schedules
        cursor = await old.execute("SELECT * FROM photo_schedules")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO photo_schedules (chat_id, target_user, target_user_id, send_time, photo_count, search_query, enabled, last_sent_date, created_by) VALUES (?,?,?,?,?,?,?,?,?)",
                (r["chat_id"], r["target_user"], r["target_user_id"], r["send_time"],
                 r["photo_count"], r["search_query"], r["enabled"], r["last_sent_date"], r["created_by"])
            )
        log.info(f"Migrated {len(rows)} photo schedules")

        # Migrate expenses and sessions
        cursor = await old.execute("SELECT * FROM expense_sessions")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO expense_sessions (chat_id, session_id, status, created_at, closed_at) VALUES (?,?,?,?,?)",
                (r["chat_id"], r["session_id"], r["status"], r["created_at"], r["closed_at"])
            )

        cursor = await old.execute("SELECT * FROM expenses")
        rows = await cursor.fetchall()
        for r in rows:
            await db.execute(
                "INSERT OR IGNORE INTO expenses (chat_id, session_id, paid_by_name, amount, description, split_among, created_at) VALUES (?,?,?,?,?,?,?)",
                (r["chat_id"], r["session_id"], r["paid_by"], r["amount"], r["description"], r["split_among"], r["timestamp"])
            )

        # Rebuild FTS index after migration
        await db.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")

        await db.commit()
        log.info("Data migration complete!")

    except Exception as e:
        log.error(f"Migration error: {e}")
        raise
    finally:
        await old.close()


async def init_db() -> None:
    """Initialize the database: connect and run migrations."""
    await get_db()
    await run_migrations()
    log.info("Database initialized successfully")
