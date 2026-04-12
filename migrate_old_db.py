"""
Migration script: import data from old bot's chat_history.db into new format.
Usage: python3 migrate_old_db.py /path/to/old/chat_history.db

This script:
1. Reads messages, users, locations, catches, expenses from old DB
2. Creates new DB with v2 schema
3. Imports all data preserving timestamps and relationships
"""

import asyncio
import sqlite3
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_old_tables(conn):
    """Get list of tables in old database."""
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return [row[0] for row in cursor.fetchall()]


def get_old_columns(conn, table):
    """Get list of column names in a table."""
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cursor.fetchall()]


async def migrate(old_db_path: str) -> None:
    from bot.storage.database import init_db, get_db

    if not os.path.exists(old_db_path):
        print(f"❌ ERROR: Old database not found: {old_db_path}")
        sys.exit(1)

    # Delete existing new DB to start fresh
    from bot.config import DATABASE_PATH
    from pathlib import Path
    db_path = Path(DATABASE_PATH)
    if db_path.exists():
        os.remove(db_path)
        print("🗑️  Removed old v2 database for clean migration")

    # Initialize new database
    await init_db()
    print("✅ New database initialized")

    # Connect to old database
    old_conn = sqlite3.connect(old_db_path)
    old_conn.row_factory = sqlite3.Row

    old_tables = get_old_tables(old_conn)
    print(f"📋 Old database tables: {', '.join(old_tables)}")

    db = await get_db()

    # ── Migrate messages ─────────────────────────────────
    print("\n📝 Migrating messages...")
    if "messages" in old_tables:
        old_cols = get_old_columns(old_conn, "messages")
        print(f"   Old columns: {', '.join(old_cols)}")

        cursor = old_conn.execute("SELECT * FROM messages ORDER BY id ASC")
        messages = cursor.fetchall()
        count = 0
        errors = 0
        for msg in messages:
            try:
                chat_id = msg["chat_id"]
                user_id = msg["user_id"] if "user_id" in old_cols else 0
                username = msg["username"] if "username" in old_cols else ""
                message_text = msg["message_text"] if "message_text" in old_cols else ""
                timestamp = msg["timestamp"] if "timestamp" in old_cols else ""
                msg_id = msg["message_id"] if "message_id" in old_cols else None

                await db.execute(
                    """INSERT INTO messages
                       (chat_id, user_id, username, message_text, message_id, message_type, created_at)
                       VALUES (?, ?, ?, ?, ?, 'text', ?)""",
                    (chat_id, user_id, username, message_text, msg_id, timestamp),
                )
                count += 1
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"   ⚠️ Error on message: {e}")
        await db.commit()
        print(f"   ✅ {count}/{len(messages)} messages migrated ({errors} errors)")
    else:
        print("   ⚠️ No messages table found")

    # ── Migrate user_aliases ─────────────────────────────
    print("\n🏷️  Migrating user aliases...")
    if "user_aliases" in old_tables:
        old_cols = get_old_columns(old_conn, "user_aliases")
        cursor = old_conn.execute("SELECT * FROM user_aliases")
        aliases = cursor.fetchall()
        count = 0
        for a in aliases:
            try:
                ts_col = "timestamp" if "timestamp" in old_cols else "created_at"
                await db.execute(
                    """INSERT OR IGNORE INTO user_aliases
                       (chat_id, alias, canonical_name, created_by, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (a["chat_id"], a["alias"], a["canonical_name"],
                     a["created_by"] if "created_by" in old_cols else "",
                     a[ts_col] if ts_col in old_cols else ""),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} aliases migrated")
    else:
        print("   ⚠️ No user_aliases table found")

    # ── Migrate locations ────────────────────────────────
    print("\n📍 Migrating locations...")
    if "locations" in old_tables:
        cursor = old_conn.execute("SELECT * FROM locations")
        locations = cursor.fetchall()
        count = 0
        for loc in locations:
            try:
                await db.execute(
                    """INSERT OR REPLACE INTO locations
                       (chat_id, user_id, username, latitude, longitude, updated_at)
                       VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                    (loc["chat_id"], loc.get("user_id", 0),
                     loc["username"] if "username" in get_old_columns(old_conn, "locations") else "",
                     loc["latitude"], loc["longitude"]),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} locations migrated")
    else:
        print("   ⚠️ No locations table found")

    # ── Migrate catches ──────────────────────────────────
    print("\n🐟 Migrating catches...")
    if "catches" in old_tables:
        old_cols = get_old_columns(old_conn, "catches")
        print(f"   Old columns: {', '.join(old_cols)}")
        cursor = old_conn.execute("SELECT * FROM catches")
        catches = cursor.fetchall()
        count = 0
        for c in catches:
            try:
                # Handle different column names between old and new
                weight = None
                for col in ["estimated_weight_kg", "weight_kg", "weight"]:
                    if col in old_cols:
                        weight = c[col]
                        break

                length = None
                for col in ["estimated_length_cm", "length_cm", "length"]:
                    if col in old_cols:
                        length = c[col]
                        break

                ts = None
                for col in ["timestamp", "created_at"]:
                    if col in old_cols:
                        ts = c[col]
                        break

                await db.execute(
                    """INSERT INTO catches
                       (chat_id, user_id, person_name, fish_species, fish_count,
                        estimated_weight_kg, estimated_length_cm, photo_file_id, analysis_text, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        c["chat_id"],
                        c["user_id"] if "user_id" in old_cols else 0,
                        c["person_name"] if "person_name" in old_cols else "",
                        c["fish_species"] if "fish_species" in old_cols else "",
                        c["fish_count"] if "fish_count" in old_cols else 1,
                        weight,
                        length,
                        c["photo_file_id"] if "photo_file_id" in old_cols else "",
                        c["analysis_text"] if "analysis_text" in old_cols else "",
                        ts,
                    ),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} catches migrated")
    else:
        print("   ⚠️ No catches table found")

    # ── Migrate expense_sessions ─────────────────────────
    print("\n💳 Migrating expense sessions...")
    if "expense_sessions" in old_tables:
        cursor = old_conn.execute("SELECT * FROM expense_sessions")
        sessions = cursor.fetchall()
        count = 0
        for s in sessions:
            try:
                await db.execute(
                    """INSERT OR IGNORE INTO expense_sessions
                       (chat_id, session_id, status, created_at, closed_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (s["chat_id"], s["session_id"], s["status"],
                     s["created_at"], s.get("closed_at")),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} sessions migrated")
    else:
        print("   ⚠️ No expense_sessions table found")

    # ── Migrate expenses ─────────────────────────────────
    print("\n💰 Migrating expenses...")
    if "expenses" in old_tables:
        old_cols = get_old_columns(old_conn, "expenses")
        print(f"   Old columns: {', '.join(old_cols)}")
        cursor = old_conn.execute("SELECT * FROM expenses")
        expenses = cursor.fetchall()
        count = 0
        for e in expenses:
            try:
                ts = None
                for col in ["timestamp", "created_at"]:
                    if col in old_cols:
                        ts = e[col]
                        break

                paid_by = ""
                for col in ["paid_by", "paid_by_name"]:
                    if col in old_cols:
                        paid_by = e[col]
                        break

                await db.execute(
                    """INSERT INTO expenses
                       (chat_id, session_id, paid_by_name, amount, description,
                        split_among, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        e["chat_id"],
                        e["session_id"] if "session_id" in old_cols else "default",
                        paid_by,
                        e["amount"] if "amount" in old_cols else 0,
                        e["description"] if "description" in old_cols else "",
                        e["split_among"] if "split_among" in old_cols else "",
                        ts,
                    ),
                )
                count += 1
            except Exception as e2:
                print(f"   ⚠️ Error: {e2}")
        await db.commit()
        print(f"   ✅ {count} expenses migrated")
    else:
        print("   ⚠️ No expenses table found")

    # ── Migrate face_registry ────────────────────────────
    print("\n👤 Migrating face registry...")
    if "face_registry" in old_tables:
        old_cols = get_old_columns(old_conn, "face_registry")
        cursor = old_conn.execute("SELECT * FROM face_registry")
        faces = cursor.fetchall()
        count = 0
        for f in faces:
            try:
                await db.execute(
                    """INSERT OR IGNORE INTO face_registry
                       (chat_id, person_name, photo_file_id, face_description)
                       VALUES (?, ?, ?, ?)""",
                    (
                        f["chat_id"],
                        f["person_name"] if "person_name" in old_cols else "",
                        f["photo_file_id"] if "photo_file_id" in old_cols else "",
                        f["face_description"] if "face_description" in old_cols else "",
                    ),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} faces migrated")
    else:
        print("   ⚠️ No face_registry table found")

    # ── Migrate weather_subs ─────────────────────────────
    print("\n🌤️  Migrating weather subscriptions...")
    if "weather_subs" in old_tables:
        cursor = old_conn.execute("SELECT * FROM weather_subs")
        subs = cursor.fetchall()
        count = 0
        for s in subs:
            try:
                await db.execute(
                    """INSERT OR IGNORE INTO weather_subs
                       (chat_id, latitude, longitude, enabled, last_sent)
                       VALUES (?, ?, ?, ?, ?)""",
                    (s["chat_id"], s.get("latitude"), s.get("longitude"),
                     s.get("enabled", 1), s.get("last_sent")),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} weather subs migrated")
    else:
        print("   ⚠️ No weather_subs table found")

    # ── Migrate photo_schedules ──────────────────────────
    print("\n📸 Migrating photo schedules...")
    if "photo_schedules" in old_tables:
        old_cols = get_old_columns(old_conn, "photo_schedules")
        cursor = old_conn.execute("SELECT * FROM photo_schedules")
        schedules = cursor.fetchall()
        count = 0
        for s in schedules:
            try:
                await db.execute(
                    """INSERT OR IGNORE INTO photo_schedules
                       (chat_id, target_user, target_user_id, send_time,
                        photo_count, search_query, enabled, last_sent_date, created_by)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        s["chat_id"],
                        s["target_user"] if "target_user" in old_cols else "",
                        s["target_user_id"] if "target_user_id" in old_cols else None,
                        s["send_time"],
                        s["photo_count"] if "photo_count" in old_cols else 5,
                        s["search_query"] if "search_query" in old_cols else "",
                        s["enabled"] if "enabled" in old_cols else 1,
                        s["last_sent_date"] if "last_sent_date" in old_cols else None,
                        s["created_by"] if "created_by" in old_cols else "",
                    ),
                )
                count += 1
            except Exception as e:
                print(f"   ⚠️ Error: {e}")
        await db.commit()
        print(f"   ✅ {count} photo schedules migrated")
    else:
        print("   ⚠️ No photo_schedules table found")

    # ── Rebuild FTS index ────────────────────────────────
    print("\n🔍 Rebuilding full-text search index...")
    try:
        await db.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        await db.commit()
        print("   ✅ FTS index rebuilt")
    except Exception as e:
        print(f"   ⚠️ FTS rebuild error: {e}")

    old_conn.close()

    # ── Verify ───────────────────────────────────────────
    from bot.storage.database import fetch_scalar
    msg_count = await fetch_scalar("SELECT COUNT(*) FROM messages")
    print(f"\n🎉 Migration complete! Total messages in new DB: {msg_count}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 migrate_old_db.py /path/to/old/chat_history.db")
        sys.exit(1)

    asyncio.run(migrate(sys.argv[1]))
