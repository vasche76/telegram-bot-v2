"""
Message storage and retrieval with FTS5 full-text search.
"""

from typing import Optional
from bot.storage.database import execute, fetch_all, fetch_one, fetch_scalar
from bot.utils.logging import get_logger

log = get_logger("storage.messages")


async def save_message(
    chat_id: int,
    user_id: int,
    username: str,
    message_text: str,
    message_id: Optional[int] = None,
    message_type: str = "text",
    reply_to: Optional[int] = None,
) -> int:
    """Save a message and return its row ID."""
    cursor = await execute(
        """INSERT INTO messages (chat_id, message_id, user_id, username, message_text, message_type, reply_to_message_id)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (chat_id, message_id, user_id, username, message_text, message_type, reply_to),
    )
    log.debug(f"Saved message from {username} in chat {chat_id}")
    return cursor.lastrowid


async def get_recent_messages(chat_id: int, limit: int = 50) -> list[dict]:
    """Get recent messages for a chat, ordered by time."""
    return await fetch_all(
        """SELECT user_id, username, message_text, message_type, created_at
           FROM messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?""",
        (chat_id, limit),
    )


async def search_messages_fts(chat_id: int, query: str, limit: int = 15) -> list[dict]:
    """Full-text search in chat messages using FTS5.

    FIX: Added LIKE fallback for empty FTS results (not just errors).
    This handles Russian morphological forms where "Астрахань" != "Астрахани"
    in the unicode61 tokenizer (no stemming by default).
    Search chain:
    1. FTS5 phrase match (exact token)
    2. FTS5 word OR match (individual words, more permissive)
    3. LIKE substring match (handles inflected/partial forms, slowest)
    """
    safe_query = query.replace('"', '""')
    try:
        # Step 1: Exact phrase
        results = await fetch_all(
            """SELECT m.user_id, m.username, m.message_text, m.created_at,
                      rank
               FROM messages_fts fts
               JOIN messages m ON m.id = fts.rowid
               WHERE fts.message_text MATCH ? AND m.chat_id = ?
               ORDER BY rank
               LIMIT ?""",
            (f'"{safe_query}"', chat_id, limit),
        )

        if not results:
            # Step 2: Individual words OR (catches multi-word queries)
            words = [w for w in query.split() if len(w) > 2]
            if words:
                fts_query = " OR ".join(f'"{w}"' for w in words)
                results = await fetch_all(
                    """SELECT m.user_id, m.username, m.message_text, m.created_at,
                              rank
                       FROM messages_fts fts
                       JOIN messages m ON m.id = fts.rowid
                       WHERE fts.message_text MATCH ? AND m.chat_id = ?
                       ORDER BY rank
                       LIMIT ?""",
                    (fts_query, chat_id, limit),
                )

        if not results:
            # Step 3: LIKE fallback (handles inflected forms, no stopwords)
            log.debug(f"FTS returned no results for '{query}', falling back to LIKE")
            results = await fetch_all(
                """SELECT user_id, username, message_text, created_at
                   FROM messages WHERE chat_id = ? AND message_text LIKE ?
                   ORDER BY created_at DESC LIMIT ?""",
                (chat_id, f"%{query}%", limit),
            )

        return results

    except Exception as e:
        log.warning(f"FTS search failed: {e}, falling back to LIKE")
        return await fetch_all(
            """SELECT user_id, username, message_text, created_at
               FROM messages WHERE chat_id = ? AND message_text LIKE ?
               ORDER BY created_at DESC LIMIT ?""",
            (chat_id, f"%{query}%", limit),
        )


async def get_message_count(chat_id: int) -> int:
    """Get total message count for a chat."""
    return await fetch_scalar(
        "SELECT COUNT(*) FROM messages WHERE chat_id = ?", (chat_id,)
    ) or 0


async def get_user_messages(chat_id: int, user_id: int, limit: int = 50) -> list[dict]:
    """Get messages from a specific user in a chat."""
    return await fetch_all(
        """SELECT message_text, created_at FROM messages
           WHERE chat_id = ? AND user_id = ?
           ORDER BY created_at DESC LIMIT ?""",
        (chat_id, user_id, limit),
    )
