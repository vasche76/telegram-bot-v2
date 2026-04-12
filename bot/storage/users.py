"""
User profile storage and alias management.
Users are identified by (user_id, chat_id) — not by username strings.
"""

from typing import Optional
from bot.storage.database import execute, fetch_all, fetch_one, fetch_scalar
from bot.utils.logging import get_logger

log = get_logger("storage.users")


async def upsert_user(
    user_id: int,
    chat_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    last_name: Optional[str] = None,
) -> None:
    """Create or update a user profile for a specific chat."""
    display_name = " ".join(filter(None, [first_name, last_name])) or username or str(user_id)
    await execute(
        """INSERT INTO users (user_id, chat_id, username, first_name, last_name, display_name, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
           ON CONFLICT(user_id, chat_id) DO UPDATE SET
             username = COALESCE(excluded.username, users.username),
             first_name = COALESCE(excluded.first_name, users.first_name),
             last_name = COALESCE(excluded.last_name, users.last_name),
             display_name = excluded.display_name,
             updated_at = CURRENT_TIMESTAMP""",
        (user_id, chat_id, username, first_name, last_name, display_name),
    )

    # Auto-create aliases
    aliases_to_add = set()
    if username:
        aliases_to_add.add(username.lower())
    if first_name:
        aliases_to_add.add(first_name.lower())
    if display_name:
        aliases_to_add.add(display_name.lower())

    for alias in aliases_to_add:
        if len(alias) >= 2:
            await execute(
                """INSERT OR IGNORE INTO user_aliases (chat_id, alias, canonical_name, created_by)
                   VALUES (?, ?, ?, 'auto')""",
                (chat_id, alias, display_name),
            )


async def get_user(user_id: int, chat_id: int) -> Optional[dict]:
    """Get user profile."""
    return await fetch_one(
        "SELECT * FROM users WHERE user_id = ? AND chat_id = ?",
        (user_id, chat_id),
    )


async def get_display_name(user_id: int, chat_id: int) -> str:
    """Get user display name, falling back to user_id."""
    row = await fetch_one(
        "SELECT display_name FROM users WHERE user_id = ? AND chat_id = ?",
        (user_id, chat_id),
    )
    return row["display_name"] if row else str(user_id)


async def resolve_name(chat_id: int, name: str) -> Optional[str]:
    """Resolve a name/alias to canonical display name."""
    row = await fetch_one(
        "SELECT canonical_name FROM user_aliases WHERE chat_id = ? AND alias = ?",
        (chat_id, name.lower()),
    )
    return row["canonical_name"] if row else None


async def get_all_users(chat_id: int) -> list[dict]:
    """Get all known users in a chat."""
    return await fetch_all(
        "SELECT * FROM users WHERE chat_id = ? ORDER BY display_name",
        (chat_id,),
    )


async def add_alias(chat_id: int, alias: str, canonical_name: str, created_by: str) -> None:
    """Manually add a user alias."""
    await execute(
        """INSERT OR REPLACE INTO user_aliases (chat_id, alias, canonical_name, created_by)
           VALUES (?, ?, ?, ?)""",
        (chat_id, alias.lower(), canonical_name, created_by),
    )
