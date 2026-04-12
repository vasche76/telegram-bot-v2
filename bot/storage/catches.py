"""
Fish catch records and statistics.
"""

from typing import Optional
from bot.storage.database import execute, fetch_all, fetch_one, fetch_scalar
from bot.utils.logging import get_logger

log = get_logger("storage.catches")


async def save_catch(
    chat_id: int,
    user_id: Optional[int],
    person_name: str,
    fish_species: str,
    fish_count: int = 1,
    weight_kg: Optional[float] = None,
    length_cm: Optional[float] = None,
    confidence: str = "medium",
    photo_file_id: Optional[str] = None,
    analysis_text: Optional[str] = None,
) -> int:
    """Save a fish catch record."""
    cursor = await execute(
        """INSERT INTO catches (chat_id, user_id, person_name, fish_species, fish_count,
           estimated_weight_kg, estimated_length_cm, confidence, photo_file_id, analysis_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (chat_id, user_id, person_name, fish_species, fish_count,
         weight_kg, length_cm, confidence, photo_file_id, analysis_text),
    )
    log.info(f"Saved catch: {person_name} caught {fish_species} ({weight_kg}kg)")
    return cursor.lastrowid


async def get_catches_by_person(chat_id: int, person_name: str) -> list[dict]:
    """Get all catches for a person in a chat."""
    return await fetch_all(
        """SELECT * FROM catches WHERE chat_id = ? AND LOWER(person_name) = LOWER(?)
           ORDER BY created_at DESC""",
        (chat_id, person_name),
    )


async def get_chat_leaderboard(chat_id: int) -> list[dict]:
    """Get catch leaderboard for a chat."""
    return await fetch_all(
        """SELECT person_name,
                  COUNT(*) as total_catches,
                  SUM(fish_count) as total_fish,
                  SUM(estimated_weight_kg) as total_weight_kg,
                  MAX(estimated_weight_kg) as biggest_catch_kg,
                  GROUP_CONCAT(DISTINCT fish_species) as species_list
           FROM catches WHERE chat_id = ?
           GROUP BY LOWER(person_name)
           ORDER BY total_weight_kg DESC""",
        (chat_id,),
    )


async def get_recent_catches(chat_id: int, limit: int = 10) -> list[dict]:
    """Get recent catches in a chat."""
    return await fetch_all(
        "SELECT * FROM catches WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
        (chat_id, limit),
    )
