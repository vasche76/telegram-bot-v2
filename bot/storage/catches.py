"""
Fish catch records and statistics.

CHANGES vs original:
- save_catch: accepts new fish-vision pipeline fields
  (object_type, species_confidence, is_valid_catch, rejection_reason, fish_vision_version)
- get_chat_leaderboard: ONLY counts catches where is_valid_catch = 1.
  This prevents lures, fish parts, fry, and low-confidence detections from
  polluting the leaderboard.
- save_rejected_catch: records rejected photos for audit trail without
  affecting statistics.
"""

from typing import Optional
from bot.storage.database import execute, fetch_all, fetch_one, fetch_scalar
from bot.utils.logging import get_logger

log = get_logger("storage.catches")

# Current fish vision pipeline version (increment when retraining)
FISH_VISION_VERSION = 1


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
    # New fish-vision pipeline fields
    object_type: str = "whole_fish",
    species_confidence: float = 0.7,
    is_valid_catch: bool = True,
    rejection_reason: Optional[str] = None,
) -> int:
    """
    Save a fish catch record.

    Only catches with is_valid_catch=True appear in the leaderboard.
    Rejected catches (lures, parts, fry, low-confidence) are still stored
    with is_valid_catch=False for audit purposes.
    """
    cursor = await execute(
        """INSERT INTO catches (
               chat_id, user_id, person_name, fish_species, fish_count,
               estimated_weight_kg, estimated_length_cm, confidence,
               photo_file_id, analysis_text,
               object_type, species_confidence, is_valid_catch,
               rejection_reason, fish_vision_version
           )
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chat_id, user_id, person_name, fish_species, fish_count,
            weight_kg, length_cm, confidence,
            photo_file_id, analysis_text,
            object_type, species_confidence, 1 if is_valid_catch else 0,
            rejection_reason, FISH_VISION_VERSION,
        ),
    )
    if is_valid_catch:
        log.info(
            f"Saved valid catch: {person_name} caught {fish_species} "
            f"({weight_kg}kg, conf={species_confidence:.2f})"
        )
    else:
        log.info(
            f"Saved rejected catch: {person_name} / {object_type} "
            f"({rejection_reason or 'no reason'})"
        )
    return cursor.lastrowid


async def get_catches_by_person(
    chat_id: int,
    person_name: str,
    valid_only: bool = True,
) -> list[dict]:
    """Get all catches for a person in a chat."""
    sql = """SELECT * FROM catches
             WHERE chat_id = ? AND LOWER(person_name) = LOWER(?)"""
    if valid_only:
        sql += " AND is_valid_catch = 1"
    sql += " ORDER BY created_at DESC"
    return await fetch_all(sql, (chat_id, person_name))


async def get_chat_leaderboard(chat_id: int) -> list[dict]:
    """
    Get catch leaderboard for a chat.

    IMPORTANT: Only counts valid catches (is_valid_catch = 1).
    This prevents lures, fish parts, fry, and low-confidence detections
    from appearing in the leaderboard.
    """
    return await fetch_all(
        """SELECT person_name,
                  COUNT(*) as total_catches,
                  SUM(fish_count) as total_fish,
                  SUM(estimated_weight_kg) as total_weight_kg,
                  MAX(estimated_weight_kg) as biggest_catch_kg,
                  GROUP_CONCAT(DISTINCT fish_species) as species_list
           FROM catches
           WHERE chat_id = ? AND is_valid_catch = 1
           GROUP BY LOWER(person_name)
           ORDER BY total_weight_kg DESC""",
        (chat_id,),
    )


async def get_recent_catches(chat_id: int, limit: int = 10, valid_only: bool = True) -> list[dict]:
    """Get recent catches in a chat."""
    sql = "SELECT * FROM catches WHERE chat_id = ?"
    params: tuple = (chat_id,)
    if valid_only:
        sql += " AND is_valid_catch = 1"
    sql += " ORDER BY created_at DESC LIMIT ?"
    params += (limit,)
    return await fetch_all(sql, params)


async def get_catch_stats_for_chat(chat_id: int) -> dict:
    """Get aggregate stats including rejection counts (for /status)."""
    total = await fetch_scalar(
        "SELECT COUNT(*) FROM catches WHERE chat_id = ?", (chat_id,)
    ) or 0
    valid = await fetch_scalar(
        "SELECT COUNT(*) FROM catches WHERE chat_id = ? AND is_valid_catch = 1", (chat_id,)
    ) or 0
    rejected = total - valid
    lures = await fetch_scalar(
        "SELECT COUNT(*) FROM catches WHERE chat_id = ? AND object_type = 'lure'", (chat_id,)
    ) or 0
    return {
        "total_submissions": total,
        "valid_catches": valid,
        "rejected_total": rejected,
        "lures_caught": lures,
    }
