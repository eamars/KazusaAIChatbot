"""Validate and seed a character personality profile into MongoDB.

Usage:
    python -m scripts.load_character_profile personalities/asuna.json
    python -m scripts.load_character_profile personalities/asuna.json --force

The service performs the same seed-or-verify operation during native startup.
``--force`` explicitly replaces static profile fields while preserving runtime
state.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from kazusa_ai_chatbot.character_profile import load_character_profile_seed
from kazusa_ai_chatbot.db import (
    close_db,
    db_bootstrap,
    ensure_character_profile_seed,
    get_character_profile,
    save_character_profile,
)

logger = logging.getLogger(__name__)


async def main(path: Path, force: bool) -> None:
    await db_bootstrap()
    seed = load_character_profile_seed(path.resolve())
    seed_result = await ensure_character_profile_seed(seed)

    existing = await get_character_profile()
    if existing.get("name") and not force:
        logger.info(
            f"Character profile '{existing['name']}' {seed_result}; "
            "static fields were preserved."
        )
    elif force:
        await save_character_profile(dict(seed))
        logger.info(
            f"Character profile '{seed['name']}' static fields overwritten; "
            "runtime state preserved."
        )
    else:
        logger.info(f"Character profile '{seed['name']}' inserted.")

    await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load character profile into MongoDB")
    parser.add_argument("path", type=Path, help="Path to the personality JSON file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing profile")
    args = parser.parse_args()

    asyncio.run(main(args.path, args.force))
