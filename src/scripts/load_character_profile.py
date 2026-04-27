"""Load a character personality profile from a JSON file into MongoDB.

Usage:
    python -m scripts.load_character_profile personalities/kazusa.json
    python -m scripts.load_character_profile personalities/kazusa.json --force

The first invocation is mandatory before starting the brain service.
Subsequent runs will skip the upload unless --force is provided.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


async def main(path: Path, force: bool) -> None:
    from kazusa_ai_chatbot.db import db_bootstrap, get_character_profile, save_character_profile, close_db

    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    if not isinstance(profile, dict) or not profile.get("name"):
        logger.error("Invalid profile — must be a JSON object with at least a 'name' field")
        sys.exit(1)

    await db_bootstrap()

    existing = await get_character_profile()
    if existing.get("name") and not force:
        logger.info(
            "Character profile '%s' already exists in the database. "
            "Use --force to overwrite.",
            existing.get("name", "(unknown)"),
        )
    else:
        await save_character_profile(profile)
        logger.info("Character profile '%s' saved to database.", profile["name"])

    await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load character profile into MongoDB")
    parser.add_argument("path", type=Path, help="Path to the personality JSON file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing profile")
    args = parser.parse_args()

    asyncio.run(main(args.path, args.force))
