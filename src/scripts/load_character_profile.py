"""Validate and seed or revision-reset a character identity profile.

Usage:
    python -m scripts.load_character_profile personalities/example.json
    python -m scripts.load_character_profile personalities/example.json \
        --force --operator-action-id change-ticket-123

A clean ledger receives immutable revision zero. An existing ledger is
preserved unless ``--force`` names an explicit operator action, in which case
the validated profile becomes a new immutable ``operator_reset`` revision.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from kazusa_ai_chatbot.character_profile import load_character_profile_seed
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import (
    IdentityLedgerNotFoundError,
    close_db,
    create_operator_reset_revision,
    db_bootstrap,
    ensure_operational_character_state,
    ensure_seed_identity,
    get_current_identity,
)


logger = logging.getLogger(__name__)


async def main(
    path: Path,
    *,
    force: bool,
    operator_action_id: str | None,
) -> None:
    """Apply one validated profile through the revision ledger."""

    if force and not operator_action_id:
        raise ValueError(
            "--force requires a nonempty --operator-action-id"
        )
    if not force and operator_action_id:
        raise ValueError(
            "--operator-action-id is valid only together with --force"
        )

    await db_bootstrap()
    try:
        seed = load_character_profile_seed(path)
        await ensure_operational_character_state()
        try:
            current = await get_current_identity(
                character_id=CHARACTER_GLOBAL_USER_ID,
            )
        except IdentityLedgerNotFoundError:
            revision = await ensure_seed_identity(
                character_id=CHARACTER_GLOBAL_USER_ID,
                seed=seed,
            )
            logger.info(
                f"Character identity {seed['name']!r} inserted as "
                f"revision {revision['revision_number']}."
            )
            return

        if not force:
            logger.info(
                f"Character identity revision "
                f"{current['revision_number']} was preserved."
            )
            return

        revision = await create_operator_reset_revision(
            character_id=CHARACTER_GLOBAL_USER_ID,
            identity=seed,
            operator_action_id=operator_action_id,
            correlation_id=uuid4().hex,
        )
        logger.info(
            f"Character identity {seed['name']!r} stored as operator-reset "
            f"revision {revision['revision_number']}."
        )
    finally:
        await close_db()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a canonical character identity into MongoDB"
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the canonical identity JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Create an immutable operator-reset revision",
    )
    parser.add_argument(
        "--operator-action-id",
        help="Required audit action identifier for --force",
    )
    args = parser.parse_args()

    asyncio.run(
        main(
            args.path,
            force=args.force,
            operator_action_id=args.operator_action_id,
        )
    )
