"""Audit, reset, or guarded-restore conversation progress V2 rows."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from kazusa_ai_chatbot.db.script_operations import (
    apply_conversation_progress_v2_migration,
    audit_conversation_progress_v2_rows,
    dry_run_conversation_progress_v2_migration,
    restore_conversation_progress_v1_backup,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


def _parser() -> argparse.ArgumentParser:
    """Build the exact mutually exclusive maintenance CLI."""

    parser = argparse.ArgumentParser(
        description='Conversation progress V2 export-and-reset migration',
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--audit', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--restore-v1', action='store_true')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--backup-input', type=Path)
    parser.add_argument('--backup-output', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    return parser


async def _run(args: argparse.Namespace) -> None:
    """Execute one explicitly selected maintenance mode."""

    timestamp = storage_utc_now_iso()
    if args.dry_run:
        if args.backup_output is None:
            raise ValueError('--dry-run requires --backup-output')
        await dry_run_conversation_progress_v2_migration(
            backup_output=args.backup_output,
            report_output=args.output,
            generated_at=timestamp,
        )
        return
    if args.audit:
        await audit_conversation_progress_v2_rows(
            output=args.output,
            generated_at=timestamp,
        )
        return
    if args.apply:
        if args.input is None or args.backup_input is None:
            raise ValueError('--apply requires --input and --backup-input')
        await apply_conversation_progress_v2_migration(
            dry_run_input=args.input,
            backup_input=args.backup_input,
            output=args.output,
            applied_at=timestamp,
        )
        return
    if args.input is None or args.backup_input is None:
        raise ValueError(
            '--restore-v1 requires --input and --backup-input'
        )
    await restore_conversation_progress_v1_backup(
        apply_input=args.input,
        backup_input=args.backup_input,
        output=args.output,
        restored_at=timestamp,
    )


def main() -> None:
    """Parse command-line arguments and run one maintenance operation."""

    asyncio.run(_run(_parser().parse_args()))


if __name__ == '__main__':
    main()
