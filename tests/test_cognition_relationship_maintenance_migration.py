"""Focused dry-run coverage for the relationship maintenance migration."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from kazusa_ai_chatbot.cognition_shared.state_models import (
    build_acquaintance_user_state,
)
from scripts import migrate_cognition_relationship_maintenance as migration


@pytest.mark.asyncio
async def test_relationship_maintenance_migration_dry_run_performs_no_writes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Create a reviewable backup/report while keeping the database read-only."""

    state = build_acquaintance_user_state(
        global_user_id="migration-dry-run-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    old_state = deepcopy(state)
    del old_state["relationship"]["relationship_maintenance"]
    calls = {"writes": 0}

    async def load_rows() -> list[dict[str, object]]:
        """Return one legacy row from the read-only owner helper."""

        return [{
            "global_user_id": "migration-dry-run-user",
            "cognition_state": old_state,
        }]

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )

    report = await migration.run_dry_run(
        backup_path=tmp_path / "backup.json",
        report_path=tmp_path / "report.json",
        generated_at="2026-08-18T01:00:00Z",
    )

    assert calls["writes"] == 0
    assert report["writes_performed"] == 0
    assert (tmp_path / "backup.json").exists()
    assert (tmp_path / "report.json").exists()


@pytest.mark.asyncio
async def test_dry_run_backfill_preserves_existing_relationship_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Backfill metadata without changing native relationship axes."""

    state = build_acquaintance_user_state(
        global_user_id="migration-axis-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    legacy = deepcopy(state)
    del legacy["relationship"]["relationship_maintenance"]

    async def load_rows() -> list[dict[str, object]]:
        return [{
            "global_user_id": "migration-axis-user",
            "cognition_state": legacy,
        }]

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )
    report = await migration.run_dry_run(
        backup_path=tmp_path / "backup.json",
        report_path=tmp_path / "report.json",
        generated_at="2026-08-18T01:00:00Z",
    )
    backup = json.loads(
        (tmp_path / "backup.json").read_text(encoding="utf-8")
    )

    replacement = backup["rows"][0]["replacement_state"]
    assert replacement["relationship"]["trust"] == state["relationship"][
        "trust"
    ]
    assert report["writes_performed"] == 0


@pytest.mark.asyncio
async def test_apply_requires_matching_dry_run_digest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Reject an apply when its reviewed backup bytes changed."""

    state = build_acquaintance_user_state(
        global_user_id="migration-digest-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    legacy = deepcopy(state)
    del legacy["relationship"]["relationship_maintenance"]

    async def load_rows() -> list[dict[str, object]]:
        return [{
            "global_user_id": "migration-digest-user",
            "cognition_state": legacy,
        }]

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )
    backup_path = tmp_path / "backup.json"
    report_path = tmp_path / "report.json"
    await migration.run_dry_run(
        backup_path=backup_path,
        report_path=report_path,
        generated_at="2026-08-18T01:00:00Z",
    )
    backup_path.write_bytes(backup_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="digest does not match"):
        await migration.run_apply(
            backup_path=backup_path,
            report_path=report_path,
            output_path=tmp_path / "output.json",
            applied_at="2026-08-18T02:00:00Z",
        )


@pytest.mark.asyncio
async def test_migration_fails_closed_on_concurrent_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Skip a row whose current state differs from the dry-run snapshot."""

    state = build_acquaintance_user_state(
        global_user_id="migration-drift-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    legacy = deepcopy(state)
    del legacy["relationship"]["relationship_maintenance"]
    current_state = deepcopy(legacy)
    current_state["relationship"]["trust"] = 1
    calls = {"cas": 0}
    load_count = {"value": 0}

    async def load_rows() -> list[dict[str, object]]:
        load_count["value"] += 1
        selected = legacy if load_count["value"] == 1 else current_state
        return [{
            "global_user_id": "migration-drift-user",
            "cognition_state": selected,
        }]

    async def compare_and_replace(**_: object) -> bool:
        calls["cas"] += 1
        return True

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )
    monkeypatch.setattr(
        migration,
        "compare_and_replace_user_cognition_state_for_migration",
        compare_and_replace,
    )
    backup_path = tmp_path / "backup.json"
    report_path = tmp_path / "report.json"
    await migration.run_dry_run(
        backup_path=backup_path,
        report_path=report_path,
        generated_at="2026-08-18T01:00:00Z",
    )
    result = await migration.run_apply(
        backup_path=backup_path,
        report_path=report_path,
        output_path=tmp_path / "output.json",
        applied_at="2026-08-18T02:00:00Z",
    )

    assert result["counts"]["drift"] == 1
    assert calls["cas"] == 0


@pytest.mark.asyncio
async def test_migration_fails_closed_on_drifted_already_valid_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Re-read and compare an already-valid row before accepting it."""

    state = build_acquaintance_user_state(
        global_user_id="migration-valid-drift-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    drifted_state = deepcopy(state)
    drifted_state["relationship"]["trust"] = 1
    load_count = {"value": 0}

    async def load_rows() -> list[dict[str, object]]:
        load_count["value"] += 1
        current = state if load_count["value"] == 1 else drifted_state
        return [{
            "global_user_id": "migration-valid-drift-user",
            "cognition_state": current,
        }]

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )
    backup_path = tmp_path / "backup.json"
    report_path = tmp_path / "report.json"
    await migration.run_dry_run(
        backup_path=backup_path,
        report_path=report_path,
        generated_at="2026-08-18T01:00:00Z",
    )
    result = await migration.run_apply(
        backup_path=backup_path,
        report_path=report_path,
        output_path=tmp_path / "output.json",
        applied_at="2026-08-18T02:00:00Z",
    )

    assert result["counts"]["drift"] == 1
    assert result["counts"]["already_valid"] == 0
    assert result["activation_ready"] is False


@pytest.mark.asyncio
async def test_migration_fails_closed_on_missing_already_valid_row(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Record a missing reviewed row instead of treating it as valid."""

    state = build_acquaintance_user_state(
        global_user_id="migration-valid-missing-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    load_count = {"value": 0}

    async def load_rows() -> list[dict[str, object]]:
        load_count["value"] += 1
        if load_count["value"] == 1:
            return [{
                "global_user_id": "migration-valid-missing-user",
                "cognition_state": state,
            }]
        return []

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )
    backup_path = tmp_path / "backup.json"
    report_path = tmp_path / "report.json"
    await migration.run_dry_run(
        backup_path=backup_path,
        report_path=report_path,
        generated_at="2026-08-18T01:00:00Z",
    )
    result = await migration.run_apply(
        backup_path=backup_path,
        report_path=report_path,
        output_path=tmp_path / "output.json",
        applied_at="2026-08-18T02:00:00Z",
    )

    assert result["counts"]["missing"] == 1
    assert result["counts"]["already_valid"] == 0
    assert result["activation_ready"] is False


@pytest.mark.asyncio
async def test_migration_uses_named_db_maintenance_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Route both migration reads and writes through named DB helpers."""

    state = build_acquaintance_user_state(
        global_user_id="migration-boundary-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    legacy = deepcopy(state)
    del legacy["relationship"]["relationship_maintenance"]
    calls = {"list": 0, "cas": 0}

    async def load_rows() -> list[dict[str, object]]:
        calls["list"] += 1
        return [{
            "global_user_id": "migration-boundary-user",
            "cognition_state": legacy,
        }]

    async def compare_and_replace(**_: object) -> bool:
        calls["cas"] += 1
        return True

    monkeypatch.setattr(
        migration,
        "list_user_cognition_states_for_relationship_maintenance_migration",
        load_rows,
    )
    monkeypatch.setattr(
        migration,
        "compare_and_replace_user_cognition_state_for_migration",
        compare_and_replace,
    )
    backup_path = tmp_path / "backup.json"
    report_path = tmp_path / "report.json"
    await migration.run_dry_run(
        backup_path=backup_path,
        report_path=report_path,
        generated_at="2026-08-18T01:00:00Z",
    )
    result = await migration.run_apply(
        backup_path=backup_path,
        report_path=report_path,
        output_path=tmp_path / "output.json",
        applied_at="2026-08-18T02:00:00Z",
    )

    assert calls == {"list": 2, "cas": 1}
    assert result["writes_performed"] == 1
