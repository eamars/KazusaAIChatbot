"""Deterministic contracts for the W5/W6 memory-scope tools."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from scripts import audit_character_memory_scope as audit_module
from scripts import repair_character_memory_scope as repair_module


def _review(*, valid: bool = True) -> dict[str, Any]:
    """Build a typed learned-memory review fixture."""

    review = {
        "global_applicability": "global" if valid else "scoped",
        "target_specific_meaning_removed": valid,
        "affects_identity_or_boundaries": False,
        "private_detail_risk": "low" if valid else "high",
        "user_details_removed": valid,
        "boundary_assessment": "bounded structural review",
        "reviewer": "automated_llm",
    }
    return review


def _row(
    memory_unit_id: str,
    *,
    memory_type: str = "fact",
    valid: bool = True,
) -> dict[str, Any]:
    """Build a production-shaped learned-memory row."""

    return {
        "_id": f"mongo-{memory_unit_id}",
        "memory_unit_id": memory_unit_id,
        "lineage_id": f"lineage-{memory_unit_id}",
        "version": 1,
        "memory_name": f"Memory {memory_unit_id}",
        "content": "A bounded repository fact.",
        "source_global_user_id": "",
        "memory_type": memory_type,
        "source_kind": "conversation_extracted",
        "authority": "conversation_accepted",
        "status": "active",
        "updated_at": "2026-08-25T00:00:00Z",
        "privacy_review": _review(valid=valid),
        "evidence_refs": [{"source": "conversation_history"}],
        "embedding": [0.1, 0.2],
    }


def test_audit_is_read_only_and_emits_exact_scope_manifest() -> None:
    """The audit preserves identity/hashes and produces no write plan elsewhere."""

    valid = _row("unit-valid")
    invalid = _row("unit-invalid", memory_type="defense_rule", valid=False)
    report = audit_module.build_scope_audit_report(
        [valid, invalid],
        generated_at="2026-08-25T00:00:00Z",
    )

    assert report["writes_attempted"] == 0
    assert report["learned_candidates"] == 2
    assert report["certified_rows"] == 1
    assert report["manifest_rows"] == 1
    entry = report["apply_manifest"][0]
    assert entry["memory_unit_id"] == "unit-invalid"
    assert entry["expected_row_hash"] == audit_module.stable_memory_row_hash(invalid)
    assert entry["expected_content_hash"] == audit_module.memory_content_hash(invalid)
    assert entry["disposition"] == "reject"
    assert entry["expected_privacy_review"] == invalid["privacy_review"]
    assert "embedding" not in report["rows"][0]["source_document"]


def test_load_manifest_restores_review_snapshot_from_audit_rows(
    tmp_path: Path,
) -> None:
    """The frozen audit report supplies the exact reviewed certificate."""

    row = _row("unit-frozen-report", valid=False)
    report = audit_module.build_scope_audit_report([row])
    report["apply_manifest"][0].pop("expected_privacy_review")
    manifest_path = tmp_path / "scope-audit.json"
    manifest_path.write_text(
        json.dumps(report, ensure_ascii=False),
        encoding="utf-8",
    )

    entries = repair_module.load_approved_manifest(manifest_path)

    assert entries[0]["expected_privacy_review"] == row["privacy_review"]


@pytest.mark.asyncio
async def test_apply_rejects_only_approved_unchanged_units_and_invalidates_cache(
    tmp_path: Path,
) -> None:
    """Apply backs up exact rows, calls lifecycle rejection, and records cache stats."""

    row = _row("unit-reject", valid=False)
    report = audit_module.build_scope_audit_report([row])
    entries = report["apply_manifest"]
    live_rows = [deepcopy(row)]
    rejected_ids: list[str] = []
    cache_stats = {"invalidations": 0, "size": 1}

    async def loader(**_kwargs: object) -> list[dict[str, Any]]:
        return deepcopy(live_rows)

    async def rejecter(**kwargs: object) -> dict[str, Any]:
        memory_unit_id = str(kwargs["active_unit_id"])
        rejected_ids.append(memory_unit_id)
        cache_stats["invalidations"] += 1
        live_rows[0]["status"] = "rejected"
        return {"memory_unit_id": memory_unit_id, "status": "rejected"}

    result = await repair_module.apply_approved_manifest(
        entries=entries,
        row_loader=loader,
        rejecter=rejecter,
        clock=lambda: "2026-08-25T00:01:00Z",
        cache_stats_provider=lambda: dict(cache_stats),
        backup_path=tmp_path / "backup.json",
    )

    assert rejected_ids == ["unit-reject"]
    assert result["writes_attempted"] == 1
    assert result["cache_verification"]["invalidations_increased"] is True
    assert (tmp_path / "backup.json").exists()


@pytest.mark.asyncio
async def test_apply_stops_on_manifest_row_drift() -> None:
    """Any row drift blocks every mutation in the approved manifest."""

    original = _row("unit-drift", valid=False)
    entries = audit_module.build_scope_audit_report([original])["apply_manifest"]
    drifted = deepcopy(original)
    drifted["content"] = "Changed before apply."
    rejected_ids: list[str] = []

    async def loader(**_kwargs: object) -> list[dict[str, Any]]:
        return [deepcopy(drifted)]

    async def rejecter(**kwargs: object) -> dict[str, Any]:
        rejected_ids.append(str(kwargs["active_unit_id"]))
        return {"status": "rejected"}

    with pytest.raises(repair_module.ManifestDriftError):
        await repair_module.apply_approved_manifest(
            entries=entries,
            row_loader=loader,
            rejecter=rejecter,
        )
    assert rejected_ids == []


@pytest.mark.asyncio
async def test_apply_rejects_currently_certified_manifest_row_before_write() -> None:
    """A live certified row blocks every lifecycle mutation."""

    invalid = _row("unit-certified", valid=False)
    entry = audit_module.build_scope_audit_report([invalid])["apply_manifest"][0]
    certified = _row("unit-certified", valid=True)
    entry["expected_row_hash"] = audit_module.stable_memory_row_hash(certified)
    entry["expected_content_hash"] = audit_module.memory_content_hash(certified)
    entry["expected_privacy_review"] = deepcopy(certified["privacy_review"])
    rejected_ids: list[str] = []

    async def loader(**_kwargs: object) -> list[dict[str, Any]]:
        return [deepcopy(certified)]

    async def rejecter(**kwargs: object) -> dict[str, Any]:
        rejected_ids.append(str(kwargs["active_unit_id"]))
        return {"status": "rejected"}

    with pytest.raises(repair_module.ManifestDriftError, match="currently_certified"):
        await repair_module.apply_approved_manifest(
            entries=[entry],
            row_loader=loader,
            rejecter=rejecter,
        )
    assert rejected_ids == []


@pytest.mark.asyncio
async def test_apply_rejects_manifest_live_certificate_mismatch_before_write() -> None:
    """A changed live certificate blocks every lifecycle mutation."""

    original = _row("unit-certificate-drift", valid=False)
    entry = audit_module.build_scope_audit_report([original])["apply_manifest"][0]
    changed = deepcopy(original)
    changed["privacy_review"]["boundary_assessment"] = "changed review"
    rejected_ids: list[str] = []

    async def loader(**_kwargs: object) -> list[dict[str, Any]]:
        return [deepcopy(changed)]

    async def rejecter(**kwargs: object) -> dict[str, Any]:
        rejected_ids.append(str(kwargs["active_unit_id"]))
        return {"status": "rejected"}

    with pytest.raises(repair_module.ManifestDriftError, match="privacy_review_changed"):
        await repair_module.apply_approved_manifest(
            entries=[entry],
            row_loader=loader,
            rejecter=rejecter,
        )
    assert rejected_ids == []


class _ConsoleProbe:
    """Record console configuration before accepting output."""

    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self._events = events

    def reconfigure(self, **kwargs: object) -> None:
        self._events.append(("reconfigure", kwargs))

    def write(self, value: str) -> int:
        self._events.append(("write", value))
        return len(value)

    def flush(self) -> None:
        return None


def _assert_cli_configures_utf8_before_json_output(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
) -> None:
    events: list[tuple[object, ...]] = []
    stdout = _ConsoleProbe(events)
    stderr = _ConsoleProbe(events)
    monkeypatch.setattr(module.sys, "stdout", stdout)
    monkeypatch.setattr(module.sys, "stderr", stderr)

    module._print_json({"value": "中文"})

    first_write = next(
        index
        for index, event in enumerate(events)
        if event[0] == "write"
    )
    assert first_write == 2
    assert all(
        event[1] == {"encoding": "utf-8", "errors": "strict"}
        for event in events[:first_write]
    )


def test_audit_cli_configures_utf8_before_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit CLI configures UTF-8 before emitting JSON."""

    _assert_cli_configures_utf8_before_json_output(monkeypatch, audit_module)


def test_repair_cli_configures_utf8_before_json_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The repair CLI configures UTF-8 before emitting JSON."""

    _assert_cli_configures_utf8_before_json_output(monkeypatch, repair_module)


def test_approved_manifest_requires_exact_reject_disposition(tmp_path: Path) -> None:
    """The apply boundary admits only exact lifecycle-reject entries."""

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({
            "manifest_version": repair_module.MANIFEST_VERSION,
            "apply_manifest": [{
                "memory_unit_id": "unit-1",
                "expected_row_hash": "hash",
                "expected_status": "active",
                "disposition": "keep",
            }],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="disposition"):
        repair_module.load_approved_manifest(manifest_path)
