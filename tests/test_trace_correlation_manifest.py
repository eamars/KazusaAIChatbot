"""Tests for strict, bounded trace-correlation manifests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.llm_tracing.correlation import (
    build_trace_correlation_manifest,
    resolve_trace_candidates,
)
from scripts import export_trace_correlation_manifest as exporter


_FIXTURE_PATH = Path(
    "tests/fixtures/trace_correlation_manifest_cases.json"
)


@pytest.mark.parametrize(
    "case",
    json.loads(_FIXTURE_PATH.read_text(encoding="utf-8")),
    ids=lambda case: case["name"],
)
def test_strict_resolver_fixture_cases(case: dict[str, object]) -> None:
    """Typed sources preserve exact, zero, and multiple outcomes."""

    resolution = resolve_trace_candidates(
        source_surface=str(case["source_surface"]),
        identifier=str(case["identifier"]),
        rows=case["rows"],  # type: ignore[arg-type]
        protected_available=bool(case.get("protected_available", True)),
    )

    assert resolution.status == case["status"]
    assert list(resolution.trace_ids) == case["trace_ids"]
    if case["status"] == "confirmed":
        assert resolution.trace_id == case["trace_ids"][0]
    else:
        assert resolution.trace_id == ""


def test_resolver_does_not_classify_identifier_by_shape() -> None:
    """An opaque browser value stays unavailable without a typed surface."""

    resolution = resolve_trace_candidates(
        source_surface="unknown",
        identifier="0" * 32,
        rows=[{"trace_id": "llmtrace_real"}],
    )

    assert resolution.status == "not_available_from_web"
    assert resolution.trace_ids == ()


@pytest.mark.asyncio
async def test_manifest_uses_exact_parent_and_identifier_only_joins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manifest output carries bounded identifiers without protected payloads."""

    candidate_reader = AsyncMock(
        return_value=[{"trace_id": "llmtrace_parent"}]
    )
    run_reader = AsyncMock(
        return_value=[{
            "trace_id": "llmtrace_parent",
            "status": "succeeded",
            "global_user_id": "global-user-1",
            "raw_response_text": "secret raw output",
        }]
    )
    companion_reader = AsyncMock(
        return_value={
            "conversation_history": [{
                "llm_trace_id": "llmtrace_parent",
                "row_id": "conversation-1",
                "body_text": "private message body",
            }],
            "llm_trace_steps": [{
                "trace_id": "llmtrace_parent",
                "capture_reason": "cognition_failure_capsule",
                "cognition_invocation_id": "invocation-1",
                "capsule": {
                    "cognition_invocation_id": "invocation-1",
                    "raw_response_text": "secret capsule output",
                },
            }],
            "self_cognition_action_attempts": [{
                "attempt_id": "attempt-1",
                "source_llm_trace_id": "llmtrace_parent",
            }],
            "background_work_jobs": [{
                "job_id": "job-1",
                "accepted_task_id": "task-1",
                "source_llm_trace_id": "llmtrace_parent",
            }],
            "calendar_schedules": [{
                "schedule_id": "schedule-1",
                "source_llm_trace_id": "llmtrace_parent",
            }],
            "calendar_runs": [{
                "run_id": "run-1",
                "source_llm_trace_id": "llmtrace_parent",
            }],
            "child_trace_runs": [{
                "trace_id": "llmtrace_child",
                "parent_llm_trace_id": "llmtrace_parent",
                "source_background_work_job_id": "job-1",
            }],
        }
    )
    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_candidates",
        candidate_reader,
    )
    monkeypatch.setattr(
        exporter.script_operations,
        "export_collection_rows",
        run_reader,
    )
    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_companions",
        companion_reader,
    )

    manifest = await exporter.build_correlation_manifest(
        identifier="console-trace-1",
        source_surface="web_control_trace_id",
    )

    assert manifest["parent_trace"]["status"] == "confirmed"
    assert manifest["parent_trace"]["trace_id"] == "llmtrace_parent"
    assert manifest["identifiers"]["global_user_id"]["value"] == (
        "global-user-1"
    )
    assert manifest["identifiers"]["cognition_invocation_id"]["value"] == (
        "invocation-1"
    )
    assert manifest["joins"]["child_trace_runs"]["match_count"] == 1
    assert manifest["joins"]["conversation_history"]["candidates"] == [{
        "llm_trace_id": "llmtrace_parent",
        "row_id": "conversation-1",
        "owner": "conversation_history",
        "collection": "conversation_history",
    }]
    serialized = json.dumps(manifest)
    for forbidden in (
        "raw_response_text",
        "private message body",
        "secret capsule output",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_manifest_reports_protected_read_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Database failures are explicit availability outcomes."""

    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_candidates",
        AsyncMock(side_effect=PyMongoError("database unavailable")),
    )

    manifest = await exporter.build_correlation_manifest(
        identifier="console-trace-unavailable",
        source_surface="web_control_trace_id",
    )

    assert manifest["parent_trace"]["status"] == "not_available"
    assert manifest["parent_trace"]["trace_id"] == ""
    assert manifest["availability"]["parent_trace"] == "not_available"


@pytest.mark.asyncio
async def test_manifest_preserves_historical_source_gap_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable source row without a forward trace stays not_captured."""

    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_candidates",
        AsyncMock(return_value=[{"attempt_id": "attempt-historical"}]),
    )
    companion_reader = AsyncMock()
    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_companions",
        companion_reader,
    )

    manifest = await exporter.build_correlation_manifest(
        identifier="attempt-historical",
        source_surface="protected_action_attempt_id",
    )

    assert manifest["parent_trace"]["status"] == "not_captured"
    assert manifest["identifiers"]["action_attempt_id"] == {
        "name": "action_attempt_id",
        "value": "attempt-historical",
        "candidate_values": ["attempt-historical"],
        "status": "confirmed",
        "owner": "self_cognition_action_attempts",
        "source": {
            "collection": "self_cognition_action_attempts",
            "field": "attempt_id",
        },
    }
    assert manifest["identifiers"]["global_user_id"]["status"] == (
        "not_applicable"
    )
    assert manifest["identifiers"]["cognition_invocation_id"]["status"] == (
        "not_applicable"
    )
    assert manifest["joins"]["conversation_history"]["status"] == (
        "not_applicable"
    )
    companion_reader.assert_not_awaited()


def test_manifest_marks_unresolved_derived_identifiers_not_applicable() -> None:
    """An unclassified browser value cannot imply protected identifiers."""

    manifest = build_trace_correlation_manifest(
        generated_at="2026-08-09T00:00:00+00:00",
        resolution=resolve_trace_candidates(
            source_surface="unknown",
            identifier="opaque-value",
            rows=[{"trace_id": "trace-ignored"}],
        ),
    )

    assert manifest["parent_trace"]["status"] == "not_available_from_web"


@pytest.mark.asyncio
async def test_manifest_reports_recorded_companion_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Conflict metadata remains bounded and blocks replacement selection."""

    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_candidates",
        AsyncMock(return_value=[{"trace_id": "llmtrace-parent"}]),
    )
    monkeypatch.setattr(
        exporter.script_operations,
        "export_collection_rows",
        AsyncMock(return_value=[{"trace_id": "llmtrace-parent"}]),
    )
    monkeypatch.setattr(
        exporter.script_operations,
        "list_trace_correlation_companions",
        AsyncMock(return_value={
            "conversation_history": [],
            "llm_trace_steps": [],
            "self_cognition_action_attempts": [{
                "attempt_id": "attempt-conflict",
                "source_llm_trace_id": "llmtrace-parent",
                "correlation_write_status": "conflict",
                "correlation_conflict_source_llm_trace_id": (
                    "llmtrace-incoming"
                ),
            }],
            "background_work_jobs": [],
            "calendar_schedules": [],
            "calendar_runs": [],
            "child_trace_runs": [],
        }),
    )

    manifest = await exporter.build_correlation_manifest(
        identifier="trace-console-1",
        source_surface="web_control_trace_id",
    )

    assert manifest["joins"]["self_cognition_action_attempts"]["status"] == (
        "conflict"
    )
    assert manifest["joins"]["self_cognition_action_attempts"][
        "conflict_count"
    ] == 1
    assert any(
        row["relation"] == "self_cognition_action_attempts"
        and row["status"] == "conflict"
        for row in manifest["unresolved"]
    )


def test_manifest_builder_keeps_explicit_context_fields_bounded() -> None:
    """The shared builder emits identifier-only source context."""

    manifest = build_trace_correlation_manifest(
        generated_at="2026-08-09T00:00:00+00:00",
        resolution=resolve_trace_candidates(
            source_surface="web_control_trace_id",
            identifier="trace-1",
            rows=[{"trace_id": "trace-1"}],
        ),
        parent_rows=[{"trace_id": "trace-1", "body_text": "hidden"}],
    )

    assert manifest["schema_version"] == "trace_correlation_manifest.v1"
    assert manifest["parent_trace"]["runs"] == [{
        "trace_id": "trace-1",
        "owner": "llm_trace_run",
        "collection": "llm_trace_runs",
    }]
    assert "body_text" not in json.dumps(manifest)
