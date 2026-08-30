"""Executable gates for the read-only Plan 3 drain checker."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest


def _load():
    """Load the planned drain script as a module."""

    try:
        from scripts import check_dsh_plan3_drain
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"planned drain checker is unavailable: {exc}")
    return check_dsh_plan3_drain


class _CountCollection:
    """Read-only fake collection returning a declared sequence of counts."""

    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.filters: list[dict[str, object]] = []

    async def count_documents(self, filter_doc: dict[str, object]) -> int:
        self.filters.append(filter_doc)
        if not self.counts:
            raise AssertionError("unexpected additional Mongo count")
        return self.counts.pop(0)


class _DrainDb:
    """Minimal named Mongo collection facade for the four drain counts."""

    def __init__(self) -> None:
        self.accepted_tasks = _CountCollection([2])
        self.background_work_jobs = _CountCollection([3, 4])
        self.dsh_interactions = _CountCollection([5])
        self.writes: list[tuple[str, object]] = []

    def __getitem__(self, name: str) -> _CountCollection:
        if name == "accepted_tasks":
            return self.accepted_tasks
        if name == "background_work_jobs":
            return self.background_work_jobs
        if name == "dsh_interactions":
            return self.dsh_interactions
        raise AssertionError(f"unexpected drain collection: {name}")


def _write_ledger(path: Path, value: object) -> None:
    """Create one deterministic legacy ledger fixture."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def _literal_mapping(node: ast.Dict) -> dict[str, ast.AST]:
    """Return string-keyed entries from a literal dictionary node."""

    mapping: dict[str, ast.AST] = {}
    for key, value in zip(node.keys, node.values):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            mapping[key.value] = value
    return mapping


def _string_literal(node: ast.AST) -> str:
    """Evaluate one source string literal for an owner-shape assertion."""

    value = ast.literal_eval(node)
    if not isinstance(value, str):
        raise AssertionError(f"expected string literal, got {value!r}")
    return value


def _string_list(node: ast.AST) -> list[str]:
    """Evaluate one source string-list literal for an owner-shape assertion."""

    value = ast.literal_eval(node)
    if not isinstance(value, list) or not all(
        isinstance(item, str) for item in value
    ):
        raise AssertionError(f"expected string list literal, got {value!r}")
    return value


def test_drain_owner_is_the_only_v1_payload_exception_and_uses_exact_filters() -> None:
    """The narrow drain exception owns exactly the two required row filters."""

    source_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "kazusa_ai_chatbot"
        / "db"
        / "script_operations.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    owners = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "count_dsh_plan3_drain_rows"
    ]
    assert len(owners) == 1
    owner = owners[0]

    payload_schema = "task_orchestrator_worker_payload." + "v1"
    assert source.count(payload_schema) == 2
    payload_literals = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Constant) and node.value == payload_schema
    ]
    assert len(payload_literals) == 2

    filter_dicts = [
        _literal_mapping(node)
        for node in ast.walk(owner)
        if isinstance(node, ast.Dict)
        and any(
            isinstance(child, ast.Constant) and child.value == payload_schema
            for child in ast.walk(node)
        )
    ]
    assert len(filter_dicts) == 2

    expected_execution_status = ["queued", "in_progress"]
    expected_undelivered_status = [
        "completed",
        "failed",
        "delivery_failed",
        "delivery_in_progress",
    ]
    base_fields = {
        "schema_version",
        "requested_worker",
        "worker_payload.schema_version",
        "status",
    }
    seen_statuses: set[tuple[str, ...]] = set()
    for fields in filter_dicts:
        assert _string_literal(fields["schema_version"]) == "background_work_job.v2"
        assert _string_literal(fields["requested_worker"]) == "task_orchestrator"
        assert (
            _string_literal(fields["worker_payload.schema_version"])
            == payload_schema
        )
        status = fields["status"]
        assert isinstance(status, ast.Dict)
        status_fields = _literal_mapping(status)
        assert set(status_fields) == {"$in"}
        status_values = _string_list(status_fields["$in"])
        seen_statuses.add(tuple(status_values))
        if status_values == expected_execution_status:
            assert set(fields) == base_fields
        elif status_values == expected_undelivered_status:
            assert set(fields) == base_fields | {"delivery_state"}
            delivery_state = fields["delivery_state"]
            assert isinstance(delivery_state, ast.Dict)
            delivery_fields = _literal_mapping(delivery_state)
            assert set(delivery_fields) == {"$ne"}
            assert _string_literal(delivery_fields["$ne"]) == "delivered"
        else:
            raise AssertionError(f"unexpected drain status filter: {status_values!r}")

    assert seen_statuses == {
        tuple(expected_execution_status),
        tuple(expected_undelivered_status),
    }


@pytest.mark.asyncio
async def test_drain_helpers_count_only_exact_legacy_active_undelivered_and_open_old_catalog_rows(
    tmp_path,
    monkeypatch,
) -> None:
    """The real helper emits all five bounded categories from named fakes."""

    module = _load()
    database = _DrainDb()
    root = (tmp_path / "legacy-coding").resolve()
    _write_ledger(
        root / "coding_runs" / "run-active" / "run.json",
        {"schema_version": "coding_run.v1", "status": "blocked"},
    )
    _write_ledger(
        root / "coding_runs" / "run-done" / "run.json",
        {"schema_version": "coding_run.v1", "status": "completed"},
    )
    _write_ledger(root / "coding_runs" / "run-invalid" / "run.json", b"{")
    _write_ledger(
        root / "coding_runs" / "run-unknown" / "run.json",
        {"schema_version": "unknown.v1", "status": "future"},
    )
    _write_ledger(
        root.parent / "coding_runs" / "escape" / "run.json",
        {"schema_version": "coding_run.v1", "status": "blocked"},
    )

    async def get_db():
        return database

    monkeypatch.setattr(module, "get_db", get_db, raising=False)
    before = {
        path: path.read_bytes()
        for path in root.rglob("run.json")
    }
    report = await module.collect_dsh_plan3_drain(
        legacy_coding_workspace_root=root,
    )
    after = {
        path: path.read_bytes()
        for path in root.rglob("run.json")
    }

    assert report["schema_version"] == "dsh_plan3_drain_report.v1"
    assert set(report) == {"schema_version", "generated_at", "counts", "ready"}
    assert report["counts"] == {
        "active_legacy_accepted_tasks": 2,
        "executing_legacy_task_jobs": 3,
        "undelivered_legacy_task_jobs": 4,
        "nonterminal_or_invalid_legacy_coding_runs": 3,
        "open_pre_cutover_dsh_interactions": 5,
    }
    assert report["ready"] is False
    assert before == after
    assert database.writes == []


@pytest.mark.asyncio
async def test_drain_cli_is_read_only_and_reports_closed_five_counts(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    """The CLI serializes exactly five counts and never exposes row content."""

    module = _load()
    database = _DrainDb()
    root = (tmp_path / "legacy-coding").resolve()
    root.mkdir(parents=True)

    async def get_db():
        return database

    monkeypatch.setattr(module, "get_db", get_db, raising=False)
    result = module.main(
        [
            "--legacy-coding-workspace-root",
            str(root),
            "--format",
            "json",
        ],
    )
    if hasattr(result, "__await__"):
        result = await result
    output = capsys.readouterr().out.strip()
    payload = result if isinstance(result, dict) else json.loads(output)

    assert set(payload) == {"schema_version", "generated_at", "counts", "ready"}
    assert set(payload["counts"]) == {
        "active_legacy_accepted_tasks",
        "executing_legacy_task_jobs",
        "undelivered_legacy_task_jobs",
        "nonterminal_or_invalid_legacy_coding_runs",
        "open_pre_cutover_dsh_interactions",
    }
    assert payload["ready"] is False
    serialized = json.dumps(payload, sort_keys=True)
    assert "run.json" not in serialized
    assert "blocked" not in serialized
    assert "coding_run.v1" not in serialized
    assert database.writes == []


@pytest.mark.asyncio
async def test_drain_cli_counts_nonterminal_and_invalid_coding_ledgers_without_exposing_content(
    tmp_path,
    monkeypatch,
) -> None:
    """Malformed and unknown ledgers count as metadata without body disclosure."""

    module = _load()
    database = _DrainDb()
    root = (tmp_path / "legacy-coding").resolve()
    _write_ledger(
        root / "coding_runs" / "malformed" / "run.json",
        b"not-json-secret-ledger-body",
    )
    _write_ledger(
        root / "coding_runs" / "unknown" / "run.json",
        {
            "schema_version": "coding_run.v999",
            "status": "invented",
            "summary": "private ledger body must stay hidden",
        },
    )

    async def get_db():
        return database

    monkeypatch.setattr(module, "get_db", get_db, raising=False)
    report = await module.collect_dsh_plan3_drain(
        legacy_coding_workspace_root=root,
    )

    assert report["counts"]["nonterminal_or_invalid_legacy_coding_runs"] == 2
    serialized = json.dumps(report, sort_keys=True)
    assert "not-json-secret-ledger-body" not in serialized
    assert "private ledger body must stay hidden" not in serialized
    assert "summary" not in serialized
