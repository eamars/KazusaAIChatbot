"""Isolation helpers for the disposable fresh-database Stage 3 harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import parse_qs

from dotenv import load_dotenv


STAGE3_TEST_DATABASE_NAME = "_test_kazusa_core_v2"
STAGE3_URI_ENV = "MONGODB_URI"
STAGE3_DATABASE_ENV = "MONGODB_DB_NAME"
STAGE3_DATABASE_GUARD_ENV = "STAGE3_DATABASE_GUARD"
STAGE3_PROFILE_ENV = "CHARACTER_PROFILE_PATH"
STAGE3_CASE_FIXTURE_SCHEMA = "stage3_fresh_database_cases.v1"
STAGE3_SESSION_FILENAME = "stage3_run_session.json"
STAGE3_EVIDENCE_SCHEMA = "stage3_fresh_database_evidence.v1"
STAGE3_NATIVE_SCHEMA_MANIFEST_SCHEMA = "stage3_native_schema_manifest.v1"
STAGE3_EVIDENCE_FILENAME = "stage3_fresh_database_evidence.v1.json"
STAGE3_NATIVE_SCHEMA_MANIFEST_FILENAME = (
    "stage3_native_schema_manifest.v1.json"
)


class Stage3MongoEndpointIdentityV1(TypedDict):
    """Canonical endpoint fields used for Stage 3 isolation checks."""

    scheme: Literal["mongodb", "mongodb+srv"]
    hosts: list[str]
    tls: bool
    replica_set: str
    direct_connection: bool


class Stage3FreshDatabaseEvidenceV1(TypedDict):
    """Technical evidence contract for the complete Stage 3 sequence."""

    schema_version: Literal["stage3_fresh_database_evidence.v1"]
    database_endpoint_fingerprint: str
    database_name: str
    database_absent_before_start: bool
    profile_bootstrap_result: str
    restart_profile_result: str
    source_cases: list[dict[str, object]]
    action_cases: list[dict[str, object]]
    internal_latch_cases: list[dict[str, object]]
    trace_cardinality: dict[str, int]
    post_turn_lifecycle_cardinality: dict[str, int]
    collections_and_indexes: list[dict[str, object]]
    llm_call_ledger: list[dict[str, object]]
    latency_summary_ms: dict[str, int]
    technical_failures: list[dict[str, object]]
    quality_review_path: str


class Stage3NativeSchemaManifestV1(TypedDict):
    """Native Stage 3 schema inventory handed to Stage 4."""

    schema_version: Literal["stage3_native_schema_manifest.v1"]
    collections: list[dict[str, object]]


def validate_stage3_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Validate Stage 3 inputs without importing service configuration.

    Args:
        environ: Environment mapping to validate, defaulting to the current
            process environment.

    Returns:
        The configured URI, exact database name, and computed endpoint
        fingerprint for the test harness evidence.

    Raises:
        ValueError: If any required guard input is missing or inconsistent.
    """

    if environ is None:
        load_dotenv(override=False)
        source = os.environ
    else:
        source = environ
    uri = _required_environment_value(source, STAGE3_URI_ENV)
    database_name = _required_environment_value(source, STAGE3_DATABASE_ENV)
    if database_name != STAGE3_TEST_DATABASE_NAME:
        raise ValueError(
            f"{STAGE3_DATABASE_ENV} must be {STAGE3_TEST_DATABASE_NAME!r}"
        )

    identity = build_stage3_endpoint_identity(uri)
    actual_fingerprint = stage3_endpoint_fingerprint(identity)
    uri_database_name = _database_name_from_uri(uri)
    if uri_database_name and uri_database_name != database_name:
        raise ValueError("MongoDB URI database disagrees with the guarded name")

    profile_path = _required_environment_value(source, STAGE3_PROFILE_ENV)
    profile = Path(profile_path)
    if not profile.is_absolute() or not profile.is_file():
        raise ValueError(
            f"{STAGE3_PROFILE_ENV} must be an existing absolute file path"
        )

    return_value = {
        "mongodb_uri": uri,
        "database_name": database_name,
        "endpoint_fingerprint": actual_fingerprint,
        "database_guard": "exact_reserved_name",
        "character_profile_path": str(profile),
    }
    return return_value


def load_stage3_case_fixture(path: Path) -> dict[str, object]:
    """Load and validate the immutable forty-case Stage 3 input fixture."""

    raw_value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_value, dict):
        raise ValueError("Stage 3 fixture root must be an object")
    if raw_value.get("schema_version") != STAGE3_CASE_FIXTURE_SCHEMA:
        raise ValueError("Stage 3 fixture schema version is invalid")
    source_artifact_paths = raw_value.get("source_artifact_paths")
    if not isinstance(source_artifact_paths, list) or not source_artifact_paths:
        raise ValueError("Stage 3 fixture must identify source artifacts")
    cases = raw_value.get("cases")
    if not isinstance(cases, list) or len(cases) != 40:
        raise ValueError("Stage 3 fixture must contain exactly 40 cases")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("Stage 3 fixture cases must be objects")
        case_id = case.get("case_id")
        sequence = case.get("sequence")
        input_text = case.get("input_text")
        target_scope = case.get("target_scope_fixture")
        technical_expectations = case.get("technical_expectations")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Stage 3 case id must be non-empty text")
        if case_id in case_ids:
            raise ValueError(f"Stage 3 case id is duplicated: {case_id}")
        if sequence not in {"group", "private"}:
            raise ValueError(f"Stage 3 case sequence is invalid: {case_id}")
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError(f"Stage 3 case input is invalid: {case_id}")
        if not isinstance(target_scope, dict):
            raise ValueError(f"Stage 3 target scope is invalid: {case_id}")
        if not isinstance(technical_expectations, dict):
            raise ValueError(
                f"Stage 3 technical expectations are invalid: {case_id}"
            )
        if case.get("expected_dialog") is not None:
            raise ValueError("Stage 3 fixture cannot contain quality answers")
        case_ids.add(case_id)
    return raw_value


def select_stage3_case(
    fixture: Mapping[str, object],
    case_id: str,
) -> Mapping[str, object]:
    """Select one case by id after the fixture has passed validation."""

    cases = fixture.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Stage 3 fixture cases are missing")
    for case in cases:
        if isinstance(case, Mapping) and case.get("case_id") == case_id:
            return case
    raise ValueError(f"Stage 3 case id not found: {case_id}")


def build_stage3_report(
    fixture_path: Path,
    input_dir: Path,
    output_path: Path,
) -> None:
    """Build technical evidence and a review report after forty cases exist."""

    evidence, native_manifest, rows = build_stage3_evidence(
        fixture_path,
        input_dir,
        output_path,
    )
    output_lines = [
        "# Stage 3 Fresh Database Forty-Case Technical Report",
        "",
        "This report contains technical evidence only. Character-quality review "
        "is performed after the complete sequence is available.",
        "",
        f"Technical failures: {len(evidence['technical_failures'])}",
        f"Trace cardinality: {evidence['trace_cardinality']}",
        f"Post-turn lifecycle cardinality: "
        f"{evidence['post_turn_lifecycle_cardinality']}",
        "",
        "| Case | Sequence | Source | Technical status | Terminal status | "
        "Duration ms | LLM calls |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in rows:
        output_lines.append(
            "| {case_id} | {sequence} | {source_kind} | "
            "{technical_status} | {terminal_status} | {duration_ms} | "
            "{llm_call_count} |".format(**row)
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    evidence_path = output_path.parent / STAGE3_EVIDENCE_FILENAME
    evidence_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_path = output_path.parent / STAGE3_NATIVE_SCHEMA_MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(native_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_stage3_evidence(
    fixture_path: Path,
    input_dir: Path,
    quality_review_path: Path,
) -> tuple[
    Stage3FreshDatabaseEvidenceV1,
    Stage3NativeSchemaManifestV1,
    list[dict[str, object]],
]:
    """Aggregate safe per-case artifacts into the Stage 3 evidence contracts."""

    fixture = load_stage3_case_fixture(fixture_path)
    cases = fixture["cases"]
    if not isinstance(cases, list):
        raise ValueError("Stage 3 fixture cases are missing")
    session_path = input_dir / STAGE3_SESSION_FILENAME
    if not session_path.is_file():
        raise ValueError("Stage 3 run session is missing")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(session, Mapping):
        raise ValueError("Stage 3 run session is invalid")
    rows: list[dict[str, object]] = []
    source_cases: list[dict[str, object]] = []
    action_cases: list[dict[str, object]] = []
    internal_latch_cases: list[dict[str, object]] = []
    llm_call_ledger: list[dict[str, object]] = []
    collections_and_indexes: list[dict[str, object]] = []
    technical_failures: list[dict[str, object]] = []
    durations: list[int] = []
    trace_count = 0
    lifecycle_count = 0
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("Stage 3 fixture cases are invalid")
        case_id = case.get("case_id")
        if not isinstance(case_id, str):
            raise ValueError("Stage 3 case id is invalid")
        artifact_path = input_dir / f"{case_id}.json"
        if not artifact_path.is_file():
            raise ValueError(f"Stage 3 case artifact is missing: {case_id}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(artifact, Mapping):
            raise ValueError(f"Stage 3 case artifact is invalid: {case_id}")
        if artifact.get("case_id") != case_id:
            raise ValueError(f"Stage 3 artifact id mismatch: {case_id}")
        source_kind = str(
            artifact.get("source_kind")
            or _mapping_value(artifact.get("trace_review"), "source_kind")
            or ""
        )
        terminal_status = str(artifact.get("terminal_status") or "unknown")
        technical_status = str(
            artifact.get("technical_status") or "unknown"
        )
        duration_ms = _nonnegative_int(artifact.get("duration_ms"))
        llm_call_count = _nonnegative_int(artifact.get("llm_call_count"))
        trace_cardinality = _nonnegative_int(
            artifact.get("trace_cardinality")
        )
        lifecycle_cardinality = _nonnegative_int(
            artifact.get("lifecycle_cardinality")
        )
        if technical_status != "passed":
            technical_failures.append({
                "case_id": case_id,
                "code": "case_technical_status",
                "status": technical_status,
            })
        if source_kind not in {
            "user_message",
            "internal_thought",
            "self_cognition",
            "scheduled_tick",
            "tool_result",
        }:
            technical_failures.append({
                "case_id": case_id,
                "code": "invalid_source_kind",
                "value": source_kind,
            })
        if trace_cardinality != 1:
            technical_failures.append({
                "case_id": case_id,
                "code": "trace_cardinality",
                "value": trace_cardinality,
            })
        if lifecycle_cardinality != 1:
            technical_failures.append({
                "case_id": case_id,
                "code": "post_turn_lifecycle_cardinality",
                "value": lifecycle_cardinality,
            })
        trace_count += trace_cardinality
        lifecycle_count += lifecycle_cardinality
        durations.append(duration_ms)
        rows.append({
            "case_id": case_id,
            "sequence": case.get("sequence", ""),
            "source_kind": source_kind,
            "technical_status": technical_status,
            "terminal_status": terminal_status,
            "duration_ms": duration_ms,
            "llm_call_count": llm_call_count,
        })
        source_cases.append({
            "case_id": case_id,
            "sequence": case.get("sequence", ""),
            "source_kind": source_kind,
            "technical_status": technical_status,
            "terminal_status": terminal_status,
            "duration_ms": duration_ms,
            "llm_call_count": llm_call_count,
        })
        trace_review = artifact.get("trace_review")
        if isinstance(trace_review, Mapping):
            action_kinds = trace_review.get("action_kinds", [])
            action_statuses = trace_review.get("action_statuses", [])
            if isinstance(action_kinds, list):
                for index, action_kind in enumerate(action_kinds):
                    action_status = (
                        action_statuses[index]
                        if isinstance(action_statuses, list)
                        and index < len(action_statuses)
                        else "unknown"
                    )
                    action_cases.append({
                        "case_id": case_id,
                        "action_kind": str(action_kind),
                        "status": str(action_status),
                    })
        latch_claimed = artifact.get("latch_claimed")
        case_review = artifact.get("case_review")
        if not isinstance(latch_claimed, bool) and isinstance(
            case_review,
            Mapping,
        ):
            latch_claimed = case_review.get("latch_requested")
        if isinstance(latch_claimed, bool) and latch_claimed:
            internal_latch_cases.append({
                "case_id": case_id,
                "claimed": True,
                "consumed": bool(artifact.get("latch_consumed_episode_id")),
            })
        call_rows = artifact.get("llm_step_review")
        if isinstance(call_rows, list):
            for call_row in call_rows:
                if isinstance(call_row, Mapping):
                    ledger_row = dict(call_row)
                    ledger_row["case_id"] = case_id
                    llm_call_ledger.append(ledger_row)
        observed_schema = artifact.get("collections_and_indexes")
        if isinstance(observed_schema, list):
            collections_and_indexes.extend(
                dict(row)
                for row in observed_schema
                if isinstance(row, Mapping)
            )
    if len(rows) != 40:
        raise ValueError("Stage 3 evidence requires exactly forty artifacts")
    if not collections_and_indexes:
        technical_failures.append({
            "code": "native_schema_observation_missing",
            "status": "missing",
        })
    database_absent = session.get("database_absent_before_start") is True
    if not database_absent:
        technical_failures.append({
            "code": "database_absent_before_start",
            "status": "unverified",
        })
    latency_summary = _latency_summary(durations)
    evidence: Stage3FreshDatabaseEvidenceV1 = {
        "schema_version": STAGE3_EVIDENCE_SCHEMA,
        "database_endpoint_fingerprint": str(
            session.get("database_endpoint_fingerprint", "")
        ),
        "database_name": str(session.get("database_name", "")),
        "database_absent_before_start": database_absent,
        "profile_bootstrap_result": str(
            session.get("profile_bootstrap_result", "unknown")
        ),
        "restart_profile_result": str(
            session.get("restart_profile_result", "unknown")
        ),
        "source_cases": source_cases,
        "action_cases": action_cases,
        "internal_latch_cases": internal_latch_cases,
        "trace_cardinality": {
            "settled_episode_traces": trace_count,
            "expected_episode_traces": 40,
            "technical_failure_count": sum(
                1
                for failure in technical_failures
                if failure.get("code") == "trace_cardinality"
            ),
        },
        "post_turn_lifecycle_cardinality": {
            "records": lifecycle_count,
            "expected_records": 40,
            "technical_failure_count": sum(
                1
                for failure in technical_failures
                if failure.get("code")
                == "post_turn_lifecycle_cardinality"
            ),
        },
        "collections_and_indexes": collections_and_indexes,
        "llm_call_ledger": llm_call_ledger,
        "latency_summary_ms": latency_summary,
        "technical_failures": technical_failures,
        "quality_review_path": str(quality_review_path),
    }
    native_manifest = _build_native_schema_manifest(collections_and_indexes)
    return evidence, native_manifest, rows


def _mapping_value(value: object, key: str) -> object:
    """Read one value from a mapping without exposing arbitrary payloads."""

    if not isinstance(value, Mapping):
        return None
    return value.get(key)


def _nonnegative_int(value: object) -> int:
    """Normalize one technical counter into a non-negative integer."""

    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)


def _latency_summary(values: list[int]) -> dict[str, int]:
    """Build bounded latency totals from the per-case ledger."""

    if not values:
        return {
            "case_count": 0,
            "total_ms": 0,
            "average_ms": 0,
            "max_ms": 0,
        }
    return {
        "case_count": len(values),
        "total_ms": sum(values),
        "average_ms": round(sum(values) / len(values)),
        "max_ms": max(values),
    }


def _build_native_schema_manifest(
    collections_and_indexes: list[dict[str, object]],
) -> Stage3NativeSchemaManifestV1:
    """Build a sanitized Stage 4 schema inventory from native observations."""

    by_collection: dict[str, dict[str, object]] = {}
    for observed in collections_and_indexes:
        collection_name = str(observed.get("collection_name") or "")
        if not collection_name:
            continue
        existing = by_collection.setdefault(collection_name, {
            "collection_name": collection_name,
            "index_names": [],
            "owner_module": "native runtime owner",
            "native_schema_version": "native schema observed",
            "creation_condition": "created by native owner operation",
            "retention_ttl": "owner-defined or none",
            "sanitized_representative_shape": {},
        })
        index_names = existing["index_names"]
        observed_index_names = observed.get("index_names", [])
        if isinstance(index_names, list) and isinstance(
            observed_index_names,
            list,
        ):
            index_names.extend(
                str(name)
                for name in observed_index_names
                if str(name) not in index_names
            )
        for field_name in (
            "owner_module",
            "native_schema_version",
            "creation_condition",
            "retention_ttl",
            "sanitized_representative_shape",
        ):
            if field_name in observed:
                existing[field_name] = observed[field_name]
    manifest: Stage3NativeSchemaManifestV1 = {
        "schema_version": STAGE3_NATIVE_SCHEMA_MANIFEST_SCHEMA,
        "collections": list(by_collection.values()),
    }
    return manifest


def main() -> int:
    """Run one guarded case, build evidence, or clean the guarded database."""

    args = _parse_args()
    if args.command == "run-case":
        _run_case_command(args)
    elif args.command == "build-report":
        build_stage3_report(args.fixture, args.input_dir, args.output)
        print(f"wrote {args.output}")
    else:
        _cleanup_stage3_database(args.output_dir)
        print(f"cleaned guarded database {STAGE3_TEST_DATABASE_NAME}")
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse the parent-owned Stage 3 evidence commands."""

    parser = argparse.ArgumentParser(description="Stage 3 fresh DB harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run-case")
    run_parser.add_argument("--fixture", type=Path, required=True)
    run_parser.add_argument("--case-id", required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    report_parser = subparsers.add_parser("build-report")
    report_parser.add_argument("--fixture", type=Path, required=True)
    report_parser.add_argument("--input-dir", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _run_case_command(args: argparse.Namespace) -> None:
    """Guard one case and delegate its live execution to the single-case test."""

    guarded_environment = validate_stage3_environment()
    fixture = load_stage3_case_fixture(args.fixture)
    case = select_stage3_case(fixture, args.case_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / STAGE3_SESSION_FILENAME
    mode = "restart" if session_path.is_file() else "cold_start"
    if mode == "cold_start":
        _assert_stage3_database_absent(guarded_environment)
    else:
        _assert_stage3_session_matches(session_path, guarded_environment)
        _assert_stage3_database_present(guarded_environment)
    child_environment = dict(os.environ)
    child_environment.update({
        "PYTHON_DOTENV_DISABLED": "1",
        "PYTEST_ADDOPTS": "",
        "MONGODB_URI": guarded_environment["mongodb_uri"],
        "MONGODB_DB_NAME": guarded_environment["database_name"],
        STAGE3_DATABASE_GUARD_ENV: "1",
        STAGE3_PROFILE_ENV: guarded_environment["character_profile_path"],
        "STAGE3_CASE_ID": args.case_id,
        "STAGE3_RUN_MODE": mode,
        "STAGE3_CASE_OUTPUT_PATH": str(args.output_dir / f"{args.case_id}.json"),
    })
    started_at = time.perf_counter()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_stage3_fresh_database_e2e_live_llm.py::"
            "test_live_fresh_database_case",
            "-q",
            "-s",
            "-o",
            "addopts=",
        ],
        cwd=Path.cwd(),
        env=child_environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_stem = args.output_dir / f"{args.case_id}_pytest"
    log_stem.with_suffix(".stdout.log").write_text(
        completed.stdout,
        encoding="utf-8",
    )
    log_stem.with_suffix(".stderr.log").write_text(
        completed.stderr,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage 3 case failed: {args.case_id} "
            f"exit={completed.returncode}"
        )
    artifact_path = args.output_dir / f"{args.case_id}.json"
    if not artifact_path.is_file():
        raise RuntimeError(f"Stage 3 case did not write an artifact: {args.case_id}")
    session: dict[str, object] = {}
    if session_path.is_file():
        existing_session = json.loads(
            session_path.read_text(encoding="utf-8")
        )
        if not isinstance(existing_session, Mapping):
            raise RuntimeError("Stage 3 run session is invalid")
        session.update(existing_session)
    if mode == "cold_start":
        session.update({
            "schema_version": "stage3_run_session.v1",
            "database_endpoint_fingerprint": (
                guarded_environment["endpoint_fingerprint"]
            ),
            "database_name": guarded_environment["database_name"],
            "database_absent_before_start": True,
            "profile_bootstrap_result": "inserted",
            "restart_profile_result": "pending",
            "first_case_id": args.case_id,
        })
    else:
        session["restart_profile_result"] = "verified"
    session["last_case_id"] = args.case_id
    session["case_count"] = _nonnegative_int(session.get("case_count")) + 1
    session_path.write_text(
        json.dumps(session, indent=2) + "\n",
        encoding="utf-8",
    )
    duration_ms = round((time.perf_counter() - started_at) * 1000)
    print(f"completed {case['case_id']} mode={mode} duration_ms={duration_ms}")


def _cleanup_stage3_database(output_dir: Path) -> None:
    """Drop only the exact guarded Stage 3 database after an explicit command."""

    guarded_environment = validate_stage3_environment()
    session_path = output_dir / STAGE3_SESSION_FILENAME
    if not session_path.is_file():
        raise RuntimeError("Stage 3 run session is missing")
    _assert_stage3_session_matches(session_path, guarded_environment)
    _assert_stage3_database_present(guarded_environment)
    from pymongo import MongoClient

    client = MongoClient(
        guarded_environment["mongodb_uri"],
        serverSelectionTimeoutMS=5_000,
    )
    try:
        client.drop_database(guarded_environment["database_name"])
    finally:
        client.close()


def _assert_stage3_database_absent(guarded_environment: Mapping[str, str]) -> None:
    """Require the guarded database to be absent before the cold start."""

    if _stage3_database_has_collections(
        guarded_environment["mongodb_uri"],
        guarded_environment["database_name"],
    ):
        raise RuntimeError("Stage 3 database already exists before cold start")


def _assert_stage3_database_present(guarded_environment: Mapping[str, str]) -> None:
    """Require the guarded database to exist before a restart case."""

    if not _stage3_database_has_collections(
        guarded_environment["mongodb_uri"],
        guarded_environment["database_name"],
    ):
        raise RuntimeError("Stage 3 database is missing for restart")


def _assert_stage3_session_matches(
    session_path: Path,
    guarded_environment: Mapping[str, str],
) -> None:
    """Require restart/cleanup to use the same guarded database session."""

    session = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(session, Mapping):
        raise RuntimeError("Stage 3 run session is invalid")
    if session.get("database_name") != guarded_environment["database_name"]:
        raise RuntimeError("Stage 3 session database name changed")
    if session.get("database_endpoint_fingerprint") != (
        guarded_environment["endpoint_fingerprint"]
    ):
        raise RuntimeError("Stage 3 session MongoDB endpoint changed")


def _stage3_database_has_collections(uri: str, database_name: str) -> bool:
    """Inspect only the exact guarded database for persistent collections."""

    from pymongo import MongoClient

    client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    try:
        collection_names = client[database_name].list_collection_names()
    finally:
        client.close()
    return bool(collection_names)


def build_stage3_endpoint_identity(
    uri: str,
) -> Stage3MongoEndpointIdentityV1:
    """Build the canonical endpoint identity without DNS or service imports."""

    scheme, separator, remainder = uri.partition("://")
    if not separator or scheme not in {"mongodb", "mongodb+srv"}:
        raise ValueError("MongoDB URI must use mongodb or mongodb+srv")

    authority, _, path_and_query = remainder.partition("/")
    authority = authority.rsplit("@", maxsplit=1)[-1]
    if not authority:
        raise ValueError("MongoDB URI must include a host")

    hosts = [
        _canonical_host(host_text, scheme)
        for host_text in authority.split(",")
        if host_text.strip()
    ]
    if not hosts:
        raise ValueError("MongoDB URI must include a host")

    query_text = path_and_query.partition("?")[2]
    query = parse_qs(query_text, keep_blank_values=True)
    tls = _query_boolean(
        query,
        "tls",
        default=scheme == "mongodb+srv",
    )
    if "tls" not in query:
        tls = _query_boolean(
            query,
            "ssl",
            default=tls,
        )
    replica_set = _query_text(query, "replicaSet")
    direct_connection = _query_boolean(
        query,
        "directConnection",
        default=False,
    )
    identity: Stage3MongoEndpointIdentityV1 = {
        "scheme": scheme,
        "hosts": sorted(hosts),
        "tls": tls,
        "replica_set": replica_set,
        "direct_connection": direct_connection,
    }
    return identity


def stage3_endpoint_fingerprint(
    identity: Stage3MongoEndpointIdentityV1,
) -> str:
    """Hash the canonical endpoint identity used by the isolation guard."""

    canonical_json = json.dumps(
        identity,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return fingerprint


def _required_environment_value(
    environ: Mapping[str, str],
    name: str,
) -> str:
    """Read one required non-empty guard input."""

    value = environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _canonical_host(host_text: str, scheme: str) -> str:
    """Normalize one Mongo host for the endpoint identity."""

    host_value = host_text.strip().lower()
    if not host_value:
        raise ValueError("MongoDB URI contains an empty host")
    if scheme == "mongodb+srv":
        if ":" in host_value:
            raise ValueError("MongoDB SRV host must not include a port")
        return host_value

    if host_value.startswith("["):
        closing_bracket = host_value.find("]")
        if closing_bracket < 0:
            raise ValueError("MongoDB IPv6 host is malformed")
        host = host_value[: closing_bracket + 1]
        port_text = host_value[closing_bracket + 1 :].lstrip(":")
    elif host_value.count(":") == 1:
        host, port_text = host_value.rsplit(":", maxsplit=1)
    else:
        host = host_value
        port_text = ""
    port = port_text or "27017"
    if not port.isdigit() or int(port) <= 0 or int(port) > 65535:
        raise ValueError("MongoDB host port is invalid")
    return f"{host}:{int(port)}"


def _query_text(query: Mapping[str, list[str]], name: str) -> str:
    """Return one normalized query value or an empty string."""

    values = query.get(name, [])
    if not values:
        return ""
    return values[0].strip()


def _query_boolean(
    query: Mapping[str, list[str]],
    name: str,
    *,
    default: bool,
) -> bool:
    """Parse one Mongo boolean query option."""

    value = _query_text(query, name).lower()
    if not value:
        return default
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise ValueError(f"MongoDB URI option {name} must be Boolean")


def _database_name_from_uri(uri: str) -> str:
    """Extract the optional database path without exposing credentials."""

    _, separator, remainder = uri.partition("://")
    if not separator:
        return ""
    _, slash, path_and_query = remainder.partition("/")
    if not slash:
        return ""
    database_name = path_and_query.partition("?")[0].strip()
    return database_name


if __name__ == "__main__":
    raise SystemExit(main())
