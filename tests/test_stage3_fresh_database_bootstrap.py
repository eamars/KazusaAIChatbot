"""Focused tests for the isolated fresh-database harness boundary."""

from __future__ import annotations

import importlib
import json
from pathlib import Path


def test_stage3_harness_uses_the_fixed_disposable_database_name() -> None:
    """The harness must target only the dedicated Stage 3 database."""

    module = importlib.import_module("tests.stage3_fresh_database")

    assert module.STAGE3_TEST_DATABASE_NAME == "_test_kazusa_core_v2"


def test_stage3_harness_exposes_pre_service_guard() -> None:
    """Isolation validation must be callable without importing the service."""

    module = importlib.import_module("tests.stage3_fresh_database")

    assert callable(module.validate_stage3_environment)


def test_stage3_environment_rejects_missing_guard_inputs() -> None:
    """Missing isolation inputs must stop before service import."""

    module = importlib.import_module("tests.stage3_fresh_database")

    try:
        module.validate_stage3_environment({})
    except ValueError as exc:
        assert "MONGODB_URI" in str(exc)
    else:
        raise AssertionError("missing Stage 3 inputs were accepted")


def test_stage3_endpoint_identity_ignores_credentials_and_host_order() -> None:
    """Endpoint identity must be stable across secret and host ordering changes."""

    module = importlib.import_module("tests.stage3_fresh_database")
    first_identity = module.build_stage3_endpoint_identity(
        "mongodb://user:secret@B:27017,a/?tls=true"
    )
    second_identity = module.build_stage3_endpoint_identity(
        "mongodb://other:changed@a:27017,b/?tls=true"
    )

    assert first_identity == second_identity
    assert module.stage3_endpoint_fingerprint(first_identity) == (
        module.stage3_endpoint_fingerprint(second_identity)
    )


def test_stage3_environment_accepts_same_endpoint_with_reserved_database(
    tmp_path: Path,
) -> None:
    """The exact reserved database name is the primary isolation guard."""

    module = importlib.import_module("tests.stage3_fresh_database")
    uri = "mongodb://localhost:27017"
    profile_path = tmp_path / "kazusa.json"
    profile_path.write_text("{}", encoding="utf-8")
    environment = {
        "MONGODB_URI": uri,
        "MONGODB_DB_NAME": "_test_kazusa_core_v2",
        "CHARACTER_PROFILE_PATH": str(profile_path),
    }

    result = module.validate_stage3_environment(environment)

    assert result["mongodb_uri"] == uri
    assert result["database_name"] == "_test_kazusa_core_v2"
    assert result["database_guard"] == "exact_reserved_name"


def test_stage3_environment_rejects_uri_database_mismatch(tmp_path: Path) -> None:
    """An embedded URI database must agree with the exact guarded name."""

    module = importlib.import_module("tests.stage3_fresh_database")
    profile_path = tmp_path / "kazusa.json"
    profile_path.write_text("{}", encoding="utf-8")
    try:
        module.validate_stage3_environment({
            "MONGODB_URI": "mongodb://localhost:27017/roleplay_bot",
            "MONGODB_DB_NAME": "_test_kazusa_core_v2",
            "CHARACTER_PROFILE_PATH": str(profile_path),
        })
    except ValueError as exc:
        assert "disagrees" in str(exc)
    else:
        raise AssertionError("URI database mismatch was accepted")


def test_stage3_case_fixture_contains_only_sanitized_technical_inputs() -> None:
    """The forty-case fixture must contain no generated quality answers."""

    module = importlib.import_module("tests.stage3_fresh_database")
    fixture = module.load_stage3_case_fixture(
        Path("tests/fixtures/stage3_fresh_database_cases.json")
    )

    cases = fixture["cases"]
    assert isinstance(cases, list)
    assert len(cases) == 40
    assert {
        case["sequence"]
        for case in cases
        if isinstance(case, dict)
    } == {"group", "private"}
    for case in cases:
        assert isinstance(case, dict)
        assert "expected_dialog" not in case


def test_stage3_group_addressing_uses_typed_bot_mention() -> None:
    """Group live fixtures must use the typed intake addressing contract."""

    module = importlib.import_module(
        "tests.test_stage3_fresh_database_e2e_live_llm"
    )

    mentions = module._stage3_typed_mentions(
        channel_type="group",
        addressed_to_character=True,
    )

    assert mentions == [{
        "platform_user_id": "stage3-bot",
        "entity_kind": "bot",
        "raw_text": "@stage3-character",
    }]
    assert module._stage3_typed_mentions(
        channel_type="private",
        addressed_to_character=True,
    ) == []


def test_stage3_live_evidence_redacts_protected_payload_fields() -> None:
    """Persistent live evidence must retain metadata without raw prompt data."""

    module = importlib.import_module(
        "tests.test_stage3_fresh_database_e2e_live_llm"
    )
    safe = module._safe_evidence_payload({
        "input": {"body_text": "private prompt"},
        "raw_llm_calls": [{
            "raw_messages": ["private prompt"],
            "prompt_chars": 14,
        }],
    })

    assert isinstance(safe, dict)
    assert safe["input"]["redacted"] is True
    assert safe["raw_llm_calls"][0]["raw_messages"]["redacted"] is True
    assert "private prompt" not in str(safe)


def test_stage3_report_emits_evidence_and_native_manifest(tmp_path: Path) -> None:
    """The report command must produce both Stage 3 handoff contracts."""

    module = importlib.import_module("tests.stage3_fresh_database")
    fixture_path = Path("tests/fixtures/stage3_fresh_database_cases.json")
    fixture = module.load_stage3_case_fixture(fixture_path)
    input_dir = tmp_path / "fresh_40"
    input_dir.mkdir()
    (input_dir / module.STAGE3_SESSION_FILENAME).write_text(
        json.dumps({
            "schema_version": "stage3_run_session.v1",
            "database_endpoint_fingerprint": "test-endpoint",
            "database_name": module.STAGE3_TEST_DATABASE_NAME,
            "database_absent_before_start": True,
            "profile_bootstrap_result": "inserted",
            "restart_profile_result": "verified",
        }),
        encoding="utf-8",
    )
    for case in fixture["cases"]:
        case_id = case["case_id"]
        (input_dir / f"{case_id}.json").write_text(
            json.dumps({
                "schema_version": "stage3_live_case_evidence.v1",
                "case_id": case_id,
                "technical_status": "passed",
                "terminal_status": "completed_visible",
                "duration_ms": 10,
                "llm_call_count": 1,
                "source_kind": "user_message",
                "trace_cardinality": 1,
                "lifecycle_cardinality": 1,
                "trace_review": {
                    "source_kind": "user_message",
                    "action_kinds": [],
                    "action_statuses": [],
                },
                "llm_step_review": [{
                    "stage_name": "test",
                    "duration_ms": 10,
                }],
                "collections_and_indexes": [{
                    "collection_name": "character_state",
                    "index_names": ["_id_"],
                }],
            }),
            encoding="utf-8",
        )
    output_path = input_dir / "review.md"
    module.build_stage3_report(fixture_path, input_dir, output_path)

    evidence_path = input_dir / module.STAGE3_EVIDENCE_FILENAME
    manifest_path = input_dir / module.STAGE3_NATIVE_SCHEMA_MANIFEST_FILENAME
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert evidence["schema_version"] == "stage3_fresh_database_evidence.v1"
    assert evidence["trace_cardinality"]["settled_episode_traces"] == 40
    assert manifest["schema_version"] == "stage3_native_schema_manifest.v1"
    assert manifest["collections"][0]["collection_name"] == "character_state"
