"""One-case-at-a-time lifecycle evidence CLI for validation-local V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import statistics
import time

from kazusa_ai_chatbot.cognition_episode import CognitiveEpisode
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    validate_cognition_core_input,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    run_lifecycle_case,
    write_diagnostic_artifact,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc


FIXTURE_PATH = Path("tests/fixtures/cognition_core_v2_emotion_lifecycle_cases.json")
BENCHMARK_FIXTURE_PATH = Path(
    "tests/fixtures/cognition_core_v2_benchmark_cases.json"
)


def main() -> int:
    """Run one explicit lifecycle case and retain its structured evidence."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lifecycle_parser = subparsers.add_parser("lifecycle")
    lifecycle_parser.add_argument("--case-id", required=True)
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--case-id", required=True)
    benchmark_parser.add_argument("--repetitions", type=int, required=True)
    arguments = parser.parse_args()
    if arguments.command == "lifecycle":
        return _run_lifecycle(arguments.case_id)
    return _run_benchmark(arguments.case_id, arguments.repetitions)


def _run_lifecycle(case_id: str) -> int:
    """Write one deterministic lifecycle artifact for the requested fixture row."""

    cases = _load_cases()
    case = next((item for item in cases if item["case_id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown lifecycle case id: {case_id}")
    result = run_lifecycle_case(case)
    artifact_path = write_diagnostic_artifact(case_id, result)
    print(json.dumps({"artifact_path": str(artifact_path), "result": result}))
    return 0


def _run_benchmark(case_id: str, repetitions: int) -> int:
    """Benchmark exact V2 input validation with a frozen fixture case."""

    if repetitions < 1:
        raise ValueError("benchmark repetitions must be positive")
    cases = _load_benchmark_cases()
    case = next((item for item in cases if item["case_id"] == case_id), None)
    if case is None:
        raise ValueError(f"unknown benchmark case id: {case_id}")
    payload = _build_benchmark_payload(case)
    samples_ms = []
    for _ in range(repetitions):
        started_at = time.perf_counter()
        validate_cognition_core_input(payload)
        samples_ms.append((time.perf_counter() - started_at) * 1000)
    result = {
        "case_id": case_id,
        "implementation": "v2",
        "operation": "input_contract_validation",
        "repetitions": repetitions,
        "minimum_ms": min(samples_ms),
        "median_ms": statistics.median(samples_ms),
        "maximum_ms": max(samples_ms),
        "state_scope": payload["state_scope"],
        "evidence_count": len(payload["evidence"]),
    }
    artifact_path = write_diagnostic_artifact(
        f"benchmark_{case_id}",
        result,
        artifact_root=Path("test_artifacts/cognition_core_v2/metrics"),
    )
    print(json.dumps({"artifact_path": str(artifact_path), "result": result}))
    return 0


def _load_cases() -> list[dict[str, object]]:
    """Load the approved lifecycle matrix as UTF-8 fixture data."""

    fixture_text = FIXTURE_PATH.read_text(encoding="utf-8")
    raw_cases = json.loads(fixture_text)
    if not isinstance(raw_cases, list):
        raise ValueError("lifecycle fixture must contain a list")
    cases: list[dict[str, object]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("lifecycle fixture rows must be objects")
        cases.append(raw_case)
    return cases


def _load_benchmark_cases() -> list[dict[str, str]]:
    """Load the frozen V2-only benchmark cases."""

    fixture_text = BENCHMARK_FIXTURE_PATH.read_text(encoding="utf-8")
    raw_cases = json.loads(fixture_text)
    if not isinstance(raw_cases, list):
        raise ValueError("benchmark fixture must contain a list")
    cases = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError("benchmark fixture rows must be objects")
        required = {
            "case_id",
            "origin_summary",
            "user_input",
            "decontextualized_input",
        }
        if set(raw_case) != required or any(
            not isinstance(raw_case[field_name], str)
            for field_name in required
        ):
            raise ValueError("benchmark fixture row fields are invalid")
        cases.append({
            field_name: raw_case[field_name]
            for field_name in required
        })
    return cases


def _build_benchmark_payload(case: dict[str, str]) -> dict[str, object]:
    """Build one synthetic native-V2 input without external services."""

    occurred_at = "2026-07-14T00:00:00Z"
    character_state = build_character_production_state(
        updated_at=occurred_at,
    )
    semantic_text = (
        f"{case['origin_summary']}: {case['decontextualized_input']}"
    )
    episode: CognitiveEpisode = {
        "episode_id": f"benchmark-{case['case_id']}",
        "trigger_source": "system_probe",
        "input_sources": ["internal_monologue"],
        "output_mode": "preview",
        "percepts": [{
            "percept_id": f"benchmark-percept-{case['case_id']}",
            "input_source": "internal_monologue",
            "content": semantic_text,
            "visibility": "model_visible",
            "metadata": {"source": "validation_benchmark"},
        }],
        "target_scope": {
            "platform": "validation_cli",
            "platform_channel_id": "benchmark",
            "channel_type": "internal",
            "current_platform_user_id": "benchmark-user",
            "current_global_user_id": "benchmark-user",
            "current_display_name": "Benchmark User",
            "target_addressed_user_ids": [],
            "target_broadcast": False,
        },
        "origin_metadata": {
            "platform": "validation_cli",
            "platform_message_id": f"benchmark-{case['case_id']}",
            "active_turn_platform_message_ids": [],
            "active_turn_conversation_row_ids": [],
            "debug_modes": {"think_only": True},
        },
        "storage_timestamp_utc": occurred_at,
        "local_time_context": local_time_context_from_storage_utc(
            occurred_at
        ),
    }
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": episode,
        "state_scope": "user",
        "mutable_state": build_acquaintance_user_state(
            global_user_id="benchmark-user",
            updated_at=occurred_at,
        ),
        "character_constraints": {
            "drives": character_state["drives"],
            "standards": character_state["standards"],
            "meaning_state": character_state["meaning_state"],
        },
        "evidence": [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": f"episode:benchmark-{case['case_id']}",
                "occurred_at": occurred_at,
                "semantic_summary": semantic_text[:500],
            },
            "semantic_text": semantic_text[:1000],
            "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "",
        "private_continuity_context": "",
        "scene_context": {
            "channel_scope": "internal",
            "character_role": "character",
            "semantic_scene": semantic_text,
            "conversation_continuity": "",
            "semantic_temporal_context": "immediate",
        },
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
