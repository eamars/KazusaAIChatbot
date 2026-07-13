"""One-case-at-a-time lifecycle evidence CLI for validation-local V2."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    run_lifecycle_case,
    write_diagnostic_artifact,
)


FIXTURE_PATH = Path("tests/fixtures/cognition_core_v2_emotion_lifecycle_cases.json")


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
    """Reject an uninjected benchmark instead of fabricating V1/V2 measurements."""

    if repetitions < 1:
        raise ValueError("benchmark repetitions must be positive")
    raise RuntimeError(
        "benchmark execution requires the parent validation harness to inject "
        "matching V1 and V2 services"
    )


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


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
