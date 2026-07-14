"""Validation-only diagnostics for the native V2 cognition pipeline."""

from __future__ import annotations

from contextvars import ContextVar
from copy import deepcopy
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
import time
from typing import Mapping

from kazusa_ai_chatbot.cognition_core_v2.emotion_definitions import (
    EMOTION_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.emotion_derivation import (
    derive_emotion_activation_v2,
)


_DIAGNOSTIC_RECORDS: list[dict[str, object]] = []
_VALIDATION_CAPTURE: ContextVar[dict[str, object] | None] = ContextVar(
    "cognition_core_v2_validation_capture",
    default=None,
)


def run_lifecycle_case(case: Mapping[str, object]) -> dict[str, object]:
    """Run a native activation lifecycle diagnostic for one typed fixture.

    Args:
        case: Fixture row containing an approved emotion id and typed cause sequence.

    Returns:
        Phase-by-phase activation, trend, and guard outcomes for test evidence.
    """

    emotion_id = case["emotion_id"]
    if not isinstance(emotion_id, str) or emotion_id not in EMOTION_DEFINITIONS:
        raise ValueError("lifecycle case must name an approved emotion")
    cause_sequence = case["cause_sequence"]
    if not isinstance(cause_sequence, list) or not cause_sequence:
        raise ValueError("lifecycle case must include a typed cause sequence")
    definition = EMOTION_DEFINITIONS[emotion_id]
    root_kind = definition.causal_entity_kinds[0]
    root_ref = {
        "scope": "character" if root_kind == "meaning" else "user",
        "kind": root_kind,
        "entity_id": f"diagnostic:{emotion_id}",
    }
    cause = {
        "root_ref": root_ref,
        "score": 60,
        "cause_status": "active",
        "salience": 60,
    }
    baseline = None
    beginning = derive_emotion_activation_v2(
        emotion_id,
        candidates=[cause],
        previous=None,
        updated_at="2026-07-14T00:00:00Z",
    )
    sustained = derive_emotion_activation_v2(
        emotion_id,
        candidates=[cause],
        previous=beginning,
        updated_at="2026-07-14T01:00:00Z",
    )
    fading = derive_emotion_activation_v2(
        emotion_id,
        candidates=[{
            **cause,
            "score": 0,
            "cause_status": "resolved",
        }],
        previous=sustained,
        updated_at="2026-07-14T02:00:00Z",
    )
    negative_control = None
    result = {
        "case_id": case["case_id"],
        "emotion_id": emotion_id,
        "phases": {
            "baseline": _phase_payload(baseline, True),
            "begin": _phase_payload(beginning, True),
            "sustain": _phase_payload(sustained, True),
            "fade": _phase_payload(fading, True),
            "negative_control": _phase_payload(negative_control, False),
        },
    }
    return result


def record_diagnostic(record: Mapping[str, object]) -> None:
    """Keep one validation-side diagnostic record outside the V1 return value."""

    _DIAGNOSTIC_RECORDS.append(dict(record))


def diagnostic_records() -> list[dict[str, object]]:
    """Return copied in-memory diagnostics for a focused test or harness run."""

    records = [dict(record) for record in _DIAGNOSTIC_RECORDS]
    return records


def clear_diagnostic_records() -> None:
    """Clear process-local sidecar records before an independent scenario."""

    _DIAGNOSTIC_RECORDS.clear()


def reset_validation_capture(case_id: str) -> dict[str, object]:
    """Start an empty context-local raw capture for one validation case.

    Args:
        case_id: Harness-owned identity for one independently inspected case.

    Returns:
        A copied empty capture record with the requested case identity.
    """

    if not case_id:
        raise ValueError("validation capture requires a case id")
    capture = {
        "case_id": case_id,
        "started_at": time.time(),
        "stages": [],
        "events": [],
        "failures": [],
    }
    _VALIDATION_CAPTURE.set(capture)
    capture_snapshot = deepcopy(capture)
    return capture_snapshot


def capture_validation_stage(
    *,
    stage_id: str,
    config: object,
    system_prompt: str,
    human_payload: str,
    raw_output: str | None,
    parsed_output: object | None,
    parse_status: str,
    started_at: float,
    ended_at: float,
    branch_id: str | None = None,
    error: str | None = None,
) -> None:
    """Append one raw LLM-stage capture to the active validation case.

    Args:
        stage_id: Stable V2 semantic stage identity.
        config: Existing LLM configuration, projected without the API key.
        system_prompt: Static prompt supplied to the model.
        human_payload: Current-run dynamic prompt payload.
        raw_output: Normalized raw model output when invocation succeeded.
        parsed_output: Parser result before structural validation when available.
        parse_status: Stage parse or validation status for raw evidence review.
        started_at: Monotonic stage start time.
        ended_at: Monotonic stage end time.
        branch_id: Optional activated goal branch identity.
        error: Concrete failure text when the stage failed.
    """

    capture = _VALIDATION_CAPTURE.get()
    if capture is None:
        return
    stage_record = {
        "stage_id": stage_id,
        "branch_id": branch_id,
        "config": _project_config_identity(config),
        "system_prompt": system_prompt,
        "human_payload": human_payload,
        "raw_output": raw_output,
        "parsed_output": _json_safe(parsed_output),
        "parse_status": parse_status,
        "started_at_monotonic": started_at,
        "ended_at_monotonic": ended_at,
        "duration_ms": max(0, int((ended_at - started_at) * 1000)),
        "error": error,
    }
    stage_records = capture["stages"]
    if not isinstance(stage_records, list):
        raise TypeError("validation capture stages must be a list")
    stage_records.append(stage_record)
    if error is not None:
        failures = capture["failures"]
        if not isinstance(failures, list):
            raise TypeError("validation capture failures must be a list")
        failures.append({"stage_id": stage_id, "branch_id": branch_id, "error": error})


def capture_validation_event(
    event_id: str,
    payload: Mapping[str, object],
) -> None:
    """Append deterministic state or orchestration evidence to the active case."""

    capture = _VALIDATION_CAPTURE.get()
    if capture is None:
        return
    event_records = capture["events"]
    if not isinstance(event_records, list):
        raise TypeError("validation capture events must be a list")
    event_records.append({
        "event_id": event_id,
        "recorded_at": time.time(),
        "payload": _json_safe(payload),
    })


def validation_capture_snapshot() -> dict[str, object] | None:
    """Return a deep copy of the current validation capture for inspection."""

    capture = _VALIDATION_CAPTURE.get()
    if capture is None:
        return None
    capture["completed_at"] = time.time()
    capture_snapshot = deepcopy(capture)
    return capture_snapshot


def write_validation_capture(
    *,
    artifact_root: Path = Path("test_artifacts/cognition_core_v2/raw"),
) -> Path:
    """Write the active complete case capture to the validation artifact root.

    Args:
        artifact_root: Ignored artifact directory chosen by the parent harness.

    Returns:
        The written UTF-8 JSON path for the active harness case.

    Raises:
        RuntimeError: No harness case was started with reset_validation_capture.
    """

    capture = validation_capture_snapshot()
    if capture is None:
        raise RuntimeError("no validation capture is active")
    case_id = capture["case_id"]
    if not isinstance(case_id, str):
        raise TypeError("validation capture case id must be text")
    capture_timestamp = time.time_ns()
    artifact_case_id = f"{case_id}_{capture_timestamp}"
    artifact_path = write_diagnostic_artifact(
        artifact_case_id,
        capture,
        artifact_root=artifact_root,
    )
    return artifact_path


def write_diagnostic_artifact(
    case_id: str,
    record: Mapping[str, object],
    *,
    artifact_root: Path = Path("test_artifacts/cognition_core_v2/raw"),
) -> Path:
    """Write structured validation evidence outside source-controlled output.

    Args:
        case_id: Stable lifecycle or benchmark case identity.
        record: Structured evidence captured by the validation harness.
        artifact_root: Ignored root used for raw inspection artifacts.

    Returns:
        The UTF-8 JSON artifact path written for the requested case.
    """

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / f"{case_id}.json"
    artifact_text = json.dumps(
        _json_safe(record),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    artifact_path.write_text(artifact_text, encoding="utf-8")
    return artifact_path


def _phase_payload(
    activation: Mapping[str, object] | None,
    guard_passed: bool,
) -> dict[str, object]:
    """Project one lifecycle state into inspectable fixture evidence."""

    phase = {
        "activation": activation["score"] if activation is not None else 0,
        "trend": activation["trend"] if activation is not None else "inactive",
        "guard_passed": guard_passed,
    }
    return phase


def _json_safe(value: object) -> object:
    """Convert local dataclasses into JSON-ready validation evidence values."""

    if is_dataclass(value):
        dataclass_value = asdict(value)
        return _json_safe(dataclass_value)
    if isinstance(value, Mapping):
        projected = {str(key): _json_safe(item) for key, item in value.items()}
        return projected
    if isinstance(value, list):
        projected = [_json_safe(item) for item in value]
        return projected
    if isinstance(value, tuple):
        projected = [_json_safe(item) for item in value]
        return projected
    return value


def _project_config_identity(config: object) -> dict[str, object]:
    """Keep route identity and generation settings without exposing API keys."""

    thinking = getattr(config, "thinking", None)
    projected = {
        "stage_name": getattr(config, "stage_name", None),
        "route_name": getattr(config, "route_name", None),
        "base_url": getattr(config, "base_url", None),
        "model": getattr(config, "model", None),
        "temperature": getattr(config, "temperature", None),
        "top_p": getattr(config, "top_p", None),
        "top_k": getattr(config, "top_k", None),
        "max_completion_tokens": getattr(config, "max_completion_tokens", None),
        "presence_penalty": getattr(config, "presence_penalty", None),
        "timeout_seconds": getattr(config, "timeout_seconds", None),
        "thinking_enabled": getattr(thinking, "enabled", None),
    }
    return projected
