"""Real-LLM reproductions of captured relational-carrier failures."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter, time_ns
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    EVIDENCE_SOURCE_QUESTION_IDS,
    CognitionExecutionError,
    validate_relational_willingness,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    validate_current_turn_relational_willingness,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.test_cognition_core_v2_quoted_message_reproduction import (
    _load_case,
    _semantic_context,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = (
    _ROOT
    / "tests"
    / "fixtures"
    / "cognition_core_v2_relational_carrier_failure_cases.json"
)
_ARTIFACT_ROOT = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2_relational_carrier_failure_live_llm"
)
_OBSERVED_ERROR_CODE = "current_turn_relational_carrier_invalid"


class _CapturingLLM:
    """Delegate to the configured live route and retain call evidence."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> Any:
        """Invoke the real model and record its request and response."""

        started_at = perf_counter()
        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(getattr(response, "content", "")),
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "route": {
                "stage_name": str(getattr(config, "stage_name", "")),
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
        })
        return response


def _load_cases() -> dict[str, dict[str, Any]]:
    """Load the checked-in summaries of both protected trace cases."""

    fixture = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    if fixture.get("schema_version") != (
        "cognition_core_v2_relational_carrier_failure_cases.v1"
    ):
        raise AssertionError("relational carrier fixture schema is invalid")
    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise AssertionError("relational carrier fixture has no cases")
    return {
        str(case["case_id"]): dict(case)
        for case in cases
        if isinstance(case, dict)
    }


def _captured_case(case_id: str) -> dict[str, Any]:
    """Return one captured case and reject accidental fixture drift."""

    case = _load_cases().get(case_id)
    if case is None:
        raise AssertionError(f"captured case is missing: {case_id}")
    expected = case.get("expected")
    carrier = case.get("carrier")
    evidence = case.get("evidence")
    if (
        not isinstance(expected, dict)
        or not isinstance(carrier, dict)
        or not isinstance(evidence, list)
    ):
        raise TypeError(f"captured case is incomplete: {case_id}")
    source_trace_label = case.get("source_trace_label")
    if (
        not isinstance(source_trace_label, str)
        or not source_trace_label.startswith("trace_case_")
    ):
        raise TypeError(f"captured trace label is invalid: {case_id}")
    if case.get("resolver_cycle_index") != 1:
        raise ValueError(f"captured cycle index is invalid: {case_id}")
    if evidence != [{"evidence_handle": "e1", "source_kind": "episode"}]:
        raise ValueError(f"captured evidence provenance is invalid: {case_id}")
    if expected.get("model_calls_during_failure") != 0:
        raise ValueError(f"captured call expectation is invalid: {case_id}")
    return case


def _validated_carrier(case: dict[str, Any]) -> dict[str, Any]:
    """Prove the captured carrier is valid before exercising its boundary."""

    episode_id = str(case["episode_id"])
    carrier = validate_current_turn_relational_willingness(
        case["carrier"],
        episode_id=episode_id,
    )
    validate_relational_willingness(
        carrier["decision"],
        evidence_handles={"e1"},
        episode_handles={"e1"},
    )
    return carrier


def _goal_context(
    *,
    episode_id: str,
    include_episode_binding: bool,
) -> dict[str, Any]:
    """Build a safe production-shaped goal context for the captured carrier."""

    context = _semantic_context(_load_case())
    if include_episode_binding:
        context["_episode_id"] = episode_id
    return context


def _episode_evidence(episode_id: str) -> dict[str, Any]:
    """Build one current-episode row citing the captured episode identity."""

    return {
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode:{episode_id}",
            "occurred_at": "2026-08-13T00:00:00Z",
            "semantic_summary": "Captured current-episode evidence.",
        },
        "semantic_text": "Current episode evidence for a direct user request.",
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
        "authority": "current_event",
    }


def _services_with_capture() -> tuple[Any, _CapturingLLM]:
    """Build live services while preserving every goal model call."""

    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    return replace(base_services, llm=capturing_llm), capturing_llm


def _failure_details(exc: CognitionExecutionError) -> dict[str, Any]:
    """Serialize the typed failure boundary for the durable artifact."""

    return {
        "error_code": exc.error_code,
        "branch_id": exc.branch_id,
        "stage": exc.stage,
        "attempt_count": exc.attempt_count,
        "safe_checkpoint": exc.safe_checkpoint,
        "retryable": exc.retryable,
        "message": str(exc),
    }


def _write_artifact(case_id: str, payload: dict[str, Any]) -> Path:
    """Persist raw structured evidence for one live reproduction."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f"{case_id}__{time_ns()}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _assert_failure(
    failure: dict[str, Any] | None,
    *,
    expected: dict[str, Any],
) -> None:
    """Assert the common typed contract emitted by both reproductions."""

    assert failure is not None
    assert failure["error_code"] == _OBSERVED_ERROR_CODE
    assert failure["branch_id"] == "ordinary_response"
    assert failure["stage"] == "goal_cognition"
    assert failure["attempt_count"] == expected["attempt_count"]
    assert failure["safe_checkpoint"] == expected["safe_checkpoint"]
    assert failure["retryable"] is False


@pytest.mark.live_llm
async def test_live_f408_carrier_failure_without_episode_binding() -> None:
    """Reproduce the pre-model carrier failure from the first trace case."""

    case = _captured_case("captured_private_missing_episode_binding")
    carrier = _validated_carrier(case)
    services, capturing_llm = _services_with_capture()
    episode_id = str(case["episode_id"])
    control_bid: dict[str, Any] | None = None
    control_failure: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    try:
        control_bid = dict(await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:captured-carrier-control",
            },
            _goal_context(
                episode_id=episode_id,
                include_episode_binding=True,
            ),
            [_episode_evidence(episode_id)],
            services,
            current_turn_relational_willingness=carrier,
        ))
    except CognitionExecutionError as exc:
        control_failure = _failure_details(exc)

    calls_before_failure = len(capturing_llm.calls)
    try:
        await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:captured-carrier-missing-context",
            },
            _goal_context(
                episode_id=episode_id,
                include_episode_binding=False,
            ),
            [_episode_evidence(episode_id)],
            services,
            current_turn_relational_willingness=carrier,
        )
    except CognitionExecutionError as exc:
        failure = _failure_details(exc)

    _write_artifact(case["case_id"], {
        "schema_version": "relational_carrier_failure_live_llm.v1",
        "source_evidence": {
            "trace_label": case["source_trace_label"],
            "observed_failure": {
                "error_code": _OBSERVED_ERROR_CODE,
                "stage": "goal_cognition",
                "branch_id": "ordinary_response",
            },
            "captured_carrier": carrier,
        },
        "control": {
            "bid": control_bid,
            "failure": control_failure,
        },
        "reproduction": {
            "variant": case["failure_variant"],
            "resolver_cycle_index": case["resolver_cycle_index"],
            "captured_evidence": case["evidence"],
            "episode_binding_present": False,
            "evidence_source_kinds": ["episode"],
            "failure": failure,
            "model_calls_before_failure": calls_before_failure,
            "model_calls_after_failure": len(capturing_llm.calls),
        },
        "model_calls": capturing_llm.calls,
    })
    _assert_failure(failure, expected=case["expected"])
    assert control_failure is None
    assert control_bid is not None
    assert len(capturing_llm.calls) == calls_before_failure


@pytest.mark.live_llm
async def test_live_fdd_carrier_failure_without_episode_binding() -> None:
    """Reproduce the pre-model carrier failure from the second trace case."""

    case = _captured_case("captured_group_missing_episode_binding")
    carrier = _validated_carrier(case)
    services, capturing_llm = _services_with_capture()
    episode_id = str(case["episode_id"])
    control_bid: dict[str, Any] | None = None
    control_failure: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None

    try:
        control_bid = dict(await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:captured-carrier-control",
            },
            _goal_context(
                episode_id=episode_id,
                include_episode_binding=True,
            ),
            [_episode_evidence(episode_id)],
            services,
            current_turn_relational_willingness=carrier,
        ))
    except CognitionExecutionError as exc:
        control_failure = _failure_details(exc)

    calls_before_failure = len(capturing_llm.calls)
    try:
        await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:captured-carrier-missing-context",
            },
            _goal_context(
                episode_id=episode_id,
                include_episode_binding=False,
            ),
            [_episode_evidence(episode_id)],
            services,
            current_turn_relational_willingness=carrier,
        )
    except CognitionExecutionError as exc:
        failure = _failure_details(exc)

    _write_artifact(case["case_id"], {
        "schema_version": "relational_carrier_failure_live_llm.v1",
        "source_evidence": {
            "trace_label": case["source_trace_label"],
            "observed_failure": {
                "error_code": _OBSERVED_ERROR_CODE,
                "stage": "goal_cognition",
                "branch_id": "ordinary_response",
            },
            "captured_carrier": carrier,
        },
        "control": {
            "bid": control_bid,
            "failure": control_failure,
        },
        "reproduction": {
            "variant": case["failure_variant"],
            "resolver_cycle_index": case["resolver_cycle_index"],
            "captured_evidence": case["evidence"],
            "episode_binding_present": False,
            "evidence_source_kinds": ["episode"],
            "failure": failure,
            "model_calls_before_failure": calls_before_failure,
            "model_calls_after_failure": len(capturing_llm.calls),
        },
        "model_calls": capturing_llm.calls,
    })
    _assert_failure(failure, expected=case["expected"])
    assert control_failure is None
    assert control_bid is not None
    assert len(capturing_llm.calls) == calls_before_failure
