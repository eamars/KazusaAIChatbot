"""Replay non-ordinary generic goal-schema failures through the live LLM."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from time import perf_counter, time_ns
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    EVIDENCE_SOURCE_QUESTION_IDS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2_self_improvement_schema"
)
_CAPTURED_CASES: tuple[dict[str, object], ...] = (
    {
        "case_id": "plan_trace_0a04_self_improvement",
        "trace_id": "llmtrace_0ef8aa8da3784e0c8a8b65b6b16defdd",
        "trace_path": (
            _ROOT
            / "test_artifacts"
            / "diagnostics"
            / "cognition_trace_0a04c1db64e24dd7870cd3d865179f37.json"
        ),
    },
    {
        "case_id": "plan_trace_a1a573_self_improvement",
        "trace_id": "llmtrace_93482f08e4a74aa5af90adc6e6f5918a",
        "trace_path": (
            _ROOT
            / "test_artifacts"
            / "diagnostics"
            / "cognition_trace_a1a573b590a3494786c4edebdee55342.json"
        ),
    },
    {
        "case_id": "postdraft_d1138_autonomy_boundary",
        "trace_id": "llmtrace_d1138c97929e442d89fc339e125a0fbb",
        "trace_path": (
            _ROOT
            / "test_artifacts"
            / "diagnostics"
            / (
                "postdraft_goal_bid_failure_llmtrace_"
                "d1138c97929e442d89fc339e125a0fbb.json"
            )
        ),
        "stage_name": "goal_cognition.autonomy_boundary.initial",
        "branch_id": "autonomy_boundary",
    },
)
_DEFAULT_STAGE_NAME = "goal_cognition.self_improvement.initial"
_DEFAULT_BRANCH_ID = "self_improvement"
_CAPTURED_OCCURRED_AT = "2026-08-08T00:00:00Z"
_EXPECTED_GENERIC_FIELDS = (
    "intention",
    "desired_outcome",
    "concrete_detail",
    "reason",
    "private_monologue",
    "target_role_handles",
    "evidence_handles",
    "expected_consequences",
    "confidence",
)
_EXPECTED_CONTAMINATED_FIELDS = tuple(
    sorted((*_EXPECTED_GENERIC_FIELDS, "relational_willingness"))
)


class _CapturingLLM:
    """Replay one preserved invalid candidate before live repair."""

    def __init__(self, delegate: Any, first_response_text: str) -> None:
        self.delegate = delegate
        self.first_response_text = first_response_text
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> Any:
        """Invoke the configured model and preserve its complete request."""

        started_at = perf_counter()
        response_source = "live_model"
        if not self.calls:
            response = SimpleNamespace(content=self.first_response_text)
            response_source = "preserved_historical_candidate"
        else:
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
            "raw_output": str(response.content),
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "route": {
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
            "response_source": response_source,
        })
        return response


def _load_captured_attempt(
    case: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one preserved non-ordinary generic goal attempt."""

    trace_path = case["trace_path"]
    trace_id = case["trace_id"]
    stage_name = case.get("stage_name", _DEFAULT_STAGE_NAME)
    if not isinstance(trace_path, Path) or not trace_path.exists():
        raise AssertionError(f"captured production trace is missing: {trace_path}")
    if not isinstance(trace_id, str):
        raise AssertionError("captured trace identifier is invalid")
    if not isinstance(stage_name, str):
        raise AssertionError("captured goal stage name is invalid")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    capsules = trace.get("cognition_failure_capsules") or []
    for capsule in capsules:
        if not isinstance(capsule, Mapping):
            continue
        if capsule.get("trace_id") != trace_id:
            continue
        for attempt in capsule.get("attempts", []):
            if (
                not isinstance(attempt, Mapping)
                or attempt.get("stage_name") != stage_name
            ):
                continue
            messages = attempt.get("messages")
            if not isinstance(messages, list):
                raise AssertionError("captured goal messages are not a list")
            human_messages = [
                message.get("content")
                for message in messages
                if (
                    isinstance(message, Mapping)
                    and message.get("role") == "human"
                )
            ]
            if not human_messages or not isinstance(human_messages[0], str):
                raise AssertionError("captured goal human payload is missing")
            payload = json.loads(human_messages[0])
            if not isinstance(payload, dict):
                raise AssertionError("captured goal payload is not an object")
            return payload, {
                "historical_output": attempt.get("parsed_output"),
                "historical_raw_response_text": str(
                    attempt.get("raw_response_text") or ""
                ),
                "historical_validation_error": str(
                    attempt.get("validation_error") or ""
                ),
                "historical_stage": attempt.get("stage_name"),
            }
    raise AssertionError(
        f"captured stage {stage_name} is missing from {trace_id}"
    )


def _replay_evidence_rows(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild typed evidence rows from the captured goal projection."""

    raw_evidence = payload.get("evidence")
    if not isinstance(raw_evidence, list) or not raw_evidence:
        raise AssertionError("captured generic goal payload has no evidence")
    evidence: list[dict[str, Any]] = []
    for row in raw_evidence:
        if not isinstance(row, Mapping):
            raise AssertionError("captured evidence row is not an object")
        handle = row.get("handle")
        source_kind = row.get("source_kind")
        semantic_text = row.get("semantic_text")
        if not all(
            isinstance(value, str)
            for value in (handle, source_kind, semantic_text)
        ):
            raise AssertionError("captured evidence row is incomplete")
        if source_kind not in EVIDENCE_SOURCE_QUESTION_IDS:
            raise AssertionError(
                f"captured evidence source kind is unsupported: {source_kind}"
            )
        evidence_row: dict[str, Any] = {
            "evidence_handle": handle,
            "evidence_ref": {
                "source_kind": source_kind,
                "source_id": f"captured-self-improvement:{handle}",
                "occurred_at": _CAPTURED_OCCURRED_AT,
                "semantic_summary": semantic_text[:200],
            },
            "semantic_text": semantic_text,
            "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        }
        memory_scope = row.get("memory_scope")
        if isinstance(memory_scope, str):
            evidence_row["memory_scope"] = memory_scope
        evidence.append(evidence_row)
    return evidence


def _replay_semantic_context(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore role bindings required by the current goal builder."""

    semantic_context = payload.get("semantic_context")
    role_handles = payload.get("role_handles")
    if not isinstance(semantic_context, Mapping):
        raise AssertionError("captured semantic context is missing")
    if not isinstance(role_handles, list):
        raise AssertionError("captured role handles are missing")
    context = deepcopy(dict(semantic_context))
    role_bindings: dict[str, dict[str, str]] = {}
    for handle in role_handles:
        if not isinstance(handle, str):
            raise AssertionError("captured role handle is not a string")
        if handle == "self":
            role_bindings[handle] = {
                "role": "actor",
                "entity_kind": "character",
                "entity_id": "character:global",
            }
        elif handle == "current_user":
            role_bindings[handle] = {
                "role": "target",
                "entity_kind": "user",
                "entity_id": "replay:current-user",
            }
        else:
            role_bindings[handle] = {
                "role": "target",
                "entity_kind": "user",
                "entity_id": f"replay:{handle}",
            }
    context["_role_bindings"] = role_bindings
    role_summaries = payload.get("role_summaries")
    context["role_summaries"] = (
        dict(role_summaries) if isinstance(role_summaries, Mapping) else {}
    )
    return context


def _write_artifact(
    case_id: str,
    artifact: Mapping[str, Any],
) -> Path:
    """Write one durable raw replay artifact for human inspection."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f"{case_id}__{time_ns()}.json"
    path.write_text(
        json.dumps(dict(artifact), ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    return path


def _assert_branch_scoped_requests(
    calls: list[dict[str, Any]],
    branch_id: str,
) -> None:
    """Require every non-ordinary replay request to stay generic."""

    assert calls
    initial_system_prompt = calls[0]["messages"][0]["content"]
    assert all(
        call["messages"][0]["content"] == initial_system_prompt
        for call in calls
    )
    first_repair_payload = json.loads(calls[1]["messages"][1]["content"])
    first_repair_feedback = first_repair_payload["repair_feedback"]
    assert isinstance(first_repair_feedback, Mapping)
    assert first_repair_feedback["observed_top_level_fields"] == list(
        _EXPECTED_CONTAMINATED_FIELDS
    )
    assert first_repair_feedback["missing_top_level_fields"] == []
    assert first_repair_feedback["unexpected_top_level_fields"] == [
        "relational_willingness"
    ]
    assert "invalid_draft" not in json.dumps(
        first_repair_feedback,
        ensure_ascii=False,
    )
    assert calls[0]["raw_output"]
    assert calls[0]["raw_output"] not in calls[1]["messages"][1]["content"]
    for call in calls:
        messages = call["messages"]
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert "relational_willingness" not in messages[0]["content"]
        human_payload = json.loads(messages[1]["content"])
        assert human_payload["branch"]["goal_kind"] == branch_id
        feedback = human_payload.get("repair_feedback")
        if not isinstance(feedback, Mapping):
            continue
        if feedback.get("validation_error") != (
            "goal bid draft fields are not exact"
        ):
            continue
        goal_output_contract = feedback.get("goal_output_contract")
        assert isinstance(goal_output_contract, Mapping)
        assert goal_output_contract["top_level_fields"] == list(
            _EXPECTED_GENERIC_FIELDS
        )
        assert set(goal_output_contract["field_types"]) == set(
            _EXPECTED_GENERIC_FIELDS
        )
        for field_name in (
            "observed_top_level_fields",
            "missing_top_level_fields",
            "unexpected_top_level_fields",
        ):
            assert isinstance(feedback[field_name], list)
        assert feedback["missing_top_level_fields"] == []
        assert feedback["unexpected_top_level_fields"] == [
            "relational_willingness"
        ]
        assert "invalid_draft" not in feedback
        assert "relational_willingness" not in goal_output_contract[
            "field_types"
        ]
        assert "relational_willingness_contract" not in feedback
        assert "relational_willingness" not in " ".join(
            str(item) for item in feedback["repair_instruction"]
        )


async def _run_captured_case(case: Mapping[str, object]) -> None:
    """Replay one captured non-ordinary generic input through V2."""

    payload, historical = _load_captured_attempt(case)
    branch_id = case.get("branch_id", _DEFAULT_BRANCH_ID)
    if not isinstance(branch_id, str):
        raise AssertionError("captured goal branch id is invalid")
    if branch_id not in DEFAULT_BRANCH_DEFINITIONS:
        raise AssertionError(f"captured goal branch is unsupported: {branch_id}")
    evidence = _replay_evidence_rows(payload)
    semantic_context = _replay_semantic_context(payload)
    base_services = build_cognition_core_services()
    historical_raw_response_text = historical[
        "historical_raw_response_text"
    ]
    if not historical_raw_response_text:
        raise AssertionError("captured goal response text is missing")
    capturing_llm = _CapturingLLM(
        base_services.llm,
        historical_raw_response_text,
    )
    services = replace(base_services, llm=capturing_llm)
    bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    try:
        result = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS[branch_id],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": f"goal:{case['case_id']}",
            },
            semantic_context,
            evidence,
            services,
        )
        if isinstance(result, dict):
            bid = result
    except CognitionExecutionError as exc:
        failure = {
            "error_code": exc.error_code,
            "message": str(exc),
            "attempt_count": exc.attempt_count,
        }
    artifact = {
        "schema_version": "cognition_core_v2_self_improvement_schema.v1",
        "case_id": case["case_id"],
        "source_trace_id": case["trace_id"],
        "source_trace_path": str(case["trace_path"]),
        "stage_name": case.get("stage_name", _DEFAULT_STAGE_NAME),
        "branch_id": branch_id,
        "replay_mode": "preserved_historical_candidate_then_live_repair",
        "historical_attempt": historical,
        "model_calls": capturing_llm.calls,
        "observed_failure": failure,
        "validated_bid": bid,
        "downstream_boundary": {
            "mode": "direct_goal_owner_replay",
            "action_planning_invoked": False,
            "dialog_input_created": False,
            "invalid_bid_produced": False,
        },
    }
    artifact_path = _write_artifact(str(case["case_id"]), artifact)
    if failure is not None:
        pytest.fail(
            f"self-improvement replay failed: {failure}; "
            f"artifact={artifact_path}"
        )
    if bid is None:
        pytest.fail(
            f"self-improvement replay returned no bid; artifact={artifact_path}"
        )
    assert 2 <= len(capturing_llm.calls) <= 3
    assert capturing_llm.calls[0]["response_source"] == (
        "preserved_historical_candidate"
    )
    assert capturing_llm.calls[1]["response_source"] == "live_model"
    assert bid["branch_id"] == branch_id
    assert "relational_willingness" not in bid
    assert set(bid) == {
        "branch_id",
        "goal_ref",
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "private_monologue",
        "target_roles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
    }
    _assert_branch_scoped_requests(capturing_llm.calls, branch_id)


async def test_plan_trace_0a04_self_improvement_schema_live_llm() -> None:
    """The first plan trace recovers its generic self-improvement goal."""

    await _run_captured_case(_CAPTURED_CASES[0])


async def test_plan_trace_a1a573_self_improvement_schema_live_llm() -> None:
    """The second plan trace recovers its generic self-improvement goal."""

    await _run_captured_case(_CAPTURED_CASES[1])


async def test_postdraft_d1138_autonomy_boundary_schema_live_llm() -> None:
    """A post-draft autonomy branch uses the generic schema contract."""

    await _run_captured_case(_CAPTURED_CASES[2])
