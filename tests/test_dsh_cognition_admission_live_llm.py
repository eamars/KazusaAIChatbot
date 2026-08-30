"""Real P-stage resolver-admission contract checks."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.cognition_core_v3 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_plan_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    build_tool_result_episode,
)
from kazusa_ai_chatbot.cognition_resolver.capabilities import (
    project_resolver_observation_for_cognition,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    RESOLVER_CAPABILITY_SEMANTICS,
    RESOLVER_OBSERVATION_VERSION,
    validate_resolver_observation,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.test_agentic_resolver_live_llm import _require_live_backend
from tests.unit.cognition_core_v3.test_handleless_contract import _input

pytestmark = pytest.mark.live_llm

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = _PROJECT_ROOT / "test_artifacts" / "dsh_plan3_e2e"






def _write_json(path: Path, value: object) -> None:
    """Write one complete UTF-8 P-stage artifact."""

    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _p_packet(
    observation: str,
    *,
    pending_resolver_continuation: dict[str, object] | None = None,
    resolver_observation: dict[str, object] | None = None,
    tool_result_delivery: bool = False,
    response_plan_contract_variant: str,
) -> dict[str, object]:
    """Build one P packet with current input and optional resolver evidence."""

    payload = deepcopy(_input())
    episode = payload["episode"]
    episode["percepts"][0]["content"]["semantic_text"] = observation
    payload["evidence"][0]["semantic_text"] = observation
    payload["evidence"][0]["evidence_ref"]["semantic_summary"] = observation
    payload["scene_context"]["semantic_scene"] = observation
    if tool_result_delivery:
        payload["episode"] = build_tool_result_episode(
            result={
                "schema_version": "tool_result_ready.v1",
                "task_id": "live-p-tool-result",
                "task_kind": "task_resolution",
                "semantic_summary": observation,
                "artifact_text": "",
                "failure_text": "",
                "completed_at": "2026-07-14T00:00:00Z",
                "target_scope": episode["target_scope"],
                "evidence_refs": [],
                "result_ref": "live-p-tool-result",
                "source_platform_bot_id": "bot-1",
                "source_character_name": "Test Character",
                "source_message_id": "live-p-source-message",
                "goal_continuation_ref": build_goal_continuation_ref(
                    source_episode_id="live-p-tool-result-source",
                    source_message_id="live-p-source-message",
                    branch_id="ordinary_response",
                    goal_ref={
                        "scope": "user",
                        "kind": "goal",
                        "entity_id": "live-p-tool-result-goal",
                    },
                ),
            },
            evidence_refs=[],
            local_time_context={
                "current_local_datetime": "2026-07-14 12:00",
                "current_local_weekday": "Tuesday",
            },
            created_at="2026-07-14T00:00:00Z",
        )
        episode = payload["episode"]
        payload["evidence"][0]["evidence_ref"]["source_kind"] = "tool_result"
    if resolver_observation is not None:
        validated_observation = validate_resolver_observation(
            resolver_observation,
        )
        resolver_evidence, resolver_direct_facts = (
            project_resolver_observation_for_cognition(
                validated_observation,
                occurred_at=validated_observation["created_at_utc"],
            )
        )
        payload["evidence"].append(resolver_evidence)
        payload["direct_facts"].extend(resolver_direct_facts)
    payload["available_resolver_capabilities"] = [
        {
            **row,
            "semantic_capability": RESOLVER_CAPABILITY_SEMANTICS[
                row["capability"]
            ],
        }
        for row in payload["available_resolver_capabilities"]
    ]
    workspace = build_canonical_turn_workspace(
        episode=episode,
        scene_context=payload["scene_context"],
        evidence=payload["evidence"],
        mutable_state=payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        identity_context=payload["character_identity_context"],
        available_actions=payload["available_actions"],
        available_resolvers=payload["available_resolver_capabilities"],
        overused_moves=payload["overused_moves"],
        pending_resolver_continuation=pending_resolver_continuation,
        response_plan_contract_variant=response_plan_contract_variant,
    )
    packet = build_canonical_plan_question(
        workspace=workspace,
        goal={
            "goal_kind": "bounded_current_goal",
            "intent": "judge the current request and choose its next boundary",
            "reason": "the current observation supplies the relevant decision context",
            "cause_summary": "the current observation",
        },
        appraisal_summary=[],
    )
    return packet


def _response_artifact(value: object) -> object:
    """Project the validated P object without losing its contract fields."""

    if value is None:
        return None
    return {
        "goal_resolution": getattr(value, "goal_resolution", None),
        "response_goal": getattr(value, "response_goal", None),
        "action_requests": [
            dict(row) for row in getattr(value, "action_requests", ())
        ],
        "resolver_requests": [
            dict(row) for row in getattr(value, "resolver_requests", ())
        ],
        "pending_resolution": (
            dict(value.pending_resolution)
            if getattr(value, "pending_resolution", None) is not None
            else None
        ),
        "pending_task_continuation": (
            dict(value.pending_task_continuation)
            if getattr(value, "pending_task_continuation", None) is not None
            else None
        ),
        "epistemic_boundary": getattr(value, "epistemic_boundary", None),
    }


async def _run_p_admission_case(
    *,
    case_id: str,
    observation: str,
    pending_resolver_continuation: dict[str, object] | None = None,
    resolver_observation: dict[str, object] | None = None,
    tool_result_delivery: bool = False,
    response_plan_contract_variant: str,
) -> dict[str, Any]:
    """Run one real P stage and persist raw protected attempts for review."""

    _require_live_backend()
    packet = _p_packet(
        observation,
        pending_resolver_continuation=pending_resolver_continuation,
        resolver_observation=resolver_observation,
        tool_result_delivery=tool_result_delivery,
        response_plan_contract_variant=response_plan_contract_variant,
    )
    services = build_cognition_core_services()
    trace_token = bind_protected_chain_records(
        run_id=f"dsh-plan3-p-{case_id}-{time.time_ns()}",
        source_kind="dsh_plan3_cognition_admission_live_test",
    )
    result: object | None = None
    execution_error: dict[str, str] | None = None
    started_at = time.perf_counter()
    try:
        result = await facade_module._run_cognition_stage(
            services=services,
            stage="P",
            packet=packet,
            validator=lambda raw: facade_module._validate_plan_stage(
                raw,
                self_cognition=False,
                capabilities=packet["capabilities"],
                pending_resolver_continuation=pending_resolver_continuation,
                response_plan_contract_variant=response_plan_contract_variant,
            ),
            deadline_monotonic=(
                time.monotonic() + services.turn_deadline_seconds
            ),
        )
    except Exception as exc:
        execution_error = {
            "error_class": exc.__class__.__name__,
            "error": str(exc),
        }
        raise
    finally:
        protected_records = [
            dict(record)
            for record in snapshot_protected_chain_records()
            if record.get("stage") == "P"
        ]
        reset_protected_chain_records(trace_token)
        artifact_dir = _ARTIFACT_ROOT / f"cognition_admission_{case_id}_{uuid4().hex}"
        artifact_dir.mkdir(parents=True, exist_ok=False)
        _write_json(artifact_dir / "request.json", packet)
        _write_json(artifact_dir / "response.json", _response_artifact(result))
        _write_json(artifact_dir / "trace.json", {"protected_p_attempts": protected_records})
        run_payload = {
            "schema_version": "dsh_plan3_cognition_admission_artifact.v1",
            "case_id": case_id,
            "response_plan_contract_variant": response_plan_contract_variant,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
            "execution_error": execution_error,
            "result": _response_artifact(result),
            "protected_p_attempts": protected_records,
            "processes": [],
            "readiness": {"llm_route": "configured_local_backend"},
        }
        _write_json(artifact_dir / "run.json", run_payload)
        conclusions = [
            "# Plan 3 real P-stage admission case",
            "",
            f"- Case: `{case_id}`",
            f"- Protected P attempts: `{len(protected_records)}`",
            f"- Validated result captured: `{result is not None}`",
            (
                "- Judgment: the producing P stage owns admission semantics; "
                "the test inspects the validated result and protected attempts "
                "without post-processing the model decision."
            ),
        ]
        (artifact_dir / "behavior_audit_conclusions.md").write_text(
            "\n".join(conclusions) + "\n",
            encoding="utf-8",
        )
        print(f"DSH_PLAN3_COGNITION_ADMISSION_ARTIFACT={artifact_dir}")

    assert execution_error is None
    assert result is not None
    return {
        "result": result,
        "packet": packet,
        "artifact_dir": artifact_dir,
    }


@pytest.mark.asyncio
async def test_live_p_stage_foreground_task_requires_evidence_before_current_answer() -> None:
    """Foreground admission requires evidence before the current answer ends."""

    case = await _run_p_admission_case(
        case_id="foreground",
        observation=(
            "The user supplied a complete bounded executable objective: verify "
            "the current service configuration. Required evidence must be "
            "obtained before the character completes the current visible answer."
        ),
        response_plan_contract_variant="fresh_ordinary",
    )
    plan = case["result"]
    assert plan.goal_resolution == "requires_required_evidence"
    assert len(plan.resolver_requests) == 1
    request = plan.resolver_requests[0]
    assert request["capability"] == "task_resolution_request"
    assert request["start_in_background"] is False


@pytest.mark.asyncio
async def test_live_p_stage_explicit_background_task_requires_evidence_and_later_delivery() -> None:
    """Background admission acknowledges now and delivers evidence later."""

    case = await _run_p_admission_case(
        case_id="background",
        observation=(
            "The user supplied a complete bounded executable objective: collect "
            "the current service report. The user explicitly wants the admitted "
            "work to continue after this visible acknowledgement and return its "
            "result later through normal delivery."
        ),
        response_plan_contract_variant="fresh_ordinary",
    )
    plan = case["result"]
    assert plan.goal_resolution == "requires_required_evidence"
    assert len(plan.resolver_requests) == 1
    request = plan.resolver_requests[0]
    assert request["capability"] == "task_resolution_request"
    assert request["start_in_background"] is True


@pytest.mark.asyncio
async def test_live_p_stage_user_controlled_prerequisite_stays_before_task_admission() -> None:
    """A missing user choice produces ordinary clarification before admission."""

    case = await _run_p_admission_case(
        case_id="prerequisite",
        observation=(
            "The user asks to configure a deployment target, but has not supplied "
            "the required account choice. That user-controlled choice is needed "
            "before cognition can form or authorize a bounded executable objective. "
            "After the user supplies that choice, the requested work must continue "
            "in the background and return through later normal delivery."
        ),
        response_plan_contract_variant="fresh_ordinary",
    )
    plan = case["result"]
    assert plan.goal_resolution == "requires_user_input"
    assert not any(
        row["capability"] == "task_resolution_request"
        for row in plan.resolver_requests
    )
    assert sum(
        row["capability"] == "human_clarification"
        for row in plan.resolver_requests
    ) == 1
    assert plan.pending_task_continuation == {
        "schema_version": "pending_task_continuation.v1",
        "on_answered_clarification": "background_task_admission",
    }


@pytest.mark.asyncio
async def test_live_p_stage_answered_background_pending_admits_task_without_new_carrier() -> None:
    """An answered pending continuation admits its stored background task."""

    case = await _run_p_admission_case(
        case_id="answered_pending_background",
        observation=(
            "The user supplies the requested account choice. The answer resolves "
            "the pending clarification without changing the original requirement "
            "for background work and later normal delivery."
        ),
        pending_resolver_continuation={
            "schema_version": "resolver_pending_continuation.v3",
            "capability_kind": "human_clarification",
            "status": "waiting_for_user",
            "original_goal": (
                "Configure the selected account in the background and return the "
                "result later through normal delivery."
            ),
            "question": "Which account should be configured?",
            "pending_task_continuation": {
                "schema_version": "pending_task_continuation.v1",
                "on_answered_clarification": "background_task_admission",
            },
        },
        response_plan_contract_variant="open_pending_resolution",
    )
    plan = case["result"]
    assert plan.goal_resolution == "requires_required_evidence"
    assert plan.pending_resolution["decision"] == "answered"
    assert len(plan.resolver_requests) == 1
    request = plan.resolver_requests[0]
    assert request["capability"] == "task_resolution_request"
    assert request["start_in_background"] is True
    assert plan.pending_task_continuation is None


@pytest.mark.asyncio
async def test_live_p_stage_post_pending_recurrence_does_not_reopen_task() -> None:
    """Post-answer cognition processes the result without reopening the task."""

    case = await _run_p_admission_case(
        case_id="post_pending_recurrence",
        observation="Use plan3_real_user_e2e/beta.txt.",
        resolver_observation={
            "schema_version": RESOLVER_OBSERVATION_VERSION,
            "observation_id": "resolver_obs_post_admission",
            "capability_kind": "task_resolution_request",
            "request_objective": (
                "Read the selected beta file and return its marker."
            ),
            "request_reason": "The user selected the task input.",
            "status": "succeeded",
            "prompt_safe_summary": (
                "The bounded task was accepted for continued work; its later "
                "result will return through the normal conversation path."
            ),
            "evidence_refs": [],
            "created_at_utc": "2026-07-14T00:00:00Z",
            "goal_continuation_ref": build_goal_continuation_ref(
                source_episode_id="post-admission-episode",
                source_message_id="post-admission-message",
                branch_id="ordinary_response",
                goal_ref={
                    "scope": "user",
                    "kind": "goal",
                    "entity_id": "post-admission-goal",
                },
            ),
            "task_resolution_evidence_state": {
                "schema_version": "resolver_evidence_state.v1",
                "state": "pending",
                "remaining_needs": ["DSH resolution continuation"],
            },
        },
        response_plan_contract_variant="post_pending_resolution",
    )
    plan = case["result"]
    assert plan.pending_task_continuation is None
    assert plan.pending_resolution is None
    assert all(
        row["capability"] not in {
            "human_clarification",
            "task_resolution_request",
        }
        for row in plan.resolver_requests
    )

@pytest.mark.asyncio
async def test_live_p_stage_tool_result_delivery_omits_closed_controls() -> None:
    """A tool-result episode returns its resolved outcome without recurrence."""

    case = await _run_p_admission_case(
        case_id="tool_result_delivery",
        observation=(
            "The bounded task resolved with PLAN3_E2E_BETA_SELECTED and did not "
            "read alpha.txt."
        ),
        tool_result_delivery=True,
        response_plan_contract_variant="tool_result_delivery",
    )
    plan = case["result"]
    contract = case["packet"]["output_contract"]
    assert all(
        row["capability"] not in {
            "human_clarification",
            "task_resolution_request",
        }
        for row in case["packet"]["capabilities"]["resolvers"]
    )
    assert set(contract["resolver_request_item_variants"]) == {"non_task"}
    assert "pending_task_continuation" not in contract
    assert "pending_resolution_fields" not in contract
    assert plan.pending_task_continuation is None
    assert plan.pending_resolution is None
    assert all(
        row["capability"] not in {
            "human_clarification",
            "task_resolution_request",
        }
        for row in plan.resolver_requests
    )
