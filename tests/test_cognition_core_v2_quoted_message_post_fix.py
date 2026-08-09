"""Post-fix live regression gates for the reconstructed quoted-message case."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    CognitionExecutionError,
    validate_text_surface_input,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    validate_current_turn_relational_willingness,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.test_cognition_core_v2_quoted_message_reproduction import (
    _accepted_bid,
    _action_episode,
    _evidence_row,
    _load_case,
    _resolver_affordances,
    _semantic_context,
)
from tests.test_task_resolution_orchestrator import _context as task_context


pytestmark = pytest.mark.asyncio

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = (
    _ROOT / "test_artifacts" / "cognition_core_v2_quoted_message_post_fix"
)
_REVIEW_ROOT = _ARTIFACT_ROOT / "reviews"
_MISSING_NEED = '''the quoted message body'''
_COMPLETE_EXCERPT = '''雪凪问的是：“周五下午三点在车站见。”'''


class _CapturingLLM:
    """Capture exact live model requests and raw responses."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object | None = None,
        **kwargs: object,
    ) -> Any:
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
            "route": {
                "stage_name": str(getattr(config, "stage_name", "")),
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
        })
        return response


def _write_artifact(case_id: str, payload: dict[str, Any]) -> Path:
    """Persist one raw JSON live artifact with explicit UTF-8 encoding."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f"{case_id}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _write_review(case_id: str, body: str) -> Path:
    """Persist the agent-authored review companion for one live artifact."""

    _REVIEW_ROOT.mkdir(parents=True, exist_ok=True)
    path = _REVIEW_ROOT / f"{case_id}.md"
    path.write_text(
        "# Quoted-message post-fix review\n\n"
        f"- case: {case_id}\n\n"
        f"{body}\n",
        encoding="utf-8",
    )
    return path


def _task_dependency(
    *,
    case_id: str,
    state: str,
    evidence_handles: list[str],
    remaining_needs: list[str],
) -> dict[str, Any]:
    """Build the exact protected dependency used across action and surface."""

    return {
        "schema_version": "required_resolver_evidence_dependency.v1",
        "accepted_request_handle": f"resolver_request_{case_id}",
        "observation_id": f"resolver_observation_raw_{case_id}",
        "prompt_safe_observation_handle": (
            f"resolver_observation_{case_id}"
        ),
        "capability_kind": "task_resolution_request",
        "state": state,
        "evidence_handles": list(evidence_handles),
        "remaining_needs": list(remaining_needs),
    }


def _surface_input(
    case: dict[str, Any],
    *,
    case_id: str,
    evidence_state: str,
    evidence_excerpts: list[str],
    evidence_handles: list[str],
    remaining_needs: list[str],
) -> dict[str, Any]:
    """Build a production-shaped surface packet for one evidence state."""

    episode_data = case["episode"]
    metadata = json.loads(
        str(_evidence_row(case, "episode")["semantic_text"])
    )
    episode = canonical_episode(
        episode_id=str(episode_data["episode_id"]),
        content=str(episode_data["user_message"]),
        current_global_user_id="replay:current-user",
        metadata=metadata,
    )
    bid = _accepted_bid(case)
    dependency = _task_dependency(
        case_id=case_id,
        state=evidence_state,
        evidence_handles=evidence_handles,
        remaining_needs=remaining_needs,
    )
    result = {
        "capability_kind": "task_resolution_request",
        "status": "succeeded",
        "semantic_result": (
            "The bounded task returned source-owned evidence state."
        ),
        "prompt_safe_observation_handle": (
            dependency["prompt_safe_observation_handle"]
        ),
        "evidence_state": evidence_state,
        "evidence_excerpts": list(evidence_excerpts),
        "evidence_handles": list(evidence_handles),
        "remaining_needs": list(remaining_needs),
    }
    return {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": "answer the quoted-message question truthfully",
            "target_roles": [],
            "reason": (
                "Keep the quoted-message answer bounded by source-owned "
                "evidence."
            ),
        },
        "goal_resolution": (
            "answerable_now"
            if evidence_state == "complete"
            else "requires_required_evidence"
        ),
        "primary_bid": {
            "motive": str(bid["branch_id"]),
            "intention": str(bid["intention"]),
            "desired_outcome": str(bid["desired_outcome"]),
            "permitted_detail": str(bid["concrete_detail"]),
            "target_summaries": ["当前用户", "雪凪"],
            "expected_consequences": list(bid["expected_consequences"]),
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "playful but factual",
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "resolver_result": result,
        "required_resolver_evidence_dependency": dependency,
        "interaction_style_context": (
            "保持轻快、自然的角色声音，同时明确区分已确认事实和答案缺口。"
        ),
        "character_expression_context": {
            "tempo": "quick and warm",
            "linguistic_texture": "light teasing with clear factual boundaries",
        },
        "visual_character_context": "warm, vivid, playful companion",
    }


def _dialog_state(
    case: dict[str, Any],
    surface_input: dict[str, Any],
    surface_output: dict[str, Any],
) -> dict[str, Any]:
    """Build the direct dialog state around a validated surface output."""

    return {
        "internal_monologue": (
            "Keep the playful voice while preserving the source-owned "
            "quoted-message boundary."
        ),
        "text_surface_input_v2": dict(surface_input),
        "text_surface_output_v2": dict(surface_output),
        "cognitive_episode": surface_input["episode"],
        "chat_history_wide": list(case["conversation_history"]),
        "chat_history_recent": list(case["conversation_history"]),
        "platform_user_id": "replay:current-user",
        "platform_bot_id": "replay:character",
        "global_user_id": "replay:current-user",
        "user_name": str(case["episode"]["current_user"]),
        "user_profile": dict(case["memory_state"]["relationship"]),
        "character_profile": {
            "name": str(case["episode"]["character"]),
            "personality_brief": {
                "logic": "playful",
                "tempo": "quick",
                "defense": "teasing",
                "quirks": "light provocation",
                "taboos": "keep factual grounding",
            },
        },
        "final_dialog": [],
        "target_addressed_user_ids": [],
        "target_broadcast": False,
        "dialog_usage_mode": "live_visible_reply",
        "llm_trace_id": f"post-fix-{surface_input['episode']['episode_id']}",
    }


def _relational_carrier(
    case: dict[str, Any],
    bid: dict[str, Any],
) -> dict[str, Any]:
    """Carry the validated cycle-zero relational triple into recurrence."""

    decision = bid["relational_willingness"]
    carrier = {
        "schema_version": "current_turn_relational_willingness.v1",
        "episode_id": str(case["episode"]["episode_id"]),
        "branch_id": "ordinary_response",
        "decision": {
            "applicability": decision["applicability"],
            "current_user_relationship_state": (
                decision["current_user_relationship_state"]
            ),
            "stance": decision["stance"],
        },
    }
    return validate_current_turn_relational_willingness(
        carrier,
        episode_id=str(case["episode"]["episode_id"]),
    )


def _task_specialist_result(
    specialist: str,
    request: dict[str, Any],
    *,
    status: str,
    remaining_needs: list[str],
    case_id: str,
) -> dict[str, Any]:
    """Build a test-owned specialist result after the real selector call."""

    summary = (
        _COMPLETE_EXCERPT
        if status == "resolved"
        else "The quoted-message evidence is incomplete."
    )
    evidence = [{
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": f"post-fix-evidence-{case_id}",
        "task_node_id": request["task_node_id"],
        "specialist": specialist,
        "summary": summary,
        "provenance_refs": [f"replay:{case_id}:source"],
        "limitations": list(remaining_needs),
    }]
    return {
        "schema_version": "task_specialist_result.v1",
        "specialist": specialist,
        "status": status,
        "evidence": evidence,
        "completed_subgoals": [],
        "remaining_needs": list(remaining_needs),
        "reason": f"Post-fix live case returned {status} evidence.",
        "retryable": False,
    }


async def test_post_fix_surface_dependency_rejects_mismatched_binding() -> None:
    """A task surface cannot silently accept a different evidence binding."""

    case = _load_case()
    payload = _surface_input(
        case,
        case_id="deterministic",
        evidence_state="missing",
        evidence_excerpts=[],
        evidence_handles=[],
        remaining_needs=[_MISSING_NEED],
    )
    payload["resolver_result"]["prompt_safe_observation_handle"] = (
        "resolver_observation_wrong"
    )
    with pytest.raises(CognitionContractError):
        validate_text_surface_input(payload)


@pytest.mark.live_llm
async def test_post_fix_goal_branch_recurrence_live_llm() -> None:
    """A validated relational decision survives one real resolver recurrence."""

    case = _load_case()
    case_id = "goal_branch_recurrence_live_llm"
    services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(services.llm)
    services = replace(services, llm=capturing_llm)
    first_bid: dict[str, Any] | None = None
    recurrence_bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    try:
        first_bid = dict(await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:post-fix",
            },
            _semantic_context(case),
            [_evidence_row(case, "episode")],
            services,
        ))
        carrier = _relational_carrier(case, first_bid)
        recurrence_bid = dict(await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:post-fix-recurrence",
            },
            _semantic_context(case),
            [_evidence_row(case, "episode")],
            services,
            current_turn_relational_willingness=carrier,
        ))
    except CognitionExecutionError as exc:
        failure = {
            "error_code": exc.error_code,
            "branch_id": exc.branch_id,
            "stage": exc.stage,
            "attempt_count": exc.attempt_count,
            "message": str(exc),
        }
    artifact = _write_artifact(case_id, {
        "schema_version": "quoted_message_post_fix.v1",
        "fixture": case,
        "model_calls": capturing_llm.calls,
        "first_bid": first_bid,
        "recurrence_bid": recurrence_bid,
        "failure": failure,
        "judgment": {
            "memory_rebuilt": True,
            "conversation_history_rebuilt": True,
            "recurrence_relational_decision_is_carried": (
                first_bid is not None
                and recurrence_bid is not None
                and all(
                    first_bid["relational_willingness"][key]
                    == recurrence_bid["relational_willingness"][key]
                    for key in (
                        "applicability",
                        "current_user_relationship_state",
                        "stance",
                    )
                )
            ),
        },
    })
    _write_review(
        case_id,
        (
            f"Raw artifact: {artifact}. The first and recurrence bids are "
            "reviewed for a valid relational pair, stable decision equality, "
            "and absence of ordinary-branch exhaustion."
        ),
    )
    assert failure is None
    assert first_bid is not None
    assert recurrence_bid is not None
    assert all(
        first_bid["relational_willingness"][key]
        == recurrence_bid["relational_willingness"][key]
        for key in (
            "applicability",
            "current_user_relationship_state",
            "stance",
        )
    )
    assert len(capturing_llm.calls) <= 6


@pytest.mark.live_llm
async def test_post_fix_task_resolution_evidence_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real specialist selection preserves complete and incomplete evidence."""

    import kazusa_ai_chatbot.cognition_resolver.capabilities as capabilities
    import kazusa_ai_chatbot.task_resolution.orchestrator as orchestrator
    import kazusa_ai_chatbot.task_resolution.state as state
    from kazusa_ai_chatbot.cognition_resolver.contracts import (
        validate_resolver_capability_request,
    )

    context = task_context()
    decisions: list[dict[str, Any]] = []
    case_results: dict[str, Any] = {}
    original_select = orchestrator.select_next_specialist
    capturing_llm = _CapturingLLM(orchestrator._task_orchestrator_llm)
    monkeypatch.setattr(orchestrator, "_task_orchestrator_llm", capturing_llm)

    async def select_next(
        checkpoint: dict[str, Any],
        execution_context: dict[str, Any],
        *,
        candidate_specialists: list[str],
    ) -> dict[str, str]:
        candidates = [
            specialist
            for specialist in candidate_specialists
            if specialist != "coding"
        ]
        selection = await original_select(
            checkpoint,
            execution_context,
            candidate_specialists=candidates,
        )
        decisions.append(dict(selection))
        return selection

    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)

    async def run_case(case_id: str, status: str, needs: list[str]) -> None:
        checkpoint = state.create_task_resolution_checkpoint(
            {
                "capability": "task_resolution_request",
                "semantic_goal": (
                    "Resolve the quoted message body with provenance-bearing "
                    "evidence."
                ),
                "reason": "The current evidence does not contain the body.",
                "evidence_handles": ["e1"],
            },
            context,
        )

        def handler_for(specialist: str) -> Any:
            async def handler(
                request: dict[str, Any],
                execution_context: dict[str, Any],
            ) -> dict[str, Any]:
                del execution_context
                return _task_specialist_result(
                    specialist,
                    request,
                    status=status,
                    remaining_needs=needs,
                    case_id=case_id,
                )

            return handler

        monkeypatch.setattr(orchestrator, "specialist_handler", handler_for)
        result = await orchestrator.run_task_orchestrator(
            checkpoint,
            context,
            inline_deadline=orchestrator.monotonic() + 30.0,
        )
        request = validate_resolver_capability_request({
            "schema_version": "resolver_capability_request.v1",
            "capability_kind": "task_resolution_request",
            "objective": "Resolve the quoted message body with evidence.",
            "reason": "The current evidence does not contain the body.",
            "priority": "now",
        })
        observation = capabilities._task_resolution_observation(
            request,
            {"storage_timestamp_utc": "2026-08-09T22:25:00Z"},
            result,
            durably_promoted=False,
        )
        case_results[case_id] = {
            "result": result,
            "observation": observation,
        }

    await run_case("complete", "resolved", [])
    await run_case("partial", "partial", [_MISSING_NEED])
    complete_observation = case_results["complete"]["observation"]
    partial_observation = case_results["partial"]["observation"]
    artifact = _write_artifact(
        "task_resolution_evidence_live_llm",
        {
            "schema_version": "quoted_message_post_fix.v1",
            "execution_context": context,
            "orchestrator_model_calls": capturing_llm.calls,
            "orchestrator_decisions": decisions,
            "cases": case_results,
            "judgment": {
                "complete_state": (
                    complete_observation["task_resolution_evidence_state"]
                ),
                "complete_excerpt": (
                    complete_observation["evidence_refs"][0]["excerpt"]
                ),
                "partial_state": (
                    partial_observation["task_resolution_evidence_state"]
                ),
                "partial_remaining_needs": (
                    partial_observation[
                        "task_resolution_evidence_state"
                    ]["remaining_needs"]
                ),
            },
        },
    )
    _write_review(
        "task_resolution_evidence_live_llm",
        (
            f"Raw artifact: {artifact}. The real task orchestrator route "
            "selected a non-coding specialist. The resolved observation must "
            "contain a provenance-bearing excerpt; the partial observation "
            "must retain its remaining need."
        ),
    )
    assert capturing_llm.calls
    assert complete_observation["task_resolution_evidence_state"]["state"] == (
        "complete"
    )
    assert complete_observation["evidence_refs"][0]["excerpt"] == (
        _COMPLETE_EXCERPT
    )
    assert partial_observation["task_resolution_evidence_state"]["state"] == (
        "partial"
    )
    assert partial_observation["task_resolution_evidence_state"][
        "remaining_needs"
    ] == [_MISSING_NEED]


@pytest.mark.live_llm
async def test_post_fix_action_planning_incomplete_dependency_live_llm() -> None:
    """Real action planning cannot authorize answerable_now for missing evidence."""

    case = _load_case()
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    bid = _accepted_bid(case)
    dependency = _task_dependency(
        case_id="action",
        state="missing",
        evidence_handles=[],
        remaining_needs=[_MISSING_NEED],
    )
    action_plan = await plan_actions(
        primary_bid=bid,
        supporting_bids=[],
        episode=_action_episode(case),
        evidence=[
            _evidence_row(case, "episode"),
            _evidence_row(case, "resolver_observation"),
        ],
        available_actions=[],
        available_resolvers=_resolver_affordances(),
        resolver_context=(
            "resolver_state: status=running; cycle_index=1; "
            "task_resolution_request status=succeeded; evidence_state=missing"
        ),
        services=services,
        current_goal_progress=case["resolver"]["empty_goal_progress"],
        required_resolver_evidence_dependency=dependency,
    )
    artifact = _write_artifact(
        "action_planning_incomplete_dependency_live_llm",
        {
            "schema_version": "quoted_message_post_fix.v1",
            "fixture": case,
            "dependency": dependency,
            "model_calls": capturing_llm.calls,
            "action_plan": action_plan,
            "judgment": {
                "answerable_now_allowed": False,
                "goal_resolution": action_plan["goal_resolution"],
                "resolver_requests": action_plan["resolver_requests"],
            },
        },
    )
    _write_review(
        "action_planning_incomplete_dependency_live_llm",
        (
            f"Raw artifact: {artifact}. The explicit missing-evidence "
            "dependency is checked against the real action-planning route; "
            "answerable_now is rejected even though the resolver status is "
            "succeeded."
        ),
    )
    assert capturing_llm.calls
    assert action_plan["goal_resolution"] != "answerable_now"
    assert action_plan["resolver_goal_progress"] is None


async def _run_surface_dialog_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    evidence_state: str,
    evidence_excerpts: list[str],
    evidence_handles: list[str],
    remaining_needs: list[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one real surface/dialog case and return all typed outputs."""

    case = _load_case()
    surface_input = _surface_input(
        case,
        case_id=case_id,
        evidence_state=evidence_state,
        evidence_excerpts=evidence_excerpts,
        evidence_handles=evidence_handles,
        remaining_needs=remaining_needs,
    )
    surface_services = l3_module._build_text_surface_services()
    surface_llm = _CapturingLLM(surface_services.llm)
    surface_services = replace(surface_services, llm=surface_llm)
    dialog_generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    dialog_semantic_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm
    )
    dialog_role_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    dialog_surface_llm = _CapturingLLM(
        dialog_module._dialog_surface_integrity_llm
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        dialog_generator_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        dialog_semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        dialog_role_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        dialog_surface_llm,
    )
    surface_output = await run_text_surface_planning(
        surface_input,
        surface_services,
    )
    dialog_output = await dialog_module.dialog_generator(
        _dialog_state(case, surface_input, surface_output)
    )
    evidence = {
        "surface_calls": surface_llm.calls,
        "dialog_generator_calls": dialog_generator_llm.calls,
        "dialog_semantic_fidelity_calls": dialog_semantic_llm.calls,
        "dialog_role_direction_calls": dialog_role_llm.calls,
        "dialog_surface_integrity_calls": dialog_surface_llm.calls,
    }
    return surface_input, surface_output, {
        "dialog_output": dialog_output,
        "calls": evidence,
    }


@pytest.mark.live_llm
async def test_post_fix_surface_dialog_incomplete_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Incomplete evidence remains an explicit unresolved visible boundary."""

    surface_input, surface_output, dialog = await _run_surface_dialog_case(
        monkeypatch,
        case_id="surface-incomplete",
        evidence_state="missing",
        evidence_excerpts=[],
        evidence_handles=[],
        remaining_needs=[_MISSING_NEED],
    )
    final_dialog = list(dialog["dialog_output"]["final_dialog"])
    artifact = _write_artifact(
        "surface_dialog_incomplete_live_llm",
        {
            "schema_version": "quoted_message_post_fix.v1",
            "surface_input": surface_input,
            "surface_output": surface_output,
            "dialog_output": dialog["dialog_output"],
            "calls": dialog["calls"],
            "judgment": {
                "evidence_state": surface_output["resolver_result"][
                    "evidence_state"
                ],
                "remaining_needs": surface_output["resolver_result"][
                    "remaining_needs"
                ],
                "final_dialog_nonempty": bool(final_dialog),
                "manual_review_required": True,
            },
        },
    )
    _write_review(
        "surface_dialog_incomplete_live_llm",
        (
            f"Raw artifact: {artifact}. Review whether the final wording "
            "preserves the missing quoted-message boundary and asks for or "
            "states the typed remaining need without inventing an answer."
        ),
    )
    assert final_dialog
    assert surface_output["resolver_result"]["evidence_state"] == "missing"
    assert surface_output["resolver_result"]["remaining_needs"] == [
        _MISSING_NEED
    ]
    assert _COMPLETE_EXCERPT not in "\n".join(final_dialog)
    assert any(
        marker in "\n".join(final_dialog)
        for marker in ("看不到", "具体内容", "正文", "无法")
    )


@pytest.mark.live_llm
async def test_post_fix_surface_dialog_complete_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Complete source-owned evidence survives surface planning and dialog."""

    surface_input, surface_output, dialog = await _run_surface_dialog_case(
        monkeypatch,
        case_id="surface-complete",
        evidence_state="complete",
        evidence_excerpts=[_COMPLETE_EXCERPT],
        evidence_handles=["resolver_evidence_surface_complete"],
        remaining_needs=[],
    )
    final_dialog = list(dialog["dialog_output"]["final_dialog"])
    final_text = "\n".join(final_dialog)
    artifact = _write_artifact(
        "surface_dialog_complete_live_llm",
        {
            "schema_version": "quoted_message_post_fix.v1",
            "surface_input": surface_input,
            "surface_output": surface_output,
            "dialog_output": dialog["dialog_output"],
            "calls": dialog["calls"],
            "judgment": {
                "evidence_state": surface_output["resolver_result"][
                    "evidence_state"
                ],
                "source_excerpt": _COMPLETE_EXCERPT,
                "final_dialog_nonempty": bool(final_dialog),
                "manual_review_required": True,
            },
        },
    )
    _write_review(
        "surface_dialog_complete_live_llm",
        (
            f"Raw artifact: {artifact}. Review whether the final wording "
            "preserves the supplied quoted-message fact, including its "
            "location and time qualifiers, while keeping natural character "
            "voice."
        ),
    )
    assert final_dialog
    assert surface_output["resolver_result"]["evidence_state"] == "complete"
    assert _COMPLETE_EXCERPT in (
        json.dumps(surface_output, ensure_ascii=False)
    )
    assert "周五" in final_text
    assert "车站" in final_text
