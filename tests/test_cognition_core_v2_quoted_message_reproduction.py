"""Reconstruct the quoted-message failure at each owning boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
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
from kazusa_ai_chatbot.cognition_core_v2 import model_attempt_policy
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = pytest.mark.asyncio

_ROOT = Path(__file__).resolve().parents[1]
_FIXTURE_PATH = (
    _ROOT
    / "tests"
    / "fixtures"
    / "cognition_core_v2_quoted_message_case.json"
)
_REVIEW_ROOT = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2_quoted_message_reproduction"
)
_CASE_NAME = "cognition_core_v2_quoted_message_reproduction"


class _SequenceLLM:
    """Return preserved candidates while retaining every prompt payload."""

    def __init__(self, responses: Sequence[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object,
        **kwargs: object,
    ) -> SimpleNamespace:
        """Return the next preserved response for one bounded model call."""

        del args, kwargs
        if not self.responses:
            raise AssertionError("the replayed model received an extra call")
        response = self.responses.pop(0)
        if isinstance(response, str):
            response_text = response
        else:
            response_text = json.dumps(response, ensure_ascii=False)
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": response_text,
            "route": {
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
        })
        return SimpleNamespace(content=response_text)


class _CapturingLLM:
    """Delegate to a configured live model and preserve raw call evidence."""

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
        """Invoke the real route and retain its exact request and response."""

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
            "raw_output": str(response.content),
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
            "route": {
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
        })
        return response


def _load_case() -> dict[str, Any]:
    """Load the checked-in reconstruction instead of querying live MongoDB."""

    with _FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        case = json.load(fixture_file)
    if case.get("schema_version") != (
        "cognition_core_v2_quoted_message_case.v1"
    ):
        raise AssertionError("quoted-message reconstruction schema is invalid")
    return case


def _evidence_row(
    case: Mapping[str, Any],
    evidence_name: str,
) -> dict[str, Any]:
    """Map one reconstructed evidence record to the V2 prompt contract."""

    raw = case["evidence"][evidence_name]
    source_kind = str(raw["source_kind"])
    semantic_text = str(raw["semantic_text"])
    return {
        "evidence_handle": str(raw["evidence_handle"]),
        "evidence_ref": {
            "source_kind": source_kind,
            "source_id": str(raw["source_id"]),
            "occurred_at": "2026-08-09T22:25:00Z",
            "semantic_summary": str(raw["semantic_summary"]),
        },
        "semantic_text": semantic_text,
        "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS[source_kind]),
        "authority": (
            "current_event"
            if source_kind == "episode"
            else "contextual_fact_only"
        ),
    }


def _semantic_context(case: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the goal prompt context from memory and conversation history."""

    episode = case["episode"]
    memory = case["memory_state"]
    history = case["conversation_history"]
    role_bindings = {
        "self": {
            "role": "actor",
            "entity_kind": "character",
            "entity_id": "character:global",
        },
        "current_user": {
            "role": "target",
            "entity_kind": "user",
            "entity_id": "replay:current-user",
        },
        "p2": {
            "role": "target",
            "entity_kind": "user",
            "entity_id": "replay:p2",
        },
    }
    role_summaries = {
        "self": f"当前角色：{episode['character']}",
        "current_user": f"当前用户：{episode['current_user']}",
        "p2": "第三方参与者：雪凪",
    }
    return {
        "character_identity": {
            "name": episode["character"],
            "personality": "活泼、调侃、主动推进对话",
        },
        "character_constraints": {
            "drives": {},
            "standards": [],
            "meaning_state": {},
            "personality_judgment": {
                "logic": "playful",
                "defense": "teasing",
                "quirks": "light provocation",
                "taboos": "keep factual grounding",
            },
        },
        "scene_context": {
            "channel_scope": "group",
            "character_role": "companion",
            "semantic_scene": str(
                memory["conversation_progress"]["scene_summary"]
            ),
            "public_group_scene": "用户正在群聊中转述第三方问题。",
            "conversation_continuity": "继续当前转述，不把问题正文当作已知事实。",
            "semantic_temporal_context": "immediate",
            "participant_bindings": [
                {
                    "handle": "p2",
                    "display_name": "雪凪",
                    "entity_kind": "third_party",
                },
            ],
        },
        "relationship": dict(memory["relationship"]),
        "past_dialog_cognition_context": "\n".join(
            str(row["body_text"])
            for row in history
            if row["role"] == "assistant"
        ),
        "group_engagement_action_context": {
            "engagement_guidelines": [],
            "confidence": "",
        },
        "private_continuity_context": (
            "保留既有互动风格，但引用正文仍然缺失。"
        ),
        "resolver_context": (
            "resolver_state: status=running; cycle_index=1; "
            "task_resolution_request status=succeeded; "
            f"knowledge_we_know_so_far: {case['resolver']['post_cycle_observation']['knowledge_we_know_so_far']}"
        ),
        "goal_projection": {
            "goal_kind": "ordinary_response",
            "lifecycle": "active",
        },
        "_role_bindings": role_bindings,
        "role_summaries": role_summaries,
    }


def _goal_services(llm: Any) -> SimpleNamespace:
    """Build the smallest service object consumed by direct goal replay."""

    return SimpleNamespace(
        llm=llm,
        goal_ordinary_response_config=SimpleNamespace(
            route_name="replay.goal_ordinary_response",
        ),
        goal_active_branch_config=SimpleNamespace(
            route_name="replay.goal_active_branch",
        ),
    )


def _resolver_affordances() -> list[dict[str, str]]:
    """Rebuild the resolver registry order that made task resolution r4."""

    return [
        {
            "capability": "approval_preparation",
            "semantic_capability": "prepare approval",
            "availability": "available",
        },
        {
            "capability": "human_clarification",
            "semantic_capability": "ask for one user-controlled detail",
            "availability": "available",
        },
        {
            "capability": "self_goal_resolution",
            "semantic_capability": "resolve one internal goal",
            "availability": "available",
        },
        {
            "capability": "task_resolution_request",
            "semantic_capability": "resolve one bounded semantic task",
            "availability": "available",
        },
    ]


def _action_services(llm: Any) -> SimpleNamespace:
    """Build the service object consumed by direct action-planning replay."""

    return SimpleNamespace(
        llm=llm,
        action_planning_config=SimpleNamespace(
            route_name="replay.action_planning",
        ),
        action_authorization_config=SimpleNamespace(
            route_name="replay.action_authorization",
        ),
        resolver_authorization_config=SimpleNamespace(
            route_name="replay.resolver_authorization",
        ),
    )


def _accepted_bid(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build the post-collapse autonomy bid used by the second capsule."""

    candidate = case["captured_goal_candidate"]
    return {
        "branch_id": "autonomy_boundary",
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal:autonomy_boundary:replay",
        },
        "intention": str(candidate["selection"]),
        "desired_outcome": str(candidate["selection"]),
        "concrete_detail": str(candidate["selection"]),
        "reason": str(candidate["reason"]),
        "private_monologue": str(candidate["private_monologue"]),
        "target_roles": [
            {
                "role": "target",
                "entity_kind": "user",
                "entity_id": "replay:current-user",
            },
            {
                "role": "target",
                "entity_kind": "user",
                "entity_id": "replay:p2",
            },
        ],
        "evidence_handles": ["e1"],
        "expected_consequences": list(candidate["expected_consequences"]),
        "confidence": str(candidate["confidence"]),
    }


def _action_plan_response(
    case: Mapping[str, Any],
    response_name: str,
) -> dict[str, Any]:
    """Expand a preserved action-plan candidate into the validator input."""

    raw = deepcopy(case["resolver"][response_name])
    if raw["resolver_goal_progress"] == "use_false_success_progress":
        raw["resolver_goal_progress"] = deepcopy(
            case["resolver"]["false_success_progress"]
        )
    return raw


def _action_episode(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bounded episode projection consumed by action planning."""

    return {
        "episode_id": str(case["episode"]["episode_id"]),
        "trigger_source": "user_message",
        "output_mode": "visible_reply",
    }


def _surface_input(
    case: Mapping[str, Any],
    bid: Mapping[str, Any],
    action_plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the downstream surface input after the false-success handoff."""

    episode_data = case["episode"]
    episode_evidence = _evidence_row(case, "episode")
    metadata = json.loads(str(episode_evidence["semantic_text"]))
    episode = canonical_episode(
        episode_id=str(episode_data["episode_id"]),
        content=str(episode_data["user_message"]),
        current_global_user_id="replay:current-user",
        metadata=metadata,
    )
    primary_bid = {
        "motive": str(bid["branch_id"]),
        "intention": str(bid["intention"]),
        "desired_outcome": str(bid["desired_outcome"]),
        "permitted_detail": str(bid["concrete_detail"]),
        "target_summaries": ["当前用户", "雪凪"],
        "expected_consequences": list(bid["expected_consequences"]),
    }
    selected_operation = dict(metadata["response_operation"])
    return {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": str(action_plan["intention"]["intention"]),
            "target_roles": [],
            "reason": str(action_plan["intention"]["reason"]),
            "selected_response_operation": selected_operation,
        },
        "selected_response_operation": selected_operation,
        "goal_resolution": str(action_plan["goal_resolution"]),
        "primary_bid": primary_bid,
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "playful and teasing",
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "resolver_result": {
            "capability_kind": "task_resolution_request",
            "status": "succeeded",
            "semantic_result": str(
                case["resolver"]["post_cycle_observation"]["semantic_summary"]
            ),
            "prompt_safe_observation_handle": "resolver_observation_replay",
            "evidence_state": "missing",
            "evidence_excerpts": [],
            "evidence_handles": [],
            "remaining_needs": ["the quoted message body"],
        },
        "interaction_style_context": (
            "保留轻快、调侃和亲近的互动风格；事实仍须来自当前证据。"
        ),
        "character_expression_context": {
            "tempo": "quick and playful",
            "linguistic_texture": "short teasing clauses with a warm edge",
        },
        "visual_character_context": "warm, vivid, playful companion",
    }


def _surface_output(case: Mapping[str, Any]) -> dict[str, Any]:
    """Build the captured semantic surface that omitted the answer fact."""

    candidate = case["captured_goal_candidate"]
    return {
        "schema_version": "text_surface_output.v2",
        "content_plan": str(candidate["selection"]),
        "content_requirements": [
            "表达角色勉强愿意继续回应的态度。",
            "保持调侃传话者的角色语气。",
            "等待用户提供或转发具体问题内容。",
        ],
        "visible_boundaries": [],
        "addressee_plan": [],
        "delivery_profile": {
            "lexical_register": "colloquial",
            "sentence_shape": "short complete clauses",
            "rhythm": "light and steady",
            "hesitation": "small playful hesitation",
            "punctuation": "restrained",
        },
        "selected_surface_intent": str(candidate["selection"]),
        "permitted_action_results": [],
        "resolver_result": {
            "capability_kind": "task_resolution_request",
            "status": "succeeded",
            "semantic_result": str(
                case["resolver"]["post_cycle_observation"]["semantic_summary"]
            ),
            "prompt_safe_observation_handle": "resolver_observation_replay",
            "evidence_state": "missing",
            "evidence_excerpts": [],
            "evidence_handles": [],
            "remaining_needs": ["the quoted message body"],
        },
    }


def _dialog_state(
    case: Mapping[str, Any],
    surface_input: Mapping[str, Any],
    surface_output: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the direct production dialog state for the reconstructed case."""

    return {
        "internal_monologue": "继续保持轻快语气，但不能凭空补出引用正文。",
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
        "llm_trace_id": "replay-quoted-message",
    }


async def test_rebuilt_case_reproduces_goal_branch_exhaustion() -> None:
    """Recurrence consumes the ordinary branch's cumulative third attempt."""

    case = _load_case()
    assert len(case["conversation_history"]) == 4
    assert case["episode"]["quoted_message_body"] is None
    assert case["direct_facts"] == []

    invalid_candidate = case["captured_goal_candidate"]
    repaired_candidate = deepcopy(invalid_candidate)
    repaired_candidate["relational_willingness"][
        "current_user_relationship_state"
    ] = "not_applicable"
    llm = _SequenceLLM([
        invalid_candidate,
        repaired_candidate,
        invalid_candidate,
    ])
    error: CognitionExecutionError | None = None
    ledger = model_attempt_policy.create_v2_attempt_ledger(
        "quoted-message-branch-replay"
    )
    ledger_token = model_attempt_policy.bind_v2_attempt_ledger(
        ledger,
        graph_attempt=1,
    )
    try:
        first_bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:replay",
            },
            _semantic_context(case),
            [_evidence_row(case, "episode")],
            _goal_services(llm),
        )
        assert first_bid["relational_willingness"][
            "current_user_relationship_state"
        ] == "not_applicable"
        try:
            await run_goal_cognition(
                DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
                {
                    "scope": "user",
                    "kind": "goal",
                    "entity_id": "goal:ordinary_response:recurrence",
                },
                _semantic_context(case),
                [_evidence_row(case, "episode")],
                _goal_services(llm),
            )
        except CognitionExecutionError as exc:
            error = exc
    except CognitionExecutionError as exc:
        error = exc
    finally:
        model_attempt_policy.reset_v2_attempt_ledger(ledger_token)

    raw_path = write_llm_trace(
        "cognition_core_v2_quoted_message_reproduction",
        "goal_branch_exhaustion_deterministic",
        {
            "fixture": case,
            "model_calls": llm.calls,
            "failure": None
            if error is None
            else {
                "error_code": error.error_code,
                "branch_id": error.branch_id,
                "stage": error.stage,
                "attempt_count": error.attempt_count,
                "message": str(error),
            },
            "judgment": {
                "memory_rebuilt": True,
                "conversation_history_rebuilt": True,
                "missing_quoted_body_preserved": True,
                "failure_mode": (
                    "ordinary_response recovered on local attempt two, then "
                    "exhausted its cumulative third attempt during recurrence"
                ),
            },
        },
    )
    review_path = _write_review(
        "goal_branch_exhaustion_deterministic",
        status="completed",
        body=(
            "The replay fed the captured relational-willingness mismatch to "
            "the first ordinary-response attempt, accepted its captured "
            "repair on attempt two, then fed the mismatch again during "
            f"recurrence. Raw evidence: `{raw_path}`. Expected failure: "
            "`goal_bid_structure_exhausted` on `ordinary_response` before "
            "downstream planning."
        ),
    )

    assert error is not None, f"expected exhaustion; review={review_path}"
    assert error.error_code == "goal_bid_structure_exhausted"
    assert error.branch_id == "ordinary_response"
    assert error.attempt_count == 3
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_rebuilt_case_reproduces_answer_loss_after_false_resolver_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A generic resolver success can reach dialog without the quoted fact."""

    case = _load_case()
    planner_llm = _SequenceLLM([
        _action_plan_response(case, "false_success_action_plan"),
        _action_plan_response(case, "repair_action_plan"),
    ])
    bid = _accepted_bid(case)
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
            "task_resolution_request status=succeeded; "
            f"knowledge_we_know_so_far: {case['resolver']['post_cycle_observation']['knowledge_we_know_so_far']}"
        ),
        services=_action_services(planner_llm),
        current_goal_progress=case["resolver"]["empty_goal_progress"],
    )

    surface_input = _surface_input(case, bid, action_plan)
    surface_output = _surface_output(case)
    dialog_llm = _SequenceLLM([{
        "final_dialog": case["captured_final_dialog"],
    }])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", dialog_llm)
    dialog_output = await dialog_module.dialog_generator(
        _dialog_state(case, surface_input, surface_output)
    )
    final_dialog = dialog_output["final_dialog"]

    raw_path = write_llm_trace(
        "cognition_core_v2_quoted_message_reproduction",
        "answer_loss_after_false_resolver_success_deterministic",
        {
            "fixture": case,
            "planner_calls": planner_llm.calls,
            "action_plan": action_plan,
            "surface_input": surface_input,
            "surface_output": surface_output,
            "dialog_calls": dialog_llm.calls,
            "final_dialog": final_dialog,
            "judgment": {
                "resolver_status": "succeeded",
                "resolver_answer_bearing_facts": case["resolver"][
                    "post_cycle_observation"
                ]["answer_bearing_facts"],
                "direct_facts": case["direct_facts"],
                "goal_resolution": action_plan["goal_resolution"],
                "final_dialog_answers_quoted_question": False,
                "semantic_verifiers": "none",
            },
        },
    )
    review_path = _write_review(
        "answer_loss_after_false_resolver_success_deterministic",
        status="completed",
        body=(
            "The preserved planner candidate first failed on the empty "
            "progress-shell contract, then its captured repair accepted "
            "`answerable_now` with no resolver request. Dialog received a "
            "structurally valid surface and the captured three-message "
            "non-answer. Raw evidence: "
            f"`{raw_path}`."
        ),
    )

    assert action_plan["goal_resolution"] == "answerable_now"
    assert action_plan["resolver_requests"] == []
    assert action_plan["resolver_goal_progress"] is None
    assert case["resolver"]["post_cycle_observation"][
        "answer_bearing_facts"
    ] == []
    assert case["direct_facts"] == []
    assert final_dialog == case["captured_final_dialog"]
    assert not any(
        case["episode"]["quoted_question"] in message
        for message in final_dialog
    ), review_path


@pytest.mark.live_llm
async def test_rebuilt_goal_branch_current_prompt_live_llm() -> None:
    """Run the current goal prompt against the rebuilt memory/history boundary."""

    case = _load_case()
    case_id = "goal_branch_current_prompt_live_llm"
    review_path = _write_review(
        case_id,
        status="started",
        body=(
            "Live call pending. The prompt will receive the reconstructed "
            "relationship memory, active goals, conversation history, e1, "
            "and the missing quoted body."
        ),
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    bid: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    try:
        result = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
            {
                "scope": "user",
                "kind": "goal",
                "entity_id": "goal:ordinary_response:live-replay",
            },
            _semantic_context(case),
            [_evidence_row(case, "episode")],
            services,
        )
        bid = dict(result)
    except CognitionExecutionError as exc:
        failure = {
            "error_code": exc.error_code,
            "branch_id": exc.branch_id,
            "stage": exc.stage,
            "attempt_count": exc.attempt_count,
            "message": str(exc),
        }
    finally:
        raw_path = write_llm_trace(
            _CASE_NAME,
            case_id,
            {
                "fixture": case,
                "model_calls": capturing_llm.calls,
                "validated_bid": bid,
                "failure": failure,
                "judgment": {
                    "quoted_body_available_to_model": (
                        case["episode"]["quoted_message_body"] is not None
                    ),
                    "direct_facts": case["direct_facts"],
                    "review_boundary": "goal contract and relational willingness",
                },
            },
        )
        _write_review(
            case_id,
            status="completed",
            body=(
                f"Live raw artifact: `{raw_path}`. Calls: "
                f"`{len(capturing_llm.calls)}`. Failure: `{failure}`. "
                f"Validated bid present: `{bid is not None}`. Review started "
                f"at `{review_path}`."
            ),
        )

    assert capturing_llm.calls
    if bid is not None:
        relational = bid.get("relational_willingness")
        assert isinstance(relational, dict)
        assert relational["evidence_handles"]
        assert relational["stance"] in {
            "not_applicable",
            "reject",
            "deflect",
            "negotiate",
            "conditional_accept",
            "accept",
        }
    else:
        assert failure is not None
        assert failure["error_code"] in {
            "goal_bid_structure_exhausted",
            "goal_bid_provider_exhausted",
        }


@pytest.mark.live_llm
async def test_rebuilt_false_success_handoff_downstream_live_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run real surface/dialog models after the captured false-success handoff."""

    case = _load_case()
    case_id = "false_success_handoff_downstream_live_llm"
    review_path = _write_review(
        case_id,
        status="started",
        body=(
            "Live downstream call pending. The action-plan boundary is "
            "replayed from the captured false-success repair so the real "
            "surface and dialog models see exactly the missing-fact state."
        ),
    )
    planner_llm = _SequenceLLM([
        _action_plan_response(case, "false_success_action_plan"),
        _action_plan_response(case, "repair_action_plan"),
    ])
    bid = _accepted_bid(case)
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
            "task_resolution_request status=succeeded; "
            f"knowledge_we_know_so_far: {case['resolver']['post_cycle_observation']['knowledge_we_know_so_far']}"
        ),
        services=_action_services(planner_llm),
        current_goal_progress=case["resolver"]["empty_goal_progress"],
    )
    surface_input = _surface_input(case, bid, action_plan)
    surface_services = l3_module._build_text_surface_services()
    surface_llm = _CapturingLLM(surface_services.llm)
    surface_services = replace(surface_services, llm=surface_llm)
    dialog_generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_generator_llm",
        dialog_generator_llm,
    )

    text_output = await run_text_surface_planning(
        surface_input,
        surface_services,
    )
    dialog_output = await dialog_module.dialog_generator(
        _dialog_state(case, surface_input, text_output)
    )
    final_dialog = list(dialog_output["final_dialog"])
    raw_path = write_llm_trace(
        _CASE_NAME,
        case_id,
        {
            "fixture": case,
            "replayed_planner_calls": planner_llm.calls,
            "action_plan": action_plan,
            "surface_input": surface_input,
            "surface_calls": surface_llm.calls,
            "surface_output": text_output,
            "dialog_generator_calls": dialog_generator_llm.calls,
            "final_dialog": final_dialog,
            "judgment": {
                "resolver_status": "succeeded",
                "resolver_answer_bearing_facts": case["resolver"][
                    "post_cycle_observation"
                ]["answer_bearing_facts"],
                "direct_facts": case["direct_facts"],
                "final_dialog_nonempty": bool(final_dialog),
                "manual_review_required": True,
            },
        },
    )
    _write_review(
        case_id,
        status="completed",
        body=(
            f"Live raw artifact: `{raw_path}`. Surface calls: "
            f"`{len(surface_llm.calls)}`; dialog generator calls: "
            f"`{len(dialog_generator_llm.calls)}`; final messages: "
            f"`{len(final_dialog)}`. The upstream resolver observation still "
            "contains no answer-bearing fact, so inspect the raw artifact "
            "for whether downstream wording answers or merely acknowledges "
            f"the question. Review started at `{review_path}`."
        ),
    )

    assert action_plan["goal_resolution"] == "answerable_now"
    assert action_plan["resolver_requests"] == []
    assert case["direct_facts"] == []
    assert case["resolver"]["post_cycle_observation"][
        "answer_bearing_facts"
    ] == []
    assert surface_llm.calls
    assert final_dialog
    assert all(isinstance(message, str) and message.strip() for message in final_dialog)
