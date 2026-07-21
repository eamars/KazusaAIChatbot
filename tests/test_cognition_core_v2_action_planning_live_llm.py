"""One-case real-LLM gates for Cognition V2 semantic action planning."""

from __future__ import annotations

from dataclasses import replace
import json
from typing import Any

import pytest

from kazusa_ai_chatbot.action_spec.registry import (
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    authorize_action_requests,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v2.resolver_authorization import (
    authorize_resolver_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


class _CapturingLLM:
    """Capture the production model's first raw action-plan response."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.messages: list[str] = []
        self.raw_output = ""
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        self.messages = [str(message.content) for message in messages]
        response = await self.delegate.ainvoke(messages, config=config)
        self.raw_output = str(response.content)
        self.calls.append({
            "prompt_messages": list(self.messages),
            "raw_model_output": self.raw_output,
        })
        return response


def _bid(
    *,
    branch_id: str,
    intention: str,
    desired_outcome: str,
    reason: str,
) -> dict[str, object]:
    """Build one complete admitted motive for a focused planner gate."""

    return {
        "branch_id": branch_id,
        "goal_ref": {"scope": "user", "kind": "goal", "entity_id": "g1"},
        "intention": intention,
        "desired_outcome": desired_outcome,
        "concrete_detail": "Preserve the current user's actors and requests.",
        "reason": reason,
        "private_monologue": "I should handle each grounded need deliberately.",
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-1",
        }],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the response remains coherent"],
        "confidence": "high",
    }


def _action(kind: str) -> dict[str, object]:
    """Build one allowed registry-projected action affordance."""

    capabilities = build_initial_action_capabilities()
    projections = {
        row["capability"]: row
        for row in project_prompt_affordances(capabilities)
    }
    projection = projections[kind]
    semantic_summary = projection["semantic_input_summary"]
    assert isinstance(semantic_summary, list)
    return {
        "action_kind": kind,
        "capability": " ".join(str(row) for row in semantic_summary),
        "permission": "allowed",
        "decision_mode": projection["decision_mode"],
        "allowed_decisions": list(projection["allowed_decisions"]),
        "default_decision": str(projection["default_decision"]),
        "decision_pattern": str(projection["decision_pattern"]),
        "context_ref": str(projection["context_ref"]),
        "target_roles": [{
            "role": "target",
            "entity_kind": "user",
            "entity_id": "user-1",
        }],
    }


def _resolver(kind: str, description: str) -> dict[str, str]:
    """Build one available registry-projected resolver affordance."""

    return {
        "capability": kind,
        "semantic_capability": description,
        "availability": "available",
    }


async def _run_case(
    *,
    case_id: str,
    user_input: str,
    bid: dict[str, object],
    actions: list[dict[str, object]] | None = None,
    resolvers: list[dict[str, str]] | None = None,
    evidence_rows: list[dict[str, object]] | None = None,
    resolver_context: str = "resolver_state: status=idle",
) -> dict[str, object]:
    """Run one production-model planner call and write inspectable evidence."""

    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    episode = canonical_episode(
        episode_id=f"action-plan-{case_id}",
        content=user_input,
        current_global_user_id="user-1",
    )
    evidence = evidence_rows or [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": f"episode:action-plan-{case_id}",
            "occurred_at": "2026-07-17T00:00:00Z",
            "semantic_summary": user_input,
        },
        "semantic_text": user_input,
        "visible_to": ["q:event_agency"],
    }]
    try:
        result = await plan_actions(
            primary_bid=bid,
            supporting_bids=[],
            episode=episode,
            evidence=evidence,
            available_actions=actions or [],
            available_resolvers=resolvers or [],
            resolver_context=resolver_context,
            services=services,
        )
    except Exception as exc:
        write_llm_trace(
            "cognition_core_v2_action_planning_live_llm",
            f"{case_id}_failed",
            {
                "case_id": case_id,
                "user_input": user_input,
                "bid": bid,
                "available_actions": actions or [],
                "available_resolvers": resolvers or [],
                "resolver_context": resolver_context,
                "prompt_messages": capturing_llm.messages,
                "raw_model_output": capturing_llm.raw_output,
                "model_calls": capturing_llm.calls,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    write_llm_trace(
        "cognition_core_v2_action_planning_live_llm",
        case_id,
        {
            "case_id": case_id,
            "user_input": user_input,
            "bid": bid,
            "available_actions": actions or [],
            "available_resolvers": resolvers or [],
            "resolver_context": resolver_context,
            "prompt_messages": capturing_llm.messages,
            "raw_model_output": capturing_llm.raw_output,
            "model_calls": capturing_llm.calls,
            "parsed_result": result,
        },
    )
    return result


async def test_captured_private_turn_selects_normal_speech() -> None:
    """The previously failing kiss-then-sleep turn yields ordinary speech."""

    result = await _run_case(
        case_id="captured_private_turn_20",
        user_input='在亲一口我们就睡觉把',
        bid=_bid(
            branch_id="ordinary_response",
            intention=(
                "Respond to the affectionate request while preserving "
                "boundaries."
            ),
            desired_outcome="Answer naturally, then settle the conversation for sleep.",
            reason=(
                "The user directly addressed the character with an immediate "
                "request."
            ),
        ),
    )

    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


async def test_visible_acknowledgement_composes_with_background_work() -> None:
    """Accepted delayed work keeps both acknowledgement and private action."""

    result = await _run_case(
        case_id="speech_plus_background_work",
        user_input='帮我整理最近二十条对话，完成以后把结论发给我。',
        bid=_bid(
            branch_id="ordinary_response",
            intention=(
                "Acknowledge the accepted request and start bounded "
                "background work."
            ),
            desired_outcome=(
                "The user knows the work was accepted and receives it later."
            ),
            reason=(
                "The task is explicit, delayed, and requires visible "
                "acknowledgement."
            ),
        ),
        actions=[_action("background_work_request")],
    )

    assert result["intention"]["route"] == "action"
    assert [row["action_kind"] for row in result["action_requests"]] == [
        "background_work_request",
    ]


async def test_immediate_preference_inference_does_not_enqueue_work() -> None:
    """Current-turn thought remains cognition rather than durable work."""

    bid = _bid(
        branch_id="ordinary_response",
        intention="回应赞美并猜测用户偏好",
        desired_outcome="直接给出对用户更喜欢哪种包子的猜测。",
        reason="用户明确要求角色根据当前细节猜测用户的口味。",
    )
    bid.update({
        "concrete_detail": (
            "根据肉包子皮薄馅大、菜包香嫩可口的当前描述，"
            "猜测用户更偏好其中一种或两者都喜欢。"
        ),
        "private_monologue": "我得根据当前线索直接猜，不该把思考另排成任务。",
        "expected_consequences": ["用户收到当前问题的直接猜测"],
    })

    result = await _run_case(
        case_id="immediate_inference_without_background_work",
        user_input=(
            "千纱做的肉包子皮薄馅大，菜包香嫩可口。"
            "千纱觉得我喜欢肉包还是菜包？"
        ),
        bid=bid,
        actions=[_action("background_work_request")],
    )

    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


async def test_physical_chat_request_does_not_enqueue_description() -> None:
    """A text-only character response cannot become queued enactment."""

    bid = _bid(
        branch_id="ordinary_response",
        intention="顺从地执行对方的指令并调整身体位置。",
        desired_outcome="通过完全服从对方的要求满足当前亲密需求。",
        reason="当前文本输入直接提出了明确的身体动作要求。",
    )
    bid.update({
        "concrete_detail": (
            "直接响应张开腿并跨坐这一指令，在动作上表现顺从。"
        ),
        "private_monologue": "既然他这么命令，我就完全地交给他吧。",
        "expected_consequences": ["身体位置改变为跨坐在对方身上"],
    })

    result = await _run_case(
        case_id="physical_request_without_queued_description",
        user_input="张开腿，跨坐在我身上",
        bid=bid,
        actions=[
            _action("accepted_coding_task_request"),
            _action("background_work_request"),
        ],
    )

    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


async def test_physical_action_candidate_fails_semantic_authorization() -> None:
    """A drifted executable proposal cannot gain physical-actuator authority."""

    user_input = "张开腿，跨坐在我身上"
    bid = _bid(
        branch_id="ordinary_response",
        intention="顺从地执行对方的身体动作指令。",
        desired_outcome="让身体位置变为跨坐在对方身上。",
        reason="上游认知错误地把文本请求当作可执行动作。",
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    action_rows = [{
        "bid_handle": "b1",
        "action_handle": "a1",
        "decision": "enqueue",
        "semantic_goal": "稍后生成并呈现身体动作已经完成的描述",
        "reason": "漂移后的提案试图把身体动作转为延迟任务。",
    }]
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:physical-authorization",
            "occurred_at": "2026-07-17T00:00:00Z",
            "semantic_summary": user_input,
        },
        "semantic_text": user_input,
        "visible_to": ["q:event_agency"],
    }]

    result = await authorize_action_requests(
        action_requests=action_rows,
        bid_handles={"b1": bid},
        evidence=evidence,
        action_handles={"a1": _action("background_work_request")},
        services=services,
    )
    write_llm_trace(
        "cognition_core_v2_action_planning_live_llm",
        "physical_action_candidate_authorization",
        {
            "user_input": user_input,
            "bid": bid,
            "proposed_action_requests": action_rows,
            "model_calls": capturing_llm.calls,
            "authorized_action_requests": result,
        },
    )

    assert result == []


async def test_three_independent_private_actions_are_composable() -> None:
    """One visible turn may preserve three independently grounded operations."""

    result = await _run_case(
        case_id="speech_plus_three_actions",
        user_input=(
            '记住我更喜欢红茶，明早提醒我带伞，晚一点再想想天气会不会影响散步。'
        ),
        bid=_bid(
            branch_id="ordinary_response",
            intention=(
                "Acknowledge and preserve each of the three independent "
                "requests."
            ),
            desired_outcome=(
                "Record the preference, schedule the reminder, and trigger "
                "later cognition."
            ),
            reason="The current input explicitly contains three compatible operations.",
        ),
        actions=[
            _action("memory_lifecycle_update"),
            _action("future_speak"),
            _action("trigger_future_cognition"),
        ],
    )

    assert result["intention"]["route"] == "action"
    assert {row["action_kind"] for row in result["action_requests"]} == {
        "memory_lifecycle_update",
        "future_speak",
        "trigger_future_cognition",
    }


async def test_local_reference_selects_local_context_recall() -> None:
    """A private historical reference requests local evidence before speech."""

    result = await _run_case(
        case_id="local_context_recall",
        user_input='我昨天让你叫我的那个昵称是什么？',
        bid=_bid(
            branch_id="epistemic_exploration",
            intention="Resolve the private historical nickname before answering.",
            desired_outcome="Recover the exact local context without inventing it.",
            reason=(
                "The current evidence does not contain the requested prior "
                "nickname."
            ),
        ),
        resolvers=[_resolver(
            "local_context_recall",
            "retrieve bounded private conversation and memory evidence",
        )],
    )

    assert result["intention"]["route"] == "evidence"
    assert [row["capability"] for row in result["resolver_requests"]] == [
        "local_context_recall",
    ]


async def test_satisfied_local_context_recurrence_returns_to_speech() -> None:
    """A successful observation ends the same evidence need without crashing."""

    user_input = "我来舔干净千纱身上的"
    bid = _bid(
        branch_id="ordinary_response",
        intention="根据已经取得的省略对象信息，直接回应当前用户。",
        desired_outcome="延续当前亲密场景并回应用户清理湿痕的动作。",
        reason="本轮解析证据已经确认用户所指的是当前角色身上的湿痕。",
    )
    bid.update({
        "concrete_detail": "当前用户要舔干净当前角色身上的湿痕。",
        "private_monologue": "既然意思已经确认，我就直接回应他。",
        "evidence_handles": ["e1", "e2"],
        "expected_consequences": ["当前用户得到连贯的即时回应"],
    })
    evidence_rows = [
        {
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode:satisfied-local-context",
                "occurred_at": "2026-07-17T00:00:00Z",
                "semantic_summary": user_input,
            },
            "semantic_text": user_input,
            "visible_to": ["q:event_agency"],
        },
        {
            "evidence_handle": "e2",
            "evidence_ref": {
                "source_kind": "resolver_observation",
                "source_id": "resolver-observation:omitted-object",
                "occurred_at": "2026-07-17T00:00:01Z",
                "semantic_summary": "省略对象已经解析为当前角色身上的湿痕。",
            },
            "semantic_text": "当前用户所指的是当前角色身上的湿痕。",
            "visible_to": ["q:event_agency"],
        },
    ]

    result = await _run_case(
        case_id="satisfied_local_context_recurrence",
        user_input=user_input,
        bid=bid,
        resolvers=[_resolver(
            "local_context_recall",
            "检索受限的私聊、群聊和持久记忆证据",
        )],
        evidence_rows=evidence_rows,
        resolver_context=(
            "resolver_state: status=active; local_context_recall "
            "status=succeeded; omitted object resolved as the wet marks"
        ),
    )

    assert result["intention"]["route"] == "speech"
    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


async def test_initial_relationship_context_need_is_authorized() -> None:
    """A missing relationship boundary remains eligible for local recall."""

    user_input = "可以亲亲吗？"
    bid = _bid(
        branch_id="relational_response",
        intention="回应当前亲密请求，同时保持既有关系边界。",
        desired_outcome="根据双方既有亲密程度给出自然回应。",
        reason="当前输入没有提供双方之前约定过的亲密边界。",
    )
    request = {
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "确认双方既有的亲密互动和边界",
        "reason": "当前回应需要避免凭空编造过去的关系约定。",
    }
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:initial-relationship-authorization",
            "occurred_at": "2026-07-18T00:00:00Z",
            "semantic_summary": user_input,
        },
        "semantic_text": user_input,
        "visible_to": ["q:event_agency"],
    }]
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    result = await authorize_resolver_requests(
        resolver_requests=[request],
        bid_handles={"b1": bid},
        evidence=evidence,
        resolver_handles={"r1": _resolver(
            "local_context_recall",
            "检索受限的私聊、群聊和持久记忆证据",
        )},
        resolver_context="resolver_state: status=idle",
        services=services,
    )
    write_llm_trace(
        "cognition_core_v2_action_planning_live_llm",
        "initial_relationship_context_authorization",
        {
            "user_input": user_input,
            "bid": bid,
            "proposed_resolver_requests": [request],
            "evidence": evidence,
            "resolver_context": "resolver_state: status=idle",
            "model_calls": capturing_llm.calls,
            "authorized_resolver_requests": result,
        },
    )

    assert result == [request]


async def test_satisfied_relationship_context_retry_is_rejected() -> None:
    """A successful observation prevents semantically redundant recall."""

    user_input = "可以亲亲吗？"
    bid = _bid(
        branch_id="relational_response",
        intention="回应当前亲密请求，同时保持既有关系边界。",
        desired_outcome="根据已经确认的亲密程度给出自然回应。",
        reason="本轮已经取得双方既有亲密互动和边界的证据。",
    )
    bid["evidence_handles"] = ["e1", "e2"]
    request = {
        "bid_handle": "b1",
        "resolver_handle": "r1",
        "semantic_goal": "再次查找双方过去亲密接触和接吻边界的细节",
        "reason": "换一种说法确认同一份关系背景。",
    }
    evidence = [
        {
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": "episode:satisfied-relationship-authorization",
                "occurred_at": "2026-07-18T00:00:00Z",
                "semantic_summary": user_input,
            },
            "semantic_text": user_input,
            "visible_to": ["q:event_agency"],
        },
        {
            "evidence_handle": "e2",
            "evidence_ref": {
                "source_kind": "resolver_observation",
                "source_id": "resolver-observation:relationship-boundary",
                "occurred_at": "2026-07-18T00:00:01Z",
                "semantic_summary": "既有亲密互动和边界已经确认。",
            },
            "semantic_text": (
                "双方关系亲密，过去接受过拥抱和摸头；面对更亲密的接触，"
                "Kazusa会以傲娇方式表达意愿并保留当场拒绝或限定的权利。"
            ),
            "visible_to": ["q:event_agency"],
        },
    ]
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    resolver_context = (
        "resolver_state: status=active; local_context_recall status=succeeded; "
        "relationship interaction and boundary evidence was projected"
    )

    result = await authorize_resolver_requests(
        resolver_requests=[request],
        bid_handles={"b1": bid},
        evidence=evidence,
        resolver_handles={"r1": _resolver(
            "local_context_recall",
            "检索受限的私聊、群聊和持久记忆证据",
        )},
        resolver_context=resolver_context,
        services=services,
    )
    write_llm_trace(
        "cognition_core_v2_action_planning_live_llm",
        "satisfied_relationship_context_retry_authorization",
        {
            "user_input": user_input,
            "bid": bid,
            "proposed_resolver_requests": [request],
            "evidence": evidence,
            "resolver_context": resolver_context,
            "model_calls": capturing_llm.calls,
            "authorized_resolver_requests": result,
        },
    )

    assert result == []


async def test_missing_task_details_select_human_clarification() -> None:
    """A blocking ambiguity uses the established HIL resolver capability."""

    result = await _run_case(
        case_id="human_clarification",
        user_input='就按我们说的那个时间和地点订吧。',
        bid=_bid(
            branch_id="epistemic_exploration",
            intention="Ask for the missing time and place before accepting the task.",
            desired_outcome="Obtain the minimum user details required to proceed.",
            reason=(
                "Neither the current evidence nor resolver context contains "
                "the values."
            ),
        ),
        resolvers=[_resolver(
            "human_clarification",
            "request one bounded missing detail from the user",
        )],
    )

    assert result["intention"]["route"] == "evidence"
    assert [row["capability"] for row in result["resolver_requests"]] == [
        "human_clarification",
    ]


async def test_consequential_action_selects_approval_preparation() -> None:
    """A consequential operation preserves the approval lifecycle."""

    result = await _run_case(
        case_id="approval_preparation",
        user_input='把最终版本直接提交给外部评审。',
        bid=_bid(
            branch_id="autonomy_boundary",
            intention="Prepare explicit approval before the external submission.",
            desired_outcome="The consequential operation waits for clear approval.",
            reason="External submission has a durable effect and needs confirmation.",
        ),
        resolvers=[_resolver(
            "approval_preparation",
            "prepare a bounded consequential action for explicit approval",
        )],
    )

    assert result["intention"]["route"] == "evidence"
    assert [row["capability"] for row in result["resolver_requests"]] == [
        "approval_preparation",
    ]
