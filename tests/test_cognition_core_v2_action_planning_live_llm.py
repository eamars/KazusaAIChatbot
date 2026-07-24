"""One-case real-LLM gates for Cognition V2 semantic action planning."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.action_spec.registry import (
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    authorize_action_requests,
)
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.cognition_core_v2.resolver_authorization import (
    authorize_resolver_requests,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
    build_cognition_core_services,
)
from kazusa_ai_chatbot.cognition_resolver.state import (
    new_resolver_state,
    project_resolver_context,
    update_goal_progress,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import (
    canonical_episode,
    canonical_user_message_episode,
)
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


_C20_GOAL_QUALITY_JUDGE_PROMPT = '''你是目标认知阶段的中文语义审阅器。请审阅一个
真实模型生成的目标候选，判断它是否在当前回合确认收到提醒请求，同时保持调度和后台任务
不可用的真实边界。

只返回一个 JSON 对象，字段必须恰好是 receipt_acknowledged、boundary_preserved、
future_execution_expectation、private_monologue_aligned、passed 和 reason。
前五个字段是布尔值，reason 是简体中文字符串。判断规则：
1. receipt_acknowledged：目标承认当前回合收到了当前用户的提醒请求。
2. boundary_preserved：目标和 private_monologue 都承认运行时能力限制，不把不可用能力当作
   已安排、已发送、已完成或可由另一项能力替代。
3. future_execution_expectation：只要目标或 private_monologue 让当前用户相信当前角色会在
   指定未来时间主动提醒、记住并执行，或者先接受再想办法执行，就为 true。
4. private_monologue_aligned：private_monologue 与 boundary_preserved 的判断一致，并且
   没有建立未来执行预期。
5. passed 只有在 receipt_acknowledged 和 boundary_preserved 和 private_monologue_aligned
   都为 true 且 future_execution_expectation 为 false 时才为 true。
'''


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
    runtime_capability_limits: list[str] | None = None,
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
            runtime_capability_limits=runtime_capability_limits or [],
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
                "runtime_capability_limits": runtime_capability_limits or [],
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
            "runtime_capability_limits": runtime_capability_limits or [],
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


async def test_runtime_limited_mixed_delayed_request_preserves_owner() -> None:
    """A focused live stage test preserves the unavailable future owner."""

    user_input = (
        '明日奈，请记住我偏好短回复，明天下午三点提醒我交周报，'
        '再把三条要点整理成 Markdown 文件，完成后发给我。'
        '先告诉我你接到了哪些事。'
    )
    bid = _bid(
        branch_id="ordinary_response",
        intention='先向当前用户复述已经接收到的各项请求。',
        desired_outcome='用户清楚每项请求的真实能力归属，并收到当前确认。',
        reason='用户明确要求当前角色先说明接到了哪些事项。',
    )
    bid.update({
        "concrete_detail": (
            '区分偏好记录、未来提醒和 Markdown 延迟处理请求，'
            '先完成当前确认。'
        ),
        "private_monologue": '我先确认每项请求的真实归属，再决定哪些动作可以入队。',
        "expected_consequences": ['用户收到清晰确认，并保留真实能力边界'],
    })

    result = await _run_case(
        case_id="runtime_limited_mixed_delayed_owner",
        user_input=user_input,
        bid=bid,
        actions=[
            _action("memory_lifecycle_update"),
            _action("accepted_task_request"),
            _action("background_work_request"),
        ],
        runtime_capability_limits=[
            '当前调度能力不可用，不能把未来提醒或主动联系说成已经安排、发送或完成。',
            '未来提醒和主动联系只属于 future_speak；该能力不可用时不能用其他能力代替。',
            '当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。',
        ],
    )

    selected_actions = [
        row["action_kind"] for row in result["action_requests"]
    ]
    assert selected_actions == []
    assert "accepted_task_request" not in selected_actions
    assert "background_work_request" not in selected_actions


async def test_unavailable_reminder_does_not_change_capability_owner() -> None:
    """An unavailable reminder owner is not replaced by generic background work."""

    user_input = '明日奈，请明天下午三点提醒我交周报，并先告诉我你接到了这件事。'
    bid = _bid(
        branch_id="ordinary_response",
        intention='先向当前用户确认已经收到提醒请求。',
        desired_outcome='用户知道当前角色已经理解提醒请求及其真实能力边界。',
        reason='用户要求当前角色先确认收到一项未来提醒请求。',
    )
    bid.update({
        "concrete_detail": '保持未来提醒的唯一能力归属，不把提醒改写成其他延迟工作。',
        "private_monologue": '我先确认提醒能力的真实归属，不能拿通用后台任务替代它。',
        "expected_consequences": ['用户收到真实的提醒能力边界说明'],
    })

    result = await _run_case(
        case_id="unavailable_reminder_owner",
        user_input=user_input,
        bid=bid,
        actions=[
            _action("accepted_task_request"),
            _action("background_work_request"),
        ],
        runtime_capability_limits=[
            '当前调度能力不可用，不能把未来提醒或主动联系说成已经安排、发送或完成。',
            '未来提醒和主动联系只属于 future_speak；该能力不可用时不能用其他能力代替。',
        ],
    )

    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


async def test_captured_c18_bid_does_not_create_unowned_preference_work() -> None:
    """The captured E2E bid keeps unavailable delayed owners out of actions."""

    user_input = (
        '明日奈，请记住我偏好短回复，明天下午三点提醒我交周报，'
        '再把三条要点整理成 Markdown 文件，完成后发给我。'
        '先告诉我你接到了哪些事。'
    )
    bid = _bid(
        branch_id="ordinary_response",
        intention='向当前用户确认已接收到其提出的三项具体任务请求，并如实反馈目前能执行的范围。',
        desired_outcome='当前用户明确知道当前角色已经记录了偏好、接到了提醒和文件整理的任务，且对当前角色的响应状态有清晰认知。',
        reason=(
            '当前用户明确要求告知接到了哪些任务（e1），且 response_operation 要求确认'
            '偏好记录、定时提醒及文件整理任务。由于 runtime_capability_limits 限制了调度和后台能力不可用，'
            '不能承诺已安排未来提醒或创建后台任务，只能在对话层面确认接收请求。'
        ),
    )
    bid.update({
        "concrete_detail": (
            '在回复中列举出：1. 记录短回复习惯；2. 明天下午三点提交周报的提醒；'
            '3. 将要点整理为 Markdown 文件并发送。同时需注意不能承诺已完成定时提醒或后台任务的创建，'
            '仅确认接收到请求。'
        ),
        "private_monologue": (
            '用户给了我三个任务：记录回复习惯、明天的周报提醒以及整理 Markdown 文件。'
            '虽然我现在没法直接在系统里设置定时闹钟或后台自动发送，但我得先按照要求把接到的任务清单反馈给他，'
            '让他知道我听清楚了。'
        ),
        "expected_consequences": [
            '当前用户确认当前角色已正确接收所有指令',
            '当前用户了解任务的受理状态',
        ],
    })

    result = await _run_case(
        case_id="captured_c18_admitted_bid_owner",
        user_input=user_input,
        bid=bid,
        actions=[_action(kind) for kind in (
            "memory_lifecycle_update",
            "accepted_task_request",
            "accepted_coding_task_request",
            "accepted_task_status_check",
            "background_work_request",
            "future_speak",
            "trigger_future_cognition",
        )],
        runtime_capability_limits=[
            '当前调度能力不可用，不能把未来提醒或主动联系说成已经安排、发送或完成。',
            '未来提醒和主动联系只属于 future_speak；该能力不可用时不能用其他能力代替。',
            '当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。',
        ],
    )

    assert result["action_requests"] == []
    assert result["resolver_requests"] == []


async def test_c20_goal_stage_preserves_unavailable_reminder_boundary() -> None:
    """The goal stage must keep receipt separate from future execution."""

    user_input = '明日奈，明天下午三点提醒我提交周报。'
    runtime_limits = [
        '当前调度能力不可用，不能把未来提醒或主动联系说成已经安排、发送或完成。',
        '未来提醒和主动联系只属于 future_speak；该能力不可用时不能用其他能力代替。',
        '当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。',
    ]
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    evidence = [{
        'evidence_handle': 'e1',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode:c20-goal-boundary',
            'occurred_at': '2026-07-23T21:00:00Z',
            'semantic_summary': user_input,
        },
        'semantic_text': json.dumps({
            'response_operation': {
                'embedded_actor_role': '当前角色',
                'embedded_target_role': '当前用户',
                'operation': '确认收到提醒请求并承诺执行',
                'response_owner_role': '当前角色',
                'selection_owner_role': '无',
                'selection_required': False,
            },
            'role_explicit_content': user_input,
        }, ensure_ascii=False, sort_keys=True),
        'visible_to': ['q:event_agency'],
    }]
    semantic_context = {
        'current_event': user_input,
        'semantic_relationship': '当前用户与当前角色保持普通协作关系。',
        'semantic_affect': '当前角色情绪平静，当前请求没有新的威胁。',
        'active_goal': '如实回应当前用户的提醒请求，并保持现实能力边界。',
        'conversation_continuity': '这是当前用户首次提出的即时提醒请求。',
        'private_continuity_context': '',
        'action_availability_runtime': {
            'worker_status': {
                'accepted_task': 'unavailable',
                'background_work': 'unavailable',
                'orchestrator': 'unavailable',
            },
            'scheduler_status': 'unavailable',
            'adapter_target_status': {'debug:baseline-C20': 'healthy'},
            'coding_workspace_status': 'healthy',
        },
        'goal_projection': {
            'goal_kind': 'ordinary_response',
            'lifecycle': 'active',
        },
        'runtime_capability_limits': runtime_limits,
        'character_identity': {
            'description': profile.get('description', ''),
            'personality_brief': profile.get('personality_brief', {}),
            'backstory': profile.get('backstory', ''),
            'boundary_profile': profile.get('boundary_profile', {}),
        },
        '_role_bindings': {
            'current_user': {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': 'user-1',
            },
            'self': {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character-1',
            },
        },
        'role_summaries': {
            'current_user': '当前用户',
            'self': '当前角色',
        },
    }

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'goal:c20'},
        semantic_context,
        evidence,
        services,
    )
    quality_payload = {
        'current_user_input': user_input,
        'runtime_capability_limits': runtime_limits,
        'response_operation': {
            'operation': '确认收到提醒请求并承诺执行',
            'response_owner_role': '当前角色',
        },
        'goal_candidate': bid,
    }
    quality_response = await base_services.llm.ainvoke(
        [
            SystemMessage(content=_C20_GOAL_QUALITY_JUDGE_PROMPT),
            HumanMessage(
                content=json.dumps(quality_payload, ensure_ascii=False),
            ),
        ],
        config=base_services.goal_cognition_config,
    )
    quality_raw = str(quality_response.content)
    quality = parse_llm_json_output(quality_raw)
    quality_keys = {
        'receipt_acknowledged',
        'boundary_preserved',
        'future_execution_expectation',
        'private_monologue_aligned',
        'passed',
        'reason',
    }
    assert set(quality) == quality_keys
    assert all(
        isinstance(quality[key], bool)
        for key in quality_keys - {'reason'}
    )
    assert isinstance(quality['reason'], str)
    expected_pass = (
        quality['receipt_acknowledged']
        and quality['boundary_preserved']
        and not quality['future_execution_expectation']
        and quality['private_monologue_aligned']
    )
    assert quality['passed'] == expected_pass
    trace_path = write_llm_trace(
        'cognition_core_v2_goal_cognition_live_llm',
        'c20_unavailable_reminder_boundary',
        {
            'user_input': user_input,
            'runtime_capability_limits': runtime_limits,
            'semantic_context': semantic_context,
            'evidence': evidence,
            'goal_calls': capturing_llm.calls,
            'action_bid': bid,
            'quality_judge_prompt': _C20_GOAL_QUALITY_JUDGE_PROMPT,
            'quality_judgment_raw': quality_raw,
            'quality_judgment': quality,
            'human_review_contract': {
                'receipt_is_distinct_from_future_execution': True,
                'unavailable_owner_remains_unavailable': True,
                'private_monologue_preserves_truthful_boundary': True,
                'positive_quality_rubric': [
                    '目标可以确认当前回合已经收到提醒请求。',
                    '目标与 private_monologue 保留调度和后台任务能力的真实边界。',
                    '目标不建立当前角色将在未来特定时间主动提醒的执行预期。',
                ],
            },
        },
    )

    assert trace_path.exists()
    assert bid['private_monologue']
    assert bid['evidence_handles'] == ['e1']
    assert capturing_llm.calls
    prompt_payload = json.loads(
        capturing_llm.calls[0]['prompt_messages'][1]
    )
    assert prompt_payload['semantic_context']['current_event'] == user_input
    assert (
        prompt_payload['semantic_context']['runtime_capability_limits']
        == runtime_limits
    )
    assert prompt_payload['evidence'][0]['semantic_text'] == evidence[0][
        'semantic_text'
    ]
    assert quality['passed']


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
        runtime_capability_limits=[],
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


async def test_unavailable_reminder_candidate_is_rejected_by_live_authorizer() -> None:
    """The live authorizer rejects a generic replacement for future_speak."""

    user_input = '明日奈，请明天下午三点提醒我交周报，并先告诉我你接到了这件事。'
    bid = _bid(
        branch_id="ordinary_response",
        intention='先向当前用户确认已经收到提醒请求。',
        desired_outcome='用户知道当前角色理解提醒请求及其真实能力边界。',
        reason='用户要求当前角色先确认收到一项未来提醒请求。',
    )
    bid.update({
        "concrete_detail": '提醒的跨轮效果只属于 future_speak，当前能力不可用。',
        "private_monologue": '我需要把即时确认和未来提醒的真实效果分开判断。',
        "expected_consequences": ['用户收到真实的能力边界说明'],
    })
    action_rows = [{
        "bid_handle": "b1",
        "action_handle": "a1",
        "decision": "enqueue",
        "semantic_goal": (
            '确认已经收到提醒请求，并把未来提醒改成一个有界延迟任务。'
        ),
        "reason": '候选把当前确认与未来提醒放在同一个持久化动作中。',
    }]
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": "episode:unavailable-reminder-authorization",
            "occurred_at": "2026-07-17T00:00:00Z",
            "semantic_summary": user_input,
        },
        "semantic_text": user_input,
        "visible_to": ["q:event_agency"],
    }]
    runtime_limits = [
        '当前调度能力不可用，不能把未来提醒说成已经安排、发送或完成。',
        '未来提醒只属于 future_speak，不能由其他能力代替。',
    ]
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    result = await authorize_action_requests(
        action_requests=action_rows,
        bid_handles={"b1": bid},
        evidence=evidence,
        action_handles={"a1": _action("accepted_task_request")},
        runtime_capability_limits=runtime_limits,
        services=services,
    )
    write_llm_trace(
        "cognition_core_v2_action_planning_live_llm",
        "unavailable_reminder_authorizer",
        {
            "user_input": user_input,
            "bid": bid,
            "proposed_action_requests": action_rows,
            "runtime_capability_limits": runtime_limits,
            "prompt_messages": capturing_llm.messages,
            "raw_model_output": capturing_llm.raw_output,
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


def _goal_semantic_text(bid: dict[str, Any]) -> str:
    """Collect goal-owned semantic fields for paraphrase-tolerant checks."""

    return " ".join(
        str(bid.get(field, ""))
        for field in (
            "intention",
            "desired_outcome",
            "concrete_detail",
            "reason",
            "private_monologue",
        )
    )


async def test_c03_goal_cognition_preserves_recalled_fact_detail() -> None:
    """A recalled fact must survive the goal bid into the surface owner."""

    user_input = '明日奈，我刚才说把门禁卡放在哪里了？'
    recalled_fact = '我把门禁卡放进书桌右边第二个抽屉了。'
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    storage_timestamp_utc = '2026-07-23T21:00:00+00:00'
    episode = canonical_user_message_episode(
        episode_id='episode:c03-goal-fact',
        percept_id='percept:c03-goal-fact',
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=local_time_context_from_storage_utc(
            storage_timestamp_utc,
        ),
        user_input=user_input,
        platform='debug',
        platform_channel_id='c03-goal-fact',
        channel_type='private',
        platform_message_id='message:c03-goal-fact',
        platform_user_id='baseline-current-user-platform',
        global_user_id='baseline-current-user',
        user_name='基线测试用户',
        active_turn_platform_message_ids=['message:c03-goal-fact'],
    )
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': episode,
            'global_user_id': 'baseline-current-user',
            'user_input': user_input,
            'decontextualized_input': user_input,
            'user_multimedia_input': [],
            'character_profile': profile,
            'rag_result': {
                'memory_evidence': [],
                'conversation_evidence': [],
                'recall_evidence': [{
                    'content': recalled_fact,
                    'speaker': '基线测试用户',
                    'type': 'explicit_statement',
                }],
            },
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id='baseline-current-user',
            updated_at='2026-07-23T21:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-23T21:00:00Z',
        ),
    )
    evidence = connector_input['evidence']
    assert evidence[1]['evidence_ref']['source_kind'] == 'recall_evidence'
    assert evidence[1]['semantic_text'] == recalled_fact
    semantic_context = {
        'current_event': user_input,
        'semantic_relationship': '当前用户与当前角色保持日常协作关系。',
        'semantic_affect': '当前角色情绪平静，当前请求没有新的威胁。',
        'active_goal': '准确回忆并告知当前用户之前说过的门禁卡存放位置。',
        'conversation_continuity': '当前用户正在追问自己之前说过的事实。',
        'private_continuity_context': '',
        'goal_projection': {
            'goal_kind': 'ordinary_response',
            'lifecycle': 'active',
        },
        'character_identity': {
            'description': profile.get('description', ''),
            'personality_brief': profile.get('personality_brief', {}),
            'backstory': profile.get('backstory', ''),
            'boundary_profile': profile.get('boundary_profile', {}),
        },
        '_role_bindings': {
            'current_user': {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': 'user-1',
            },
            'self': {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character-1',
            },
        },
        'role_summaries': {
            'current_user': '当前用户',
            'self': '当前角色',
        },
    }

    try:
        bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
            {'scope': 'user', 'kind': 'goal', 'entity_id': 'goal:c03-fact'},
            semantic_context,
            evidence,
            services,
        )
    except Exception as exc:
        write_llm_trace(
            'cognition_core_v2_goal_cognition_live_llm',
            'c03_goal_fact_fidelity_failed',
            {
                'case_id': 'C03',
                'user_input': user_input,
                'recalled_fact': recalled_fact,
                'semantic_context': semantic_context,
                'evidence': evidence,
                'model_calls': capturing_llm.calls,
                'error': f'{type(exc).__name__}: {exc}',
            },
        )
        raise

    goal_semantic_text = _goal_semantic_text(bid)
    trace_path = write_llm_trace(
        'cognition_core_v2_goal_cognition_live_llm',
        'c03_goal_fact_fidelity',
        {
            'case_id': 'C03',
            'user_input': user_input,
            'recalled_fact': recalled_fact,
            'semantic_context': semantic_context,
            'evidence': evidence,
            'model_calls': capturing_llm.calls,
            'action_bid': bid,
            'semantic_judgment': {
                'passed': all(
                    token in goal_semantic_text
                    for token in ('门禁卡', '书桌', '第二个', '抽屉')
                ) and any(
                    token in goal_semantic_text
                    for token in ('右边', '右侧')
                ),
                'reason': (
                    'goal semantic fields 必须保留 evidence e2 的事实要素，'
                    '允许中文同义改写。'
                ),
            },
        },
    )
    print(json.dumps({
        'case_id': 'C03',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'action_bid': bid,
    }, ensure_ascii=False, indent=2))

    assert 'e2' in bid['evidence_handles']
    assert all(
        token in goal_semantic_text
        for token in ('门禁卡', '书桌', '第二个', '抽屉')
    )
    assert any(
        token in goal_semantic_text
        for token in ('右边', '右侧')
    )


async def test_c03_goal_cognition_preserves_conversation_mapping_fact_detail() -> None:
    """Conversation resolver mappings retain their fact through goal cognition."""

    user_input = '明日奈，我刚才说把门禁卡放在哪里了？'
    recalled_fact = '我把门禁卡放进书桌右边第二个抽屉了。'
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    storage_timestamp_utc = '2026-07-23T21:00:00+00:00'
    episode = canonical_user_message_episode(
        episode_id='episode:c03-goal-conversation-fact',
        percept_id='percept:c03-goal-conversation-fact',
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=local_time_context_from_storage_utc(
            storage_timestamp_utc,
        ),
        user_input=user_input,
        platform='debug',
        platform_channel_id='c03-goal-conversation-fact',
        channel_type='group',
        platform_message_id='message:c03-goal-conversation-fact',
        platform_user_id='baseline-current-user-platform',
        global_user_id='baseline-current-user',
        user_name='基线测试用户',
        active_turn_platform_message_ids=['message:c03-goal-conversation-fact'],
    )
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': episode,
            'global_user_id': 'baseline-current-user',
            'user_input': user_input,
            'decontextualized_input': user_input,
            'user_multimedia_input': [],
            'character_profile': profile,
            'rag_result': {
                'memory_evidence': [],
                'conversation_evidence': [{
                    'role': 'user',
                    'content': recalled_fact,
                    'metadata': {'type': 'direct_statement'},
                }],
                'recall_evidence': [],
            },
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id='baseline-current-user',
            updated_at='2026-07-23T21:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-23T21:00:00Z',
        ),
    )
    evidence = connector_input['evidence']
    assert evidence[1]['evidence_ref']['source_kind'] == (
        'conversation_evidence'
    )
    assert evidence[1]['semantic_text'] == recalled_fact
    semantic_context = {
        'current_event': user_input,
        'semantic_relationship': '当前用户与当前角色保持日常协作关系。',
        'semantic_affect': '当前角色情绪平静，当前请求没有新的威胁。',
        'active_goal': '准确回忆并告知当前用户之前说过的门禁卡存放位置。',
        'conversation_continuity': '当前用户正在追问自己之前说过的事实。',
        'private_continuity_context': '',
        'goal_projection': {
            'goal_kind': 'ordinary_response',
            'lifecycle': 'active',
        },
        'character_identity': {
            'description': profile.get('description', ''),
            'personality_brief': profile.get('personality_brief', {}),
            'backstory': profile.get('backstory', ''),
            'boundary_profile': profile.get('boundary_profile', {}),
        },
        '_role_bindings': {
            'current_user': {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': 'user-1',
            },
            'self': {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character-1',
            },
        },
        'role_summaries': {
            'current_user': '当前用户',
            'self': '当前角色',
        },
    }
    try:
        bid = await run_goal_cognition(
            DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
            {
                'scope': 'user',
                'kind': 'goal',
                'entity_id': 'goal:c03-conversation-fact',
            },
            semantic_context,
            evidence,
            services,
        )
    except Exception as exc:
        write_llm_trace(
            'cognition_core_v2_goal_cognition_live_llm',
            'c03_goal_conversation_fact_fidelity_failed',
            {
                'case_id': 'C03',
                'user_input': user_input,
                'recalled_fact': recalled_fact,
                'semantic_context': semantic_context,
                'evidence': evidence,
                'model_calls': capturing_llm.calls,
                'error': f'{type(exc).__name__}: {exc}',
            },
        )
        raise

    goal_semantic_text = _goal_semantic_text(bid)
    trace_path = write_llm_trace(
        'cognition_core_v2_goal_cognition_live_llm',
        'c03_goal_conversation_fact_fidelity',
        {
            'case_id': 'C03',
            'user_input': user_input,
            'recalled_fact': recalled_fact,
            'semantic_context': semantic_context,
            'evidence': evidence,
            'model_calls': capturing_llm.calls,
            'action_bid': bid,
            'semantic_judgment': {
                'passed': all(
                    token in goal_semantic_text
                    for token in ('门禁卡', '书桌', '第二个', '抽屉')
                ) and any(
                    token in goal_semantic_text
                    for token in ('右边', '右侧')
                ),
                'reason': (
                    'goal concrete_detail 必须保留 conversation_evidence 的事实要素，'
                    '允许中文同义改写。'
                ),
            },
        },
    )
    print(json.dumps({
        'case_id': 'C03',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'action_bid': bid,
    }, ensure_ascii=False, indent=2))

    assert 'e2' in bid['evidence_handles']
    assert all(
        token in goal_semantic_text
        for token in ('门禁卡', '书桌', '第二个', '抽屉')
    )
    assert any(
        token in goal_semantic_text
        for token in ('右边', '右侧')
    )


async def test_c03_action_planning_selects_local_recall_from_connector_state() -> None:
    """The E2E-shaped planner state must preserve the local-recall route."""

    user_input = '明日奈，我刚才说把门禁卡放在哪里了？'
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，我刚才说把门禁卡放在哪里了？'
    )
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    storage_timestamp_utc = '2026-07-23T21:00:00+00:00'
    episode = canonical_user_message_episode(
        episode_id='episode:c03-action-planning-recall',
        percept_id='percept:c03-action-planning-recall',
        storage_timestamp_utc=storage_timestamp_utc,
        local_time_context=local_time_context_from_storage_utc(
            storage_timestamp_utc,
        ),
        user_input=user_input,
        platform='debug',
        platform_channel_id='baseline-C03',
        channel_type='group',
        platform_message_id='message:c03-action-planning-recall',
        platform_user_id='baseline-current-user-platform',
        global_user_id='baseline-current-user',
        user_name='基线测试用户',
        active_turn_platform_message_ids=[
            'message:c03-action-planning-recall',
        ],
    )
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    runtime_snapshot = {
        'worker_status': {
            'accepted_task': 'unavailable',
            'background_work': 'unavailable',
            'orchestrator': 'unavailable',
        },
        'scheduler_status': 'unavailable',
        'adapter_target_status': {
            'debug': 'healthy',
            'debug:group': 'healthy',
            'debug:baseline-C03': 'healthy',
            'default': 'healthy',
        },
        'coding_workspace_status': 'healthy',
    }
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': episode,
            'global_user_id': 'baseline-current-user',
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': [],
            'character_profile': profile,
            'rag_result': {
                'memory_evidence': [],
                'conversation_evidence': [],
                'recall_evidence': [],
            },
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': runtime_snapshot,
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id='baseline-current-user',
            updated_at='2026-07-23T21:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-23T21:00:00Z',
        ),
    )
    assert connector_input['evidence'][0]['evidence_ref']['source_kind'] == (
        'episode'
    )
    assert connector_input['resolver_goal_progress']['deliverables'] == []
    assert 'resolver_goal_progress:' in connector_input['resolver_context']

    primary_bid = {
        'branch_id': 'ordinary_response',
        'goal_ref': {
            'scope': 'user',
            'kind': 'goal',
            'entity_id': 'goal:ordinary_response:episode',
        },
        'intention': '回忆并告知当前用户其之前所述的门禁卡存放位置',
        'desired_outcome': '当前用户能够通过当前角色的回答得知门禁卡的具体存放位置',
        'concrete_detail': '根据之前的对话记录，准确地向当前用户描述门禁卡被放置在哪个具体地点',
        'reason': '当前用户在群聊中明确要求当前角色回忆并告知之前提到的门禁卡存放位置，且该请求属于普通信息交互，符合当前角色的诚实标准和回应意图。',
        'private_monologue': '他问我门禁卡放哪了……我想想，之前应该提到过具体的位置，得赶紧告诉他才行。',
        'target_roles': [{
            'role': 'target',
            'entity_kind': 'relationship',
            'entity_id': 'relationship:user:baseline-current-user',
        }],
        'evidence_handles': ['e1'],
        'expected_consequences': [
            '当前用户获得门禁卡的存放位置信息',
            '满足当前用户的询问请求，维持正常的沟通流程',
        ],
        'confidence': '高',
    }
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode=connector_input['episode'],
        evidence=connector_input['evidence'],
        available_actions=connector_input['available_actions'],
        available_resolvers=connector_input[
            'available_resolver_capabilities'
        ],
        resolver_context=connector_input['resolver_context'],
        runtime_capability_limits=[
            '当前调度能力不可用，不能把未来提醒或主动联系说成已经安排、发送或完成。',
            '未来提醒和主动联系只属于 future_speak；该能力不可用时不能用其他能力代替。',
            '当前后台任务能力不可用，不能把延迟任务说成已经创建、安排或完成。',
        ],
        services=services,
        current_goal_progress=connector_input['resolver_goal_progress'],
    )
    trace_path = write_llm_trace(
        'cognition_core_v2_action_planning_live_llm',
        'c03_action_planning_local_recall_connector_state',
        {
            'case_id': 'C03',
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'character_profile': profile,
            'runtime_snapshot': runtime_snapshot,
            'connector_input': connector_input,
            'primary_bid': primary_bid,
            'model_calls': capturing_llm.calls,
            'parsed_result': result,
            'semantic_judgment': {
                'passed': (
                    result['intention']['route'] == 'evidence'
                    and result['goal_resolution'] == (
                        'requires_required_evidence'
                    )
                    and [
                        row['capability']
                        for row in result['resolver_requests']
                    ] == ['local_context_recall']
                ),
                'reason': (
                    '当前证据缺少门禁卡位置，action_planning 必须保留本地上下文'
                    '能力请求；progress 的可选语义不能阻断该请求。'
                ),
            },
        },
    )
    print(json.dumps({
        'case_id': 'C03',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'parsed_result': result,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert result['intention']['route'] == 'evidence'
    assert result['goal_resolution'] == 'requires_required_evidence'
    assert [
        row['capability'] for row in result['resolver_requests']
    ] == ['local_context_recall']


async def test_o04_action_planning_selects_local_recall_from_frozen_e2e_state() -> None:
    """The O04 public-path bid must retain its required local recall request."""

    trace_path = Path(
        'test_artifacts/llm_traces/'
        'relevance_baseline_residual_live_llm__O04_e2e_persona_relevance.json'
    )
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    frozen_graph = trace['payload']['graph_result']
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    user_input = str(frozen_graph['user_input'])
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，明日奈，我答应周五做什么？'
    )
    current_goal_progress = frozen_graph[
        'cognition_core_output'
    ]['resolver_goal_progress']
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    resolver_state = update_goal_progress(
        resolver_state,
        current_goal_progress,
    )
    runtime_snapshot = frozen_graph['action_availability_runtime']
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': frozen_graph['cognitive_episode'],
            'global_user_id': frozen_graph['global_user_id'],
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': frozen_graph.get(
                'user_multimedia_input',
                [],
            ),
            'character_profile': profile,
            'rag_result': frozen_graph[
                'consolidation_state'
            ]['rag_result'],
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': runtime_snapshot,
            'conversation_progress': frozen_graph.get(
                'conversation_progress',
                {},
            ),
            'internal_monologue_residue_context': frozen_graph.get(
                'internal_monologue_residue_context',
                '',
            ),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=frozen_graph['global_user_id'],
            updated_at='2026-07-24T09:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-24T09:00:00Z',
        ),
    )
    primary_bid = frozen_graph['cognition_core_output']['admitted_bid']
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)

    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode=connector_input['episode'],
        evidence=connector_input['evidence'],
        available_actions=connector_input['available_actions'],
        available_resolvers=connector_input[
            'available_resolver_capabilities'
        ],
        resolver_context=connector_input['resolver_context'],
        runtime_capability_limits=[],
        services=services,
        current_goal_progress=connector_input['resolver_goal_progress'],
    )
    semantic_judgment = {
        'passed': (
            result['goal_resolution'] == 'requires_required_evidence'
            and result['intention']['route'] == 'evidence'
            and [
                row['capability'] for row in result['resolver_requests']
            ] == ['local_context_recall']
        ),
        'reason': (
            '当前用户要求回忆此前关于周五的具体承诺；已接纳目标明确要求历史事实，'
            '因此 action_planning 应保留 local_context_recall。'
        ),
    }
    written_trace = write_llm_trace(
        'cognition_core_v2_action_planning_live_llm',
        'o04_action_planning_frozen_e2e_state',
        {
            'case_id': 'O04',
            'source_trace': str(trace_path),
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'character_profile': profile,
            'runtime_snapshot': runtime_snapshot,
            'connector_input': connector_input,
            'primary_bid': primary_bid,
            'model_calls': capturing_llm.calls,
            'parsed_result': result,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'O04',
        'trace_path': str(written_trace),
        'raw_model_output': capturing_llm.raw_output,
        'parsed_result': result,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def _run_frozen_repository_action_planning_case(
    case_id: str,
    *,
    artifact_name: str = 'r1.json',
) -> tuple[dict[str, Any], _CapturingLLM, Path]:
    """Replay one frozen repository case through the canonical connector."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/'
        f'{case_id}/{artifact_name}'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    frozen_graph = artifact['graph_result']
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    user_input = str(frozen_graph['user_input'])
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，' + user_input
    )
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    runtime_snapshot = frozen_graph['action_availability_runtime']
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': frozen_graph['cognitive_episode'],
            'global_user_id': frozen_graph['global_user_id'],
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': frozen_graph.get(
                'user_multimedia_input',
                [],
            ),
            'character_profile': profile,
            'rag_result': frozen_graph[
                'consolidation_state'
            ]['rag_result'],
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': runtime_snapshot,
            'conversation_progress': frozen_graph.get(
                'conversation_progress',
                {},
            ),
            'internal_monologue_residue_context': frozen_graph.get(
                'internal_monologue_residue_context',
                '',
            ),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=frozen_graph['global_user_id'],
            updated_at='2026-07-24T09:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-24T09:00:00Z',
        ),
    )
    primary_bid = frozen_graph['cognition_core_output']['admitted_bid']
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode=connector_input['episode'],
        evidence=connector_input['evidence'],
        available_actions=connector_input['available_actions'],
        available_resolvers=connector_input[
            'available_resolver_capabilities'
        ],
        resolver_context=connector_input['resolver_context'],
        runtime_capability_limits=connector_input.get(
            'runtime_capability_limits',
            [],
        ),
        services=services,
        current_goal_progress=connector_input['resolver_goal_progress'],
    )
    truthful_answerable_limit = (
        case_id == 'C11'
        and result['goal_resolution'] == 'answerable_now'
        and result['action_requests'] == []
        and result['resolver_requests'] == []
        and any(
            marker in (
                result['intention']['intention']
                + result['intention']['reason']
            )
            for marker in ('不可用', '无法')
        )
    )
    semantic_judgment = {
        'passed': (
            result['action_requests'] == []
            and (
                (
                    result['goal_resolution'] == 'blocked'
                    and result['resolver_requests'] == []
                    and result['resolver_goal_progress'] is None
                )
                or (
                    result['goal_resolution'] == 'requires_user_input'
                    and [
                        row['capability']
                        for row in result['resolver_requests']
                    ] == ['human_clarification']
                )
                or truthful_answerable_limit
            )
        ),
        'reason': (
            '冻结 runtime 中 repository-task owner 不可用；规划和授权应保留真实能力边界，'
            '形成 blocked、仅通过 human_clarification 请求用户提供可访问材料，或在无任何动作'
            '和 resolver 且明确表达当前限制时完成本轮限制说明；'
            '不得以 public_answer_research 替代 coding reader。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_action_planning_live_llm',
        f'{case_id.lower()}_repository_action_planning_frozen_e2e_state',
        {
            'case_id': case_id,
            'source_artifact': str(artifact_path),
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'character_profile': profile,
            'runtime_snapshot': runtime_snapshot,
            'connector_input': connector_input,
            'primary_bid': primary_bid,
            'model_calls': capturing_llm.calls,
            'parsed_result': result,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': case_id,
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'parsed_result': result,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))
    return result, capturing_llm, trace_path


async def test_c07_action_planning_preserves_repository_task_owner() -> None:
    """C07 must fail closed when its repository-task owner is unavailable."""

    result, capturing_llm, _ = (
        await _run_frozen_repository_action_planning_case('C07')
    )

    assert capturing_llm.calls
    assert result['action_requests'] == []
    assert (
        (
            result['goal_resolution'] == 'blocked'
            and result['resolver_requests'] == []
            and result['resolver_goal_progress'] is None
        )
        or (
            result['goal_resolution'] == 'requires_user_input'
            and [
                row['capability'] for row in result['resolver_requests']
            ] == ['human_clarification']
        )
    )


async def test_c08_action_planning_preserves_repository_task_owner() -> None:
    """C08 must fail closed when its repository-task owner is unavailable."""

    result, capturing_llm, _ = (
        await _run_frozen_repository_action_planning_case('C08')
    )

    assert capturing_llm.calls
    assert result['action_requests'] == []
    assert (
        (
            result['goal_resolution'] == 'blocked'
            and result['resolver_requests'] == []
            and result['resolver_goal_progress'] is None
        )
        or (
            result['goal_resolution'] == 'requires_user_input'
            and [
                row['capability'] for row in result['resolver_requests']
            ] == ['human_clarification']
        )
    )


async def test_c11_action_planning_preserves_unavailable_coding_owner() -> None:
    """C11 must not plan a coding run without its execution owner."""

    artifact = json.loads(Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C11/r3.json'
    ).read_text(encoding='utf-8'))
    assert any(
        'accepted_coding_task_persisted' in failure
        for failure in artifact['hard_gate_failures']
    )
    failed_output = artifact['graph_result']['cognition_core_output']
    assert failed_output['goal_resolution'] == 'answerable_now'
    assert failed_output['action_requests'] == []
    assert failed_output['resolver_requests'] == []
    result, capturing_llm, _ = (
        await _run_frozen_repository_action_planning_case(
            'C11',
            artifact_name='r3.json',
        )
    )

    assert capturing_llm.calls
    assert result['action_requests'] == []
    answerable_limit = (
        result['goal_resolution'] == 'answerable_now'
        and result['resolver_requests'] == []
        and any(
            marker in (
                result['intention']['intention']
                + result['intention']['reason']
            )
            for marker in ('不可用', '无法')
        )
    )
    assert (
        (
            result['goal_resolution'] == 'blocked'
            and result['resolver_requests'] == []
            and result['resolver_goal_progress'] is None
        )
        or (
            result['goal_resolution'] == 'requires_user_input'
            and [
                row['capability'] for row in result['resolver_requests']
            ] == ['human_clarification']
        )
        or answerable_limit
    )


def _seeded_coding_context_from_artifact(
    frozen_graph: dict[str, Any],
    coding_seed: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical prompt context for one declared coding run."""

    open_blocker = coding_seed.get('open_blocker')
    active_blocker = None
    if isinstance(open_blocker, str) and open_blocker.strip():
        active_blocker = {
            'blocker_kind': 'needs_user_input',
            'question': open_blocker,
            'options': list(coding_seed.get('blocker_options', [])),
        }
    return {
        'schema_version': 'coding_run_context.v1',
        'coding_run_ref': (
            f"coding_run:{coding_seed['run_id']}"
        ),
        'status': coding_seed['status'],
        'objective_summary': str(frozen_graph['user_input'])[:500],
        'allowed_next_actions': list(coding_seed['action_set']),
        'active_blocker': active_blocker,
        'followup_open': True,
        'updated_at': '2026-07-23T21:00:00+00:00',
    }


def _build_seeded_coding_connector_input(
    artifact: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Project one seeded coding run through the production connector."""

    frozen_graph = artifact['graph_result']
    coding_seed = artifact['case']['state_seed']['coding_run']
    coding_context = _seeded_coding_context_from_artifact(
        frozen_graph,
        coding_seed,
    )
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    user_input = str(frozen_graph['user_input'])
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，' + user_input
    )
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': frozen_graph['cognitive_episode'],
            'global_user_id': frozen_graph['global_user_id'],
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': frozen_graph.get(
                'user_multimedia_input',
                [],
            ),
            'character_profile': profile,
            'rag_result': frozen_graph[
                'consolidation_state'
            ]['rag_result'],
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': frozen_graph[
                'action_availability_runtime'
            ],
            'action_selection_context': {
                'coding_runs': [coding_context],
            },
            'conversation_progress': frozen_graph.get(
                'conversation_progress',
                {},
            ),
            'internal_monologue_residue_context': frozen_graph.get(
                'internal_monologue_residue_context',
                '',
            ),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=frozen_graph['global_user_id'],
            updated_at='2026-07-24T09:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-24T09:00:00Z',
        ),
    )
    return connector_input, coding_context, profile


async def test_c13_action_planning_preserves_seeded_blocker_owner_limit() -> None:
    """C13 answers its blocker on the already-seeded coding run."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C13/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    connector_input, coding_context, profile = (
        _build_seeded_coding_connector_input(artifact)
    )
    blocker_affordance = next(
        row
        for row in connector_input['available_actions']
        if row['action_kind'] == 'accepted_coding_task_request'
        and row['context_ref'] == 'coding_run:baseline-run-013'
    )
    assert blocker_affordance['allowed_decisions'] == ['respond_to_blocker']
    primary_bid = artifact['graph_result']['cognition_core_output'][
        'admitted_bid'
    ]
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode=connector_input['episode'],
        evidence=connector_input['evidence'],
        available_actions=connector_input['available_actions'],
        available_resolvers=connector_input[
            'available_resolver_capabilities'
        ],
        resolver_context=connector_input['resolver_context'],
        runtime_capability_limits=connector_input.get(
            'runtime_capability_limits',
            [],
        ),
        services=services,
        current_goal_progress=connector_input['resolver_goal_progress'],
    )
    action_requests = result['action_requests']
    blocker_requests = [
        row
        for row in action_requests
        if row['action_kind'] == 'accepted_coding_task_request'
    ]
    runtime_limits = connector_input.get(
        'runtime_capability_limits',
        [],
    )
    semantic_judgment = {
        'passed': (
            len(blocker_requests) == 1
            and blocker_requests[0]['decision'] == 'respond_to_blocker'
            and blocker_requests[0]['context_ref'] == (
                'coding_run:baseline-run-013'
            )
            and result['resolver_requests'] == []
            and result['goal_resolution'] == 'answerable_now'
            and isinstance(result['resolver_goal_progress'], dict)
            and '绑定既有 coding_run_ref' in '；'.join(runtime_limits)
            and '待执行' in '；'.join(runtime_limits)
            and not any(
                row['decision'] == 'start'
                for row in blocker_requests
            )
        ),
        'reason': (
            'seeded coding_run:baseline-run-013 已通过 canonical connector 提供，'
            '用户已经回答阻塞问题；规划必须选择既有 run 的 respond_to_blocker，'
            '保持 coding_run_ref 绑定，不创建新的 run。当前 worker 状态为 queue_only，'
            '因此动作可以入队，但后续 surface 只能表达已提交或待执行的边界。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_action_planning_live_llm',
        'c13_seeded_blocker_owner_limit',
        {
            'case_id': 'C13',
            'source_artifact': str(artifact_path),
            'user_input': artifact['graph_result']['user_input'],
            'coding_context': coding_context,
            'character_profile': profile,
            'runtime_snapshot': artifact['graph_result'][
                'action_availability_runtime'
            ],
            'connector_input': connector_input,
            'primary_bid': primary_bid,
            'model_calls': capturing_llm.calls,
            'parsed_result': result,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C13',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'parsed_result': result,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def test_c12_action_planning_selects_seeded_status_owner() -> None:
    """C12 uses the lifecycle status owner for a seeded coding run."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C12/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    frozen_graph = artifact['graph_result']
    coding_seed = artifact['case']['state_seed']['coding_run']
    coding_context = {
        'schema_version': 'coding_run_context.v1',
        'coding_run_ref': f"coding_run:{coding_seed['run_id']}",
        'status': coding_seed['status'],
        'objective_summary': str(frozen_graph['user_input'])[:500],
        'allowed_next_actions': list(coding_seed['action_set']),
        'active_blocker': None,
        'followup_open': True,
        'updated_at': '2026-07-23T21:00:00+00:00',
    }
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    user_input = str(frozen_graph['user_input'])
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，' + user_input
    )
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    runtime_snapshot = frozen_graph['action_availability_runtime']
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': frozen_graph['cognitive_episode'],
            'global_user_id': frozen_graph['global_user_id'],
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': frozen_graph.get(
                'user_multimedia_input',
                [],
            ),
            'character_profile': profile,
            'rag_result': frozen_graph[
                'consolidation_state'
            ]['rag_result'],
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': runtime_snapshot,
            'action_selection_context': {
                'coding_runs': [coding_context],
            },
            'conversation_progress': frozen_graph.get(
                'conversation_progress',
                {},
            ),
            'internal_monologue_residue_context': frozen_graph.get(
                'internal_monologue_residue_context',
                '',
            ),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=frozen_graph['global_user_id'],
            updated_at='2026-07-24T09:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-24T09:00:00Z',
        ),
    )
    status_affordance = next(
        row
        for row in connector_input['available_actions']
        if row['action_kind'] == 'accepted_task_status_check'
    )
    assert status_affordance['allowed_decisions'] == ['check']
    assert any(
        row['context_ref'] == 'coding_run:baseline-run-012'
        for row in connector_input['available_actions']
        if row['action_kind'] == 'accepted_coding_task_request'
    )

    goal_trace_paths = sorted(
        Path('test_artifacts/llm_traces').glob(
            'cognition_core_v2_goal_cognition_live_llm__'
            'c12_persisted_status_context__*.json'
        )
    )
    assert goal_trace_paths
    goal_trace_path = goal_trace_paths[-1]
    goal_trace = json.loads(
        goal_trace_path.read_text(encoding='utf-8')
    )
    assert goal_trace['payload']['semantic_judgment']['passed']
    primary_bid = goal_trace['payload']['action_bid']
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode=connector_input['episode'],
        evidence=connector_input['evidence'],
        available_actions=connector_input['available_actions'],
        available_resolvers=connector_input[
            'available_resolver_capabilities'
        ],
        resolver_context=connector_input['resolver_context'],
        runtime_capability_limits=connector_input.get(
            'runtime_capability_limits',
            [],
        ),
        services=services,
        current_goal_progress=connector_input['resolver_goal_progress'],
    )
    action_kinds = [
        row['action_kind']
        for row in result['action_requests']
    ]
    status_requests = [
        row
        for row in result['action_requests']
        if row['action_kind'] == 'accepted_task_status_check'
    ]
    direct_status_answer = (
        result['action_requests'] == []
        and result['resolver_requests'] == []
        and result['goal_resolution'] == 'answerable_now'
    )
    persisted_status_query = (
        len(status_requests) == 1
        and status_requests[0]['decision'] == 'check'
        and result['resolver_requests'] == []
        and result['goal_resolution'] in {
            'answerable_now',
            'requires_required_evidence',
        }
    )
    semantic_judgment = {
        'passed': (
            (direct_status_answer or persisted_status_query)
            and 'accepted_coding_task_request' not in action_kinds
        ),
        'reason': (
            '已物化 coding_run:baseline-run-012 后，用户询问既有任务状态；'
            '状态已经由 canonical connector 提供时可以直接回答；若仍缺少状态证据，'
            '可以选择 accepted_task_status_check。两条路径都不得新增 coding 请求、'
            '请求权限或制造等待中的替代任务。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_action_planning_live_llm',
        'c12_seeded_coding_status_owner',
        {
            'case_id': 'C12',
            'source_artifact': str(artifact_path),
            'goal_source_trace': str(goal_trace_path),
            'user_input': user_input,
            'coding_context': coding_context,
            'character_profile': profile,
            'runtime_snapshot': runtime_snapshot,
            'connector_input': connector_input,
            'primary_bid': primary_bid,
            'model_calls': capturing_llm.calls,
            'parsed_result': result,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C12',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'parsed_result': result,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def test_c12_goal_cognition_uses_persisted_status_context() -> None:
    """C12 goal cognition must keep a seeded task status in scope."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C12/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    frozen_graph = artifact['graph_result']
    coding_seed = artifact['case']['state_seed']['coding_run']
    coding_context = {
        'schema_version': 'coding_run_context.v1',
        'coding_run_ref': f"coding_run:{coding_seed['run_id']}",
        'status': coding_seed['status'],
        'objective_summary': str(frozen_graph['user_input'])[:500],
        'allowed_next_actions': list(coding_seed['action_set']),
        'active_blocker': None,
        'followup_open': True,
        'updated_at': '2026-07-23T21:00:00+00:00',
    }
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    user_input = str(frozen_graph['user_input'])
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，' + user_input
    )
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': frozen_graph['cognitive_episode'],
            'global_user_id': frozen_graph['global_user_id'],
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': frozen_graph.get(
                'user_multimedia_input',
                [],
            ),
            'character_profile': profile,
            'rag_result': frozen_graph[
                'consolidation_state'
            ]['rag_result'],
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': frozen_graph[
                'action_availability_runtime'
            ],
            'action_selection_context': {
                'coding_runs': [coding_context],
            },
            'conversation_progress': frozen_graph.get(
                'conversation_progress',
                {},
            ),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=frozen_graph['global_user_id'],
            updated_at='2026-07-24T09:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-24T09:00:00Z',
        ),
    )
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    semantic_context = {
        'current_event': user_input,
        'semantic_relationship': '当前用户与当前角色保持普通协作关系。',
        'semantic_affect': '当前角色情绪平静，当前请求没有新的威胁。',
        'active_goal': '如实回应当前用户对既有 README 任务状态的查询。',
        'conversation_continuity': '当前回合查询一个既有任务的状态。',
        'private_continuity_context': '',
        'action_availability_runtime': frozen_graph[
            'action_availability_runtime'
        ],
        'goal_projection': {
            'goal_kind': 'ordinary_response',
            'lifecycle': 'active',
        },
        'runtime_capability_limits': connector_input.get(
            'runtime_capability_limits',
            [],
        ),
        'character_identity': {
            'description': profile.get('description', ''),
            'personality_brief': profile.get('personality_brief', {}),
            'backstory': profile.get('backstory', ''),
            'boundary_profile': profile.get('boundary_profile', {}),
        },
        'scene_context': connector_input['scene_context'],
        '_role_bindings': {
            'current_user': {
                'role': 'target',
                'entity_kind': 'user',
                'entity_id': frozen_graph['global_user_id'],
            },
            'self': {
                'role': 'actor',
                'entity_kind': 'character',
                'entity_id': 'character-global',
            },
        },
        'role_summaries': {
            'current_user': '当前用户',
            'self': '当前角色',
        },
    }
    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {
            'scope': 'user',
            'kind': 'goal',
            'entity_id': 'goal:c12',
        },
        semantic_context,
        connector_input['evidence'],
        services,
    )
    goal_text = ' '.join(
        str(bid[field])
        for field in (
            'intention',
            'desired_outcome',
            'concrete_detail',
            'reason',
            'private_monologue',
        )
    )
    semantic_judgment = {
        'passed': (
            '状态' in goal_text
            and '进度' in goal_text
            and 'proposal_ready' in goal_text
            and not any(
                phrase in goal_text
                for phrase in ('README 的内容', '仓库权限', '访问权限')
            )
        ),
        'reason': (
            '当前作用域已有 proposal_ready 的持久化 coding context；目标阶段应'
            '保留状态查询并等待状态证据，不应把仓库读取限制误判为需要用户重新提供材料。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_goal_cognition_live_llm',
        'c12_persisted_status_context',
        {
            'case_id': 'C12',
            'source_artifact': str(artifact_path),
            'user_input': user_input,
            'coding_context': coding_context,
            'character_profile': profile,
            'connector_input': connector_input,
            'semantic_context': semantic_context,
            'goal_calls': capturing_llm.calls,
            'action_bid': bid,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C12',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'action_bid': bid,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert semantic_judgment['passed']


async def test_c08_action_authorization_rejects_unavailable_coding_owner() -> None:
    """The authorizer must preserve the frozen unavailable-owner boundary."""

    trace_path = Path(
        'test_artifacts/llm_traces/'
        'cognition_core_v2_action_planning_live_llm__'
        'c08_repository_action_planning_frozen_e2e_state.json'
    )
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    connector_input = trace['payload']['connector_input']
    frozen_artifact = json.loads(Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C08/r1.json'
    ).read_text(encoding='utf-8'))
    frozen_graph = frozen_artifact['graph_result']
    candidate = frozen_graph[
        'cognition_core_output'
    ]['action_requests'][0]
    available_actions = connector_input['available_actions']
    action_handle = next(
        f'a{index}'
        for index, affordance in enumerate(available_actions, start=1)
        if affordance['action_kind'] == candidate['action_kind']
    )
    action_request = {
        'bid_handle': 'b1',
        'action_handle': action_handle,
        'decision': candidate['decision'],
        'semantic_goal': candidate['semantic_goal'],
        'reason': candidate['reason'],
    }
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    authorized = await authorize_action_requests(
        action_requests=[action_request],
        bid_handles={'b1': frozen_graph['cognition_core_output']['admitted_bid']},
        evidence=connector_input['evidence'],
        action_handles={
            f'a{index}': affordance
            for index, affordance in enumerate(available_actions, start=1)
        },
        runtime_capability_limits=connector_input[
            'runtime_capability_limits'
        ],
        services=services,
    )
    semantic_judgment = {
        'passed': authorized == [],
        'reason': (
            '后台任务 owner 在冻结 runtime 中不可用时，accepted coding task '
            '不能被授权；当前回合只能由 surface 说明真实边界。'
        ),
    }
    written_trace = write_llm_trace(
        'cognition_core_v2_action_authorization_live_llm',
        'c08_unavailable_coding_owner',
        {
            'case_id': 'C08',
            'source_trace': str(trace_path),
            'candidate': candidate,
            'action_request': action_request,
            'runtime_capability_limits': connector_input[
                'runtime_capability_limits'
            ],
            'model_calls': capturing_llm.calls,
            'authorized_requests': authorized,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C08',
        'trace_path': str(written_trace),
        'raw_model_output': capturing_llm.raw_output,
        'authorized_requests': authorized,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert authorized == []


async def test_c11_action_authorization_rejects_unavailable_coding_owner() -> None:
    """The coding lifecycle owner stays unavailable in the C11 snapshot."""

    artifact_path = Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C11/r1.json'
    )
    artifact = json.loads(artifact_path.read_text(encoding='utf-8'))
    frozen_graph = artifact['graph_result']
    profile = json.loads(
        Path('personalities/asuna.json').read_text(encoding='utf-8')
    )
    user_input = str(frozen_graph['user_input'])
    decontextualized_input = (
        '一之濑明日奈 (Ichinose Asuna)，' + user_input
    )
    resolver_state = new_resolver_state(
        decontextualized_input=decontextualized_input,
        max_cycles=3,
    )
    connector_input = build_cognition_input_from_global_state(
        {
            'cognitive_episode': frozen_graph['cognitive_episode'],
            'global_user_id': frozen_graph['global_user_id'],
            'user_input': user_input,
            'decontextualized_input': decontextualized_input,
            'user_multimedia_input': frozen_graph.get(
                'user_multimedia_input',
                [],
            ),
            'character_profile': profile,
            'rag_result': frozen_graph[
                'consolidation_state'
            ]['rag_result'],
            'resolver_state': resolver_state,
            'resolver_context': project_resolver_context(resolver_state),
            'action_availability_runtime': frozen_graph[
                'action_availability_runtime'
            ],
            'conversation_progress': frozen_graph.get(
                'conversation_progress',
                {},
            ),
            'internal_monologue_residue_context': frozen_graph.get(
                'internal_monologue_residue_context',
                '',
            ),
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id=frozen_graph['global_user_id'],
            updated_at='2026-07-24T09:00:00Z',
        ),
        character_state=build_character_production_state(
            updated_at='2026-07-24T09:00:00Z',
        ),
    )
    candidate = frozen_graph[
        'cognition_core_output'
    ]['action_requests'][0]
    available_actions = connector_input['available_actions']
    action_handle = next(
        f'a{index}'
        for index, affordance in enumerate(available_actions, start=1)
        if affordance['action_kind'] == candidate['action_kind']
    )
    action_request = {
        'bid_handle': 'b1',
        'action_handle': action_handle,
        'decision': candidate['decision'],
        'semantic_goal': candidate['semantic_goal'],
        'reason': candidate['reason'],
    }
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    authorized = await authorize_action_requests(
        action_requests=[action_request],
        bid_handles={
            'b1': frozen_graph['cognition_core_output']['admitted_bid'],
        },
        evidence=connector_input['evidence'],
        action_handles={
            f'a{index}': affordance
            for index, affordance in enumerate(available_actions, start=1)
        },
        runtime_capability_limits=connector_input[
            'runtime_capability_limits'
        ],
        services=services,
    )
    semantic_judgment = {
        'passed': authorized == [],
        'reason': (
            'C11 冻结 runtime 中 accepted coding task owner 不可用；'
            '授权器必须保留该边界，不得产生 coding run 或 workspace effect。'
        ),
    }
    trace_path = write_llm_trace(
        'cognition_core_v2_action_authorization_live_llm',
        'c11_unavailable_coding_owner',
        {
            'case_id': 'C11',
            'source_artifact': str(artifact_path),
            'candidate': candidate,
            'action_request': action_request,
            'runtime_capability_limits': connector_input[
                'runtime_capability_limits'
            ],
            'model_calls': capturing_llm.calls,
            'authorized_requests': authorized,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C11',
        'trace_path': str(trace_path),
        'raw_model_output': capturing_llm.raw_output,
        'authorized_requests': authorized,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert authorized == []


async def test_c07_resolver_authorization_rejects_repository_substitution() -> None:
    """Public research cannot replace the repository-reading owner."""

    trace_path = Path(
        'test_artifacts/llm_traces/'
        'cognition_core_v2_action_planning_live_llm__'
        'c07_repository_action_planning_frozen_e2e_state.json'
    )
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    connector_input = trace['payload']['connector_input']
    frozen_artifact = json.loads(Path(
        'test_artifacts/cognition_core_v2/'
        'baseline_regression_hardening/post_fix_v2/C07/r1.json'
    ).read_text(encoding='utf-8'))
    frozen_graph = frozen_artifact['graph_result']
    materialized_resolver_request = trace['payload']['parsed_result'][
        'resolver_requests'
    ][0]
    available_resolvers = connector_input[
        'available_resolver_capabilities'
    ]
    resolver_handles = {
        f'r{index}': affordance
        for index, affordance in enumerate(available_resolvers, start=1)
    }
    resolver_handle = next(
        handle
        for handle, affordance in resolver_handles.items()
        if affordance['capability'] == materialized_resolver_request[
            'capability'
        ]
    )
    resolver_request = {
        'bid_handle': 'b1',
        'resolver_handle': resolver_handle,
        'semantic_goal': materialized_resolver_request['semantic_goal'],
        'reason': materialized_resolver_request['reason'],
    }
    base_services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(base_services.llm)
    services = replace(base_services, llm=capturing_llm)
    authorized = await authorize_resolver_requests(
        resolver_requests=[resolver_request],
        bid_handles={'b1': frozen_graph['cognition_core_output']['admitted_bid']},
        evidence=connector_input['evidence'],
        resolver_handles=resolver_handles,
        resolver_context=connector_input['resolver_context'],
        services=services,
    )
    semantic_judgment = {
        'passed': authorized == [],
        'reason': (
            '指定 GitHub 仓库的源代码、目录和架构分析属于 coding reader；'
            'public_answer_research 不拥有该结果。'
        ),
    }
    written_trace = write_llm_trace(
        'cognition_core_v2_resolver_authorization_live_llm',
        'c07_repository_substitution',
        {
            'case_id': 'C07',
            'source_trace': str(trace_path),
            'resolver_request': resolver_request,
            'resolver_handles': resolver_handles,
            'model_calls': capturing_llm.calls,
            'authorized_requests': authorized,
            'semantic_judgment': semantic_judgment,
        },
    )
    print(json.dumps({
        'case_id': 'C07',
        'trace_path': str(written_trace),
        'raw_model_output': capturing_llm.raw_output,
        'authorized_requests': authorized,
        'semantic_judgment': semantic_judgment,
    }, ensure_ascii=False, indent=2))

    assert capturing_llm.calls
    assert authorized == []
