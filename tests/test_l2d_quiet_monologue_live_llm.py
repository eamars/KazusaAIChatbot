"""Live LLM reproduction for quiet monologue leaking into L2d speech."""

from __future__ import annotations

import json
import logging
import sys

import httpx
import pytest

from kazusa_ai_chatbot.action_spec.registry import (
    build_initial_action_capabilities,
    project_prompt_affordances,
)
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d_module
from kazusa_ai_chatbot.cognition_chain_core.stages.l2d import (
    build_action_selection_payload_text,
    select_semantic_actions,
)
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def test_l2d_live_quiet_group_monologue_does_not_select_speak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the L2d quiet-monologue failure from a frozen upstream state.

    The upstream state mirrors the production log where cognition labels the
    turn CONFIRM/PROVIDE while the internal monologue says there is no
    particular thing to say.  Desired L2d behavior is no visible speak action.
    """

    await _skip_if_llm_unavailable()
    frozen_state = _quiet_group_graphics_state()
    prompt_payload = build_action_selection_payload_text(frozen_state)

    action_selection_llm = build_cognition_chain_services().action_selection_llm
    capturing_llm = _CapturingLLM(action_selection_llm)
    monkeypatch.setattr(l2d_module, '_action_selection_llm', capturing_llm)

    result = await select_semantic_actions(frozen_state)
    raw_output = capturing_llm.raw_output
    raw_parsed_output = parse_llm_json_output(raw_output)
    action_specs = action_connector.materialize_semantic_action_requests(
        result.get('semantic_action_requests', []),
        frozen_state,
    )
    observed_speak = _observed_user_visible_speak(action_specs)
    speak_reasons = _observed_user_visible_speak_reasons(action_specs)
    trace_path = write_llm_trace(
        'l2d_quiet_monologue_live_llm',
        'qq_group_graphics_no_particular_thing_to_say',
        {
            'case_id': 'qq_group_graphics_no_particular_thing_to_say',
            'source_log_timestamp': '2026-06-12 19:55:54,573',
            'evaluation_goal': (
                'L2d should not select visible speech when the explicit '
                'internal monologue says there is no particular thing to say.'
            ),
            'prompt_payload': prompt_payload,
            'raw_model_output': raw_output,
            'raw_parsed_output': raw_parsed_output,
            'parsed_result': result,
            'materialized_action_specs': action_specs,
            'observed_user_visible_speak': observed_speak,
            'observed_user_visible_speak_reasons': speak_reasons,
            'judgment': 'manual_review_required_for_l2d_quiet_monologue',
        },
    )
    logger.info(
        'L2D_QUIET_MONOLOGUE case=qq_group_graphics_no_particular_thing_to_say '
        f'trace={trace_path} observed_speak={observed_speak} '
        f'reasons={json.dumps(speak_reasons, ensure_ascii=True)}'
    )

    assert not observed_speak, (
        'L2d selected a user-visible speak action even though the monologue '
        'states there is no particular thing to say; '
        f'trace={trace_path}; reasons={speak_reasons!r}'
    )


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{COGNITION_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(f'LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}')

    if response.status_code >= 500:
        pytest.skip(
            f'LLM endpoint returned server error {response.status_code}: '
            f'{COGNITION_LLM_BASE_URL}'
        )


def _quiet_group_graphics_state() -> dict:
    """Build the frozen pre-L2d state from the reported production log."""

    state = {
        'character_profile': {
            'name': '杏山千纱',
            'mood': '平静',
            'global_vibe': '轻松的群聊旁观状态',
        },
        'storage_timestamp_utc': '2026-06-12T07:55:54.573000+00:00',
        'local_time_context': {
            'current_local_datetime': '2026-06-12 19:55',
            'current_local_weekday': 'Friday',
        },
        'prompt_message_context': {
            'body_text': '打游戏的话 i 卡没优化吧我记得',
            'broadcast': True,
        },
        'cognitive_episode': {
            'episode_id': 'log-2026-06-12-195554-l2d',
            'trigger_source': 'user_message',
            'input_sources': ['dialog_text'],
            'output_mode': 'visible_reply',
        },
        'platform': 'qq',
        'platform_channel_id': '961633605',
        'channel_type': 'group',
        'platform_message_id': '344135473',
        'platform_user_id': '1145187581',
        'global_user_id': 'quiet-monologue-user',
        'user_name': 'Karma',
        'user_profile': {
            'display_name': 'Karma',
            'affinity': 520,
            'last_relationship_insight': '普通群友，当前没有直接点名压力。',
        },
        'platform_bot_id': 'pytest-bot',
        'chat_history_recent': [
            {
                'role': 'user',
                'name': '群友A',
                'content': '最近显卡怎么选啊，a 卡 n 卡还是 i 卡？',
            },
            {
                'role': 'user',
                'name': 'Karma',
                'content': '打游戏的话 i 卡没优化吧我记得',
            },
        ],
        'reply_context': {},
        'indirect_speech_context': '',
        'channel_topic': '群聊里随口讨论显卡和游戏优化',
        'referents': [],
        'decontexualized_input': (
            'Karma 在群聊中说：打游戏的话 i 卡没优化吧我记得。'
        ),
        'media_summary': '',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'judgment_note': (
            '上游判断已明确角色意图为 PROVIDE，且 RAG 证据已确认 Karma '
            '的具体说法；群聊氛围轻松、无边界问题，适合直接可见回复。'
        ),
        'internal_monologue': (
            '群里在聊显卡，Karma 说打游戏的话 i 卡没优化吧我记得。'
            '气氛挺轻松的，我刚才也没插话，现在看着他们聊这个技术话题，'
            '没什么特别的想说的。'
        ),
        'emotional_appraisal': '群聊里有人提了显卡，气氛挺轻松的。',
        'interaction_subtext': (
            'Karma 在讨论 i 卡优化问题，我作为旁观者注意到这个技术话题正在展开。'
        ),
        'boundary_core_assessment': {
            'boundary_issue': 'none',
            'boundary_summary': '普通群聊技术讨论，没有身份、控制或压力问题。',
            'behavior_primary': 'comply',
            'behavior_secondary': 'none',
            'acceptance': 'allow',
            'stance_bias': 'confirm',
        },
        'social_distance': '普通群聊里的轻松接话距离。',
        'emotional_intensity': '低，轻松但不强烈。',
        'vibe_check': '群聊技术话题自然展开，适合轻描淡写地接话。',
        'relational_dynamic': 'Karma 的说法给了一个自然接话点。',
        'conversation_progress': {
            'source': 'production_log_fixture',
            'current_thread': '群聊显卡与游戏优化闲聊',
            'status': 'active',
            'next_affordances': [
                '回应 Karma 关于 i 卡游戏优化的观点',
                '以旁观者身份轻描淡写地接话',
            ],
        },
        'rag_result': {
            'answer': 'Karma 的当前说法是：打游戏的话 i 卡没优化吧我记得。',
            'memory_evidence': [],
            'conversation_evidence': [
                {
                    'summary': (
                        '群聊正在围绕显卡和游戏优化闲聊；没有成员直接点名角色回答。'
                    ),
                },
            ],
            'external_evidence': [],
            'recall_evidence': [],
            'user_image': {
                'user_memory_context': {
                    'stable_patterns': [],
                    'recent_shifts': [],
                    'objective_facts': [],
                    'milestones': [],
                    'active_commitments': [],
                },
            },
        },
        'resolver_context': '',
        'available_action_affordances': project_prompt_affordances(
            build_initial_action_capabilities(),
        ),
        'max_action_requests': 3,
        'max_resolver_requests': 3,
        'background_work_output_char_limit': 4000,
    }
    return state


def _observed_user_visible_speak(action_specs: list[dict]) -> bool:
    """Return whether materialized action specs contain visible speech."""

    for action_spec in action_specs:
        if action_spec.get('kind') != 'speak':
            continue
        if action_spec.get('visibility') == 'user_visible':
            return True
    return False


def _observed_user_visible_speak_reasons(action_specs: list[dict]) -> list[str]:
    """Return prompt-facing reasons for selected visible speech."""

    reasons: list[str] = []
    for action_spec in action_specs:
        if action_spec.get('kind') != 'speak':
            continue
        if action_spec.get('visibility') != 'user_visible':
            continue
        reason = action_spec.get('reason')
        if isinstance(reason, str) and reason.strip():
            reasons.append(reason.strip())
    return reasons


class _CapturingLLM:
    """Capture raw LLM output while preserving the production call path."""

    def __init__(self, inner_llm: object) -> None:
        self._inner_llm = inner_llm
        self.raw_output = ''

    async def ainvoke(self, messages: object) -> object:
        """Call the wrapped LLM and store the raw message content."""

        response = await self._inner_llm.ainvoke(messages)
        self.raw_output = str(response.content)
        return response
