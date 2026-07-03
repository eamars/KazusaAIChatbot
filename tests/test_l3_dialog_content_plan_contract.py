"""Focused contract tests for the L3-to-dialog content-plan handoff."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.cognition_chain_core.stages import l3 as l3_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    StateContractError,
    validate_dialog_action_directives,
)
from llm_test_helpers import bind_test_llm


class _StaticContentPlanLLM:
    """Return a fixed content-plan payload."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    async def ainvoke(
        self,
        _messages: list[object],
        *,
        config,
    ) -> SimpleNamespace:
        del config
        response = SimpleNamespace(
            content=json.dumps(self._payload, ensure_ascii=False),
        )
        return response


def _dialog_state_with_linguistic(linguistic_directives: dict) -> dict:
    """Build the smallest dialog state needed by directive validation."""

    state = {
        'action_directives': {
            'contextual_directives': {
                'social_distance': 'friendly',
                'emotional_intensity': 'low',
                'vibe_check': 'light',
                'relational_dynamic': 'casual reply',
            },
            'linguistic_directives': linguistic_directives,
        },
    }
    return state


def _content_plan_state() -> dict:
    """Build a compact L3 state for content-plan agent tests."""

    state = {
        'user_input': '继续刚才那个轻松话题。',
        'prompt_message_context': {
            'body_text': '继续刚才那个轻松话题。',
            'mentions': [],
            'attachments': [],
            'broadcast': False,
        },
        'reply_context': {},
        'decontexualized_input': '用户想延续轻松调侃。',
        'referents': [],
        'rag_result': {
            'answer': '',
            'memory_evidence': [],
            'conversation_evidence': [],
            'external_evidence': [],
            'recall_evidence': [],
            'user_image': {},
        },
        'internal_monologue': 'I should answer only the resolved light tease.',
        'logical_stance': 'CONFIRM',
        'character_intent': 'BANTAR',
        'judgment_note': 'Keep it light and do not invent a new topic.',
        'social_distance': 'friendly',
        'emotional_intensity': 'low',
        'vibe_check': 'playful',
        'relational_dynamic': 'casual teasing',
        'memory_lifecycle_context': {
            'content_plan_roles': [],
        },
        'interaction_style_context': {},
        'conversation_progress': {},
        'selected_text_surface_intent': '接住轻松调侃，不补新话题。',
        'resolver_goal_progress': {},
        'resolver_state': {'observations': []},
        'character_profile': {
            'name': 'Test Character',
        },
        'cognitive_episode': {
            'trigger_source': 'user_message',
            'input_sources': ['dialog_text'],
            'output_mode': 'visible_reply',
            'origin_metadata': {'debug_modes': {}},
        },
        'channel_type': 'private',
        'platform': 'debug',
        'platform_channel_id': 'channel-1',
        'global_user_id': 'user-1',
        'user_name': 'Test User',
    }
    return state


def test_dialog_validation_requires_content_plan() -> None:
    """Dialog should fail before LLM calls when content_plan is absent."""

    state = _dialog_state_with_linguistic({
        'rhetorical_strategy': 'answer directly',
        'linguistic_style': 'brief',
        'forbidden_phrases': [],
    })

    with pytest.raises(StateContractError, match='content_plan'):
        validate_dialog_action_directives(state, usage_mode='unit_test')


def test_dialog_validation_rejects_old_content_anchors_contract() -> None:
    """The bigbang contract must not accept the retired old anchor shape."""

    state = _dialog_state_with_linguistic({
        'rhetorical_strategy': 'answer directly',
        'linguistic_style': 'brief',
        'content_anchors': [
            '[DECISION] Answer directly.',
            '[ANSWER] Known fact.',
        ],
        'forbidden_phrases': [],
    })

    with pytest.raises(StateContractError, match='content_plan'):
        validate_dialog_action_directives(state, usage_mode='unit_test')


def test_dialog_validation_normalizes_freeform_content_plan_keys() -> None:
    """Validation should accept dict[str, str] without hard key enums."""

    state = _dialog_state_with_linguistic({
        'rhetorical_strategy': 'answer directly',
        'linguistic_style': 'brief',
        'content_plan': {
            ' semantic_content ': ' 只表达被逗乐和舒服，不补新话题。 ',
            'rendering ': ' 1 条普通文字消息；2-3个自然短句。 ',
            'weaker_model_key_drift': '允许自由键名，只要值是字符串。',
            'empty_value': '   ',
        },
        'forbidden_phrases': [],
    })

    linguistic_directives, _ = validate_dialog_action_directives(
        state,
        usage_mode='unit_test',
    )

    assert linguistic_directives['content_plan'] == {
        'semantic_content': '只表达被逗乐和舒服，不补新话题。',
        'rendering': '1 条普通文字消息；2-3个自然短句。',
        'weaker_model_key_drift': '允许自由键名，只要值是字符串。',
    }


def test_content_plan_prompt_describes_message_sequence_rendering() -> None:
    """L3 should describe rendering as an outbound message sequence."""

    prompt = l3_module._CONTENT_PLAN_AGENT_PROMPT
    retired_layout = ''.join(('单个', '\u804a\u5929\u6c14\u6ce1'))

    assert '`rendering` 写出站消息序列的形状' in prompt
    assert '1 条普通文字消息' in prompt
    assert '2 条连续发送的普通文字消息' in prompt
    assert '完整固定格式块可以单独成为一条消息' in prompt
    assert retired_layout not in prompt


@pytest.mark.asyncio
async def test_content_plan_agent_returns_native_dict_with_string_values() -> None:
    """L3 content-plan agent should output a native dict, not old anchors."""

    llm = _StaticContentPlanLLM({
            'content_plan': {
                ' visible_goal ': '接住轻松调侃。',
                'semantic_content': '被对方逗乐了，这种相处方式很舒服。',
                'voice': '轻快随和。',
                'rendering': '约35字；1 条普通文字消息；2-3个自然短句。',
            },
    })

    with patch.object(
        l3_module,
        '_content_plan_agent_llm',
        bind_test_llm(llm, 'content_plan_agent'),
    ):
        result = await l3_module.call_content_plan_agent(_content_plan_state())

    assert result == {
        'content_plan': {
            'visible_goal': '接住轻松调侃。',
            'semantic_content': '被对方逗乐了，这种相处方式很舒服。',
            'voice': '轻快随和。',
            'rendering': '约35字；1 条普通文字消息；2-3个自然短句。',
        },
    }


@pytest.mark.asyncio
async def test_surface_collector_emits_content_plan_without_anchor_append() -> None:
    """L4 should pass through L3's plan instead of appending extra anchors."""

    state = {
        'social_distance': 'friendly',
        'emotional_intensity': 'low',
        'vibe_check': 'playful',
        'relational_dynamic': 'casual teasing',
        'rhetorical_strategy': 'soft tease',
        'linguistic_style': 'brief',
        'accepted_user_preferences': [],
        'content_plan': {
            'semantic_content': '被对方逗乐了，这种相处方式很舒服。',
            'rendering': '1 条普通文字消息；2-3个自然短句。',
        },
        'forbidden_phrases': [],
        'facial_expression': [],
        'body_language': [],
        'gaze_direction': [],
        'visual_vibe': [],
        'resolver_goal_progress': {
            'schema_version': 'resolver_goal_progress.v1',
            'original_goal': '不应由收集器追加到内容计划。',
            'current_focus': '',
            'deliverables': [],
            'missing_user_inputs': [],
            'evidence_dependencies': [],
            'attempted_paths': [],
            'source_backed_facts': [],
            'assumptions_or_inferences': [],
            'blockers': [],
            'final_response_requirements': [],
        },
        'resolver_state': {
            'observations': [
                {
                    'schema_version': 'resolver_observation.v1',
                    'observation_id': 'raw-observation-id',
                    'capability_kind': 'public_answer_research',
                    'request_objective': 'Do not append.',
                    'request_reason': 'Do not append.',
                    'status': 'succeeded',
                    'prompt_safe_summary': 'Do not append this summary.',
                    'created_at_utc': '2026-06-10T00:00:00+00:00',
                },
            ],
        },
    }

    result = await l3_module.call_surface_directive_collector(state)

    linguistic = result['action_directives']['linguistic_directives']
    assert linguistic['content_plan'] == state['content_plan']
    assert 'content_anchors' not in linguistic
    assert 'Do not append this summary.' not in json.dumps(
        linguistic['content_plan'],
        ensure_ascii=False,
    )


def test_dialog_prompts_use_content_plan_as_semantic_authority() -> None:
    """Prompt text should describe the new content-plan contract."""

    generator_prompt = dialog_module._DIALOG_GENERATOR_PROMPT
    assert 'content_plan' in generator_prompt
    assert 'semantic_content' in generator_prompt
    assert 'content_anchors' not in generator_prompt
    assert '[DECISION]' not in generator_prompt
    assert '[ANSWER]' not in generator_prompt
    assert '[SCOPE]' not in generator_prompt
