"""Live LLM evidence for group-style effects on visible dialog shape."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import re
import sys
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest

from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_episode import (
    validate_cognitive_episode_v1,
)
from kazusa_ai_chatbot.config import (
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MODEL,
    DIALOG_GENERATOR_LLM_BASE_URL,
    DIALOG_GENERATOR_LLM_MODEL,
)
from kazusa_ai_chatbot.db import interaction_style_images as style_store
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.llm_trace import write_llm_trace
from tests.live_llm_mongo import live_db


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_llm,
    pytest.mark.live_db,
]

_CHARACTER_PATH = (
    Path(__file__).resolve().parents[1] / 'personalities' / 'asuna.json'
)
_TRACE_SUITE = 'group_style_output_shape_live_llm'
_GLOBAL_USER_ID = 'group-style-live-user'
_PLATFORM_USER_ID = 'group-style-live-platform-user'
_PLATFORM_BOT_ID = 'group-style-live-platform-bot'

_STYLE_OVERLAYS: dict[str, dict[str, object]] = {
    'neutral': {
        'speech_guidelines': [],
        'social_guidelines': [],
        'pacing_guidelines': [],
        'engagement_guidelines': [],
        'confidence': '',
    },
    'short_burst': {
        'speech_guidelines': [
            '优先使用短句，先表达一个主要意思，减少不必要的展开。',
        ],
        'social_guidelines': [],
        'pacing_guidelines': [
            '节奏轻快，句子之间保留短促的自然停顿。',
        ],
        'engagement_guidelines': [],
        'confidence': 'controlled',
    },
    'long_flowing': {
        'speech_guidelines': [
            '使用完整连贯的句子，围绕当前意思适度展开背景、原因和感受。',
        ],
        'social_guidelines': [],
        'pacing_guidelines': [
            '节奏舒缓，让相关内容自然连成一段。',
        ],
        'engagement_guidelines': [],
        'confidence': 'controlled',
    },
}

_PROMPT_CASES: dict[str, str] = {
    'status': '你今天过得怎么样？告诉我你现在的状态。',
    'recent_event': '刚才发生了什么？你现在在想什么？',
    'next_topic': '如果现在继续聊下去，你最想先说哪一件事？',
}


class _CapturingLiveLLM:
    """Delegate to a real route while retaining raw prompts and responses."""

    def __init__(self, stage_name: str, delegate: Any) -> None:
        self.stage_name = stage_name
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
            'stage': self.stage_name,
            'configured_stage': str(
                getattr(config, 'stage_name', '')
            ),
            'messages': [
                {
                    'type': type(message).__name__,
                    'content': str(getattr(message, 'content', '')),
                }
                for message in messages
            ],
            'raw_output': str(getattr(response, 'content', '')),
        })
        return response


async def _skip_if_endpoint_unavailable(
    name: str,
    base_url: str,
) -> None:
    """Skip one live case when its OpenAI-compatible route is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f'{name} endpoint is unavailable: {base_url}; {exc}')
    if response.status_code >= 500:
        pytest.skip(
            f'{name} endpoint returned server error '
            f'{response.status_code}: {base_url}'
        )


async def _skip_if_live_routes_unavailable() -> None:
    """Check both real model routes before creating a style-image fixture."""

    await _skip_if_endpoint_unavailable(
        'cognition',
        COGNITION_LLM_BASE_URL,
    )
    await _skip_if_endpoint_unavailable(
        'dialog generator',
        DIALOG_GENERATOR_LLM_BASE_URL,
    )


@pytest.fixture(autouse=True)
def _stub_dialog_event_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the prompt experiment scoped to its explicit test artifacts."""

    for recorder_name in (
        'record_llm_stage_event',
        'record_model_contract_event',
        'record_dialog_quality_event',
    ):
        monkeypatch.setattr(
            dialog_module.event_logging,
            recorder_name,
            AsyncMock(),
        )


def _character_profile() -> dict[str, Any]:
    """Load the frozen production Asuna profile used by every arm."""

    profile = json.loads(_CHARACTER_PATH.read_text(encoding='utf-8'))
    if not isinstance(profile, dict):
        raise TypeError('frozen character profile must be an object')
    return profile


def _style_overlay(style_arm: str) -> dict[str, object]:
    """Return a fresh controlled overlay for one experimental arm."""

    raw_overlay = _STYLE_OVERLAYS.get(style_arm)
    if raw_overlay is None:
        raise ValueError(f'unknown group style arm: {style_arm}')
    return {
        field_name: (
            list(value) if isinstance(value, list) else value
        )
        for field_name, value in raw_overlay.items()
    }


def _group_episode(
    *,
    case_id: str,
    user_input: str,
    platform_channel_id: str,
) -> dict[str, Any]:
    """Build a validated group episode with fixed semantic intent."""

    episode = canonical_episode(
        episode_id=f'group-style-{case_id}',
        content=user_input,
        current_global_user_id=_GLOBAL_USER_ID,
    )
    target_scope = episode['target_scope']
    target_scope.update({
        'platform': 'debug',
        'platform_channel_id': platform_channel_id,
        'channel_type': 'group',
        'current_platform_user_id': _PLATFORM_USER_ID,
        'current_global_user_id': _GLOBAL_USER_ID,
        'current_display_name': '测试用户',
        'target_addressed_user_ids': [_GLOBAL_USER_ID],
        'target_broadcast': True,
    })
    origin_metadata = episode['origin_metadata']
    origin_metadata.update({
        'platform': 'debug',
        'platform_channel_id': platform_channel_id,
        'platform_message_id': f'message-{case_id}',
        'active_turn_platform_message_ids': [f'message-{case_id}'],
        'privacy_scope': 'group',
        'debug_modes': {'no_visual_directives': True},
    })
    episode['privacy_scope'] = 'group'
    return validate_cognitive_episode_v1(episode)


def _surface_input(
    *,
    case_id: str,
    user_input: str,
    platform_channel_id: str,
    interaction_style_context: str,
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Build one constant-intent surface input for a style comparison."""

    expression_context, visual_context = l3_module._character_surface_contexts({
        'character_profile': profile,
    })
    return {
        'schema_version': 'text_surface_input.v2',
        'episode': _group_episode(
            case_id=case_id,
            user_input=user_input,
            platform_channel_id=platform_channel_id,
        ),
        'intention': {
            'route': 'speech',
            'intention': '回应当前用户的问题，表达当前角色的状态或想法',
            'target_roles': [],
            'reason': '当前用户提出了一个可以直接回应的问题。',
            'goal_continuation_ref': None,
        },
        'goal_resolution': 'answerable_now',
        'supporting_bids': [],
        'expression_policy': {
            'visibility': 'visible',
            'emotional_tone': 'warm',
            'intensity': 'moderate',
            'directness': 'balanced',
        },
        'semantic_affect': [],
        'permitted_action_results': [],
        'interaction_style_context': interaction_style_context,
        'character_expression_context': expression_context,
        'visual_character_context': visual_context,
        'addressee_plan': [{
            'handle': 'current_user',
            'display_name': '测试用户',
            'semantic_role': 'direct_recipient',
            'wording_policy': 'second_person_allowed',
        }],
    }


def _dialog_state(
    *,
    surface_input: dict[str, Any],
    surface_output: dict[str, Any],
    profile: dict[str, Any],
    style_arm: str,
    case_id: str,
) -> dict[str, Any]:
    """Build the direct production dialog state for one live case."""

    return {
        'internal_monologue': '当前角色先回答用户的问题，再按风格组织表达。',
        'text_surface_input_v2': surface_input,
        'text_surface_output_v2': surface_output,
        'cognitive_episode': surface_input['episode'],
        'chat_history_wide': [],
        'chat_history_recent': [],
        'platform_user_id': _PLATFORM_USER_ID,
        'platform_bot_id': _PLATFORM_BOT_ID,
        'global_user_id': _GLOBAL_USER_ID,
        'user_name': '测试用户',
        'user_profile': {},
        'character_profile': profile,
        'final_dialog': [],
        'target_addressed_user_ids': [_GLOBAL_USER_ID],
        'target_broadcast': True,
        'dialog_usage_mode': 'live_visible_reply',
        'llm_trace_id': f'group-style-{style_arm}-{case_id}',
    }


def _dialog_metrics(final_dialog: object) -> dict[str, Any]:
    """Extract comparable shape metrics without judging semantic quality."""

    if not isinstance(final_dialog, list) or any(
        not isinstance(message, str) for message in final_dialog
    ):
        return {
            'valid_string_list': False,
            'message_count': 0,
            'character_count': 0,
            'newline_count': 0,
            'blank_line_count': 0,
            'sentence_count': 0,
        }

    joined = '\n'.join(final_dialog)
    sentence_fragments = re.findall(r'[^。！？!?]+[。！？!?]', joined)
    return {
        'valid_string_list': True,
        'message_count': len(final_dialog),
        'character_count': len(joined),
        'newline_count': joined.count('\n'),
        'blank_line_count': len(re.findall(r'\n[ \t]*\n', joined)),
        'sentence_count': len(sentence_fragments),
        'sentence_lengths': [len(fragment.strip()) for fragment in sentence_fragments],
        'joined_dialog': joined,
    }


async def _run_live_case(
    *,
    live_database: Any,
    monkeypatch: pytest.MonkeyPatch,
    style_arm: str,
    case_id: str,
    user_input: str,
) -> dict[str, Any]:
    """Run one real style arm and write all raw model evidence."""

    await _skip_if_live_routes_unavailable()
    profile = _character_profile()
    platform_channel_id = (
        f'group-style-live-{style_arm}-{uuid4().hex}'
    )
    overlay = _style_overlay(style_arm)
    style_document = await style_store.upsert_group_channel_style_image(
        platform='debug',
        platform_channel_id=platform_channel_id,
        overlay=overlay,
        source_reflection_run_ids=[],
        storage_timestamp_utc='2026-08-15T00:00:00+00:00',
    )

    try:
        style_snapshot = await style_store.build_interaction_style_context(
            global_user_id='',
            channel_type='group',
            platform='debug',
            platform_channel_id=platform_channel_id,
        )
        interaction_style_context = (
            l3_module._render_interaction_style_context(style_snapshot)
        )
        surface_input = _surface_input(
            case_id=case_id,
            user_input=user_input,
            platform_channel_id=platform_channel_id,
            interaction_style_context=interaction_style_context,
            profile=profile,
        )

        text_services = l3_module._build_text_surface_services()
        text_llm = _CapturingLiveLLM('surface', text_services.llm)
        text_services = replace(text_services, llm=text_llm)
        text_output = await run_text_surface_planning(
            surface_input,
            text_services,
        )

        generator_llm = _CapturingLiveLLM(
            'dialog_generator',
            dialog_module._dialog_generator_llm,
        )
        semantic_llm = _CapturingLiveLLM(
            'dialog_semantic_fidelity',
            dialog_module._dialog_semantic_fidelity_llm,
        )
        integrity_llm = _CapturingLiveLLM(
            'dialog_surface_integrity',
            dialog_module._dialog_surface_integrity_llm,
        )
        monkeypatch.setattr(
            dialog_module,
            '_dialog_generator_llm',
            generator_llm,
        )
        monkeypatch.setattr(
            dialog_module,
            '_dialog_semantic_fidelity_llm',
            semantic_llm,
        )
        monkeypatch.setattr(
            dialog_module,
            '_dialog_surface_integrity_llm',
            integrity_llm,
        )
        dialog_output = await dialog_module.dialog_generator(
            _dialog_state(
                surface_input=surface_input,
                surface_output=text_output,
                profile=profile,
                style_arm=style_arm,
                case_id=case_id,
            )
        )

        final_dialog = dialog_output.get('final_dialog')
        evidence = {
            'experiment': {
                'style_arm': style_arm,
                'case_id': case_id,
                'user_input': user_input,
                'semantic_intent': surface_input['intention'],
                'model_routes': {
                    'surface': COGNITION_LLM_BASE_URL,
                    'surface_model': COGNITION_LLM_MODEL,
                    'dialog': DIALOG_GENERATOR_LLM_BASE_URL,
                    'dialog_model': DIALOG_GENERATOR_LLM_MODEL,
                },
            },
            'group_style_image_document': style_document,
            'interaction_style_snapshot': style_snapshot,
            'rendered_interaction_style_context': interaction_style_context,
            'surface_input': surface_input,
            'surface_model_calls': text_llm.calls,
            'text_surface_output': text_output,
            'dialog_generator_calls': generator_llm.calls,
            'dialog_semantic_fidelity_calls': semantic_llm.calls,
            'dialog_surface_integrity_calls': integrity_llm.calls,
            'dialog_output': dialog_output,
            'final_dialog_metrics': _dialog_metrics(final_dialog),
            'human_review_contract': {
                'hold_character_intent_constant': True,
                'style_input_is_group_channel_projection': True,
                'compare_delivery_profile_before_final_dialog': True,
                'inspect_raw_prompts_and_outputs': True,
            },
        }
        artifact_path = write_llm_trace(
            _TRACE_SUITE,
            f'{style_arm}__{case_id}',
            evidence,
        )

        assert artifact_path.exists()
        assert style_snapshot['group_channel_style'] == style_snapshot[
            'surface'
        ]['group_channel']['overlay']
        assert text_llm.calls
        assert generator_llm.calls
        assert isinstance(final_dialog, list) and final_dialog
        evidence['trace_path'] = str(artifact_path)
        return evidence
    finally:
        await live_database[
            style_store.INTERACTION_STYLE_IMAGE_COLLECTION
        ].delete_one({
            'style_image_id': style_document['style_image_id'],
        })


async def test_group_style_neutral_status_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the neutral baseline for the current-status prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='neutral',
        case_id='status',
        user_input=_PROMPT_CASES['status'],
    )


async def test_group_style_short_status_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture short-burst guidance for the current-status prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='short_burst',
        case_id='status',
        user_input=_PROMPT_CASES['status'],
    )


async def test_group_style_long_status_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture long-flowing guidance for the current-status prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='long_flowing',
        case_id='status',
        user_input=_PROMPT_CASES['status'],
    )


async def test_group_style_neutral_recent_event_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the neutral baseline for the recent-event prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='neutral',
        case_id='recent_event',
        user_input=_PROMPT_CASES['recent_event'],
    )


async def test_group_style_short_recent_event_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture short-burst guidance for the recent-event prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='short_burst',
        case_id='recent_event',
        user_input=_PROMPT_CASES['recent_event'],
    )


async def test_group_style_long_recent_event_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture long-flowing guidance for the recent-event prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='long_flowing',
        case_id='recent_event',
        user_input=_PROMPT_CASES['recent_event'],
    )


async def test_group_style_neutral_next_topic_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture the neutral baseline for the next-topic prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='neutral',
        case_id='next_topic',
        user_input=_PROMPT_CASES['next_topic'],
    )


async def test_group_style_short_next_topic_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture short-burst guidance for the next-topic prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='short_burst',
        case_id='next_topic',
        user_input=_PROMPT_CASES['next_topic'],
    )


async def test_group_style_long_next_topic_live_llm(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capture long-flowing guidance for the next-topic prompt."""

    await _run_live_case(
        live_database=live_db,
        monkeypatch=monkeypatch,
        style_arm='long_flowing',
        case_id='next_topic',
        user_input=_PROMPT_CASES['next_topic'],
    )
