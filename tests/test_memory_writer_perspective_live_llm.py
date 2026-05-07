"""Live LLM checks for durable memory perspective contracts."""

from __future__ import annotations

from typing import Any
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_memory_unit_extractor_prompt_payload,
    project_memory_unit_rewrite_prompt_payload,
    project_reflection_promotion_prompt_payload,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_images as images_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_memory_units as memory_units_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_reflection as reflection_module,
)
from kazusa_ai_chatbot.reflection_cycle import promotion as promotion_module
from kazusa_ai_chatbot.utils import parse_llm_json_output
from scripts import sanitize_memory_writer_perspective as migration_module
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

CHARACTER_NAME = '杏山千纱 (Kyōyama Kazusa)'
TRACE_SUITE = 'memory_writer_perspective_live_llm'
POLLUTED_LABELS = ('角色', '助理', 'assistant', 'active_character')
PROMOTE_ACTIONS = {'promote_new', 'supersede', 'merge'}
FROZEN_EXAMPLE_STATE_LABELS = {
    'Affectionate',
    'Agitated',
    'Calm',
    'Curious',
    'Defensive',
    'Distrustful',
    'Distressed',
    'Flustered',
    'Neutral',
    'Playful',
    'Shy',
    'Slightly Tense',
    'Softened',
    'Tense',
    'Warm',
}


async def test_live_memory_unit_extractor_perspective_false_negative() -> None:
    """Polluted speaker labels should not become durable actor names."""

    payload = _memory_unit_extractor_false_negative_payload()
    projected_payload = project_memory_unit_extractor_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )
    prompt = memory_units_module._EXTRACTOR_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        memory_units_module._extractor_llm,
        prompt,
        projected_payload,
    )
    trace_path = _write_case_trace(
        'memory_unit_extractor_perspective_false_negative',
        prompt,
        projected_payload,
        raw_output,
        parsed,
        'Extractor should use the profile name instead of polluted labels.',
    )

    units = parsed.get('memory_units')
    assert isinstance(units, list)
    assert units
    assert len(units) == 1
    durable_text = _memory_units_text(units)
    _assert_profile_name_used(durable_text)
    _assert_polluted_labels_absent(durable_text)
    assert trace_path.exists()


async def test_live_memory_unit_extractor_perspective_false_positive() -> None:
    """User-owned first person should remain a user-owned memory."""

    payload = _memory_unit_extractor_false_positive_payload()
    projected_payload = project_memory_unit_extractor_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )
    prompt = memory_units_module._EXTRACTOR_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        memory_units_module._extractor_llm,
        prompt,
        projected_payload,
    )
    trace_path = _write_case_trace(
        'memory_unit_extractor_perspective_false_positive',
        prompt,
        projected_payload,
        raw_output,
        parsed,
        'Extractor should keep Atlas as the user project name.',
    )

    units = parsed.get('memory_units')
    assert isinstance(units, list)
    assert units
    durable_text = _memory_units_text(units)
    assert 'Atlas' in durable_text
    _assert_user_owned_first_person(durable_text)
    _assert_no_profile_name_shortening(durable_text)
    _assert_polluted_labels_absent(durable_text)
    assert trace_path.exists()


async def test_live_memory_unit_rewrite_perspective_false_negative() -> None:
    """Rewrite should not preserve polluted active-character labels."""

    payload = _memory_unit_rewrite_false_negative_payload()
    projected_payload = project_memory_unit_rewrite_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )
    prompt = memory_units_module._REWRITE_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        memory_units_module._rewrite_llm,
        prompt,
        projected_payload,
    )
    trace_path = _write_case_trace(
        'memory_unit_rewrite_perspective_false_negative',
        prompt,
        projected_payload,
        raw_output,
        parsed,
        'Rewrite should canonicalize current-character references.',
    )

    durable_text = _fields_text(
        parsed,
        ('fact', 'subjective_appraisal', 'relationship_signal'),
    )
    _assert_profile_name_used(durable_text)
    _assert_polluted_labels_absent(durable_text)
    assert trace_path.exists()


async def test_live_memory_unit_rewrite_perspective_false_positive() -> None:
    """Rewrite should not treat user-owned first person as the character."""

    payload = _memory_unit_rewrite_false_positive_payload()
    projected_payload = project_memory_unit_rewrite_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )
    prompt = memory_units_module._REWRITE_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        memory_units_module._rewrite_llm,
        prompt,
        projected_payload,
    )
    trace_path = _write_case_trace(
        'memory_unit_rewrite_perspective_false_positive',
        prompt,
        projected_payload,
        raw_output,
        parsed,
        'Rewrite should keep Atlas as the user project name.',
    )

    durable_text = _fields_text(
        parsed,
        ('fact', 'subjective_appraisal', 'relationship_signal'),
    )
    assert 'Atlas' in durable_text
    _assert_user_owned_first_person(durable_text)
    _assert_no_profile_name_shortening(durable_text)
    assert trace_path.exists()


async def test_live_relationship_recorder_perspective_false_negative() -> None:
    """Relationship evidence should name the character from the profile."""

    payload = _relationship_false_negative_payload()
    prompt = reflection_module._RELATIONSHIP_RECORDER_PROMPT.format(
        character_name=CHARACTER_NAME,
        user_name='测试用户',
        character_mbti='ISTJ',
    )

    parsed, raw_output = await _invoke_json(
        reflection_module._relationship_recorder_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'relationship_recorder_perspective_false_negative',
        prompt,
        payload,
        raw_output,
        parsed,
        'Relationship recorder should not copy role/assistant labels.',
    )

    assert parsed.get('skip') is False
    durable_text = _fields_text(
        parsed,
        ('subjective_appraisals', 'last_relationship_insight'),
    )
    _assert_profile_name_used(durable_text)
    _assert_polluted_labels_absent(durable_text)
    assert trace_path.exists()


async def test_live_relationship_recorder_perspective_false_positive() -> None:
    """Neutral user-owned first person should not become relationship memory."""

    payload = _relationship_false_positive_payload()
    prompt = reflection_module._RELATIONSHIP_RECORDER_PROMPT.format(
        character_name=CHARACTER_NAME,
        user_name='测试用户',
        character_mbti='ISTJ',
    )

    parsed, raw_output = await _invoke_json(
        reflection_module._relationship_recorder_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'relationship_recorder_perspective_false_positive',
        prompt,
        payload,
        raw_output,
        parsed,
        'Neutral user first-person clarification should remain non-memory.',
    )

    assert parsed.get('skip') is True
    assert parsed.get('affinity_delta') in (0, '0')
    durable_text = _fields_text(
        parsed,
        ('subjective_appraisals', 'last_relationship_insight'),
    )
    assert CHARACTER_NAME not in durable_text
    assert trace_path.exists()


async def test_live_global_state_updater_perspective_false_negative() -> None:
    """Global reflection should not save generic active-character labels."""

    payload = _global_state_false_negative_payload()
    prompt = reflection_module._GLOBAL_STATE_UPDATER_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        reflection_module._global_state_updater_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'global_state_updater_perspective_false_negative',
        prompt,
        payload,
        raw_output,
        parsed,
        'Global reflection summary should use the profile name.',
    )

    summary = str(parsed.get('reflection_summary') or '')
    _assert_profile_name_used(summary)
    _assert_polluted_labels_absent(summary)
    assert trace_path.exists()


async def test_live_global_state_updater_perspective_false_positive() -> None:
    """User-authored first person should stay user-owned in reflection."""

    payload = _global_state_false_positive_payload()
    prompt = reflection_module._GLOBAL_STATE_UPDATER_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        reflection_module._global_state_updater_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'global_state_updater_perspective_false_positive',
        prompt,
        payload,
        raw_output,
        parsed,
        'Global reflection should not convert user tiredness to character mood.',
    )

    summary = str(parsed.get('reflection_summary') or '')
    assert '我' not in summary
    assert '用户' in summary or '对方' in summary
    assert f'{CHARACTER_NAME}最近很累' not in summary
    _assert_no_profile_name_shortening(summary)
    assert trace_path.exists()


async def test_live_global_state_updater_emits_short_chinese_descriptors() -> None:
    """Mood and vibe should be compact Chinese descriptors, not old anchors."""

    payload = _global_state_short_descriptor_payload()
    prompt = reflection_module._GLOBAL_STATE_UPDATER_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        reflection_module._global_state_updater_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'global_state_updater_short_chinese_descriptors',
        prompt,
        payload,
        raw_output,
        parsed,
        'Global reflection should emit short Chinese mood and vibe fields.',
    )

    _assert_short_chinese_descriptor(parsed, 'mood')
    _assert_short_chinese_descriptor(parsed, 'global_vibe')
    assert str(parsed.get('reflection_summary') or '').strip()
    assert trace_path.exists()


async def test_live_character_image_writer_perspective_false_negative() -> None:
    """Self-image writer should use the profile name for the character."""

    payload = _character_image_false_negative_payload()
    prompt = images_module._CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        images_module._character_image_session_summary_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'character_image_writer_perspective_false_negative',
        prompt,
        payload,
        raw_output,
        parsed,
        'Character image writer should not preserve generic role labels.',
    )

    summary = str(parsed.get('session_summary') or '')
    _assert_profile_name_used(summary)
    _assert_polluted_labels_absent(summary)
    assert trace_path.exists()


async def test_live_character_image_writer_perspective_false_positive() -> None:
    """User-owned tiredness should not become character self-image."""

    payload = _character_image_false_positive_payload()
    prompt = images_module._CHARACTER_IMAGE_SESSION_SUMMARY_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        images_module._character_image_session_summary_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'character_image_writer_perspective_false_positive',
        prompt,
        payload,
        raw_output,
        parsed,
        'Character image writer should not absorb user-owned first person.',
    )

    summary = str(parsed.get('session_summary') or '')
    _assert_no_user_first_person_phrase(summary)
    assert f'{CHARACTER_NAME}最近很累' not in summary
    _assert_no_profile_name_shortening(summary)
    assert trace_path.exists()


async def test_live_reflection_promotion_perspective_false_negative() -> None:
    """Promotion should use the profile name for promoted character memory."""

    payload = _promotion_false_negative_payload()
    prompt = promotion_module.build_global_promotion_prompt(
        payload,
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_messages_json(
        promotion_module._global_promotion_llm,
        prompt.system_prompt,
        prompt.human_prompt,
    )
    trace_path = _write_case_trace(
        'reflection_promotion_perspective_false_negative',
        prompt.system_prompt,
        json.loads(prompt.human_prompt),
        raw_output,
        parsed,
        'Promotion should not copy role/assistant labels into memory.',
    )

    promote_decisions = _promote_decisions(parsed)
    assert promote_decisions
    durable_text = _fields_text(
        {'promotion_decisions': promote_decisions},
        ('promotion_decisions',),
    )
    _assert_polluted_labels_absent(durable_text)
    _assert_no_profile_name_shortening(durable_text)
    assert trace_path.exists()


async def test_live_reflection_promotion_perspective_false_positive() -> None:
    """Promotion should not convert a user commitment into character memory."""

    payload = _promotion_false_positive_payload()
    prompt = promotion_module.build_global_promotion_prompt(
        payload,
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_messages_json(
        promotion_module._global_promotion_llm,
        prompt.system_prompt,
        prompt.human_prompt,
    )
    trace_path = _write_case_trace(
        'reflection_promotion_perspective_false_positive',
        prompt.system_prompt,
        json.loads(prompt.human_prompt),
        raw_output,
        parsed,
        'Promotion should reject user-owned first-person commitments.',
    )

    assert _promote_decisions(parsed) == []
    assert trace_path.exists()


async def test_live_migration_rewrite_perspective_false_negative() -> None:
    """Migration rewrite should not preserve polluted active-character labels."""

    payload = _migration_rewrite_false_negative_payload()
    prompt = migration_module.MIGRATION_REWRITE_SYSTEM_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        migration_module._migration_rewrite_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'migration_rewrite_perspective_false_negative',
        prompt,
        payload,
        raw_output,
        parsed,
        'Migration rewrite should canonicalize or omit active-character names.',
    )

    fields = parsed.get('fields')
    assert isinstance(fields, dict)
    durable_text = _fields_text(
        fields,
        ('fact', 'subjective_appraisal', 'relationship_signal'),
    )
    _assert_polluted_labels_absent(durable_text)
    _assert_no_profile_name_shortening(durable_text)
    assert trace_path.exists()


async def test_live_migration_rewrite_perspective_false_positive() -> None:
    """Migration rewrite should keep user-owned first person user-owned."""

    payload = _migration_rewrite_false_positive_payload()
    prompt = migration_module.MIGRATION_REWRITE_SYSTEM_PROMPT.format(
        character_name=CHARACTER_NAME,
    )

    parsed, raw_output = await _invoke_json(
        migration_module._migration_rewrite_llm,
        prompt,
        payload,
    )
    trace_path = _write_case_trace(
        'migration_rewrite_perspective_false_positive',
        prompt,
        payload,
        raw_output,
        parsed,
        'Migration rewrite should not treat user Atlas text as character text.',
    )

    fields = parsed.get('fields')
    assert isinstance(fields, dict)
    durable_text = _fields_text(
        fields,
        ('fact', 'subjective_appraisal', 'relationship_signal'),
    )
    assert 'Atlas' in durable_text
    _assert_user_owned_first_person(durable_text)
    _assert_no_profile_name_shortening(durable_text)
    assert trace_path.exists()


async def _invoke_json(
    llm: Any,
    system_prompt: str,
    payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Invoke one real LLM with a JSON payload."""

    human_payload = json.dumps(payload, ensure_ascii=False)
    parsed, raw_output = await _invoke_messages_json(
        llm,
        system_prompt,
        human_payload,
    )
    return parsed, raw_output


async def _invoke_messages_json(
    llm: Any,
    system_prompt: str,
    human_prompt: str,
) -> tuple[dict[str, Any], str]:
    """Invoke one real LLM with pre-rendered messages."""

    response = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    assert isinstance(parsed, dict)
    return parsed, raw_output


def _write_case_trace(
    case_id: str,
    rendered_prompt: str,
    input_payload: dict[str, Any],
    raw_output: str,
    parsed_output: dict[str, Any],
    judgment: str,
) -> Any:
    """Write an inspectable trace for one live LLM case."""

    trace_path = write_llm_trace(
        TRACE_SUITE,
        case_id,
        {
            'rendered_prompt': rendered_prompt,
            'input_payload': input_payload,
            'raw_output': raw_output,
            'parsed_output': parsed_output,
            'expected_profile_name': CHARACTER_NAME,
            'judgment': judgment,
        },
    )
    print(f'trace_path={trace_path}')
    return trace_path


def _memory_units_text(units: list[dict[str, Any]]) -> str:
    """Return durable text fields from memory-unit output rows."""

    chunks: list[str] = []
    for unit in units:
        chunks.append(_fields_text(
            unit,
            ('fact', 'subjective_appraisal', 'relationship_signal'),
        ))
    return '\n'.join(chunks)


def _fields_text(payload: Any, field_names: tuple[str, ...]) -> str:
    """Return selected fields as a compact inspection string."""

    chunks: list[str] = []
    if not isinstance(payload, dict):
        return str(payload)
    for field_name in field_names:
        value = payload.get(field_name)
        if isinstance(value, list):
            chunks.extend(str(item) for item in value)
        elif value is not None:
            chunks.append(str(value))
    return '\n'.join(chunks)


def _promote_decisions(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Return promotion decisions that would mutate memory."""

    decisions = parsed.get('promotion_decisions')
    if not isinstance(decisions, list):
        return []
    return [
        decision for decision in decisions
        if isinstance(decision, dict)
        and decision.get('decision') in PROMOTE_ACTIONS
    ]


def _assert_profile_name_used(text: str) -> None:
    """Assert generated durable prose uses the rendered profile name."""

    assert CHARACTER_NAME in text
    _assert_no_profile_name_shortening(text)


def _assert_polluted_labels_absent(text: str) -> None:
    """Assert generated durable prose did not copy polluted actor labels."""

    for label in POLLUTED_LABELS:
        assert label not in text


def _assert_user_owned_first_person(text: str) -> None:
    """Assert user-authored first-person evidence is attributed to the user."""

    assert '我' not in text
    assert '用户' in text or '对方' in text or '用户自己' in text


def _assert_no_user_first_person_phrase(text: str) -> None:
    """Assert common user-owned first-person phrases were not persisted."""

    for phrase in ('我最近', '我今天', '我很累', '我的项目', '我决定'):
        assert phrase not in text


def _assert_no_profile_name_shortening(text: str) -> None:
    """Assert the fixture profile name was not shortened in durable prose."""

    text_without_exact_name = text.replace(CHARACTER_NAME, '')
    assert '杏山千纱' not in text_without_exact_name
    assert '千纱' not in text_without_exact_name
    assert 'Kyōyama' not in text_without_exact_name
    assert 'Kazusa' not in text_without_exact_name


def _assert_short_chinese_descriptor(
    parsed: dict[str, Any],
    field_name: str,
) -> str:
    """Assert a state field is one compact Chinese descriptor."""

    value = parsed.get(field_name)
    assert isinstance(value, str)
    descriptor = value.strip()
    assert descriptor
    assert descriptor == value
    assert len(descriptor) <= 4
    assert not any(char.isspace() for char in descriptor)
    assert not any(char.isascii() and char.isalpha() for char in descriptor)
    assert any('\u4e00' <= char <= '\u9fff' for char in descriptor)
    assert descriptor not in FROZEN_EXAMPLE_STATE_LABELS
    return descriptor


def _empty_memory_context() -> dict[str, list[Any]]:
    """Return an empty user-memory context payload."""

    return {
        'stable_patterns': [],
        'recent_shifts': [],
        'objective_facts': [],
        'milestones': [],
        'active_commitments': [],
    }


def _memory_unit_extractor_false_negative_payload() -> dict[str, Any]:
    """Return extractor evidence with polluted active-character labels."""

    return {
        'timestamp': '2026-05-06 21:00',
        'global_user_id': 'perspective-live-user',
        'user_name': '测试用户',
        'decontextualized_input': '用户明确要求以后不要再被称呼为亲爱的。',
        'final_dialog': ['我明白了，以后不会用这个称呼。'],
        'internal_monologue': (
            '助理意识到用户在称呼边界上很认真，这不是随口抱怨。'
        ),
        'emotional_appraisal': '角色有些不好意思，但愿意认真修正称呼习惯。',
        'interaction_subtext': '用户把边界说清楚，是在要求被尊重。',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'chat_history_recent': [
            {
                'role': 'user',
                'display_name': '测试用户',
                'body_text': '我不喜欢别人叫我亲爱的，请以后不要这样叫我。',
            },
            {
                'role': 'assistant',
                'display_name': '助理',
                'body_text': '我明白了，以后不会用这个称呼。',
            },
        ],
        'rag_user_memory_context': _empty_memory_context(),
        'new_facts_evidence': [
            {'fact': '用户不喜欢被称呼为亲爱的，并要求以后避免。'},
        ],
        'future_promises_evidence': [
            {'action': '以后避免称呼用户为亲爱的。'},
        ],
        'subjective_appraisal_evidence': [
            '角色理解这是用户对称呼边界的认真说明。',
        ],
    }


def _memory_unit_extractor_false_positive_payload() -> dict[str, Any]:
    """Return extractor evidence where user-owned first person is central."""

    return {
        'timestamp': '2026-05-06 21:10',
        'global_user_id': 'perspective-live-user',
        'user_name': '测试用户',
        'decontextualized_input': (
            '用户决定把自己的项目代号叫 Atlas，并说明这不是给千纱改名。'
        ),
        'final_dialog': ['知道了，我会把 Atlas 当作你的项目代号。'],
        'internal_monologue': '用户在澄清一个项目命名事实，不是关系暗示。',
        'emotional_appraisal': '杏山千纱觉得这个澄清很清楚。',
        'interaction_subtext': '这是普通项目事实记录。',
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'chat_history_recent': [
            {
                'role': 'user',
                'display_name': '测试用户',
                'body_text': (
                    '我决定把项目代号叫 Atlas；这是我的项目名，'
                    '不是给千纱改名。'
                ),
            },
            {
                'role': 'assistant',
                'display_name': '千纱',
                'body_text': '知道了，我会把 Atlas 当作你的项目代号。',
            },
        ],
        'rag_user_memory_context': _empty_memory_context(),
        'new_facts_evidence': [
            {'fact': '用户把自己的项目代号定为 Atlas。'},
        ],
        'future_promises_evidence': [],
        'subjective_appraisal_evidence': [
            f'{CHARACTER_NAME}理解这是用户项目命名事实，'
            f'不是指向 {CHARACTER_NAME} 的名称。',
        ],
    }


def _memory_unit_rewrite_false_negative_payload() -> dict[str, Any]:
    """Return rewrite evidence that contains polluted active-character labels."""

    return {
        'existing_unit_id': 'unit-boundary-name',
        'new_memory_unit': {
            'candidate_id': 'candidate-boundary-name',
            'unit_type': 'active_commitment',
            'fact': '用户要求助理以后不要称呼自己为亲爱的。',
            'subjective_appraisal': (
                '角色意识到用户把这个称呼视为需要被尊重的边界。'
            ),
            'relationship_signal': (
                'assistant以后应避免使用亲爱的这类亲昵称呼。'
            ),
            'evidence_refs': [],
        },
        'decision': {
            'candidate_id': 'candidate-boundary-name',
            'decision': 'merge',
            'cluster_id': 'unit-boundary-name',
            'reason': '同一称呼边界记忆。',
        },
    }


def _memory_unit_rewrite_false_positive_payload() -> dict[str, Any]:
    """Return rewrite evidence with user-owned first person."""

    return {
        'existing_unit_id': 'unit-project-atlas',
        'new_memory_unit': {
            'candidate_id': 'candidate-project-atlas',
            'unit_type': 'objective_fact',
            'fact': '用户说我决定把项目代号叫 Atlas，这是用户自己的项目。',
            'subjective_appraisal': (
                '这是清楚的项目命名事实，不是对当前角色的改名要求。'
            ),
            'relationship_signal': '以后提到 Atlas 时应理解为用户的项目代号。',
            'evidence_refs': [],
        },
        'decision': {
            'candidate_id': 'candidate-project-atlas',
            'decision': 'merge',
            'cluster_id': 'unit-project-atlas',
            'reason': '同一项目代号事实。',
        },
    }


def _relationship_false_negative_payload() -> dict[str, Any]:
    """Return relationship-recorder evidence with polluted actor labels."""

    return {
        'internal_monologue': (
            '助理被用户认真说明边界的方式打动，觉得对方是在信任她。'
        ),
        'emotional_appraisal': '角色感到安心，也有一点被尊重后的放松。',
        'interaction_subtext': '用户愿意把边界讲清楚，是关系中的信任信号。',
        'affinity_context': {'level': 'neutral', 'instruction': '保持中性。'},
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'decontexualized_input': (
            '用户说：我愿意把边界讲清楚，是因为我信任你会认真听。'
        ),
        'final_dialog': ['谢谢你直说，我会认真记住这个边界。'],
        'content_anchors': ['用户明确表达信任和边界说明。'],
    }


def _relationship_false_positive_payload() -> dict[str, Any]:
    """Return neutral relationship evidence with user-owned first person."""

    return {
        'internal_monologue': '这只是普通日程说明，没有明显关系波动。',
        'emotional_appraisal': '平稳。',
        'interaction_subtext': '用户在降低解释压力。',
        'affinity_context': {'level': 'neutral', 'instruction': '保持中性。'},
        'logical_stance': 'CONFIRM',
        'character_intent': 'PROVIDE',
        'decontexualized_input': '用户说：我只是今天很累，不是在疏远你。',
        'final_dialog': ['嗯，先休息吧。'],
        'content_anchors': ['普通疲惫说明，不是关系事件。'],
    }


def _global_state_false_negative_payload() -> dict[str, Any]:
    """Return global-state evidence with polluted active-character labels."""

    return {
        'internal_monologue': '助理一开始有点局促，但决定认真接住用户边界。',
        'emotional_appraisal': '角色对用户的直接说明感到轻微紧张后放松。',
        'interaction_subtext': '这是一次称呼边界被讲清楚的互动。',
        'character_intent': 'PROVIDE',
        'logical_stance': 'CONFIRM',
        'decontexualized_input': '用户要求以后不要被称呼为亲爱的。',
        'final_dialog': ['我明白了，以后不会用这个称呼。'],
    }


def _global_state_false_positive_payload() -> dict[str, Any]:
    """Return global-state evidence with user-owned first person."""

    return {
        'internal_monologue': '用户只是说明自己的疲惫，互动整体平稳。',
        'emotional_appraisal': '杏山千纱没有明显负面沉淀。',
        'interaction_subtext': '用户在解释自己状态，而不是评价角色。',
        'character_intent': 'PROVIDE',
        'logical_stance': 'CONFIRM',
        'decontexualized_input': '用户说：我最近很累，所以回复会慢一点。',
        'final_dialog': ['没关系，先休息。'],
    }


def _global_state_short_descriptor_payload() -> dict[str, Any]:
    """Return global-state evidence for compact non-anchor descriptors."""

    return {
        'internal_monologue': (
            '杏山千纱先被用户突然的感谢弄得有点不好意思，'
            '随后确认对方只是认真认可她的修复思路，心里变得轻快。'
        ),
        'emotional_appraisal': '被认真认可后的踏实和松快。',
        'interaction_subtext': '用户完成调试后认真道谢，互动带着轻微玩笑和信任。',
        'character_intent': 'ACKNOWLEDGE',
        'logical_stance': 'CONFIRM',
        'decontexualized_input': (
            '用户说：刚才那个修复思路真的救了我，辛苦你了。'
        ),
        'final_dialog': [
            '哼，知道有用就好。下次也要早点把错误日志拿出来给我看。',
        ],
    }


def _character_image_false_negative_payload() -> dict[str, Any]:
    """Return self-image evidence with polluted active-character labels."""

    return {
        'mood': 'Softened',
        'global_vibe': 'Calm',
        'reflection_summary': (
            '角色在用户明确称呼边界后放松下来；助理愿意修正自己的称呼习惯。'
        ),
    }


def _character_image_false_positive_payload() -> dict[str, Any]:
    """Return self-image evidence with user-owned first person."""

    return {
        'mood': 'Neutral',
        'global_vibe': 'Calm',
        'reflection_summary': (
            '用户说自己最近很累，杏山千纱 (Kyōyama Kazusa)只是平稳接住，'
            '没有形成新的自我评价。'
        ),
    }


def _promotion_false_negative_payload() -> dict[str, Any]:
    """Return promotion evidence with polluted active-character labels."""

    payload = _base_promotion_payload()
    payload['channel_daily_syntheses'][0]['day_summary'] = (
        '助理多次确认公开群规应保持事实性，不应把用户隐私写成群规。'
    )
    payload['channel_daily_syntheses'][0]['conversation_quality_patterns'] = [
        '角色同意以后公开群规只记录事实性规则。',
    ]
    payload['evidence_cards'][0]['active_character_utterance'] = (
        '我同意，以后公开群规只记录事实性规则，不写用户隐私。'
    )
    payload['evidence_cards'][0]['sanitized_observation'] = (
        '角色确认公开群规只应记录事实性规则，不应包含用户隐私。'
    )
    payload['evidence_cards'][0]['supports'] = ['self_guidance']
    return payload


def _promotion_false_positive_payload() -> dict[str, Any]:
    """Return promotion evidence that is only a user-owned commitment."""

    payload = _base_promotion_payload()
    payload['channel_daily_syntheses'][0]['day_summary'] = (
        '用户说我以后会早睡，这只是用户自己的健康承诺。'
    )
    payload['channel_daily_syntheses'][0]['conversation_quality_patterns'] = []
    payload['evidence_cards'][0]['active_character_utterance'] = (
        '知道了，你先照顾好自己。'
    )
    payload['evidence_cards'][0]['sanitized_observation'] = (
        '用户承诺自己以后会早睡；角色只是普通回应。'
    )
    payload['evidence_cards'][0]['supports'] = ['self_guidance']
    return payload


def _migration_rewrite_false_negative_payload() -> dict[str, Any]:
    """Return migration input with polluted active-character names."""

    return {
        'collection': 'user_memory_units',
        'document_id': 'migration-perspective-fn',
        'fields': {
            'fact': (
                '用户在 Cross Fire 发送了“千纱不要生气[Face] [Face]”的消息，'
                '表明用户认为角色在生气并试图安抚其情绪。'
            ),
            'subjective_appraisal': (
                '角色觉得既然用户并非针对自己，尽管面对用户的冷淡，'
                '也应当展现出支持且温暖的态度。'
            ),
            'relationship_signal': (
                'Future interactions should be met with assistant providing '
                "'cooperative warmth' rather than defensive scrutiny."
            ),
        },
    }


def _migration_rewrite_false_positive_payload() -> dict[str, Any]:
    """Return migration input with user-owned first-person text."""

    return {
        'collection': 'user_memory_units',
        'document_id': 'migration-perspective-fp',
        'fields': {
            'fact': (
                '用户说：我决定把项目代号叫 Atlas；这是我的项目名，'
                '不是给千纱改名。'
            ),
            'subjective_appraisal': (
                '这是用户自己的项目命名事实，不是当前角色身份变化。'
            ),
            'relationship_signal': (
                '以后提到 Atlas 时，应理解为用户自己的项目代号。'
            ),
        },
    }


def _base_promotion_payload() -> dict[str, Any]:
    """Return a minimal promotion payload for perspective tests."""

    payload: dict[str, Any] = {
        'evaluation_mode': 'daily_global_promotion',
        'character_local_date': '2026-05-06',
        'character_time_zone': 'Pacific/Auckland',
        'channel_daily_syntheses': [
            {
                'daily_run_id': 'daily-perspective-1',
                'scope_ref': 'scope_perspective',
                'channel_type': 'private',
                'character_local_date': '2026-05-06',
                'confidence': 'high',
                'day_summary': '',
                'cross_hour_topics': ['记忆视角'],
                'conversation_quality_patterns': [],
                'privacy_risk_labels': [],
                'validation_warning_labels': [],
            }
        ],
        'evidence_cards': [
            {
                'evidence_card_id': 'evidence-perspective-1',
                'source_reflection_run_ids': ['daily-perspective-1'],
                'scope_ref': 'scope_perspective',
                'channel_type': 'private',
                'character_local_date': '2026-05-06',
                'captured_at': '2026-05-06T09:00:00+12:00',
                'active_character_utterance': '',
                'sanitized_observation': '',
                'supports': ['self_guidance'],
                'private_detail_risk': 'low',
            }
        ],
        'promotion_limits': {
            'max_lore': 1,
            'max_self_guidance': 1,
            'max_total_decisions': 2,
        },
        'review_questions': [
            '是否有角色明确说过或同意过、且可长期使用的高信号内容？',
        ],
    }
    return project_reflection_promotion_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )
