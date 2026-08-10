"""Focused regressions for typed third-party targets and wording direction."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage
import pytest

from kazusa_ai_chatbot.cognition_core_v2 import contracts as contracts_module
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    validate_surface_addressee_plan,
)
from kazusa_ai_chatbot.cognition_core_v2.facade import _role_summary
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontextualizer as decontext_module,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2 import (
    _build_scene_participant_bindings,
)
from tests.cognition_core_v2_test_helpers import canonical_episode


_FIXTURE_PATH = Path(
    'tests/fixtures/dialog_third_party_target_fidelity_cases.json'
)
_CASES = json.loads(_FIXTURE_PATH.read_text(encoding='utf-8'))['cases']


def _scene_state(channel_type: str) -> dict[str, object]:
    """Build the identity fields used by the participant roster owner."""

    return {
        'channel_type': channel_type,
        'global_user_id': 'current-global-id',
        'platform_user_id': 'current-platform-id',
        'user_name': 'YCHDDZZ',
        'platform_bot_id': 'bot-platform-id',
        'character_profile': {
            'name': 'Asuna',
            'global_user_id': 'character-global-id',
        },
    }


def _scope_users(display_names: list[str]) -> list[dict[str, str]]:
    """Build a resolved scene roster with non-prompt-safe ids."""

    identities = {
        'YCHDDZZ': ('current-platform-id', 'current-global-id'),
        '蚝爹油': ('third-party-platform-id', 'third-party-global-id'),
        'Asuna': ('bot-platform-id', 'character-global-id'),
    }
    return [
        {
            'display_name': display_name,
            'platform_user_id': identities[display_name][0],
            'global_user_id': identities[display_name][1],
        }
        for display_name in display_names
    ]


def _scene_context(
    participant_bindings: list[dict[str, str]],
) -> dict[str, object]:
    """Build a validated semantic scene with optional participant bindings."""

    context: dict[str, object] = {
        'channel_scope': 'group',
        'character_role': 'Asuna',
        'current_user_role': 'YCHDDZZ',
        'character_sleep_phase': 'awake',
        'semantic_scene': 'A visible group scene contains one named participant.',
        'public_group_scene': 'YCHDDZZ and 蚝爹油 are visible in the group.',
        'conversation_continuity': 'The current turn is grounded in the visible scene.',
        'semantic_temporal_context': '当前时刻',
    }
    if participant_bindings:
        context['participant_bindings'] = participant_bindings
    return context


def test_fixture_roster_allocates_only_resolved_non_current_participants() -> None:
    """The three fixture topologies preserve current-user transport scope."""

    for case in _CASES:
        state = _scene_state(case['channel_type'])
        bindings = _build_scene_participant_bindings(
            state,
            _scope_users(case['scene_display_names']),
        )
        expected = case['expected_participant']
        if expected is None:
            assert bindings == []
        else:
            assert [dict(row) for row in bindings] == [expected]
            serialized = json.dumps(bindings, ensure_ascii=False)
            assert 'current-global-id' not in serialized
            assert 'third-party-global-id' not in serialized
            assert 'bot-platform-id' not in serialized
        assert case['transport_recipient'] == 'current_user'


def test_scene_context_and_goal_projection_keep_p1_episode_local() -> None:
    """The p1 handle is available to cognition without leaking identity ids."""

    bindings = [{
        'handle': 'p1',
        'display_name': '蚝爹油',
        'entity_kind': 'third_party',
    }]
    scene_context = _scene_context(bindings)
    contracts_module._validate_scene_context(scene_context)

    state = {
        'state_scope': 'user:current-global-id',
        'updated_at': '2026-08-08T00:00:00Z',
        'goals': [],
        'threats': [],
        'active_events': [],
        'knowledge_gaps': [],
        'affect_activations': [],
        'drives': {},
        'standards': [],
        'meaning_state': None,
        'owner_user_id': 'current-global-id',
    }
    constraints = {
        'drives': {},
        'standards': [],
        'meaning_state': {
            'purpose_coherence': 50,
            'agency': 50,
            'identity_continuity': 50,
            'salience': 50,
        },
        'personality_judgment': {
            'logic': 'clear',
            'defense': 'reserved',
            'quirks': 'dry',
            'taboos': 'none',
        },
    }
    projection = project_state_for_prompt(
        state,
        character_constraints=constraints,
        character_identity_context={
            'goal_cognition': {'voice': 'Asuna'},
        },
        scene_context=scene_context,
    )

    assert projection.handle_to_ref['p1'] == {
        'scope': 'episode',
        'kind': 'third_party',
        'entity_id': 'scene:p1',
    }
    assert _role_summary(
        'p1',
        projection.handle_to_ref['p1'],
        scene_context=scene_context,
    ) == 'p1=蚝爹油（群聊其他参与者）'
    prompt_payload = json.dumps(projection.payload, ensure_ascii=False)
    assert 'current-global-id' not in prompt_payload
    assert 'third-party-global-id' not in prompt_payload
    assert 'scene:p1' not in prompt_payload


def _decontextualizer_candidate(
    referents: list[dict[str, str]],
) -> dict[str, object]:
    """Build a complete candidate for the referent owner validator."""

    return {
        'output': '他需要特训。',
        'reasoning': '桥接了可见群聊中的参与者。',
        'is_modified': False,
        'referents': referents,
        'role_explicit_content': '当前角色针对蚝爹油作出回应。',
        'response_operation': {
            'operation': '当前角色针对群聊中的第三方作出回应',
            'response_owner_role': '当前角色',
            'selection_owner_role': '无',
            'selection_required': False,
            'embedded_actor_role': '当前角色',
            'embedded_target_role': '其他参与者',
        },
    }


def test_decontextualized_referent_preserves_p1_and_accepts_pronoun() -> None:
    """A resolved pronoun keeps its typed participant handle."""

    result = decontext_module._validate_decontextualizer_result(
        _decontextualizer_candidate([{
            'phrase': '他',
            'referent_role': 'subject',
            'status': 'resolved',
            'participant_handle': 'p1',
        }]),
        participant_bindings=[{
            'handle': 'p1',
            'display_name': '蚝爹油',
            'entity_kind': 'third_party',
        }],
    )

    assert result['referents'] == [{
        'phrase': '他',
        'referent_role': 'subject',
        'status': 'resolved',
        'participant_handle': 'p1',
    }]


@pytest.mark.parametrize(
    'referent',
    [
        {
            'phrase': '蚝爹油',
            'referent_role': 'object',
            'status': 'resolved',
        },
        {
            'phrase': '小明',
            'referent_role': 'object',
            'status': 'resolved',
            'participant_handle': 'p1',
        },
        {
            'phrase': '蚝爹油',
            'referent_role': 'object',
            'status': 'resolved',
            'participant_handle': 'p9',
        },
        {
            'phrase': '蚝爹油',
            'referent_role': 'object',
            'status': 'unresolved',
            'participant_handle': 'p1',
        },
    ],
)
def test_decontextualizer_rejects_unbound_or_mismatched_participant(
    referent: dict[str, str],
) -> None:
    """Invalid participant ownership returns to the bounded contract path."""

    with pytest.raises(ValueError):
        decontext_module._validate_decontextualizer_result(
            _decontextualizer_candidate([referent]),
            participant_bindings=[{
                'handle': 'p1',
                'display_name': '蚝爹油',
                'entity_kind': 'third_party',
            }, {
                'handle': 'p2',
                'display_name': '小明',
                'entity_kind': 'third_party',
            }],
        )


def test_l3_target_projection_distinguishes_p1_from_current_user() -> None:
    """L3 emits a named third-party row and permits second person for users."""

    state = {
        'user_name': 'YCHDDZZ',
        'character_profile': {'name': 'Asuna'},
        'scene_participant_bindings': [{
            'handle': 'p1',
            'display_name': '蚝爹油',
            'entity_kind': 'third_party',
        }],
    }
    third_party_role = {
        'role': 'target',
        'entity_kind': 'third_party',
        'entity_id': 'scene:p1',
    }
    current_user_role = {
        'role': 'target',
        'entity_kind': 'user',
        'entity_id': 'current-global-id',
    }
    third_party_plan = l3_module._surface_addressee_plan(
        [third_party_role],
        state=state,
    )
    current_user_plan = l3_module._surface_addressee_plan(
        [current_user_role],
        state=state,
    )

    assert third_party_plan == [{
        'handle': 'p1',
        'display_name': '蚝爹油',
        'semantic_role': 'embedded_target',
        'wording_policy': 'named_or_third_person_required',
    }]
    assert current_user_plan == [{
        'handle': 'current_user',
        'display_name': 'YCHDDZZ',
        'semantic_role': 'embedded_target',
        'wording_policy': 'second_person_allowed',
    }]
    validate_surface_addressee_plan(third_party_plan)
    validate_surface_addressee_plan(current_user_plan)
    with pytest.raises(CognitionContractError):
        validate_surface_addressee_plan([{
            **third_party_plan[0],
            'wording_policy': 'second_person_allowed',
        }])


def _surface_output(
    addressee_plan: list[dict[str, str]],
) -> dict[str, object]:
    """Build the dialog verifier's complete surface contract."""

    return {
        'schema_version': 'text_surface_output.v2',
        'content_plan': 'Address the named participant in the control clause.',
        'content_requirements': ['Keep the participant and recipient distinct.'],
        'visible_boundaries': [],
        'addressee_plan': addressee_plan,
        'delivery_profile': {
            'lexical_register': 'warm',
            'sentence_shape': 'concise',
            'rhythm': 'steady',
            'hesitation': 'minimal',
            'punctuation': 'restrained',
        },
        'selected_surface_intent': 'tease the named participant',
        'permitted_action_results': [],
    }


def _surface_input(
    addressee_plan: list[dict[str, str]],
) -> dict[str, object]:
    """Build the retained input required by bounded dialog repair."""

    content_plan = 'Address the named participant in the control clause.'
    return {
        'schema_version': 'text_surface_input.v2',
        'episode': canonical_episode(
            episode_id='typed-target-repair',
            content='A named participant is the semantic target.',
            current_global_user_id='current-global-id',
        ),
        'intention': {
            'route': 'speech',
            'intention': 'tease the named participant',
            'target_roles': [],
            'reason': 'Keep the participant and recipient distinct.',
        },
        'goal_resolution': 'answerable_now',
        'supporting_bids': [],
        'expression_policy': {
            'visibility': 'visible',
            'emotional_tone': 'playful',
            'intensity': 'moderate',
            'directness': 'direct',
        },
        'semantic_affect': [],
        'permitted_action_results': [],
        'interaction_style_context': 'Keep the target explicit.',
        'character_expression_context': {
            'tempo': 'brief',
            'linguistic_texture': 'playful',
        },
        'visual_character_context': 'not used for text',
        'addressee_plan': addressee_plan,
        'primary_bid': {
            'motive': 'Keep the target explicit.',
            'intention': 'tease the named participant',
            'desired_outcome': content_plan,
            'permitted_detail': content_plan,
            'target_summaries': [content_plan],
            'expected_consequences': [
                'The recipient can distinguish the named target.',
            ],
        },
    }


def test_dialog_role_frame_lists_only_pn_non_current_targets() -> None:
    """The character actor does not pollute the third-party target roster."""

    surface_output = _surface_output([
        {
            'handle': 'p1',
            'display_name': '蚝爹油',
            'semantic_role': 'embedded_target',
            'wording_policy': 'named_or_third_person_required',
        },
        {
            'handle': 'self',
            'display_name': 'Asuna',
            'semantic_role': 'embedded_actor',
            'wording_policy': 'named_or_third_person_required',
        },
    ])

    frame = dialog_module._candidate_role_frame(surface_output)

    assert frame['typed_non_current_targets'] == [
        surface_output['addressee_plan'][0],
    ]


@pytest.mark.asyncio
async def test_role_verifier_receives_typed_p1_and_can_reject_second_person(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The existing role owner receives p1 evidence for a wrong second person."""

    surface_output = _surface_output([{
        'handle': 'p1',
        'display_name': '蚝爹油',
        'semantic_role': 'embedded_target',
        'wording_policy': 'named_or_third_person_required',
    }])
    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            'score': 0.1,
            'violations': [{
                'kind': 'typed_operation_role_reversal',
                'evidence': '你',
                'explanation': '该句把第三方目标错误地写成了当前用户。',
            }],
        }, ensure_ascii=False),
    ))
    monkeypatch.setattr(dialog_module, '_dialog_role_direction_llm', role_llm)
    monkeypatch.setattr(
        dialog_module.llm_tracing,
        'record_llm_trace_step',
        AsyncMock(),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        'record_llm_stage_event',
        AsyncMock(),
    )

    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=surface_output,
        generated_dialog=['哈哈，快让我看看你现在是不是缩成一团？'],
        current_visible_percepts=[],
        llm_trace_id='typed-target-test',
    )

    assert verdict['score'] == 0.1
    payload = json.loads(role_llm.ainvoke.await_args.args[0][1].content)
    assert payload['typed_addressee_plan'] == [surface_output['addressee_plan'][0]]
    assert payload['candidate_role_frame']['typed_non_current_targets'] == [{
        'handle': 'p1',
        'display_name': '蚝爹油',
        'semantic_role': 'embedded_target',
        'wording_policy': 'named_or_third_person_required',
    }]
    assert 'current-global-id' not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_dialog_repair_handoff_carries_p1_role_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected p1 second-person candidate reaches bounded regeneration."""

    addressee_plan = [{
        'handle': 'p1',
        'display_name': '蚝爹油',
        'semantic_role': 'embedded_target',
        'wording_policy': 'named_or_third_person_required',
    }]
    surface_output = _surface_output(addressee_plan)
    surface_input = _surface_input(addressee_plan)
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            'final_dialog': ['哈哈，快让我看看你现在是不是缩成一团？'],
        }, ensure_ascii=False)),
        AIMessage(content=json.dumps({
            'final_dialog': ['哈哈，快让我看看蚝爹油现在是不是缩成一团？'],
        }, ensure_ascii=False)),
    ])
    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(side_effect=[
        AIMessage(content=json.dumps({
            'score': 0.1,
            'violations': [{
                'kind': 'typed_operation_role_reversal',
                'evidence': '你',
                'explanation': '该句把第三方目标错误地写成了当前用户。',
            }],
        }, ensure_ascii=False)),
        AIMessage(content=json.dumps({
            'score': 1.0,
            'violations': [],
        }, ensure_ascii=False)),
    ])
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"score": 1.0, "hard_errors": []}',
    ))
    integrity_llm = MagicMock()
    integrity_llm.ainvoke = AsyncMock(return_value=AIMessage(
        content='{"score": 1.0, "issues": []}',
    ))
    repair_calls: list[dict[str, object]] = []

    async def repair_surface(**kwargs: object) -> dict[str, object]:
        repair_calls.append(kwargs)
        return surface_output

    monkeypatch.setattr(dialog_module, '_dialog_generator_llm', generator_llm)
    monkeypatch.setattr(dialog_module, '_dialog_role_direction_llm', role_llm)
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
    monkeypatch.setattr(
        dialog_module,
        'repair_text_surface_for_dialog',
        repair_surface,
    )

    state = {
        'dialog_usage_mode': 'unit_test',
        'text_surface_input_v2': surface_input,
        'text_surface_output_v2': surface_output,
        'cognitive_episode': surface_input['episode'],
        'user_name': 'YCHDDZZ',
        'llm_trace_id': 'typed-target-repair',
    }
    result = await dialog_module.dialog_generator(state)

    assert result['final_dialog'] == [
        '哈哈，快让我看看蚝爹油现在是不是缩成一团？',
    ]
    assert generator_llm.ainvoke.await_count == 2
    assert role_llm.ainvoke.await_count == 2
    assert len(repair_calls) == 1
    assert any(
        'typed_operation_role_reversal' in issue
        for issue in repair_calls[0]['verified_hard_issues']
    )
    repair_payload = json.loads(
        generator_llm.ainvoke.await_args_list[1].args[0][1].content,
    )
    assert repair_payload['repair_context']['verified_hard_issues']
    assert 'typed_operation_role_reversal' in json.dumps(
        repair_payload['repair_context'],
        ensure_ascii=False,
    )


@pytest.mark.asyncio
async def test_dialog_transport_stays_addressed_to_current_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A third-party wording target does not change adapter delivery fields."""

    surface_output = _surface_output([{
        'handle': 'p1',
        'display_name': '蚝爹油',
        'semantic_role': 'embedded_target',
        'wording_policy': 'named_or_third_person_required',
    }])
    monkeypatch.setattr(
        dialog_module,
        'dialog_generator',
        AsyncMock(return_value={
            'final_dialog': ['蚝爹油，特训现在开始。'],
            'text_surface_output_v2': surface_output,
        }),
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        'record_dialog_quality_event',
        AsyncMock(),
    )
    state = {
        'internal_monologue': 'I will keep the target explicit.',
        'text_surface_output_v2': surface_output,
        'cognitive_episode': canonical_episode(
            episode_id='typed-target-transport',
            content='A named participant is the semantic target.',
        ),
        'chat_history_wide': [],
        'chat_history_recent': [],
        'platform_user_id': 'current-platform-id',
        'platform_bot_id': 'bot-platform-id',
        'global_user_id': 'current-global-id',
        'user_name': 'YCHDDZZ',
        'user_profile': {},
        'character_profile': {},
        'dialog_usage_mode': 'unit_test',
        'llm_trace_id': 'typed-target-transport',
    }

    result = await dialog_module.dialog_agent(state)

    assert result['final_dialog'] == ['蚝爹油，特训现在开始。']
    assert result['target_addressed_user_ids'] == ['current-global-id']
    assert result['target_broadcast'] is False
