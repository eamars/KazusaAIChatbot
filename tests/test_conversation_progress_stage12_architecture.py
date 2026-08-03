"""Stage 12 responsibility-boundary and ambiguity regression tests."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2 import goal_cognition
from kazusa_ai_chatbot.conversation_progress import delta_merge, recorder
from tests.conversation_progress_v2_helpers import (
    event,
    packet,
    record_input,
)


def _scene_observation() -> dict[str, object]:
    """Build one exact scene-only semantic observation."""

    return {
        'schema_version': 'conversation_progress_scene_observation.v2',
        'scene_relation': 'same',
        'episode_change': 'none',
        'episode_narrative': 'The current interaction continues.',
        'current_thread': 'current interaction thread',
        'character_stance': 'engaged',
        'user_goal': '',
        'current_blocker': '',
        'emotional_trajectory': 'stable',
        'overused_moves': [],
    }


def _unchanged(event_handle: str) -> dict[str, str]:
    """Build one explicit unchanged prior-event observation."""

    return {
        'event_handle': event_handle,
        'observation': 'unchanged',
    }


def _changed(
    event_handle: str,
    *,
    lifecycle_change: str = 'declined',
) -> dict[str, object]:
    """Build one exact changed prior-event observation."""

    return {
        'event_handle': event_handle,
        'observation': 'changed',
        'semantic_summary': f'event {event_handle} changed',
        'outcome': 'the current turn settled this event',
        'lifecycle_change': lifecycle_change,
        'relevance': 'decision',
        'source_turn_handles': ['current_input'],
    }


def _new_event() -> dict[str, object]:
    """Build one concretely identifiable new-event observation."""

    return {
        'semantic_summary': 'the character will provide a voice summary',
        'is_obligation': True,
        'actor': 'current character',
        'action': 'provide',
        'object': 'voice summary',
        'beneficiary': 'current user',
        'precondition': 'tomorrow',
        'outcome': '',
        'lifecycle_change': 'none',
        'relevance': 'decision',
        'source_turn_handles': ['current_input'],
    }


def _event_batch(
    *,
    existing_events: list[dict[str, object]],
    new_events: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    """Build one exact event-reconciliation batch."""

    return {
        'schema_version': (
            'conversation_progress_event_observation_batch.v2'
        ),
        'existing_events': deepcopy(existing_events),
        'new_events': deepcopy(new_events or []),
    }


def test_event_reconciliation_rejects_silent_prior_event_omission() -> None:
    """A forgotten clause cannot be interpreted as an unchanged event."""

    record = record_input(
        prior_packet=packet(events=[
            event(event_id='event_one', summary='first option'),
            event(event_id='event_two', summary='written notes promise'),
        ])
    )
    candidate = _event_batch(existing_events=[_changed('e2')])

    with pytest.raises(
        delta_merge.ConversationProgressContractError,
        match='exact prior-event handle coverage',
    ):
        delta_merge.validate_event_observation_batch(
            candidate,
            record_input=record,
            supplied_event_handles={'e1', 'e2'},
            supplied_source_handles={'current_input'},
        )


def test_event_reconciliation_accepts_explicit_full_coverage() -> None:
    """Changed and unchanged prior events are both explicit and inspectable."""

    record = record_input(prior_packet=packet(events=[
        event(event_id='event_one', summary='first option'),
        event(event_id='event_two', summary='written notes promise'),
    ]))
    candidate = _event_batch(
        existing_events=[
            _changed('e1'),
            _changed('e2', lifecycle_change='replaced'),
        ],
        new_events=[_new_event()],
    )

    updates = delta_merge.validate_event_observation_batch(
        candidate,
        record_input=record,
        supplied_event_handles={'e1', 'e2'},
        supplied_source_handles={'current_input'},
    )

    assert len(updates) == 3
    assert {update['event_id'] for update in updates[:2]} == {
        'event_one',
        'event_two',
    }
    assert updates[2]['object'] == 'voice summary'


@pytest.mark.parametrize(
    'existing_events',
    [
        [_unchanged('e1'), _unchanged('e1')],
        [_unchanged('e1'), _unchanged('e3')],
        [
            {
                **_unchanged('e1'),
                'semantic_summary': 'invented extra authority',
            },
            _unchanged('e2'),
        ],
    ],
    ids=[
        'duplicate-prior-handle',
        'unknown-prior-handle',
        'extra-unchanged-field',
    ],
)
def test_event_reconciliation_rejects_handle_domain_mutations(
    existing_events: list[dict[str, object]],
) -> None:
    """Reject duplicate, unknown, and shape-expanded prior observations."""

    record = record_input(prior_packet=packet(events=[
        event(event_id='event_one'),
        event(event_id='event_two'),
    ]))

    with pytest.raises(delta_merge.ConversationProgressContractError):
        delta_merge.validate_event_observation_batch(
            _event_batch(existing_events=existing_events),
            record_input=record,
            supplied_event_handles={'e1', 'e2'},
            supplied_source_handles={'current_input'},
        )


def test_new_event_requires_concrete_actor_action_and_object() -> None:
    """A generic ordinal summary cannot become decision-critical identity."""

    candidate = _event_batch(
        existing_events=[],
        new_events=[{
            **_new_event(),
            'object': '',
        }],
    )

    with pytest.raises(
        delta_merge.ConversationProgressContractError,
        match='actor, action, and object',
    ):
        delta_merge.validate_event_observation_batch(
            candidate,
            record_input=record_input(),
            supplied_event_handles=set(),
            supplied_source_handles={'current_input'},
        )


def test_scene_and_event_prompts_have_disjoint_authority() -> None:
    """Each producer sees and returns only its owned semantic question."""

    scene_prompt = recorder.render_scene_recorder_prompt()
    event_prompt = recorder.render_event_recorder_prompt()

    assert 'conversation_progress_scene_observation.v2' in scene_prompt
    assert 'existing_events' not in scene_prompt
    assert 'new_events' not in scene_prompt
    assert 'lifecycle_change' not in scene_prompt

    assert (
        'conversation_progress_event_observation_batch.v2'
        in event_prompt
    )
    assert 'existing_events' in event_prompt
    assert 'new_events' in event_prompt
    assert 'episode_narrative' not in event_prompt
    assert 'episode_change' not in event_prompt
    assert 'overused_moves' not in event_prompt


def test_scene_observation_and_event_batch_compose_without_semantic_repair(
) -> None:
    """Validated specialist outputs map directly into one internal delta."""

    record = record_input()
    scene = delta_merge.validate_scene_observation(
        _scene_observation(),
        record_input=record,
    )
    events = delta_merge.validate_event_observation_batch(
        _event_batch(
            existing_events=[],
            new_events=[_new_event()],
        ),
        record_input=record,
        supplied_event_handles=set(),
        supplied_source_handles={'current_input'},
    )

    combined = delta_merge.compose_recorder_delta(
        scene_observation=scene,
        event_updates=events,
    )

    assert combined['schema_version'] == (
        'conversation_progress_recorder_delta.v2'
    )
    assert len(combined['event_updates']) == 1
    assert combined['event_updates'][0]['object'] == 'voice summary'


def _selection_episode_evidence() -> dict[str, object]:
    """Build one typed selection-required episode row."""

    semantic_text = json.dumps({
        'role_explicit_content': (
            'The current user asks the current character to choose next.'
        ),
        'response_operation': {
            'operation': 'the current character chooses the next option',
            'response_owner_role': 'current character',
            'selection_owner_role': 'current character',
            'selection_required': True,
            'embedded_actor_role': 'current user',
            'embedded_target_role': 'current character',
        },
    })
    return {
        'evidence_handle': 'e1',
        'evidence_ref': {
            'source_kind': 'episode',
            'source_id': 'episode-selection',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': semantic_text,
        },
        'semantic_text': semantic_text,
        'visible_to': ['q:event_agency'],
    }


def _terminal_conversation_evidence() -> dict[str, object]:
    """Build one concrete completed conversation event."""

    semantic_text = (
        'the prior neck massage; state=completed; '
        'retention=decision_critical; actor=current user; action=massage; '
        'object=current character neck'
    )
    return {
        'evidence_handle': 'e2',
        'evidence_ref': {
            'source_kind': 'conversation_evidence',
            'source_id': 'conversation-progress-event:neck',
            'occurred_at': '2026-07-30T00:00:00Z',
            'semantic_summary': 'the prior neck massage',
        },
        'semantic_text': semantic_text,
        'visible_to': ['q:event_agency'],
    }


def _selection_goal_draft() -> dict[str, object]:
    """Build one authoritative selection-goal producer result."""

    return {
        'selection_kind': 'choice',
        'selection': 'choose the current character palm',
        'reason': 'the prior neck choice is already complete',
        'private_monologue': 'I want a genuinely different choice.',
        'target_role_handles': [],
        'evidence_handles': ['e1', 'e2'],
        'expected_consequences': [
            'the current user receives one concrete new choice',
        ],
        'confidence': 'high',
        'relational_willingness': {
            'schema_version': 'relational_willingness.v1',
            'applicability': 'not_relationship_sensitive',
            'stance': 'not_applicable',
            'reason': '当前回合证据不涉及关系许可判断',
            'evidence_handles': ['e1'],
        },
    }


def test_selection_goal_accepts_uncited_unrelated_progress_evidence() -> None:
    """The producer may leave unrelated progress evidence uncited."""

    candidate = _selection_goal_draft()
    candidate['evidence_handles'] = ['e1']

    validated = goal_cognition.validate_selection_goal_draft(
        candidate,
        evidence_handles={'e1', 'e2'},
        role_handles=set(),
        required_evidence_handles={'e1'},
        require_relational_willingness=True,
        maximum_evidence_handles=9,
    )

    assert validated['evidence_handles'] == ['e1']


def test_selection_goal_rejects_retired_relation_field() -> None:
    """Reject the removed relation vocabulary in the exact producer schema."""

    candidate = _selection_goal_draft()
    candidate['conversation_evidence_relations'] = []

    with pytest.raises(ValueError, match='fields are not exact'):
        goal_cognition.validate_selection_goal_draft(
            candidate,
            evidence_handles={'e1', 'e2'},
            role_handles=set(),
            required_evidence_handles={'e1'},
            require_relational_willingness=True,
            maximum_evidence_handles=9,
        )


def test_selection_goal_rejects_missing_mandatory_citation() -> None:
    """Keep the typed required-operation citation explicit."""

    candidate = _selection_goal_draft()
    candidate['evidence_handles'] = ['e2']

    with pytest.raises(ValueError, match='required evidence coverage'):
        goal_cognition.validate_selection_goal_draft(
            candidate,
            evidence_handles={'e1', 'e2'},
            role_handles=set(),
            required_evidence_handles={'e1'},
            require_relational_willingness=True,
            maximum_evidence_handles=9,
        )


def test_selection_goal_accepts_ten_relevant_citations() -> None:
    """Relevant progress can expand the operation-only citation cap."""

    evidence_handles = {f'e{index}' for index in range(1, 11)}
    candidate = _selection_goal_draft()
    candidate['evidence_handles'] = sorted(
        evidence_handles,
        key=lambda handle: int(handle[1:]),
    )

    validated = goal_cognition.validate_selection_goal_draft(
        candidate,
        evidence_handles=evidence_handles,
        role_handles=set(),
        required_evidence_handles={'e1', 'e2'},
        require_relational_willingness=True,
        maximum_evidence_handles=10,
    )

    assert set(validated['evidence_handles']) == evidence_handles


@pytest.mark.asyncio
async def test_selection_goal_uses_one_producer_and_zero_semantic_verifiers(
) -> None:
    """One owned selection replaces goal, verifier, repair, and recheck."""

    class _LLM:
        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.calls.append(messages)
            return SimpleNamespace(content=json.dumps(
                _selection_goal_draft(),
            ))

    llm = _LLM()
    bid = await goal_cognition.run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'g1'},
        {'_role_bindings': {}, 'role_summaries': {}},
        [
            _selection_episode_evidence(),
            _terminal_conversation_evidence(),
        ],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.calls) == 1
    producer_prompt = str(llm.calls[0][0].content)
    producer_payload = json.loads(str(llm.calls[0][1].content))
    assert 'conversation_evidence_relations' not in producer_prompt
    assert [
        row['evidence_handle']
        for row in producer_payload['required_selection_operations']
    ] == ['e1']
    assert producer_payload['conversation_progress_evidence'] == [{
        'evidence_handle': 'e2',
        'semantic_text': _terminal_conversation_evidence()['semantic_text'],
    }]
    assert producer_payload['supporting_evidence'] == []
    assert 'evidence' not in producer_payload
    assert 'candidate_bid' not in producer_prompt
    assert bid['intention'] == 'choose the current character palm'
    assert bid['desired_outcome'] == bid['intention']
    assert bid['concrete_detail'] == bid['intention']
    assert not hasattr(goal_cognition, 'REQUIRED_SELECTION_VERIFIER_PROMPT')
    assert hasattr(
        goal_cognition,
        'REQUIRED_SELECTION_GOAL_REPAIR_PROMPT',
    )


@pytest.mark.asyncio
async def test_selection_json_failure_returns_to_same_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the shared JSON-repair model outside required selection."""

    hidden_repair_calls: list[str] = []

    def _hidden_json_repair(
        broken_string: str,
        **_kwargs: object,
    ) -> dict[str, object]:
        hidden_repair_calls.append(broken_string)
        return _selection_goal_draft()

    monkeypatch.setattr(
        'kazusa_ai_chatbot.utils.parse_json_with_llm',
        _hidden_json_repair,
    )

    class _LLM:
        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.calls.append(messages)
            if len(self.calls) == 1:
                return SimpleNamespace(content='not-json-at-all')
            return SimpleNamespace(content=json.dumps(
                _selection_goal_draft(),
            ))

    llm = _LLM()
    bid = await goal_cognition.run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'g1'},
        {'_role_bindings': {}, 'role_summaries': {}},
        [
            _selection_episode_evidence(),
            _terminal_conversation_evidence(),
        ],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.calls) == 2
    assert hidden_repair_calls == []
    assert bid['intention'] == 'choose the current character palm'


@pytest.mark.asyncio
async def test_selection_capacity_preserves_progress_before_optional_rows(
) -> None:
    """Size the citation cap from required and progress evidence lanes."""

    progress_handles = [f'e{index}' for index in range(2, 10)]
    selection_draft = _selection_goal_draft()
    selection_draft['evidence_handles'] = ['e1', *progress_handles]
    progress_rows: list[dict[str, object]] = []
    for handle in progress_handles:
        row = _terminal_conversation_evidence()
        row['evidence_handle'] = handle
        row['evidence_ref'] = {
            **row['evidence_ref'],
            'source_id': f'conversation-progress-event:{handle}',
        }
        progress_rows.append(row)
    rag_row = _terminal_conversation_evidence()
    rag_row['evidence_handle'] = 'e10'
    rag_row['evidence_ref'] = {
        **rag_row['evidence_ref'],
        'source_id': 'rag-conversation:1',
    }

    class _LLM:
        def __init__(self) -> None:
            self.calls: list[list[object]] = []

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del config
            self.calls.append(messages)
            return SimpleNamespace(content=json.dumps(selection_draft))

    llm = _LLM()
    bid = await goal_cognition.run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS['ordinary_response'],
        {'scope': 'user', 'kind': 'goal', 'entity_id': 'g1'},
        {'_role_bindings': {}, 'role_summaries': {}},
        [
            _selection_episode_evidence(),
            *progress_rows,
            rag_row,
        ],
        SimpleNamespace(
            llm=llm,
            goal_ordinary_response_config=object(),
        ),
    )

    assert len(llm.calls) == 1
    assert bid['evidence_handles'] == ['e1', *progress_handles]
