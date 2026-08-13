"""Independent recorder-owner, failure-isolation, and budget contracts."""

from __future__ import annotations

import asyncio
import json
from copy import deepcopy

import pytest

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS,
)
from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress.delta_merge import (
    ConversationProgressContractError,
    event_handle_map,
    source_handle_map,
    validate_event_observation_batch,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    MAX_EVENT_OUTCOME_CHARS,
)
from kazusa_ai_chatbot.time_boundary import format_storage_utc_for_llm
from tests.conversation_progress_v2_helpers import (
    changed_event_observation,
    event,
    event_observation_batch,
    new_event_observation,
    packet,
    record_input,
    scene_observation,
    unchanged_event_observation,
)


class _Response:
    """Small provider response with inspectable usage."""

    def __init__(self, payload: object) -> None:
        self.content = (
            payload
            if isinstance(payload, str)
            else json.dumps(payload, ensure_ascii=False)
        )
        self.usage_metadata = {
            'input_tokens': 50,
            'output_tokens': 20,
        }


class _OneResponseLLM:
    """Return or raise one configured result and retain every call."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[list[object], object]] = []

    async def ainvoke(self, messages, *, config):
        self.calls.append((list(messages), config))
        if isinstance(self.response, BaseException):
            raise self.response
        return _Response(self.response)


class _BarrierLLM(_OneResponseLLM):
    """Require the peer producer to start before either call can finish."""

    def __init__(
        self,
        response: object,
        *,
        started: asyncio.Event,
        peer_started: asyncio.Event,
    ) -> None:
        super().__init__(response)
        self.started = started
        self.peer_started = peer_started

    async def ainvoke(self, messages, *, config):
        self.calls.append((list(messages), config))
        self.started.set()
        await self.peer_started.wait()
        return _Response(self.response)


def _validate_event_batch(
    candidate: dict,
    *,
    submitted=None,
) -> list[dict]:
    """Validate one batch against its exact private handle domains."""

    actual_input = submitted if submitted is not None else record_input()
    return validate_event_observation_batch(
        candidate,
        record_input=actual_input,
        supplied_event_handles=set(event_handle_map(actual_input)),
        supplied_source_handles=set(source_handle_map(actual_input)),
    )


def _model_scene_observation(**kwargs) -> dict:
    """Build a scene candidate in the reduced model-facing shape."""

    payload = scene_observation(**kwargs)
    payload.pop('schema_version')
    return payload


def _model_event_observation_batch(**kwargs) -> dict:
    """Build an event candidate in the reduced model-facing shape."""

    payload = event_observation_batch(**kwargs)
    payload.pop('schema_version')
    return payload


def test_recorder_prompts_keep_protocol_metadata_code_owned() -> None:
    """Keep packet metadata outside the model-owned recorder output."""

    scene_prompt = recorder.render_scene_recorder_prompt()
    event_prompt = recorder.render_event_recorder_prompt()

    assert 'schema_version' not in scene_prompt
    assert 'conversation_progress_scene_observation.v2' not in scene_prompt
    assert 'event_handle' not in scene_prompt
    assert 'lifecycle_change' not in scene_prompt
    assert 'source_turn_handles' not in scene_prompt

    assert 'schema_version' not in event_prompt
    assert (
        'conversation_progress_event_observation_batch.v2'
        not in event_prompt
    )
    assert '"observation": "unchanged"' in event_prompt
    assert '"object": ""' in event_prompt
    assert 'episode_narrative' not in event_prompt
    assert 'character_stance' not in event_prompt

    combined = scene_prompt + event_prompt
    for forbidden in (
        'discard_event_ids',
        'next_affordances',
        'progression_guidance',
        'source_ref_allowlist',
        '"compaction"',
    ):
        assert forbidden not in combined
    assert not hasattr(recorder, '_RECORDER_REPAIR_PROMPT')


def test_recorder_payload_preserves_identity_and_semantic_clock() -> None:
    """Give both owners safe speaker identity and temporal context."""

    submitted = record_input()
    submitted['character_name'] = 'Test Character'
    assistant_turn = deepcopy(submitted['interaction_logical_turns'][0])
    assistant_turn['turn_id'] = 'trace:prior-response'
    assistant_turn['role'] = 'assistant'
    assistant_turn['display_name'] = 'machine-role-label'
    assistant_turn['conversation_row_ids'] = ['assistant-row']
    assistant_turn['llm_trace_id'] = 'prior-response'
    submitted['interaction_logical_turns'] = [assistant_turn]

    event_context = recorder.build_event_recorder_context(submitted)
    scene_payload = recorder.build_scene_recorder_human_payload(submitted)
    expected_local_time = format_storage_utc_for_llm(
        submitted['storage_timestamp_utc']
    )

    for payload in (event_context.payload, scene_payload):
        assert payload['semantic_context'] == {
            'character_name': 'Test Character',
            'current_local_time': expected_local_time,
        }

    for turn in (
        event_context.payload['source_turns'][0],
        scene_payload['recent_turns'][0],
    ):
        assert 'role' not in turn
        assert turn['speaker_kind'] == 'character'
        assert turn['speaker_name'] == 'Test Character'

    for prompt in (
        recorder.render_scene_recorder_prompt(),
        recorder.render_event_recorder_prompt(),
    ):
        assert 'semantic_context.character_name' in prompt
        assert 'semantic_context.current_local_time' in prompt
        assert 'YYYY-MM-DD' in prompt


def test_event_payload_coverage_uses_authoritative_prior_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fitter mutation cannot redefine the required prior-event domain."""

    submitted = record_input(prior_packet=packet(events=[
        event(event_id='event-one'),
        event(event_id='event-two'),
    ]))

    def _drop_prior_event(payload: dict[str, object]) -> None:
        prior_events = payload['prior_events']
        assert isinstance(prior_events, list)
        prior_events.pop()

    monkeypatch.setattr(recorder, '_fit_event_payload', _drop_prior_event)

    with pytest.raises(
        ConversationProgressContractError,
        match='complete prior-event ledger',
    ):
        recorder.build_event_recorder_context(submitted)


def test_event_validator_rejects_caller_reduced_prior_domain() -> None:
    """Canonical validation cannot accept a caller-defined ledger subset."""

    submitted = record_input(prior_packet=packet(events=[
        event(event_id='event-one'),
        event(event_id='event-two'),
    ]))
    candidate = event_observation_batch(existing_events=[
        unchanged_event_observation(event_handle='e1'),
    ])

    with pytest.raises(
        ConversationProgressContractError,
        match='complete prior-event ledger',
    ):
        validate_event_observation_batch(
            candidate,
            record_input=submitted,
            supplied_event_handles={'e1'},
            supplied_source_handles=set(source_handle_map(submitted)),
        )


def test_event_validator_requires_prior_input_order() -> None:
    """Exact coverage remains ordered for inspectable weak-model output."""

    submitted = record_input(prior_packet=packet(events=[
        event(event_id='event-one'),
        event(event_id='event-two'),
    ]))
    candidate = event_observation_batch(existing_events=[
        unchanged_event_observation(event_handle='e2'),
        unchanged_event_observation(event_handle='e1'),
    ])

    with pytest.raises(
        ConversationProgressContractError,
        match='prior-event input order',
    ):
        _validate_event_batch(candidate, submitted=submitted)


def test_both_recorder_owners_use_the_consolidation_route_budget() -> None:
    """Keep the split calls on the approved existing post-turn route."""

    configs = (
        recorder._scene_recorder_llm_config,
        recorder._event_recorder_llm_config,
    )
    assert {config.route_name for config in configs} == {
        'CONSOLIDATION_LLM'
    }
    assert {
        config.max_completion_tokens for config in configs
    } == {CONSOLIDATION_LLM_MAX_COMPLETION_TOKENS}
    assert {config.temperature for config in configs} == {0.0}


@pytest.mark.asyncio
async def test_recorder_attaches_schema_versions_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind public packet versions after semantic recorder output."""

    scene_started = asyncio.Event()
    event_started = asyncio.Event()
    scene_payload = scene_observation()
    scene_payload.pop('schema_version')
    event_payload = event_observation_batch(
        new_events=[new_event_observation()],
    )
    event_payload.pop('schema_version')
    scene_llm = _BarrierLLM(
        scene_payload,
        started=scene_started,
        peer_started=event_started,
    )
    event_llm = _BarrierLLM(
        event_payload,
        started=event_started,
        peer_started=scene_started,
    )
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    result = await asyncio.wait_for(
        recorder.record_with_llm(record_input()),
        timeout=1.0,
    )

    assert len(scene_llm.calls) == 1
    assert len(event_llm.calls) == 1
    assert result.recorder_call_count == 2
    assert result.scene_attempt_count == 1
    assert result.event_attempt_count == 1
    assert result.scene_disposition == 'accepted'
    assert result.event_disposition == 'accepted'
    assert result.delta['schema_version'] == (
        'conversation_progress_recorder_delta.v2'
    )
    assert len(result.delta['event_updates']) == 1


@pytest.mark.asyncio
async def test_scene_recorder_rejects_model_authored_schema_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject scene protocol metadata before binding the public version."""

    monkeypatch.setattr(
        recorder,
        '_scene_recorder_llm',
        _OneResponseLLM(scene_observation()),
    )

    with pytest.raises(
        recorder.ConversationProgressSceneOutputError,
        match='schema_version is code-owned',
    ):
        await recorder._record_scene(record_input())


@pytest.mark.asyncio
async def test_event_recorder_rejects_model_authored_schema_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject event protocol metadata before binding the public version."""

    monkeypatch.setattr(
        recorder,
        '_event_recorder_llm',
        _OneResponseLLM(event_observation_batch()),
    )

    with pytest.raises(
        recorder.ConversationProgressEventOutputError,
        match='schema_version is code-owned',
    ):
        await recorder._record_events(record_input())


@pytest.mark.asyncio
async def test_invalid_event_output_fails_closed_after_both_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject silent prior-event omission without a repair call."""

    submitted = record_input(
        prior_packet=packet(events=[
            event(event_id='prior-event'),
        ]),
    )
    scene_llm = _OneResponseLLM(_model_scene_observation())
    event_llm = _OneResponseLLM(_model_event_observation_batch())
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    with pytest.raises(
        recorder.ConversationProgressRecorderOutputError,
        match='coverage',
    ) as exc_info:
        await recorder.record_with_llm(submitted)

    error = exc_info.value
    assert len(scene_llm.calls) == 1
    assert len(event_llm.calls) == 1
    assert error.recorder_call_count == 2
    assert error.event_attempt_count == 1
    assert error.scene_attempt_count == 1
    assert error.event_disposition == 'failed_contract_or_provider'
    assert error.scene_disposition == 'accepted'


@pytest.mark.asyncio
async def test_invalid_scene_preserves_prior_scene_and_writes_valid_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degrade only the lower-authority scene lane."""

    prior = packet(events=[event(event_id='prior-event')])
    submitted = record_input(prior_packet=prior)
    scene_llm = _OneResponseLLM({'invalid': True})
    event_llm = _OneResponseLLM(_model_event_observation_batch(
        existing_events=[unchanged_event_observation()],
    ))
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    result = await recorder.record_with_llm(submitted)

    assert result.recorder_call_count == 2
    assert result.scene_attempt_count == 1
    assert result.scene_disposition == 'preserved_prior'
    assert result.delta['episode_narrative'] == prior['episode_narrative']
    assert result.delta['current_thread'] == prior['current_thread']
    assert result.delta['event_updates'] == []


@pytest.mark.asyncio
async def test_scene_context_limit_counts_only_the_event_provider_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep diagnostics equal to actual provider invocations."""

    monkeypatch.setattr(
        recorder,
        'MAX_SCENE_RECORDER_HUMAN_PAYLOAD_CHARS',
        1,
    )
    event_llm = _OneResponseLLM(_model_event_observation_batch(
        new_events=[new_event_observation()],
    ))
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    result = await recorder.record_with_llm(record_input())

    assert result.recorder_call_count == 1
    assert result.scene_attempt_count == 0
    assert result.event_attempt_count == 1
    assert result.scene_disposition == 'initialized_from_accepted_turn'
    assert len(event_llm.calls) == 1


@pytest.mark.asyncio
async def test_event_context_limit_retains_concurrent_scene_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Report the scene call even when the event lane fails preflight."""

    monkeypatch.setattr(
        recorder,
        'MAX_RECORDER_HUMAN_PAYLOAD_CHARS',
        1,
    )
    scene_llm = _OneResponseLLM(_model_scene_observation())
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)

    with pytest.raises(
        recorder.ConversationProgressContextLimitError,
    ) as exc_info:
        await recorder.record_with_llm(record_input())

    error = exc_info.value
    assert error.owner == 'event'
    assert error.recorder_call_count == 1
    assert error.event_attempt_count == 0
    assert error.scene_attempt_count == 1
    assert error.event_disposition == 'context_limit'
    assert error.scene_disposition == 'accepted'
    assert len(scene_llm.calls) == 1


@pytest.mark.asyncio
async def test_event_provider_failure_has_no_semantic_repair_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate one provider failure into one fail-closed disposition."""

    scene_llm = _OneResponseLLM(_model_scene_observation())
    event_llm = _OneResponseLLM(RuntimeError('provider unavailable'))
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    with pytest.raises(
        recorder.ConversationProgressRecorderOutputError,
        match='provider call failed',
    ):
        await recorder.record_with_llm(record_input())

    assert len(scene_llm.calls) == 1
    assert len(event_llm.calls) == 1


@pytest.mark.asyncio
async def test_canonical_transport_cleanup_does_not_add_model_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept fenced JSON through deterministic canonical parsing."""

    scene_payload = json.dumps(
        _model_scene_observation(),
        ensure_ascii=False,
    )
    event_payload = json.dumps(
        _model_event_observation_batch(
            new_events=[new_event_observation()],
        ),
        ensure_ascii=False,
    )
    scene_llm = _OneResponseLLM(f'```json\n{scene_payload}\n```')
    event_llm = _OneResponseLLM(f'```json\n{event_payload}\n```')
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    result = await recorder.record_with_llm(record_input())

    assert result.recorder_call_count == 2
    assert len(scene_llm.calls) == 1
    assert len(event_llm.calls) == 1


@pytest.mark.asyncio
async def test_recoverable_event_bound_is_clamped_and_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalize declared bounds without changing semantic ownership."""

    candidate = new_event_observation()
    candidate['outcome'] = 'x' * (MAX_EVENT_OUTCOME_CHARS + 25)
    scene_llm = _OneResponseLLM(_model_scene_observation())
    event_llm = _OneResponseLLM(_model_event_observation_batch(
        new_events=[candidate],
    ))
    monkeypatch.setattr(recorder, '_scene_recorder_llm', scene_llm)
    monkeypatch.setattr(recorder, '_event_recorder_llm', event_llm)

    result = await recorder.record_with_llm(record_input())

    assert len(result.delta['event_updates'][0]['outcome']) == (
        MAX_EVENT_OUTCOME_CHARS
    )
    assert {
        'owner': 'event',
        'field_path': 'new_events[0].outcome',
        'original_length': MAX_EVENT_OUTCOME_CHARS + 25,
        'normalized_length': MAX_EVENT_OUTCOME_CHARS,
    } in result.bound_normalizations


def test_event_payload_exposes_handles_without_storage_metadata() -> None:
    """Keep persistence IDs, timestamps, and source objects private."""

    prior = packet(events=[
        event(
            event_id='private-event-id',
            state='completed',
            retention='decision_critical',
        ),
    ])
    context = recorder.build_event_recorder_context(
        record_input(prior_packet=prior),
    )
    payload_text = json.dumps(
        context.payload,
        ensure_ascii=False,
        default=str,
    )
    prior_projection = context.payload['prior_events'][0]

    assert prior_projection['event_handle'] == 'e1'
    assert prior_projection['object']
    assert 'lifecycle_fact' in prior_projection
    assert 'relevance_fact' in prior_projection
    assert 'private-event-id' not in payload_text
    for forbidden in (
        'event_id',
        'source_refs',
        'first_seen_at',
        'updated_at',
        'retention',
        'state',
        'boundary_profile',
        'content_plan',
        'logical_stance',
        'character_intent',
    ):
        assert forbidden not in payload_text


def test_scene_payload_contains_no_event_or_storage_authority() -> None:
    """Keep the scene owner independent from event reconciliation."""

    payload = recorder.build_scene_recorder_human_payload(
        record_input(
            prior_packet=packet(events=[event(event_id='private-event-id')]),
        ),
    )
    payload_text = json.dumps(payload, ensure_ascii=False, default=str)

    assert 'prior_scene' in payload
    assert 'accepted_turn' in payload
    assert 'content_plan' in payload['accepted_turn']
    assert 'prior_events' not in payload
    assert 'private-event-id' not in payload_text
    assert 'source_refs' not in payload_text
    assert 'boundary_profile' not in payload_text


def test_exact_changed_observation_preserves_stable_event_definition() -> None:
    """Permit semantic change without rewriting actor/action/object identity."""

    prior_event = event(
        event_id='stable-event',
        summary='specific prior event',
    )
    prior_event.update({
        'actor': 'the character',
        'action': 'evaluate',
        'object': 'the user-selected implementation',
        'beneficiary': 'the user',
        'precondition': 'the implementation is submitted',
    })
    submitted = record_input(
        prior_packet=packet(events=[prior_event]),
    )
    updates = _validate_event_batch(
        event_observation_batch(existing_events=[
            changed_event_observation(
                summary='the implementation received a conclusive evaluation',
                lifecycle_change='concluded',
                relevance='decision',
            ),
        ]),
        submitted=submitted,
    )

    assert updates[0]['event_id'] == 'stable-event'
    assert updates[0]['actor'] == 'the character'
    assert updates[0]['action'] == 'evaluate'
    assert updates[0]['object'] == 'the user-selected implementation'
    assert updates[0]['beneficiary'] == 'the user'
    assert updates[0]['state'] == 'completed'


def test_changed_observation_rejects_definition_rewrite_fields() -> None:
    """Reject duplicated identity ownership in an existing-event row."""

    submitted = record_input(
        prior_packet=packet(events=[event(event_id='stable-event')]),
    )
    candidate = changed_event_observation()
    candidate['object'] = 'a replacement object'

    with pytest.raises(
        ConversationProgressContractError,
        match='fields are not exact',
    ):
        _validate_event_batch(
            event_observation_batch(existing_events=[candidate]),
            submitted=submitted,
        )


def test_new_event_requires_concrete_actor_action_and_object() -> None:
    """Reject generic event identities before persistence."""

    candidate = new_event_observation(object_='')

    with pytest.raises(
        ConversationProgressContractError,
        match='requires actor, action, and object',
    ):
        _validate_event_batch(
            event_observation_batch(new_events=[candidate]),
        )


@pytest.mark.parametrize(
    ('lifecycle_change', 'expected_state'),
    [
        ('none', 'open'),
        ('began', 'in_progress'),
        ('concluded', 'completed'),
        ('declined', 'rejected'),
        ('replaced', 'superseded'),
    ],
)
def test_new_event_lifecycle_is_mapped_deterministically(
    lifecycle_change: str,
    expected_state: str,
) -> None:
    """Map one semantic lifecycle observation without interpreting text."""

    updates = _validate_event_batch(event_observation_batch(new_events=[
        new_event_observation(lifecycle_change=lifecycle_change),
    ]))

    assert updates[0]['state'] == expected_state


def test_none_preserves_existing_lifecycle() -> None:
    """Prevent relevance-only updates from regressing lifecycle state."""

    submitted = record_input(prior_packet=packet(events=[
        event(event_id='active-event', state='in_progress'),
    ]))
    updates = _validate_event_batch(
        event_observation_batch(existing_events=[
            changed_event_observation(
                lifecycle_change='none',
                relevance='decision',
            ),
        ]),
        submitted=submitted,
    )

    assert updates[0]['state'] == 'in_progress'


def test_reopened_requires_a_prior_terminal_event() -> None:
    """Accept reopening only when the semantic transition has a target."""

    nonterminal = record_input(prior_packet=packet(events=[
        event(event_id='active-event', state='in_progress'),
    ]))
    candidate = event_observation_batch(existing_events=[
        changed_event_observation(lifecycle_change='reopened'),
    ])

    with pytest.raises(
        ConversationProgressContractError,
        match='prior terminal',
    ):
        _validate_event_batch(candidate, submitted=nonterminal)

    terminal = record_input(prior_packet=packet(events=[
        event(event_id='done-event', state='completed'),
    ]))
    updates = _validate_event_batch(candidate, submitted=terminal)
    assert updates[0]['state'] == 'open'


@pytest.mark.parametrize(
    ('relevance', 'expected_retention'),
    [
        ('decision', 'decision_critical'),
        ('scene', 'active_scene'),
        ('history', 'background'),
    ],
)
def test_relevance_is_mapped_deterministically(
    relevance: str,
    expected_retention: str,
) -> None:
    """Map model-owned relevance without lexical classification."""

    updates = _validate_event_batch(event_observation_batch(new_events=[
        new_event_observation(relevance=relevance),
    ]))
    assert updates[0]['retention'] == expected_retention


def test_event_payload_pressure_preserves_every_prior_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove old turn text while retaining exact full-ledger coverage."""

    prior = packet(events=[
        event(
            event_id='critical-event',
            summary='critical ' + ('x' * 180),
            retention='decision_critical',
        ),
        event(
            event_id='background-event',
            summary='background ' + ('y' * 180),
            retention='background',
        ),
    ])
    submitted = record_input(prior_packet=prior)
    submitted['interaction_logical_turns'][0]['fragments'] = [
        'older source detail ' + ('z' * 1900)
    ]
    full_context = recorder.build_event_recorder_context(submitted)
    mandatory_payload = deepcopy(full_context.payload)
    mandatory_payload['source_turns'] = []
    mandatory_chars = len(json.dumps(
        mandatory_payload,
        ensure_ascii=False,
        sort_keys=True,
    ))
    monkeypatch.setattr(
        recorder,
        'MAX_RECORDER_HUMAN_PAYLOAD_CHARS',
        mandatory_chars,
    )

    context = recorder.build_event_recorder_context(submitted)

    assert context.event_handles == frozenset({'e1', 'e2'})
    assert [
        row['event_handle'] for row in context.payload['prior_events']
    ] == ['e1', 'e2']
    assert context.payload['source_turns'] == []


def test_event_payload_fails_closed_when_required_context_cannot_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse to truncate current accepted-turn authority."""

    monkeypatch.setattr(
        recorder,
        'MAX_RECORDER_HUMAN_PAYLOAD_CHARS',
        1,
    )

    with pytest.raises(
        recorder.ConversationProgressContextLimitError,
    ) as exc_info:
        recorder.build_event_recorder_context(record_input())

    assert exc_info.value.owner == 'event'


def test_event_handle_domains_are_private_and_exact() -> None:
    """Map short handles to prior IDs and canonical refs only in code."""

    submitted = record_input(prior_packet=packet(events=[
        event(event_id='private-id'),
    ]))
    handles = event_handle_map(submitted)
    sources = source_handle_map(submitted)

    assert handles['e1']['event_id'] == 'private-id'
    assert set(sources) == {'t1', 'current_input', 'current_response'}
    updates = _validate_event_batch(
        event_observation_batch(existing_events=[
            deepcopy(unchanged_event_observation()),
        ]),
        submitted=submitted,
    )
    assert updates == []


def test_current_input_handle_preserves_all_collapsed_row_lineage() -> None:
    """One semantic input handle resolves to every exact collapsed row."""

    submitted = record_input()
    submitted['current_turn_source_refs'] = [
        {
            'ref_kind': 'conversation_row',
            'ref_id': 'current-row-one',
            'occurred_at': '2026-07-28T09:30:00+00:00',
        },
        {
            'ref_kind': 'conversation_row',
            'ref_id': 'current-row-two',
            'occurred_at': '2026-07-28T09:30:09+00:00',
        },
        {
            'ref_kind': 'llm_trace',
            'ref_id': 'trace-current',
            'occurred_at': '2026-07-28T09:30:10+00:00',
        },
    ]

    sources = source_handle_map(submitted)

    assert sources['current_input'] == submitted[
        'current_turn_source_refs'
    ][:2]
