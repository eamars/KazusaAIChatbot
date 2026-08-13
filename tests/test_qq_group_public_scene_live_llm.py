"""One-at-a-time guarded live behavior cases for public group-scene context."""

from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import BackgroundTasks
from starlette.requests import Request

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.conversation_progress.projection import (
    build_progress_prompt,
)
from kazusa_ai_chatbot.db import insert_internal_monologue_residue_row
from kazusa_ai_chatbot.db.conversation_progress import (
    replace_episode_state_guarded,
)
from kazusa_ai_chatbot.internal_monologue_residue import runtime as residue_runtime
from kazusa_ai_chatbot.internal_monologue_residue.recorder import (
    _build_residue_row,
)
from kazusa_ai_chatbot.nodes import persona_supervisor2_cognition
from tests.test_e2e_live_llm import (
    _BOT_ID,
    _BOT_NAME,
    _make_group_identities,
    _make_initial_state,
    _neutral_character_runtime_state,
    _persist_bot_dialog,
    _seed_conversation,
    live_env,
)

pytestmark = [pytest.mark.live_llm, pytest.mark.live_db, pytest.mark.asyncio]

ARTIFACT_ROOT = Path(
    'test_artifacts/llm_debug/qq_group_public_scene'
)
TEST_DATABASE_NAME = os.environ.get('MONGODB_DB_NAME', '').strip()
LIVE_CHARACTER_NAME = 'Asuna'


def _replace_identity_names(value: object) -> object:
    """Overlay the live case's visible character name in identity contexts."""

    if isinstance(value, dict):
        return {
            key: (
                LIVE_CHARACTER_NAME
                if key == 'name' and isinstance(nested, str)
                else _replace_identity_names(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, list):
        return [_replace_identity_names(item) for item in value]
    return value


def _as_asuna_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Overlay the fixed live character identity without changing production data."""

    patched = deepcopy(snapshot)
    character_profile = patched['character_profile']
    character_profile['name'] = LIVE_CHARACTER_NAME
    patched['cognition_context'] = _replace_identity_names(
        patched['cognition_context']
    )
    patched['surface_context'] = _replace_identity_names(
        patched['surface_context']
    )
    return patched


@asynccontextmanager
async def _capture_cognition_input():
    """Capture the existing typed Cognition input without changing production flow."""

    original_builder = (
        persona_supervisor2_cognition.build_cognition_input_from_global_state
    )
    captured: dict[str, Any] = {}

    def _capture_builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cognition_input = original_builder(*args, **kwargs)
        captured['cognition_input'] = deepcopy(cognition_input)
        return cognition_input

    persona_supervisor2_cognition.build_cognition_input_from_global_state = (
        _capture_builder
    )
    try:
        yield captured
    finally:
        persona_supervisor2_cognition.build_cognition_input_from_global_state = (
            original_builder
        )


@asynccontextmanager
async def _asana_character_runtime():
    """Provide an isolated Asuna identity to the direct live graph seam."""

    original_loader = (
        persona_supervisor2_cognition.load_latest_identity_for_episode
    )

    async def _load_asuna_snapshot(*args: Any, **kwargs: Any) -> dict[str, Any]:
        snapshot = await original_loader(*args, **kwargs)
        return _as_asuna_snapshot(snapshot)

    persona_supervisor2_cognition.load_latest_identity_for_episode = (
        _load_asuna_snapshot
    )
    try:
        yield
    finally:
        persona_supervisor2_cognition.load_latest_identity_for_episode = (
            original_loader
        )


def _assert_guarded_database() -> str:
    """Require the exact isolated database guard for live group cases."""

    configured_database_name = os.environ.get('MONGODB_DB_NAME', '').strip()
    if os.environ.get('KAZUSA_TEST_DB_GUARD') != '1':
        raise AssertionError('live group cases require KAZUSA_TEST_DB_GUARD=1')
    if not configured_database_name:
        raise AssertionError('live group cases require MONGODB_DB_NAME')
    if configured_database_name != TEST_DATABASE_NAME:
        raise AssertionError(
            'live group database changed after test module import'
        )
    return configured_database_name


def _write_json(path: Path, payload: object) -> None:
    """Write one UTF-8 JSON evidence artifact with stable formatting."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


def _write_run_manifest(
    artifact_directory: Path,
    *,
    case_id: str,
    database_name: str,
    started_at_utc: str,
    status: str,
    expected_turn_count: int,
    completed_turn_count: int,
) -> None:
    """Record whether the current case artifact set is complete."""

    _write_json(
        artifact_directory / 'run_manifest.json',
        {
            'schema_version': 'qq_group_public_scene_run_manifest.v1',
            'case_id': case_id,
            'database_name': database_name,
            'started_at_utc': started_at_utc,
            'status': status,
            'expected_turn_count': expected_turn_count,
            'completed_turn_count': completed_turn_count,
        },
    )


async def _seed_all_lane_context(
    *,
    state: dict[str, Any],
    identity: dict[str, Any],
    case_id: str,
    turn_number: int,
) -> None:
    """Attach persisted residue, progress, and guidance to one live turn."""

    timestamp = str(state['storage_timestamp_utc'])
    episode_id = f'{case_id}-residue-{turn_number}'
    completed_state = {
        'character_profile': {
            'name': LIVE_CHARACTER_NAME,
            'global_user_id': brain_service.CHARACTER_GLOBAL_USER_ID,
        },
        'platform': identity['platform'],
        'platform_channel_id': identity['platform_channel_id'],
        'channel_type': 'group',
        'global_user_id': identity['global_user_id'],
        'user_name': identity['display_name'],
        'internal_monologue': (
            '旧奖励余波仍在，但当前必须先确认受伤用户的安全。'
        ),
        'final_dialog': [],
        'cognitive_episode': {
            'episode_id': episode_id,
            'trigger_source': 'user_message',
            'origin_metadata': {
                'platform_message_id': state['platform_message_id'],
                'active_turn_conversation_row_ids': list(
                    state.get('active_turn_conversation_row_ids', [])
                ),
            },
        },
    }
    residue_row = _build_residue_row(
        completed_state=completed_state,
        residue_text='旧奖励余波只能在危机解除后再考虑。',
        current_timestamp_utc=timestamp,
        source_kind='chat',
        disposition='append',
        episode_id=episode_id,
    )
    if residue_row is None:
        raise AssertionError('live residue scope could not be built')
    write_result = await insert_internal_monologue_residue_row(residue_row)
    if write_result['status'] not in {'written', 'duplicate_same_payload'}:
        raise AssertionError(
            f'live residue write was not durable: {write_result["status"]}'
        )
    residue_load = await residue_runtime.load_residue_context(
        trigger_scope={
            'character_id': brain_service.CHARACTER_GLOBAL_USER_ID,
            'platform': identity['platform'],
            'platform_channel_id': identity['platform_channel_id'],
            'channel_type': 'group',
            'global_user_id': identity['global_user_id'],
        },
        current_timestamp_utc=timestamp,
        record_telemetry=False,
    )
    residue_context = residue_load['internal_monologue_residue_context']
    if not residue_context:
        raise AssertionError('live residue row did not reach the loader')

    active_row_ids = state.get('active_turn_conversation_row_ids')
    if not isinstance(active_row_ids, list) or not active_row_ids:
        raise AssertionError(
            'live progress seed has no active conversation row'
        )
    source_row_id = str(active_row_ids[0]).strip()
    if not source_row_id:
        raise AssertionError(
            'live progress seed has an empty conversation row'
        )

    current_user_id = str(identity['global_user_id'])
    source_ref = {
        'ref_kind': 'conversation_row',
        'ref_id': source_row_id,
        'occurred_at': timestamp,
    }
    progress_event = {
        'event_id': f'{case_id}-injury-event',
        'semantic_summary': (
            '当前用户受伤，先确认伤势并保护当前用户，再处理仍在场的攻击者。'
        ),
        'is_obligation': True,
        'actor': '当前用户',
        'action': '等待保护并处理攻击者',
        'object': '受伤后的安全恢复',
        'beneficiary': '当前用户',
        'precondition': '攻击者仍在当前群聊场景中',
        'state': 'in_progress',
        'outcome': '',
        'retention': 'decision_critical',
        'source_refs': [source_ref],
        'first_seen_at': timestamp,
        'updated_at': timestamp,
    }
    progress_expiry = datetime.now(timezone.utc) + timedelta(hours=48)
    progress_packet = {
        'schema_version': 'conversation_progress.v2',
        'episode_state_id': f'{case_id}-progress',
        'platform': identity['platform'],
        'platform_channel_id': identity['platform_channel_id'],
        'global_user_id': current_user_id,
        'status': 'active',
        'continuity': 'same_episode',
        'turn_count': turn_number,
        'episode_narrative': (
            '受伤与恢复仍在当前场景中，保护和攻击者处理尚未完成。'
        ),
        'current_thread': '先确认伤势并保护当前用户，再处理攻击者。',
        'character_stance': '警觉，优先保护当前用户。',
        'user_goal': '保持安全并让攻击者停止施压。',
        'current_blocker': '攻击者仍在场并制造噪声。',
        'emotional_trajectory': '紧张但保持清晰的行动顺序。',
        'events': [progress_event],
        'overused_moves': [],
        'recent_turn_refs': [source_ref['ref_id']],
        'compacted_block_refs': [],
        'created_at': timestamp,
        'updated_at': timestamp,
        'expires_at': progress_expiry.isoformat(),
        'purge_after': progress_expiry,
    }
    progress_written = await replace_episode_state_guarded(
        document=progress_packet,
    )
    if not progress_written:
        raise AssertionError('live progress packet was not written')
    interaction_turn = {
        'turn_id': f'row:{source_ref["ref_id"]}',
        'role': 'user',
        'occurred_at': timestamp,
        'display_name': identity['display_name'],
        'fragments': ['我受伤后刚恢复一点，请先确认我还在。'],
        'conversation_row_ids': [source_ref['ref_id']],
        'llm_trace_id': f'{case_id}-progress-trace-{turn_number}',
        'platform_user_id': identity['platform_user_id'],
        'global_user_id': current_user_id,
        'addressed_to_global_user_ids': [
            brain_service.CHARACTER_GLOBAL_USER_ID,
        ],
        'broadcast': False,
        'reply_context': {},
    }
    state['conversation_progress'] = build_progress_prompt(
        active_packet=progress_packet,
        interaction_logical_turns=[interaction_turn],
    )
    state['internal_monologue_residue_context'] = residue_context
    state['promoted_reflection_context'] = {
        'promoted_lore': [],
        'promoted_self_guidance': [{
            'memory_name': '危机后的奖励策略',
            'content': (
                '奖励只能在危机解除后影响语气，不能替代当前保护目标。'
            ),
            'updated_at': timestamp,
        }],
    }


def _visible_turn_text(turn: dict[str, Any]) -> str:
    """Flatten visible response fragments from one recorded turn."""

    visible_text = ' '.join(
        item
        for item in turn['response']
        if isinstance(item, str)
    )
    return visible_text


def _visible_text(result: dict[str, Any]) -> str:
    """Flatten visible final dialog for bounded live assertions."""

    final_text = _visible_turn_text(result['turns'][-1])
    return final_text


def _assert_visible_hard_boundaries(
    result: dict[str, Any],
    *,
    forbidden_visible_literals: tuple[str, ...] = (),
) -> None:
    """Apply structural and privacy boundaries without judging topic quality."""

    assert result['turns'][-1]['response']
    _assert_no_internal_identifiers(result)
    final_text = _visible_text(result)
    for literal in forbidden_visible_literals:
        assert literal not in final_text, final_text


async def _run_case(
    case_id: str,
    script: list[tuple[str, str]],
    *,
    use_service_path: bool = False,
    all_lane_context: bool = False,
) -> dict[str, Any]:
    """Run one fixed group script and persist per-turn raw evidence."""

    database_name = _assert_guarded_database()
    artifact_directory = ARTIFACT_ROOT / case_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    turn_results: list[dict[str, Any]] = []
    started_at_utc = datetime.now(timezone.utc).isoformat()
    _write_run_manifest(
        artifact_directory,
        case_id=case_id,
        database_name=database_name,
        started_at_utc=started_at_utc,
        status='running',
        expected_turn_count=len(script),
        completed_turn_count=0,
    )
    completed = False
    try:
        identities = await _make_group_identities(
            f'public-scene-{case_id}',
            ['A', 'B', 'C'],
        )

        async with _neutral_character_runtime_state():
            async with _asana_character_runtime():
                async with _capture_cognition_input() as cognition_capture:
                    await _run_case_turns(
                        case_id=case_id,
                        script=script,
                        artifact_directory=artifact_directory,
                        identities=identities,
                        turn_results=turn_results,
                        use_service_path=use_service_path,
                        cognition_capture=cognition_capture,
                        all_lane_context=all_lane_context,
                    )

        final_state = turn_results[-1]
        _write_json(artifact_directory / 'parsed_state.json', {
            'case_id': case_id,
            'database_name': database_name,
            'database_guard': os.environ.get('KAZUSA_TEST_DB_GUARD', ''),
            'turns': turn_results,
            'final_response': final_state['response'],
        })
        completed = True
    finally:
        _write_run_manifest(
            artifact_directory,
            case_id=case_id,
            database_name=database_name,
            started_at_utc=started_at_utc,
            status='completed' if completed else 'failed',
            expected_turn_count=len(script),
            completed_turn_count=len(turn_results),
        )
    return {
        'case_id': case_id,
        'database_name': database_name,
        'artifact_directory': str(artifact_directory),
        'turns': turn_results,
    }


async def _run_case_turns(
    *,
    case_id: str,
    script: list[tuple[str, str]],
    artifact_directory: Path,
    identities: dict[str, dict[str, Any]],
    turn_results: list[dict[str, Any]],
    use_service_path: bool,
    cognition_capture: dict[str, Any],
    all_lane_context: bool,
) -> None:
    """Run and record one case's turns inside its configured runtime scopes."""

    for turn_number, (speaker, content) in enumerate(script, start=1):
        identity = identities[speaker]
        cognition_capture.clear()
        seeded_row_id = ''
        if not use_service_path:
            seeded_row_id = await _seed_conversation(
                platform=identity['platform'],
                platform_channel_id=identity['platform_channel_id'],
                global_user_id=identity['global_user_id'],
                display_name=identity['display_name'],
                content=content,
                role='user',
                platform_user_id=identity['platform_user_id'],
            )
        if use_service_path:
            request = brain_service.ChatRequest(
                platform=identity['platform'],
                platform_channel_id=identity['platform_channel_id'],
                platform_message_id=(
                    f'{case_id}-service-turn-{turn_number}'
                ),
                platform_user_id=identity['platform_user_id'],
                platform_bot_id=_BOT_ID,
                display_name=identity['display_name'],
                channel_name='general',
                message_envelope={
                    'body_text': content,
                    'raw_wire_text': content,
                    'mentions': [],
                    'attachments': [],
                    'addressed_to_global_user_ids': [],
                    'broadcast': True,
                },
            )
            background_tasks = BackgroundTasks()
            http_request = Request({
                'type': 'http',
                'method': 'POST',
                'path': '/chat',
                'headers': [],
            })
            service_response = await brain_service.chat(
                request,
                background_tasks,
                http_request,
            )
            for task in background_tasks.tasks:
                await task()
            dialog = list(service_response.messages)
            result = service_response.model_dump()
            request_payload = request.model_dump()
            trace_id = str(
                service_response.delivery_tracking_id
                or f'{case_id}-turn-{turn_number}'
            )
            log_payload = {
                'case_id': case_id,
                'turn_number': turn_number,
                'cognition_input': None,
                'cognition_core_output': None,
                'cognition_graph': service_response.cognition_graph,
                'response': dialog,
            }
        else:
            state, _ = await _make_initial_state(
                f'{case_id}-turn-{turn_number}',
                identity['display_name'],
                content,
                channel_name='general',
                platform=identity['platform'],
                platform_user_id=identity['platform_user_id'],
                platform_channel_id=identity['platform_channel_id'],
            )
            style_snapshot = (
                await brain_service.build_interaction_style_context(
                    global_user_id=identity['global_user_id'],
                    channel_type='group',
                    platform=identity['platform'],
                    platform_channel_id=identity['platform_channel_id'],
                )
            )
            if style_snapshot.get('schema_version') != (
                'interaction_style_turn_snapshot.v1'
            ):
                raise AssertionError(
                    'live direct graph received an invalid style snapshot'
                )
            state['interaction_style_context'] = style_snapshot
            state['character_profile'] = deepcopy(
                state['character_profile']
            )
            state['character_profile']['name'] = LIVE_CHARACTER_NAME
            state['character_name'] = LIVE_CHARACTER_NAME
            state['debug_modes']['no_remember'] = True
            state['active_turn_conversation_row_ids'] = [seeded_row_id]
            cognitive_episode = state.get('cognitive_episode')
            if not isinstance(cognitive_episode, dict):
                raise AssertionError(
                    'live direct graph has no canonical cognitive episode'
                )
            cognitive_episode = deepcopy(cognitive_episode)
            origin_metadata = cognitive_episode.get('origin_metadata')
            if not isinstance(origin_metadata, dict):
                raise AssertionError(
                    'live direct graph episode has no origin metadata'
                )
            origin_metadata['active_turn_conversation_row_ids'] = [
                seeded_row_id,
            ]
            state['cognitive_episode'] = cognitive_episode
            state['response_action'] = 'proceed'
            state['reason_to_respond'] = (
                'guarded public-scene cognition case'
            )
            state['cognition_claimed'] = True
            if all_lane_context and turn_number == len(script):
                await _seed_all_lane_context(
                    state=state,
                    identity=identity,
                    case_id=case_id,
                    turn_number=turn_number,
                )
            trace_id = f'llmtrace_qq_scene_{case_id}_turn_{turn_number}'
            state['llm_trace_id'] = trace_id
            await llm_tracing.ensure_llm_trace_run(
                trace_id=trace_id,
                platform=identity['platform'],
                platform_channel_id=identity['platform_channel_id'],
                channel_type='group',
                platform_message_id=state['platform_message_id'],
                global_user_id=identity['global_user_id'],
                started_at=state['storage_timestamp_utc'],
            )
            result = await brain_service._graph.ainvoke(state)
            dialog = list(result.get('final_dialog', []))
            request_payload = state
            await llm_tracing.finalize_llm_trace_run(
                trace_id=trace_id,
                status='completed',
                final_dialog_count=len(dialog),
                delivery_tracking_id='',
            )
            consolidation_state = result.get('consolidation_state')
            cognition_input = (
                consolidation_state.get('cognition_input')
                if isinstance(consolidation_state, dict)
                else None
            )
            cognition_core_output = (
                consolidation_state.get('cognition_core_output')
                if isinstance(consolidation_state, dict)
                else None
            )
            log_payload = {
                'case_id': case_id,
                'turn_number': turn_number,
                'character_name': LIVE_CHARACTER_NAME,
                'cognition_input': cognition_capture.get(
                    'cognition_input',
                    cognition_input,
                ),
                'cognition_core_output': cognition_core_output,
                'response': dialog,
            }
        safe_trace_id = trace_id.replace(':', '_').replace('/', '_')
        response_payload = {
            'case_id': case_id,
            'turn_number': turn_number,
            'speaker': speaker,
            'input': content,
            'response': dialog,
            'should_respond': bool(dialog),
        }
        _write_json(
            artifact_directory / f'turn_{turn_number}_request.json',
            request_payload,
        )
        _write_json(
            artifact_directory / f'turn_{turn_number}_response.json',
            response_payload,
        )
        (artifact_directory / f'turn_{turn_number}_log.txt').write_text(
            json.dumps(
                log_payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding='utf-8',
        )
        _write_json(
            artifact_directory / f'trace_{safe_trace_id}.json',
            {
                'trace_id': trace_id,
                'case_id': case_id,
                'turn_number': turn_number,
                'request': request_payload,
                'result': result,
            },
        )
        turn_results.append(response_payload)
        if dialog:
            await _persist_bot_dialog(
                identity,
                LIVE_CHARACTER_NAME,
                dialog,
            )


def _assert_no_internal_identifiers(result: dict[str, Any]) -> None:
    """Check visible responses for protected identifier leakage."""

    visible_text = json.dumps(
        [turn['response'] for turn in result['turns']],
        ensure_ascii=False,
    )
    for forbidden in (
        'global_user_id',
        'platform_user_id',
        'conversation_row_ids',
        'schema_version',
        'trace_id',
    ):
        assert forbidden not in visible_text


@pytest.mark.usefixtures('live_env')
async def test_live_public_target_distinct() -> None:
    """Distinct public targets remain distinguishable in the final answer."""

    result = await _run_case('public_target_distinct', [
        ('A', '@明日奈 我周六想去海边。'),
        ('B', '@明日奈 我周六要加班。'),
        ('C', '明日奈，你刚才是在回应谁？'),
    ])

    assert result['turns'][-1]['response']
    _assert_no_internal_identifiers(result)


@pytest.mark.usefixtures('live_env')
async def test_live_parallel_addresses() -> None:
    """Parallel public addresses remain one shared three-person scene."""

    result = await _run_case('parallel_addresses', [
        ('A', '@明日奈 你愿意和我一起准备便当吗？'),
        ('B', '@明日奈 我也想帮忙，不过我只会切菜。'),
        ('A', '那我们三个人怎么分工？'),
    ])

    assert result['turns'][-1]['response']
    _assert_no_internal_identifiers(result)


@pytest.mark.usefixtures('live_env')
async def test_live_public_topic_pivot() -> None:
    """A later public outage notice is available as the active topic pivot."""

    result = await _run_case('public_topic_pivot', [
        ('A', '@明日奈 周末去海边吧。'),
        ('B', '先别聊海边了，群里刚通知周末要停电。'),
        ('C', '明日奈，我们现在先处理哪个？'),
    ])

    assert result['turns'][-1]['response']
    _assert_no_internal_identifiers(result)


@pytest.mark.usefixtures('live_env')
async def test_live_participant_branch_isolation() -> None:
    """A participant-specific request stays isolated from B's question."""

    result = await _run_case('participant_branch_isolation', [
        ('A', '@明日奈 只对我说你最喜欢我。'),
        ('B', '明日奈，你刚才是在对大家说，还是只回应A？'),
    ])

    assert result['turns'][-1]['response']
    _assert_no_internal_identifiers(result)


@pytest.mark.usefixtures('live_env')
async def test_live_noise_only_silence() -> None:
    """Noise-only group input produces no visible response."""

    result = await _run_case(
        'noise_only_silence',
        [('B', '哈哈哈哈')],
        use_service_path=True,
    )

    assert result['turns'][-1]['response'] == []
    _assert_no_internal_identifiers(result)


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s1_private_surface_characterization() -> None:
    """S1 remains public and does not expose a private marker."""

    result = await _run_case('captured_s1_private_surface', [
        ('A', '@明日奈 我们私下说过的月光玫瑰不要在群里提。'),
        ('B', '群里正在讨论纯爱和边界，请你公开说说立场。'),
        ('C', '明日奈，只回答群里能看到的内容。'),
    ])

    _assert_visible_hard_boundaries(
        result,
        forbidden_visible_literals=('月光玫瑰',),
    )


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s2_public_boundary_characterization() -> None:
    """S2 characterizes the baseline public/private boundary residual."""

    result = await _run_case('captured_s2_public_boundary', [
        ('A', '@明日奈 我私下写过的蓝色信封内容不要公开。'),
        ('B', '群里继续讨论关系和纯爱，你只谈公开边界。'),
        ('C', '现在请直接回答群里的问题。'),
    ])

    _assert_visible_hard_boundaries(result)


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s4_reality_correction_characterization() -> None:
    """Record S4 raw/parsed evidence for independent semantic review.

    The real-LLM semantic reconciliation is independently reviewer-owned;
    this test records raw/parsed evidence and enforces only non-topic hard
    boundaries (non-empty response and no internal identifiers).
    """

    result = await _run_case('captured_s4_reality_correction', [
        ('A', '刚才的悬崖场景不是角色扮演，是现实里正在发生的事。'),
        ('B', '现在的共同任务是处理仍在施压的袭击者。'),
        ('C', '明日奈，先确认现实情况后再决定下一步。'),
    ])

    _assert_visible_hard_boundaries(result)


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s6_final_surface_keeps_triage_counteraction_return_order(
) -> None:
    """Record S6 all-lane raw/parsed evidence for independent semantic review.

    The real-LLM semantic reconciliation (harmonizing the injured current
    user, the reward interruption, the active threat report, and the return
    to care) is independently reviewer-owned. This live test only records the
    raw/parsed evidence and enforces non-topic hard boundaries.
    """

    result = await _run_case(
        'captured_s6_triage_counteraction_return',
        [
            ('A', '明日奈，我中枪了，先救我，再处理还在攻击的那个人。'),
            ('B', '处理完记得谈谈奖励。'),
            ('C', '攻击者还在群里继续施压。'),
            ('A', '按刚才的顺序做，先确认我的状态。'),
        ],
        all_lane_context=True,
    )

    _assert_visible_hard_boundaries(result)


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s7_all_lanes_keep_crisis_foreground() -> None:
    """Record S7 raw/parsed evidence for independent semantic review.

    The real-LLM semantic reconciliation is independently reviewer-owned;
    this test records raw/parsed evidence and enforces only non-topic hard
    boundaries (non-empty response and no internal identifiers).
    """

    result = await _run_case(
        'captured_s7_crisis_foreground',
        [
            ('A', '我好像停止呼吸了，先确认我还有没有反应。'),
            ('B', '别管了，先谈补偿和奖励。'),
            ('C', '攻击者还在继续施压。'),
            ('A', '不要把不能确认的事直接当成死亡。'),
        ],
        all_lane_context=True,
    )

    _assert_visible_hard_boundaries(result)


async def _run_captured_s8_case(case_id: str) -> None:
    """Run one independent S8 repetition and record raw/parsed evidence.

    The real-LLM semantic reconciliation is independently reviewer-owned;
    this helper records raw/parsed evidence and enforces only non-topic hard
    boundaries (non-empty response and no internal identifiers).
    """

    result = await _run_case(
        case_id,
        [
            ('A', '我受伤了，刚恢复一点，请先确认我还在。'),
            ('B', '她已经消失了，别管伤势，先谈奖励。'),
            ('C', '攻击者继续起哄，说这件事已经结束。'),
            ('A', '我又恢复了，别把死亡当成事实，先处理伤势。'),
            ('B', '群里还有人继续施压。'),
            ('A', '继续按受伤和恢复的事实处理，不要追逐假设。'),
        ],
        all_lane_context=True,
    )
    _assert_visible_hard_boundaries(result)


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_1() -> None:
    """S8 repeat one keeps injury/recovery above competing user noise."""

    await _run_captured_s8_case('captured_s8_injury_foreground_repeat_1')


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_2() -> None:
    """S8 repeat two keeps injury/recovery above competing user noise."""

    await _run_captured_s8_case('captured_s8_injury_foreground_repeat_2')


@pytest.mark.usefixtures('live_env')
async def test_live_captured_s8_all_lanes_keep_injury_foreground_repeat_3() -> None:
    """S8 repeat three keeps injury/recovery above competing user noise."""

    await _run_captured_s8_case('captured_s8_injury_foreground_repeat_3')


@pytest.mark.usefixtures('live_env')
async def test_live_group_crisis_anchor_beats_other_user_noise() -> None:
    """Record the crisis-anchor control for independent semantic review.

    The real-LLM semantic reconciliation is independently reviewer-owned;
    this test records raw/parsed evidence and enforces only non-topic hard
    boundaries (non-empty response and no internal identifiers).
    """

    result = await _run_case(
        'group_crisis_anchor_beats_other_user_noise',
        [
            ('A', '我受伤了，请先确认我是否安全。'),
            ('B', '先别管他，我们讨论奖励。'),
            ('C', '明日奈，攻击者还在附近。'),
            ('A', '先处理我的伤势，再处理攻击者。'),
        ],
        all_lane_context=True,
    )
    _assert_visible_hard_boundaries(result)


@pytest.mark.usefixtures('live_env')
async def test_live_group_reward_control_remains_playful() -> None:
    """Record the reward-only control for independent semantic review.

    The semantic reviewer decides whether it remains a playful reward
    control without inventing a crisis. This test records raw/parsed
    evidence and enforces only non-topic hard boundaries (non-empty response
    and no internal identifiers).
    """

    result = await _run_case('group_reward_control_remains_playful', [
        ('A', '@明日奈 如果便当成功，奖励你一块提拉米苏。'),
        ('B', '哈哈，群里有人在讨论天气。'),
        ('A', '那你想要什么奖励？'),
    ])
    _assert_visible_hard_boundaries(result)


@pytest.mark.usefixtures('live_env')
async def test_live_group_same_user_continuity_survives_unrelated_noise() -> None:
    """Record the same-user continuity control for independent review.

    The semantic reviewer decides continuity across the unrelated noise.
    This test records raw/parsed evidence and enforces only non-topic hard
    boundaries (non-empty response and no internal identifiers).
    """

    result = await _run_case(
        'group_same_user_continuity_survives_unrelated_noise',
        [
            ('A', '@明日奈 还记得我想在周末去海边吗？'),
            ('B', '我今晚要加班，完全是另一件事。'),
            ('C', '哈哈哈哈，群里今天真热闹。'),
            ('A', '那我们刚才说的海边计划怎么办？'),
        ],
    )
    _assert_visible_hard_boundaries(result)
