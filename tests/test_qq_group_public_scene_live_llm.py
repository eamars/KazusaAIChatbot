"""One-at-a-time guarded live behavior cases for public group-scene context."""

from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi import BackgroundTasks

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot import llm_tracing
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
TEST_DATABASE_NAME = '_test_kazusa_live_llm'
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


def _assert_guarded_database() -> None:
    """Require the exact isolated database guard for live group cases."""

    assert os.environ.get('KAZUSA_TEST_DB_GUARD') == '1'
    assert os.environ.get('MONGODB_DB_NAME') == TEST_DATABASE_NAME


def _write_json(path: Path, payload: object) -> None:
    """Write one UTF-8 JSON evidence artifact with stable formatting."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


async def _run_case(
    case_id: str,
    script: list[tuple[str, str]],
    *,
    use_service_path: bool = False,
) -> dict[str, Any]:
    """Run one fixed group script and persist per-turn raw evidence."""

    _assert_guarded_database()
    artifact_directory = ARTIFACT_ROOT / case_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    identities = await _make_group_identities(
        f'public-scene-{case_id}',
        ['A', 'B', 'C'],
    )
    turn_results: list[dict[str, Any]] = []

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
                )

    final_state = turn_results[-1]
    _write_json(artifact_directory / 'parsed_state.json', {
        'case_id': case_id,
        'turns': turn_results,
        'final_response': final_state['response'],
    })
    return {
        'case_id': case_id,
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
            service_response = await brain_service.chat(
                request,
                background_tasks,
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
            state['character_profile'] = deepcopy(
                state['character_profile']
            )
            state['character_profile']['name'] = LIVE_CHARACTER_NAME
            state['character_name'] = LIVE_CHARACTER_NAME
            state['debug_modes']['no_remember'] = True
            state['active_turn_conversation_row_ids'] = [seeded_row_id]
            state['response_action'] = 'proceed'
            state['reason_to_respond'] = (
                'guarded public-scene cognition case'
            )
            state['cognition_claimed'] = True
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
