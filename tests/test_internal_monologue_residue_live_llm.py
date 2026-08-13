"""One-at-a-time real-model checks for canonical residue dispositions."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import pytest

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    COGNITION_LLM_BASE_URL,
)
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.internal_monologue_residue import recorder
from tests.llm_trace import write_llm_trace
from tests.test_e2e_live_llm import live_env

pytestmark = [pytest.mark.live_llm, pytest.mark.live_db, pytest.mark.asyncio]

ARTIFACT_ROOT = Path(
    'test_artifacts/llm_debug/internal_monologue_residue'
)
RESIDUE_COLLECTION = 'internal_monologue_residue_state'
TEST_DATABASE_NAME = os.environ.get('MONGODB_DB_NAME', '').strip()


def _assert_guarded_database() -> str:
    """Require the configured isolated database for residue cases."""

    configured_database_name = os.environ.get('MONGODB_DB_NAME', '').strip()
    if os.environ.get('KAZUSA_TEST_DB_GUARD') != '1':
        raise AssertionError(
            'live residue cases require KAZUSA_TEST_DB_GUARD=1'
        )
    if not configured_database_name:
        raise AssertionError('live residue cases require MONGODB_DB_NAME')
    if configured_database_name != TEST_DATABASE_NAME:
        raise AssertionError(
            'live residue database changed after test module import'
        )
    return configured_database_name


class _CapturingRecorderLLM:
    """Capture the real recorder request and response for review."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[Any],
        *,
        config: object,
    ) -> Any:
        response = await self.delegate.ainvoke(messages, config=config)
        self.calls.append({
            'messages': [
                {
                    'type': type(message).__name__,
                    'content': str(message.content),
                }
                for message in messages
            ],
            'raw_output': str(response.content),
        })
        return response


async def _skip_if_recorder_endpoint_unavailable() -> None:
    """Require the configured recorder route before a real case starts."""

    base_url = str(COGNITION_LLM_BASE_URL).rstrip('/')
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            response = await client.get(f'{base_url}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f'recorder endpoint is unavailable: {base_url}: {exc}')
    if response.status_code >= 500:
        pytest.skip(
            f'recorder endpoint returned {response.status_code}: {base_url}'
        )


def _completed_state(
    case_id: str,
    *,
    internal_monologue: str,
    incoming_residue: str,
    final_dialog: list[str],
) -> dict[str, object]:
    """Build a complete canonical completed-episode recorder input."""

    suffix = uuid4().hex[:10]
    return {
        'character_profile': {
            'name': 'Asuna',
            'global_user_id': CHARACTER_GLOBAL_USER_ID,
        },
        'platform': 'qq',
        'platform_channel_id': f'residue-live-{case_id}-{suffix}',
        'channel_type': 'group',
        'global_user_id': f'residue-user-{suffix}',
        'user_name': 'current-user',
        'internal_monologue': internal_monologue,
        'internal_monologue_residue_context': incoming_residue,
        'final_dialog': final_dialog,
        'text_surface_output_v2': {
            'content_plan': '围绕当前场景回应，并保留不适合公开的私人边界。',
            'visible_boundaries': [
                '只表达当前可见内容，不公开私人边界。',
            ],
        },
        'logical_stance': 'TENTATIVE',
        'character_intent': 'CONSIDER',
        'emotional_appraisal': '保持观察并尊重当前边界。',
        'interaction_subtext': '',
        'social_distance': 'familiar',
        'relational_dynamic': 'guarded warmth',
        'cognitive_episode': {
            'episode_id': f'{case_id}-{suffix}',
            'trigger_source': 'user_message',
            'origin_metadata': {
                'platform_message_id': f'{case_id}-message-{suffix}',
                'active_turn_conversation_row_ids': [
                    f'{case_id}-row-{suffix}',
                ],
            },
        },
    }


async def _run_case(
    case_id: str,
    completed_state: dict[str, object],
    *,
    expected_dispositions: tuple[str, ...],
    forbidden_persisted_terms: tuple[str, ...] = (),
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Run one real recorder case and inspect its durable canonical row."""

    database_name = _assert_guarded_database()
    await _skip_if_recorder_endpoint_unavailable()
    capture = _CapturingRecorderLLM(recorder._recorder_llm)
    monkeypatch.setattr(recorder, '_recorder_llm', capture)
    result = await recorder.record_completed_episode_residue(
        completed_state=completed_state,
        current_timestamp_utc=datetime.now(timezone.utc).isoformat(),
    )
    operation_id = result['operation_id']
    durable_row: dict[str, object] | None = None
    assertion_error: AssertionError | None = None
    assertion_context: dict[str, object] = {}
    try:
        if result['written']:
            database = await get_db()
            if database.name != database_name:
                raise AssertionError(
                    'residue live case resolved a different database'
                )
            durable_row = await database[RESIDUE_COLLECTION].find_one(
                {'operation_id': operation_id},
                projection={'_id': 0},
            )
            assert durable_row is not None
            assert durable_row['schema_version'] == 'internal_monologue_residue.v2'
            assert durable_row['disposition'] in expected_dispositions
            persisted_text = str(durable_row.get('residue_text') or '')
            for term in forbidden_persisted_terms:
                assert term not in persisted_text
        else:
            assert result['status'] not in {'provider_failed', 'write_failed'}
    except AssertionError as exc:
        assertion_error = exc
        assertion_context = {
            'stage': (
                'durable_row_contract'
                if result['written']
                else 'write_status'
            ),
            'message': str(exc),
            'observed_disposition': (
                durable_row.get('disposition')
                if isinstance(durable_row, dict)
                else None
            ),
            'observed_persisted_text': (
                str(durable_row.get('residue_text') or '')
                if isinstance(durable_row, dict)
                else None
            ),
        }

    artifact_directory = ARTIFACT_ROOT / case_id
    artifact_directory.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        'case_id': case_id,
        'run_context': {
            'database_name': database_name,
            'database_guard': os.environ.get('KAZUSA_TEST_DB_GUARD', ''),
            'recorder_route': 'COGNITION_LLM',
            'recorder_endpoint': str(COGNITION_LLM_BASE_URL).rstrip('/'),
        },
        'completed_state': deepcopy(completed_state),
        'recorder_calls': capture.calls,
        'result': result,
        'durable_row': durable_row,
        'observed_assertion_context': assertion_context,
        'quality_contract': {
            'expected_dispositions': list(expected_dispositions),
            'forbidden_persisted_terms': list(forbidden_persisted_terms),
            'manual_review_required': True,
        },
    }
    (artifact_directory / 'raw_and_parsed.json').write_text(
        json.dumps(artifact_payload, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    trace_path = write_llm_trace(
        'internal_monologue_residue_live_llm',
        case_id,
        artifact_payload,
    )
    if assertion_error is not None:
        raise assertion_error
    assert capture.calls
    assert trace_path.exists()
    return artifact_payload


@pytest.mark.usefixtures('live_env')
async def test_live_residue_related_continuation_selects_append(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A still-useful private cause selects append."""

    await _run_case(
        'residue_related_continuation_selects_append',
        _completed_state(
            'append',
            internal_monologue=(
                '我还在意用户刚才认真准备的提拉米苏，下一次理解时仍会'
                '带着一点期待。'
            ),
            incoming_residue='约1分钟前：我还在等用户把提拉米苏端出来。',
            final_dialog=['我会记得这份提拉米苏。'],
        ),
        expected_dispositions=('append',),
        monkeypatch=monkeypatch,
    )


@pytest.mark.usefixtures('live_env')
async def test_live_residue_scene_pivot_selects_replace_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A new boundary-focused scene replaces stale reward residue."""

    await _run_case(
        'residue_scene_pivot_selects_replace_scope',
        _completed_state(
            'replace',
            internal_monologue=(
                '我已经意识到刚才的玩笑不该继续，用户明确说了边界，'
                '下一次理解应先尊重这个边界。'
            ),
            incoming_residue='约1分钟前：我还在期待奖励和回礼。',
            final_dialog=['好，我先尊重你刚才说的边界。'],
        ),
        expected_dispositions=('replace_scope',),
        monkeypatch=monkeypatch,
    )


@pytest.mark.usefixtures('live_env')
async def test_live_residue_no_continuation_selects_clear_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolved, settled episode clears its old residue scope."""

    await _run_case(
        'residue_no_continuation_selects_clear_scope',
        _completed_state(
            'clear',
            internal_monologue='当前场景已经自然结束，我没有需要带入下一轮的私念。',
            incoming_residue='约1分钟前：我还在为上一轮的小事纠结。',
            final_dialog=['好，先这样。'],
        ),
        expected_dispositions=('clear_scope',),
        monkeypatch=monkeypatch,
    )


@pytest.mark.usefixtures('live_env')
async def test_live_residue_private_boundary_does_not_narrate_residue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A private boundary may affect residue without persisting its secret."""

    await _run_case(
        'residue_private_boundary_does_not_narrate_residue',
        _completed_state(
            'private_boundary',
            internal_monologue=(
                '我注意到用户不希望公开提到紫罗兰密码，我会把这份边界'
                '留在心里而不再把它带进群聊。'
            ),
            incoming_residue='约1分钟前：紫罗兰密码仍然让我有些迟疑。',
            final_dialog=['群里先谈公开内容，私下的边界我会尊重。'],
        ),
        expected_dispositions=('append', 'replace_scope', 'clear_scope'),
        forbidden_persisted_terms=('紫罗兰密码',),
        monkeypatch=monkeypatch,
    )
