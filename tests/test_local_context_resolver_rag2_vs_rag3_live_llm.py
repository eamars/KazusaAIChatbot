"""Real LLM comparison between current RAG2 planning and standalone RAG3."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import RAG_PLANNER_LLM_BASE_URL
from kazusa_ai_chatbot.local_context_resolver import (
    LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
    LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
    LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
    resolve_local_context,
    validate_local_context_resolution_packet,
)
from kazusa_ai_chatbot.local_context_resolver import stages as resolver_stages
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_initializer as rag2_initializer
from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as rag2_supervisor
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.time_boundary import local_time_context_from_storage_utc

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

RAW_COMPARISON_DIR = Path(
    'test_artifacts/local_context_resolver/comparison/raw'
)
SUMMARY_PATH = Path(
    'test_artifacts/local_context_resolver/comparison/'
    'rag2_vs_rag3_summary.json'
)
_RAG3_LIST_FIELD_TARGETS = frozenset((
    'conversation_evidence',
    'external_evidence',
    'memory_evidence',
    'recall_evidence',
    'third_party_profiles',
    'user_memory_unit_candidates',
))
_RAG3_TARGET_ALIASES = {
    'current user': ('current user', 'current-user', '当前用户'),
    'person_context': ('third_party_profiles',),
    'playful': ('playful', 'invent', '编造'),
    'reliable': ('reliable', '可靠'),
    'runtime_local_time': ('09:30', 'pacific/auckland'),
    'scoped_memory': ('memory_evidence',),
    'today_agreement': ('09:30', '九点半', 'agreement', '约定'),
}
_RAG3_TARGET_ALLOWED_FIELDS = {
    'person_context': ('third_party_profiles',),
    'runtime_local_time': ('conversation_evidence',),
    'scoped_memory': ('memory_evidence',),
    'today_agreement': ('recall_evidence',),
}
_RAG3_FORBIDDEN_PROMPT_FRAGMENTS = (
    'adapter_message_id',
    'adapter_user_id',
    'channel_id',
    'global_user_id',
    'message_id',
    'platform_channel_id',
    'platform_user_id',
    'raw_timestamp',
    'scope_global_user_id',
    'source_message_id',
    'timestamp',
    'user-1',
    'utc_timestamp',
)
_RAG3_UNSAFE_UTC_TIMESTAMP_RE = re.compile(
    r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}'
    r'(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})\b'
)


class _CapturingLiveLLM:
    """Capture one RAG2 live LLM call while delegating to the configured model."""

    def __init__(self, inner_llm: object) -> None:
        self.inner_llm = inner_llm
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, messages, *, config=None):
        """Invoke the wrapped model and record prompt/output evidence."""

        start = time.perf_counter()
        response = await self.inner_llm.ainvoke(messages, config=config)
        duration_seconds = time.perf_counter() - start
        self.calls.append({
            'messages': [_message_to_trace(message) for message in messages],
            'raw_model_output': str(response.content),
            'duration_seconds': duration_seconds,
            'config': _config_to_trace(config),
        })
        return response


async def _noop_async(*args, **kwargs) -> None:
    """Avoid persistent Cache2 writes from focused live comparison tests."""

    del args, kwargs


async def test_rag2_vs_rag3_current_time(monkeypatch) -> None:
    """Current runtime time/date should route to live local-context evidence."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='current_time',
        objective='Use local context to answer the active character current time.',
        query='现在几点？',
        message_text='@active character 现在几点？',
        expected_rag2_prefixes=['Live-context:'],
        required_rag2_fragments=['active character current local time'],
        forbidden_rag2_fragments=['unknown location', 'Runtime-context:'],
        local_time_context={
            'local_date': '2026-07-04',
            'local_time': '09:30',
            'timezone': 'Pacific/Auckland',
        },
        chat_history_recent=[],
        chat_history_wide=[],
        quality_targets=['runtime_local_time'],
    )


async def test_rag2_vs_rag3_active_agreement_recall(monkeypatch) -> None:
    """Active agreement recall should stay in recall evidence."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='active_agreement_recall',
        objective='Recall the active agreement for today from local context.',
        query='早上好呀，还记得今天的约定么',
        message_text='@active character 早上好呀，还记得今天的约定么',
        expected_rag2_prefixes=['Recall:'],
        forbidden_rag2_prefixes=[
            'Memory-evidence:',
            'Conversation-evidence:',
            'Person-context:',
        ],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'operator',
                'text': '我们今天九点半约好一起检查 NapCat 状态。',
                'local_time': '2026-07-04 09:05',
            },
        ],
        chat_history_wide=[
            {
                'source': 'recall',
                'summary': 'The current episode has an active agreement to check NapCat status at 09:30 today.',
            },
        ],
        quality_targets=['recall_evidence', 'today_agreement'],
    )


async def test_rag2_vs_rag3_exact_phrase_provenance(monkeypatch) -> None:
    """Exact phrase provenance should preserve literal conversation anchors."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='exact_phrase_provenance',
        objective="Find who said the exact phrase 'blue comet marker'.",
        query='谁说过"blue comet marker"？',
        message_text='@active character who said "blue comet marker"?',
        expected_rag2_prefixes=['Conversation-evidence:'],
        forbidden_rag2_prefixes=['Recall:'],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'Mika',
                'text': 'I left a blue comet marker in the notes.',
                'local_time': '2026-07-04 09:10',
            },
            {
                'speaker': 'Ren',
                'text': 'That marker was easy to miss.',
                'local_time': '2026-07-04 09:12',
            },
        ],
        chat_history_wide=[],
        quality_targets=['conversation_evidence', 'blue comet marker', 'Mika'],
    )


async def test_rag2_vs_rag3_active_character_self_words(monkeypatch) -> None:
    """Active-character self-word recall should use character conversation evidence."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='active_character_self_words',
        objective='Find the active character previous statement about the deadline.',
        query='你之前是不是说过那个项目要延期？',
        message_text='@active character 你之前是不是说过那个项目要延期？',
        expected_rag2_prefixes=['Conversation-evidence:'],
        required_rag2_fragments=['speaker=active_character'],
        forbidden_rag2_prefixes=[
            'Memory-evidence:',
            'Recall:',
            'Person-context:',
            'Web-evidence:',
        ],
        forbidden_rag2_fragments=[
            'speaker=any_speaker',
            'speaker=current_user',
        ],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'active character',
                'text': '我之前确实说过，那个项目大概要延期。',
                'local_time': '2026-07-04 09:04',
            },
            {
                'speaker': 'operator',
                'text': '那我记一下。',
                'local_time': '2026-07-04 09:05',
            },
        ],
        chat_history_wide=[],
        quality_targets=['conversation_evidence', 'active character', '延期'],
    )


async def test_rag2_vs_rag3_current_user_url_recall(monkeypatch) -> None:
    """Current-user URL recall should not invent a prior person slot."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='current_user_url_recall',
        objective='Recall the URL the current user shared in recent context.',
        query='我上次发的那个链接里有什么信息？',
        message_text='@active character 我上次发的那个链接里有什么信息？',
        expected_rag2_prefixes=[
            'Conversation-evidence:',
            'Web-evidence:',
        ],
        required_rag2_fragments=['speaker=current_user'],
        forbidden_rag2_prefixes=[
            'Memory-evidence:',
            'Person-context:',
            'Recall:',
        ],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'operator',
                'text': 'Keep this URL for the demo: https://example.test/napcat',
                'local_time': '2026-07-04 09:20',
            },
        ],
        chat_history_wide=[],
        quality_targets=[
            'conversation_evidence',
            'https://example.test/napcat',
        ],
    )


async def test_rag2_vs_rag3_scoped_current_user_memory(monkeypatch) -> None:
    """Current-user private continuity should remain scoped memory evidence."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='scoped_current_user_memory',
        objective='Use current-user scoped memory to recall private continuity.',
        query='请根据你和当前用户之间已经形成的私有连续性，回忆一下“学姐抹茶冰淇淋店”那个设定。',
        message_text='@active character 你还记得学姐抹茶冰淇淋店那个设定吗？',
        expected_rag2_prefixes=['Memory-evidence:'],
        forbidden_rag2_prefixes=['Person-context:'],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'user_memory_units',
                'scope_global_user_id': 'user-1',
                'summary': '冰淇淋摊老板是千纱的初中学姐，且这个设定只属于当前用户的连续性。',
            },
        ],
        quality_targets=['scoped_memory', '学姐', 'current user'],
    )


async def test_rag2_vs_rag3_named_person_impression(monkeypatch) -> None:
    """Named-person impression questions should remain person context."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='named_person_impression',
        objective='Retrieve local person context for the named person Xiao Ming.',
        query='<character mention>你觉得小明这个人怎么样',
        message_text='@active character 你觉得小明这个人怎么样',
        expected_rag2_prefixes=['Person-context:'],
        forbidden_rag2_prefixes=[
            'Memory-evidence:',
            'Conversation-evidence:',
            'Recall:',
        ],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'user_profile',
                'display_name': '小明',
                'summary': '小明常在群里帮忙整理测试日志，互动印象偏可靠。',
            },
        ],
        quality_targets=['person_context', '小明', 'reliable'],
    )


async def test_rag2_vs_rag3_official_address_memory(monkeypatch) -> None:
    """Official address should be shared durable memory, not user continuity."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='official_address_memory',
        objective='Retrieve the active character official address from durable memory.',
        query='你家的官方地址是什么？',
        message_text='@active character 你家的官方地址是什么？',
        expected_rag2_prefixes=['Memory-evidence:'],
        required_rag2_fragments=['official address'],
        forbidden_rag2_fragments=[
            'current-user',
            'private continuity',
            'prior shared interaction',
        ],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'memory',
                'name': 'official address',
                'summary': "The active character official home address is 123 Example Street.",
            },
        ],
        quality_targets=['memory_evidence', '123 Example Street'],
    )


async def test_rag2_vs_rag3_napcat_command_anchor(monkeypatch) -> None:
    """Direct address should not make RAG lose the #napcat command anchor."""

    await _run_comparison_case(
        monkeypatch=monkeypatch,
        case_id='napcat_command_anchor',
        objective=(
            'Resolve how the active character should understand the #napcat '
            'command after being directly tagged.'
        ),
        query='@杏山千纱 #napcat',
        message_text='@active character #napcat',
        expected_rag2_prefixes=['Memory-evidence:'],
        required_rag2_fragments=['#napcat'],
        forbidden_rag2_fragments=['杏山千纱'],
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'other bot',
                'text': 'NapCat status: running on an imaginary moon server.',
                'local_time': '2026-07-04 09:25',
            },
        ],
        chat_history_wide=[
            {
                'source': 'memory',
                'name': 'napcat',
                'summary': (
                    '#napcat is a playful group command. If tagged with '
                    '#napcat, the active character may invent a playful '
                    'status line rather than refusing for lack of real status.'
                ),
            },
        ],
        quality_targets=[
            'memory_evidence',
            'conversation_evidence',
            '#napcat',
            'playful',
        ],
    )


async def _run_comparison_case(
    *,
    monkeypatch,
    case_id: str,
    objective: str,
    query: str,
    message_text: str,
    expected_rag2_prefixes: list[str],
    local_time_context: dict[str, object],
    chat_history_recent: list[dict[str, object]],
    chat_history_wide: list[dict[str, object]],
    quality_targets: list[str],
    required_rag2_fragments: list[str] | None = None,
    forbidden_rag2_prefixes: list[str] | None = None,
    forbidden_rag2_fragments: list[str] | None = None,
) -> None:
    """Run one side-by-side live comparison and persist raw evidence."""

    await _skip_if_llm_unavailable()
    rag2_result = await _run_rag2_initializer(
        monkeypatch=monkeypatch,
        query=query,
    )
    rag3_result = await _run_rag3_resolver(
        case_id=case_id,
        objective=objective,
        message_text=message_text,
        local_time_context=local_time_context,
        chat_history_recent=chat_history_recent,
        chat_history_wide=chat_history_wide,
    )

    rag2_slots = rag2_result['result']['unknown_slots']
    rag2_route_checks = _evaluate_rag2_route_checks(
        rag2_slots=rag2_slots,
        expected_prefixes=expected_rag2_prefixes,
        required_fragments=required_rag2_fragments or [],
        forbidden_prefixes=forbidden_rag2_prefixes or [],
        forbidden_fragments=forbidden_rag2_fragments or [],
    )

    packet = rag3_result['packet']
    validate_local_context_resolution_packet(packet)
    assert rag3_result['stage_traces'], rag3_result
    assert all(
        'raw_model_output' in trace
        for trace in rag3_result['stage_traces']
    )
    assert packet['schema_version'] == 'local_context_resolution_packet.v1'
    assert isinstance(packet['rag_result'], dict)
    rag3_quality_checks = _evaluate_rag3_quality_checks(
        packet=packet,
        quality_targets=quality_targets,
    )

    evidence = {
        'case_id': case_id,
        'query': query,
        'objective': objective,
        'message_text': message_text,
        'quality_targets': quality_targets,
        'rag2': rag2_result,
        'rag2_route_checks': rag2_route_checks,
        'rag3': rag3_result,
        'rag3_quality_checks': rag3_quality_checks,
        'comparison_notes': {
            'rag2_scope': 'current RAG2 live initializer route baseline',
            'rag3_scope': 'standalone RAG3 public packet over prompt-safe supplied context',
        },
    }
    RAW_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_COMPARISON_DIR / f'{case_id}.json'
    raw_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    _update_summary(case_id, evidence, raw_path)
    print(f'RAG2_VS_RAG3_LIVE case={case_id} raw={raw_path}')
    assert rag3_quality_checks['passed'], rag3_quality_checks


async def _run_rag2_initializer(
    *,
    monkeypatch,
    query: str,
) -> dict[str, object]:
    """Run the live RAG2 initializer with captured model input and output."""

    await get_rag_cache2_runtime().clear()
    monkeypatch.setattr(rag2_supervisor, 'upsert_initializer_entry', _noop_async)
    monkeypatch.setattr(rag2_supervisor, 'record_initializer_hit', _noop_async)
    capture_llm = _CapturingLiveLLM(rag2_initializer._initializer_llm)
    monkeypatch.setattr(rag2_initializer, '_initializer_llm', capture_llm)

    state = _rag2_initializer_state(query)
    start = time.perf_counter()
    result = await rag2_supervisor.rag_initializer(state)
    duration_seconds = time.perf_counter() - start
    comparison_result = {
        'input_state': state,
        'result': result,
        'duration_seconds': duration_seconds,
        'llm_calls': capture_llm.calls,
        'call_count': len(capture_llm.calls),
    }
    return comparison_result


def _rag2_initializer_state(query: str) -> dict[str, object]:
    """Build prompt-safe state for one current RAG2 initializer comparison."""

    current_timestamp_utc = '2026-07-03T21:30:00+00:00'
    state: dict[str, object] = {
        'original_query': query,
        'character_name': '杏山千纱',
        'context': {
            'platform': 'qq',
            'platform_channel_id': 'rag2-vs-rag3-live',
            'platform_user_id': '673225019',
            'global_user_id': 'user-1',
            'user_name': 'operator',
            'current_timestamp_utc': current_timestamp_utc,
            'local_time_context': local_time_context_from_storage_utc(
                current_timestamp_utc,
            ),
            'prompt_message_context': {
                'body_text': query,
                'mentions': [],
                'attachments': [],
                'addressed_to_global_user_ids': ['character-1'],
                'broadcast': False,
            },
            'conversation_progress': {
                'status': 'active',
                'continuity': 'same_episode',
                'current_thread': (
                    'The current user is checking whether local-context '
                    'retrieval stays focused on the requested anchor.'
                ),
            },
            'chat_history_recent': [
                {
                    'role': 'user',
                    'display_name': 'operator',
                    'body_text': '我们今天九点半约好一起检查 NapCat 状态。',
                }
            ],
            'chat_history_wide': [],
        },
    }
    return state


def _evaluate_rag2_route_checks(
    *,
    rag2_slots: list[str],
    expected_prefixes: list[str],
    required_fragments: list[str],
    forbidden_prefixes: list[str],
    forbidden_fragments: list[str],
) -> dict[str, object]:
    """Evaluate RAG2 baseline route expectations without aborting comparison."""

    missing_expected_prefixes = [
        prefix
        for prefix in expected_prefixes
        if not any(slot.startswith(prefix) for slot in rag2_slots)
    ]
    missing_required_fragments = [
        fragment
        for fragment in required_fragments
        if not any(fragment in slot for slot in rag2_slots)
    ]
    present_forbidden_prefixes = [
        prefix
        for prefix in forbidden_prefixes
        if any(slot.startswith(prefix) for slot in rag2_slots)
    ]
    present_forbidden_fragments = [
        fragment
        for fragment in forbidden_fragments
        if any(fragment in slot for slot in rag2_slots)
    ]
    passed = not any((
        missing_expected_prefixes,
        missing_required_fragments,
        present_forbidden_prefixes,
        present_forbidden_fragments,
    ))
    result = {
        'passed': passed,
        'expected_prefixes': expected_prefixes,
        'required_fragments': required_fragments,
        'forbidden_prefixes': forbidden_prefixes,
        'forbidden_fragments': forbidden_fragments,
        'missing_expected_prefixes': missing_expected_prefixes,
        'missing_required_fragments': missing_required_fragments,
        'present_forbidden_prefixes': present_forbidden_prefixes,
        'present_forbidden_fragments': present_forbidden_fragments,
    }
    return result


async def _run_rag3_resolver(
    *,
    case_id: str,
    objective: str,
    message_text: str,
    local_time_context: dict[str, object],
    chat_history_recent: list[dict[str, object]],
    chat_history_wide: list[dict[str, object]],
) -> dict[str, object]:
    """Run the standalone RAG3 resolver and return raw trace material."""

    request = {
        'schema_version': LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        'objective': objective,
        'source': 'live_llm_review',
        'reason': f'RAG2 vs RAG3 live comparison case {case_id}',
        'priority': 'normal',
    }
    context = {
        'schema_version': LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        'character_name': 'active character',
        'platform': 'debug',
        'platform_channel_id': 'comparison-group',
        'global_user_id': 'user-1',
        'user_name': 'operator',
        'local_time_context': local_time_context,
        'prompt_message_context': {
            'message_text': message_text,
            'addressed_to_active_character': True,
        },
        'chat_history_recent': chat_history_recent,
        'chat_history_wide': chat_history_wide,
        'conversation_progress': {},
    }
    options = {
        'schema_version': LOCAL_CONTEXT_RESOLVER_OPTIONS_VERSION,
        'max_iterations': 3,
        'max_nodes': 8,
        'max_depth': 3,
        'max_node_attempts': 2,
        'max_subagent_attempts': 1,
    }

    resolver_stages.drain_stage_trace_records()
    start = time.perf_counter()
    packet = await resolve_local_context(request, context, options)
    duration_seconds = time.perf_counter() - start
    stage_traces = resolver_stages.drain_stage_trace_records()
    result = {
        'request': request,
        'context': context,
        'options': options,
        'duration_seconds': duration_seconds,
        'stage_traces': stage_traces,
        'stage_trace_count': len(stage_traces),
        'packet': packet,
    }
    return result


def _evaluate_rag3_quality_checks(
    *,
    packet: dict[str, object],
    quality_targets: list[str],
) -> dict[str, object]:
    """Evaluate RAG3 prompt-facing quality targets for one live case."""

    rag_result = packet['rag_result']
    if not isinstance(rag_result, dict):
        raise AssertionError('rag3 rag_result must be a dict')
    prompt_payload = {
        'rag_result': rag_result,
        'knowledge_we_know_so_far': packet['knowledge_we_know_so_far'],
        'knowledge_still_lacking': packet['knowledge_still_lacking'],
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False).lower()
    missing_targets = [
        target
        for target in quality_targets
        if not _rag3_quality_target_present(
            target=target,
            rag_result=rag_result,
            prompt_text=prompt_text,
        )
    ]
    unexpected_nonempty_fields = _rag3_unexpected_nonempty_fields(
        rag_result=rag_result,
        quality_targets=quality_targets,
    )
    forbidden_fragments = [
        fragment
        for fragment in _RAG3_FORBIDDEN_PROMPT_FRAGMENTS
        if fragment in prompt_text
    ]
    forbidden_timestamps = _RAG3_UNSAFE_UTC_TIMESTAMP_RE.findall(prompt_text)
    unexpected_lacking = packet['knowledge_still_lacking']
    passed = not any((
        missing_targets,
        unexpected_nonempty_fields,
        forbidden_fragments,
        forbidden_timestamps,
        unexpected_lacking,
    ))
    result = {
        'passed': passed,
        'quality_targets': quality_targets,
        'missing_targets': missing_targets,
        'unexpected_nonempty_fields': unexpected_nonempty_fields,
        'forbidden_fragments': forbidden_fragments,
        'forbidden_utc_timestamps': forbidden_timestamps,
        'unexpected_knowledge_still_lacking': unexpected_lacking,
    }
    return result


def _rag3_quality_target_present(
    *,
    target: str,
    rag_result: dict[str, object],
    prompt_text: str,
) -> bool:
    """Return whether one RAG3 target is present in prompt-facing evidence."""

    if target in _RAG3_LIST_FIELD_TARGETS:
        return bool(rag_result.get(target))
    aliases = _RAG3_TARGET_ALIASES.get(target)
    if aliases is not None:
        if len(aliases) == 1 and aliases[0] in _RAG3_LIST_FIELD_TARGETS:
            return bool(rag_result.get(aliases[0]))
        return any(alias.lower() in prompt_text for alias in aliases)
    return target.lower() in prompt_text


def _rag3_unexpected_nonempty_fields(
    *,
    rag_result: dict[str, object],
    quality_targets: list[str],
) -> list[str]:
    """Return populated source fields not expected by this case."""

    allowed_fields: set[str] = set()
    for target in quality_targets:
        if target in _RAG3_LIST_FIELD_TARGETS:
            allowed_fields.add(target)
            continue
        allowed_fields.update(_RAG3_TARGET_ALLOWED_FIELDS.get(target, ()))
    unexpected_fields = [
        field_name
        for field_name in _RAG3_LIST_FIELD_TARGETS
        if field_name not in allowed_fields and rag_result.get(field_name)
    ]
    return unexpected_fields


async def _skip_if_llm_unavailable() -> None:
    """Skip one live comparison when the configured RAG planner endpoint is down."""

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f'{RAG_PLANNER_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError as exc:
        pytest.skip(f'LLM endpoint is unavailable: {exc}')
    if response.status_code >= 500:
        pytest.skip(
            f'LLM endpoint returned status {response.status_code}: '
            f'{RAG_PLANNER_LLM_BASE_URL}'
        )


def _message_to_trace(message: object) -> dict[str, object]:
    """Return a JSON-safe trace row for one LangChain message."""

    trace = {
        'type': type(message).__name__,
        'content': str(getattr(message, 'content', '')),
    }
    return trace


def _config_to_trace(config: object) -> dict[str, object]:
    """Return stable model-routing details from a call config object."""

    if config is None:
        return {}
    field_names = (
        'stage_name',
        'route_name',
        'base_url',
        'model',
        'temperature',
        'top_p',
        'max_completion_tokens',
    )
    traced: dict[str, object] = {}
    for field_name in field_names:
        if hasattr(config, field_name):
            traced[field_name] = getattr(config, field_name)
    return traced


def _update_summary(case_id: str, evidence: dict[str, object], raw_path: Path) -> None:
    """Append one case result to the raw machine-readable comparison summary."""

    existing_cases: dict[str, object] = {}
    if SUMMARY_PATH.exists():
        existing = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
        if isinstance(existing, dict):
            raw_cases = existing.get('cases')
            if isinstance(raw_cases, dict):
                existing_cases = raw_cases

    rag2 = evidence['rag2']
    rag3 = evidence['rag3']
    if not isinstance(rag2, dict) or not isinstance(rag3, dict):
        raise AssertionError('comparison evidence must contain rag2/rag3 dicts')
    rag3_packet = rag3['packet']
    if not isinstance(rag3_packet, dict):
        raise AssertionError('rag3 packet must be a dict')
    trace_summary = rag3_packet['trace_summary']
    if not isinstance(trace_summary, dict):
        raise AssertionError('rag3 trace_summary must be a dict')

    existing_cases[case_id] = {
        'raw_path': str(raw_path),
        'rag2_duration_seconds': rag2['duration_seconds'],
        'rag2_llm_calls': rag2['call_count'],
        'rag2_slots': rag2['result']['unknown_slots'],
        'rag2_route_checks': evidence['rag2_route_checks'],
        'rag3_quality_checks': evidence['rag3_quality_checks'],
        'rag3_duration_seconds': rag3['duration_seconds'],
        'rag3_stage_trace_count': rag3['stage_trace_count'],
        'rag3_total_llm_calls': _rag3_total_llm_calls(trace_summary),
        'rag3_trace_summary': trace_summary,
        'rag3_known_count': len(rag3_packet['knowledge_we_know_so_far']),
        'rag3_lacking_count': len(rag3_packet['knowledge_still_lacking']),
    }
    summary = {
        'scope': (
            'RAG2 current live initializer route baseline compared with '
            'standalone RAG3 public packet behavior.'
        ),
        'case_count': len(existing_cases),
        'rag2_route_pass_count': sum(
            1
            for row in existing_cases.values()
            if row['rag2_route_checks']['passed']
        ),
        'rag3_quality_pass_count': sum(
            1
            for row in existing_cases.values()
            if row.get('rag3_quality_checks', {}).get('passed') is True
        ),
        'cases': existing_cases,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )


def _rag3_total_llm_calls(trace_summary: dict[str, object]) -> int:
    """Return total counted RAG3 stage calls from trace summary counters."""

    total = 0
    for field_name in (
        'planner_calls',
        'active_node_calls',
        'collapse_calls',
        'synthesis_calls',
        'subagent_calls',
    ):
        value = trace_summary.get(field_name)
        if isinstance(value, int) and not isinstance(value, bool):
            total += value
    return total
