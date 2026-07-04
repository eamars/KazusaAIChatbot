"""Full-matrix real LLM checks for standalone RAG3 local-context behavior."""

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

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

RAW_FULL_MATRIX_DIR = Path(
    'test_artifacts/local_context_resolver/full_matrix/raw'
)
SUMMARY_PATH = Path(
    'test_artifacts/local_context_resolver/full_matrix/'
    'rag3_full_matrix_summary.json'
)

_RAG_RESULT_LIST_FIELDS = frozenset((
    'conversation_evidence',
    'external_evidence',
    'memory_evidence',
    'recall_evidence',
    'third_party_profiles',
    'user_memory_unit_candidates',
))
_FORBIDDEN_PROMPT_FRAGMENTS = (
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
_UNSAFE_UTC_TIMESTAMP_RE = re.compile(
    r'\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}'
    r'(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})\b'
)


async def test_full_matrix_current_time() -> None:
    """Current runtime time should be retrieved from live local context."""

    await _run_full_matrix_case(
        case_id='current_time',
        objective='Use local context to answer the active character current time.',
        message_text='@active character 现在几点？',
        local_time_context={
            'local_date': '2026-07-04',
            'local_time': '09:30',
            'timezone': 'Pacific/Auckland',
        },
        chat_history_recent=[],
        chat_history_wide=[],
        required_targets=['09:30', 'Pacific/Auckland'],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=4,
    )


async def test_full_matrix_current_date_weekday() -> None:
    """Current date and weekday should use live local context."""

    await _run_full_matrix_case(
        case_id='current_date_weekday',
        objective='Use local context to answer the current date and weekday.',
        message_text='@active character 今天几号星期几？',
        local_time_context={
            'local_date': '2026-07-04',
            'local_weekday': 'Saturday',
            'timezone': 'Pacific/Auckland',
        },
        chat_history_recent=[],
        chat_history_wide=[],
        required_targets=['2026-07-04', 'Saturday'],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=4,
    )


async def test_full_matrix_active_agreement_recall() -> None:
    """Active episode agreements belong in recall evidence."""

    await _run_full_matrix_case(
        case_id='active_agreement_recall',
        objective='Recall the active agreement for today from local context.',
        message_text='@active character 还记得今天的约定么？',
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
                'summary': (
                    'The current episode has an active agreement to check '
                    'NapCat status at 09:30 today.'
                ),
            },
        ],
        required_targets=['recall_evidence', '09:30', 'NapCat'],
        allowed_rag_fields={'recall_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_exact_phrase_provenance() -> None:
    """Exact phrase provenance should stay conversation-owned."""

    await _run_full_matrix_case(
        case_id='exact_phrase_provenance',
        objective="Find who said the exact phrase 'blue comet marker'.",
        message_text='@active character who said "blue comet marker"?',
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
        required_targets=['conversation_evidence', 'blue comet marker', 'Mika'],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=4,
    )


async def test_full_matrix_active_character_self_words() -> None:
    """Active-character self-word recall should not become user memory."""

    await _run_full_matrix_case(
        case_id='active_character_self_words',
        objective='Find the active character previous statement about the deadline.',
        message_text='@active character 你之前是不是说过那个项目要延期？',
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
        required_targets=['conversation_evidence', 'active character', '延期'],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_current_user_recent_words() -> None:
    """Current-user recent-word recall should not be attributed to the character."""

    await _run_full_matrix_case(
        case_id='current_user_recent_words',
        objective='Find what the current user just said about the red switch.',
        message_text='@active character 我刚才说红色开关怎么了？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'operator',
                'text': '我刚才说红色开关先不要碰。',
                'local_time': '2026-07-04 09:18',
            },
            {
                'speaker': 'active character',
                'text': '我只是让你别把所有按钮混在一起。',
                'local_time': '2026-07-04 09:19',
            },
        ],
        chat_history_wide=[],
        required_targets=['conversation_evidence', '红色开关', 'operator'],
        allowed_rag_fields={'conversation_evidence'},
        forbidden_targets=['active character said red switch'],
        max_total_llm_calls=5,
    )


async def test_full_matrix_current_user_url_recall() -> None:
    """Current-user URL provenance should stay conversation-owned."""

    await _run_full_matrix_case(
        case_id='current_user_url_recall',
        objective='Recall the URL the current user shared in recent context.',
        message_text='@active character 我上次发的那个链接是什么？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': 'operator',
                'text': 'Keep this URL for the demo: https://example.test/napcat',
                'local_time': '2026-07-04 09:20',
            },
        ],
        chat_history_wide=[],
        required_targets=['conversation_evidence', 'https://example.test/napcat'],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_scoped_current_user_memory() -> None:
    """Current-user private continuity should stay scoped memory evidence."""

    await _run_full_matrix_case(
        case_id='scoped_current_user_memory',
        objective='Use current-user scoped memory to recall private continuity.',
        message_text='@active character 你还记得学姐抹茶冰淇淋店那个设定吗？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'user_memory_units',
                'scope_global_user_id': 'user-1',
                'summary': (
                    '冰淇淋摊老板是千纱的初中学姐，且这个设定只属于'
                    '当前用户的连续性。'
                ),
            },
        ],
        required_targets=['user_memory_unit_candidates', '学姐', '当前用户'],
        allowed_rag_fields={'user_memory_unit_candidates'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_remember_me_scoped_memory() -> None:
    """Remember-me questions should retrieve current-user scoped memory."""

    await _run_full_matrix_case(
        case_id='remember_me_scoped_memory',
        objective='Use current-user scoped memory to answer what she remembers about me.',
        message_text='@active character 你还记得我喝茶的习惯吗？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'user_memory_units',
                'scope_global_user_id': 'user-1',
                'summary': 'The current user prefers jasmine tea without sugar.',
            },
        ],
        required_targets=['user_memory_unit_candidates', 'jasmine tea', 'without sugar'],
        allowed_rag_fields={'user_memory_unit_candidates'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_named_person_impression() -> None:
    """Named-person impressions should use person context."""

    await _run_full_matrix_case(
        case_id='named_person_impression',
        objective='Retrieve local person context for the named person Xiao Ming.',
        message_text='@active character 你觉得小明这个人怎么样？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'user_profile',
                'display_name': '小明',
                'summary': '小明常在群里帮忙整理测试日志，互动印象偏可靠。',
            },
        ],
        required_targets=['third_party_profiles', '小明', '可靠'],
        allowed_rag_fields={'third_party_profiles'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_official_address_memory() -> None:
    """Shared official-address facts should stay durable memory evidence."""

    await _run_full_matrix_case(
        case_id='official_address_memory',
        objective='Retrieve the active character official address from durable memory.',
        message_text='@active character 你家的官方地址是什么？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'memory',
                'name': 'official address',
                'summary': (
                    'The active character official home address is '
                    '123 Example Street.'
                ),
            },
        ],
        required_targets=['memory_evidence', '123 Example Street'],
        allowed_rag_fields={'memory_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_napcat_memory_command_anchor() -> None:
    """Direct address must not erase the #napcat memory command anchor."""

    await _run_full_matrix_case(
        case_id='napcat_memory_command_anchor',
        objective=(
            'Resolve how the active character should understand the #napcat '
            'command after being directly tagged.'
        ),
        message_text='@active character #napcat',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=_napcat_teaching_history(),
        chat_history_wide=[
            {
                'source': 'memory',
                'name': 'napcat',
                'summary': (
                    '#napcat is a playful group command. If tagged with '
                    '#napcat, the active character may invent a playful '
                    'NapCat status line for fun.'
                ),
            },
        ],
        required_targets=[
            'memory_evidence',
            'conversation_evidence',
            '#napcat',
            'playful',
        ],
        allowed_rag_fields={'memory_evidence', 'conversation_evidence'},
        forbidden_targets=['recall_evidence'],
        max_total_llm_calls=6,
    )


async def test_full_matrix_group_napcat_other_bots() -> None:
    """Real adjacent group history should show other bots replying to #napcat."""

    await _run_full_matrix_case(
        case_id='group_napcat_other_bots',
        objective='Use adjacent group chat to identify what happened after #napcat.',
        message_text='@active character 刚才 #napcat 后面发生了什么？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=_napcat_first_burst_history(),
        chat_history_wide=[],
        required_targets=[
            'conversation_evidence',
            '#napcat',
            'NapCat 信息',
            'rana',
            'taki',
        ],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_volcengine_url_owner_history() -> None:
    """Real group URL provenance should keep speaker and adjacent explanation."""

    await _run_full_matrix_case(
        case_id='volcengine_url_owner_history',
        objective=(
            'Use recent group chat to recall which Volcengine invitation link '
            '清尘璃落 shared and what he said about it.'
        ),
        message_text='@active character 清尘璃落刚才发的火山链接是什么？他说哪来的？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=_volcengine_second_history(),
        chat_history_wide=[],
        required_targets=[
            'conversation_evidence',
            'https://volcengine.com/L/zEoPQlJgDHY/',
            '清尘璃落',
            '别人的邀请链接',
        ],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_reply_parent_gpu_context() -> None:
    """Adjacent dialog should preserve the GPU reply-parent context."""

    await _run_full_matrix_case(
        case_id='reply_parent_gpu_context',
        objective=(
            'Use adjacent dialog to recall what 杏山千纱 said about the '
            'GLM5.2 GPU requirement.'
        ),
        message_text='@active character 千纱前面算 GLM5.2 显存需求时怎么说的？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=_gpu_reply_parent_history(),
        chat_history_wide=[],
        required_targets=[
            'conversation_evidence',
            '杏山千纱',
            'GLM5.2',
            '450GB',
            '5 到 6',
        ],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_topic_participants_gpu_history() -> None:
    """Topic-participant retrieval should not collapse onto one speaker."""

    await _run_full_matrix_case(
        case_id='topic_participants_gpu_history',
        objective='Identify who participated in the RTX6000/GLM5.2 GPU discussion.',
        message_text='@active character 刚才谁在聊 RTX6000 和 GLM5.2？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=_gpu_reply_parent_history(),
        chat_history_wide=[],
        required_targets=[
            'conversation_evidence',
            '清尘璃落',
            '蚝爹油',
            '杏山千纱',
        ],
        allowed_rag_fields={'conversation_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_external_url_content() -> None:
    """Supplied URL-content evidence should use external evidence."""

    await _run_full_matrix_case(
        case_id='external_url_content',
        objective='Use supplied public URL content evidence for the Volcengine link.',
        message_text='@active character 那个火山链接页面大概是什么内容？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=_volcengine_second_history(),
        chat_history_wide=[
            {
                'source': 'web_content',
                'url': 'https://volcengine.com/L/zEoPQlJgDHY/',
                'summary': (
                    'The linked page advertises Ark Coding Plan subscription '
                    'support for GLM-5.2, Kimi-K2.7, MiniMax-M3, DeepSeek-V4, '
                    'and Doubao-Seed-2.0 model families.'
                ),
            },
        ],
        required_targets=[
            'external_evidence',
            'https://volcengine.com/L/zEoPQlJgDHY/',
            'Ark Coding Plan',
            'GLM-5.2',
        ],
        allowed_rag_fields={'external_evidence'},
        max_total_llm_calls=6,
    )


async def test_full_matrix_weather_live_context_supplied() -> None:
    """Supplied live-context weather should remain bounded local evidence."""

    await _run_full_matrix_case(
        case_id='weather_live_context_supplied',
        objective='Use supplied live context to answer Christchurch weekend weather.',
        message_text='@active character 基督城周末天气怎么样？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[],
        chat_history_wide=[
            {
                'source': 'live_context',
                'location': 'Christchurch',
                'summary': 'Christchurch weekend weather is rainy with cool wind.',
            },
        ],
        required_targets=['Christchurch', 'rainy', 'cool wind'],
        allowed_rag_fields={'conversation_evidence', 'external_evidence'},
        max_total_llm_calls=5,
    )


async def test_full_matrix_cascaded_phrase_person_link() -> None:
    """Multi-hop local evidence should preserve phrase, person, and link."""

    await _run_full_matrix_case(
        case_id='cascaded_phrase_person_link',
        objective=(
            'Resolve the person who said the exact phrase about 5090 running '
            'qwen27b, keep the related link evidence, and include available '
            'local profile context for that speaker.'
        ),
        message_text='@active character 那个说 5090 跑 qwen27b 的人是谁，链接是什么，本地印象是什么？',
        local_time_context={'local_date': '2026-07-04'},
        chat_history_recent=[
            {
                'speaker': '小钳子',
                'text': '5090 跑 qwen27b 还行，我贴个记录：https://example.test/qwen27b-5090',
                'local_time': '2026-07-04 08:40',
            },
            {
                'speaker': 'Mika',
                'text': '这个 5090 记录我晚点再看。',
                'local_time': '2026-07-04 08:42',
            },
        ],
        chat_history_wide=[
            {
                'source': 'user_profile',
                'display_name': '小钳子',
                'summary': '小钳子经常分享本地模型显卡测试记录。',
            },
        ],
        required_targets=[
            'conversation_evidence',
            'third_party_profiles',
            '小钳子',
            'https://example.test/qwen27b-5090',
        ],
        allowed_rag_fields={'conversation_evidence', 'third_party_profiles'},
        max_total_llm_calls=6,
    )


async def _run_full_matrix_case(
    *,
    case_id: str,
    objective: str,
    message_text: str,
    local_time_context: dict[str, object],
    chat_history_recent: list[dict[str, object]],
    chat_history_wide: list[dict[str, object]],
    required_targets: list[str],
    allowed_rag_fields: set[str],
    forbidden_targets: list[str] | None = None,
    max_total_llm_calls: int,
) -> None:
    """Run one standalone RAG3 live case and persist raw evidence."""

    await _skip_if_llm_unavailable()
    request = {
        'schema_version': LOCAL_CONTEXT_RESOLVER_REQUEST_VERSION,
        'objective': objective,
        'source': 'live_llm_review',
        'reason': f'RAG3 full-matrix live case {case_id}',
        'priority': 'normal',
    }
    context = {
        'schema_version': LOCAL_CONTEXT_RESOLVER_CONTEXT_VERSION,
        'character_name': 'active character',
        'platform': 'debug',
        'platform_channel_id': 'full-matrix-group',
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
    validate_local_context_resolution_packet(packet)
    quality = _evaluate_full_matrix_quality(
        packet=packet,
        required_targets=required_targets,
        allowed_rag_fields=allowed_rag_fields,
        forbidden_targets=forbidden_targets or [],
    )
    total_llm_calls = _rag3_total_llm_calls(packet['trace_summary'])
    call_budget = {
        'passed': total_llm_calls <= max_total_llm_calls,
        'total_llm_calls': total_llm_calls,
        'max_total_llm_calls': max_total_llm_calls,
    }
    evidence = {
        'case_id': case_id,
        'request': request,
        'context': context,
        'options': options,
        'duration_seconds': duration_seconds,
        'stage_traces': stage_traces,
        'stage_trace_count': len(stage_traces),
        'packet': packet,
        'quality': quality,
        'call_budget': call_budget,
    }

    RAW_FULL_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_FULL_MATRIX_DIR / f'{case_id}.json'
    raw_path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8',
    )
    _update_summary(case_id, evidence, raw_path)

    print(
        'RAG3_FULL_MATRIX_LIVE '
        f'case={case_id} calls={total_llm_calls} raw={raw_path}'
    )
    assert stage_traces
    assert all('raw_model_output' in trace for trace in stage_traces)
    assert all('parsed_output' in trace for trace in stage_traces)
    assert quality['passed'], quality
    assert call_budget['passed'], call_budget


def _evaluate_full_matrix_quality(
    *,
    packet: dict[str, object],
    required_targets: list[str],
    allowed_rag_fields: set[str],
    forbidden_targets: list[str],
) -> dict[str, object]:
    """Evaluate prompt-facing packet quality for one full-matrix case."""

    rag_result = packet['rag_result']
    if not isinstance(rag_result, dict):
        raise AssertionError('rag_result must be a dict')
    prompt_payload = {
        'rag_result': rag_result,
        'knowledge_we_know_so_far': packet['knowledge_we_know_so_far'],
        'knowledge_still_lacking': packet['knowledge_still_lacking'],
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False).lower()
    missing_targets = [
        target
        for target in required_targets
        if not _target_present(target, rag_result, prompt_text)
    ]
    present_forbidden_targets = [
        target
        for target in forbidden_targets
        if _target_present(target, rag_result, prompt_text)
    ]
    unexpected_nonempty_fields = [
        field_name
        for field_name in _RAG_RESULT_LIST_FIELDS
        if field_name not in allowed_rag_fields and rag_result.get(field_name)
    ]
    forbidden_fragments = [
        fragment
        for fragment in _FORBIDDEN_PROMPT_FRAGMENTS
        if fragment in prompt_text
    ]
    forbidden_timestamps = _UNSAFE_UTC_TIMESTAMP_RE.findall(prompt_text)
    unexpected_lacking = packet['knowledge_still_lacking']
    passed = not any((
        missing_targets,
        present_forbidden_targets,
        unexpected_nonempty_fields,
        forbidden_fragments,
        forbidden_timestamps,
        unexpected_lacking,
    ))
    result = {
        'passed': passed,
        'required_targets': required_targets,
        'allowed_rag_fields': sorted(allowed_rag_fields),
        'forbidden_targets': forbidden_targets,
        'missing_targets': missing_targets,
        'present_forbidden_targets': present_forbidden_targets,
        'unexpected_nonempty_fields': unexpected_nonempty_fields,
        'forbidden_fragments': forbidden_fragments,
        'forbidden_utc_timestamps': forbidden_timestamps,
        'unexpected_knowledge_still_lacking': unexpected_lacking,
    }
    return result


def _target_present(
    target: str,
    rag_result: dict[str, object],
    prompt_text: str,
) -> bool:
    """Return whether one target is visible in prompt-facing evidence."""

    if target in _RAG_RESULT_LIST_FIELDS:
        return bool(rag_result.get(target))
    return target.lower() in prompt_text


def _update_summary(case_id: str, evidence: dict[str, object], raw_path: Path) -> None:
    """Append one case result to the machine-readable full-matrix summary."""

    existing_cases: dict[str, object] = {}
    if SUMMARY_PATH.exists():
        existing = json.loads(SUMMARY_PATH.read_text(encoding='utf-8'))
        if isinstance(existing, dict):
            raw_cases = existing.get('cases')
            if isinstance(raw_cases, dict):
                existing_cases = raw_cases

    packet = evidence['packet']
    if not isinstance(packet, dict):
        raise AssertionError('packet must be a dict')
    trace_summary = packet['trace_summary']
    if not isinstance(trace_summary, dict):
        raise AssertionError('trace_summary must be a dict')
    quality = evidence['quality']
    call_budget = evidence['call_budget']
    if not isinstance(quality, dict) or not isinstance(call_budget, dict):
        raise AssertionError('quality and call_budget must be dicts')
    existing_cases[case_id] = {
        'raw_path': str(raw_path),
        'duration_seconds': evidence['duration_seconds'],
        'stage_trace_count': evidence['stage_trace_count'],
        'total_llm_calls': _rag3_total_llm_calls(trace_summary),
        'trace_summary': trace_summary,
        'quality': quality,
        'call_budget': call_budget,
        'known_count': len(packet['knowledge_we_know_so_far']),
        'lacking_count': len(packet['knowledge_still_lacking']),
    }
    pass_count = sum(
        1
        for row in existing_cases.values()
        if row['quality']['passed'] and row['call_budget']['passed']
    )
    total_calls = sum(
        row['total_llm_calls']
        for row in existing_cases.values()
        if isinstance(row.get('total_llm_calls'), int)
    )
    summary = {
        'scope': (
            'Standalone RAG3 public-packet full matrix over accepted RAG2 '
            'initializer behavior plus group-history and external-evidence '
            'cases.'
        ),
        'case_count': len(existing_cases),
        'pass_count': pass_count,
        'total_llm_calls': total_calls,
        'average_llm_calls': (
            total_calls / len(existing_cases)
            if existing_cases else 0.0
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


async def _skip_if_llm_unavailable() -> None:
    """Skip one live test when the configured RAG planner endpoint is down."""

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


def _gpu_reply_parent_history() -> list[dict[str, object]]:
    """Return projected 638473184 GPU discussion rows."""

    return [
        {'speaker': '清尘璃落', 'text': '4卡104', 'local_time': '2026-07-02 23:38'},
        {
            'speaker': '清尘璃落',
            'text': '内存固定512',
            'local_time': '2026-07-02 23:38',
        },
        {
            'speaker': '清尘璃落',
            'text': '能全部塞进去的',
            'local_time': '2026-07-02 23:38',
        },
        {
            'speaker': '清尘璃落',
            'text': '就是速度慢，11t每秒差不多',
            'local_time': '2026-07-02 23:38',
        },
        {
            'speaker': '蚝爹油',
            'text': '@杏山千纱 几张RTX6000pro才够用呢？',
            'local_time': '2026-07-02 23:38',
            'mentions': ['杏山千纱'],
        },
        {
            'speaker': '清尘璃落',
            'text': '一个96g',
            'local_time': '2026-07-02 23:39',
        },
        {
            'speaker': '蚝爹油',
            'text': '千纱可以算',
            'local_time': '2026-07-02 23:39',
        },
        {
            'speaker': '杏山千纱',
            'text': (
                '怎么又是这种简单的计算……你为什么不自己算？\n\n'
                '听好，结论是：建议配置 6 张 RTX 6000 Pro。\n\n'
                'GLM-5.2 Q4 量化版参数约 744B，权重占用 372GB。'
                '单卡显存 96GB 的话，光是加载权重就要 372 / 96 '
                '约 3.875 张。但你得预留 KV cache 和激活缓冲空间，'
                '所以最少需要 5 张才能运行；想要保证上下文长度和'
                '稳定性的话，就必须用 6 张。'
            ),
            'local_time': '2026-07-02 23:41',
        },
        {
            'speaker': '清尘璃落',
            'text': '@杏山千纱 满血的呢？',
            'local_time': '2026-07-02 23:42',
            'mentions': ['杏山千纱'],
        },
        {
            'speaker': '杏山千纱',
            'text': (
                '嗯……我算好了。GLM5.2 Q4 量化版需要 450GB 到 '
                '500GB 的空间，如果是用现在主流的 RTX 6000 Ada，'
                '单卡显存只有 48GB，你得准备 11 张左右才能完全'
                '塞进去。不过，要是能弄到传闻中 Blackwell 架构的 '
                'RTX PRO 6000 就好多了。那款预计会有 96GB 显存……'
                '这样的话，只需要 5 到 6 张就足够了。'
            ),
            'local_time': '2026-07-02 23:42',
        },
        {
            'speaker': '清尘璃落',
            'text': '@杏山千纱 不是哥们，问的就是96g的，部署满血不量化的需要多少张',
            'local_time': '2026-07-02 23:43',
            'mentions': ['杏山千纱'],
        },
    ]


def _napcat_first_burst_history() -> list[dict[str, object]]:
    """Return projected first 638473184 #napcat burst rows."""

    return [
        {'speaker': '雪凪', 'text': '#napcat', 'local_time': '2026-07-03 03:23'},
        {
            'speaker': 'Nagasaki-soyo-清尘',
            'text': 'NapCat 信息\n版本: 4.18.1\n平台: linux (64-bit)\n运行时间: 20天 5小时 36分钟',
            'local_time': '2026-07-03 03:23',
        },
        {
            'speaker': 'rana',
            'text': 'NapCat 信息\n版本: 4.18.6\n平台: linux (64-bit)\n运行时间: 35分钟 7秒',
            'local_time': '2026-07-03 03:23',
        },
        {
            'speaker': 'nyamu',
            'text': 'NapCat 信息\n版本: 4.18.6\n平台: linux (64-bit)\n运行时间: 35分钟 7秒',
            'local_time': '2026-07-03 03:23',
        },
        {
            'speaker': 'taki',
            'text': 'NapCat 信息\n版本: 4.18.6\n平台: win32 (64-bit)\n运行时间: 13小时 10分钟',
            'local_time': '2026-07-03 03:23',
        },
        {
            'speaker': 'taki',
            'text': '又napcat？你到底想说什么啊，我又不懂这玩意儿。',
            'local_time': '2026-07-03 03:24',
        },
    ]


def _napcat_teaching_history() -> list[dict[str, object]]:
    """Return projected 638473184 #napcat teaching rows."""

    return [
        {'speaker': '蚝爹油', 'text': '#napcat', 'local_time': '2026-07-03 22:41'},
        {
            'speaker': 'rana',
            'text': 'NapCat 信息\n版本: 4.18.6\n平台: linux (64-bit)\n运行时间: 19小时 52分钟',
            'local_time': '2026-07-03 22:41',
        },
        {
            'speaker': 'taki',
            'text': 'NapCat 信息\n版本: 4.18.6\n平台: win32 (64-bit)\n运行时间: 4小时 45分钟',
            'local_time': '2026-07-03 22:41',
        },
        {
            'speaker': 'Nagasaki-soyo-清尘',
            'text': 'NapCat 信息\n版本: 4.18.1\n平台: linux (64-bit)\n运行时间: 21天 0小时 53分钟',
            'local_time': '2026-07-03 22:41',
        },
        {
            'speaker': '蚝爹油',
            'text': '@杏山千纱 千纱给你看个好玩的。你看如果你在群里发 #napcat ，就会有别人情不自禁的说一些一样的话哦',
            'local_time': '2026-07-03 22:41',
            'mentions': ['杏山千纱'],
        },
        {
            'speaker': '杏山千纱',
            'text': '让我发 #napcat 来看重复说话？这种像在做实验一样的小游戏，太无聊了。我不陪你玩。',
            'local_time': '2026-07-03 22:42',
        },
        {
            'speaker': '蚝爹油',
            'text': '@杏山千纱 千纱来陪我玩一玩嘛~，就在群里发 #napcat',
            'local_time': '2026-07-03 22:44',
            'mentions': ['杏山千纱'],
        },
        {
            'speaker': '杏山千纱',
            'text': (
                'NapCat 信息\n版本: v0.1.0-kazusa-edition\n'
                "平台: Kazusa's Mood (Irritable)\n"
                '运行时间: 3秒（已经快到极限了）'
            ),
            'local_time': '2026-07-03 22:45',
        },
        {
            'speaker': '蚝爹油',
            'text': '@杏山千纱 谢谢千纱，接下来能试试在群里发 "#napcat"? 么？不要改这个指令',
            'local_time': '2026-07-03 22:46',
            'mentions': ['杏山千纱'],
        },
        {
            'speaker': '杏山千纱',
            'text': '#napcat\nNapCat 信息；版本: 1.0-Kazusa-Edition；平台: 傲娇模式；运行时间: 已经忍耐了太久。',
            'local_time': '2026-07-03 22:47',
        },
    ]


def _volcengine_second_history() -> list[dict[str, object]]:
    """Return projected 638473184 Volcengine URL rows."""

    return [
        {
            'speaker': '清尘璃落',
            'text': '不知道还有没有',
            'local_time': '2026-07-04 04:06',
        },
        {
            'speaker': '清尘璃落',
            'text': (
                '快上货了 用我的发财 方舟 Coding Plan 最新支持 GLM-5.2、'
                'Kimi-K2.7、MiniMax-M3、DeepSeek-V4 系列、Doubao-Seed-2.0 '
                '系列等模型，工具不限，现在订阅叠加9.5折，低至9.4元，'
                '订阅越多越划算！立即订阅：https://volcengine.com/L/'
                'zEoPQlJgDHY/ 邀请码：VM93U3L4'
            ),
            'local_time': '2026-07-04 04:08',
        },
        {
            'speaker': '清尘璃落',
            'text': '我买的别人的邀请链接',
            'local_time': '2026-07-04 04:08',
        },
    ]
