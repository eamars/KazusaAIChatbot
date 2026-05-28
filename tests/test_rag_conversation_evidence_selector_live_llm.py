"""Real LLM checks for the conversation-evidence selector prompt."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import RAG_SUBAGENT_LLM_BASE_URL
from kazusa_ai_chatbot.rag.conversation_evidence import selector as selector_module
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured RAG subagent endpoint is unavailable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{RAG_SUBAGENT_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError:
        pytest.skip(f'LLM endpoint is unavailable: {RAG_SUBAGENT_LLM_BASE_URL}')

    if response.status_code >= 500:
        pytest.skip(
            f'LLM endpoint returned server error {response.status_code}: '
            f'{RAG_SUBAGENT_LLM_BASE_URL}'
        )


async def test_live_selector_routes_current_user_content_search() -> None:
    """Fallback selector should route current-user content evidence to search."""
    await _skip_if_llm_unavailable()
    task = (
        'Conversation-evidence: retrieve messages from current_user '
        'containing sexual harassment or explicit accusations '
        'speaker=current_user'
    )
    context = {
        'platform': 'qq',
        'platform_channel_id': 'selector-live-test',
        'platform_user_id': 'platform-user-1',
        'global_user_id': 'user-1',
        'user_name': 'Tester',
        'known_facts': [],
        'current_slot': task,
    }
    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=selector_module._SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )

    response = await selector_module._selector_llm.ainvoke(
        [system_prompt, human_message]
    )
    parsed = parse_llm_json_output(response.content)
    if not isinstance(parsed, dict):
        pytest.fail(f'Conversation selector returned non-object JSON: {response.content}')

    normalized_plan = selector_module._normalize_selector_plan(parsed, task)
    trace_path = write_llm_trace(
        'rag_conversation_evidence_selector_live_llm',
        'current_user_content_search',
        {
            'task': task,
            'llm_input': llm_input,
            'raw_output': response.content,
            'parsed_output': parsed,
            'normalized_plan': normalized_plan,
            'judgment': (
                'Selector should choose the conversation search worker for '
                'semantic message-content evidence; person-ref dependency is '
                'owned by deterministic slot validation.'
            ),
        },
    )
    logger.info(
        f'RAG_CONVERSATION_SELECTOR_LIVE trace={trace_path} '
        f'parsed={parsed} normalized_plan={normalized_plan}'
    )

    assert parsed.get('worker') == 'conversation_search_agent'
    assert normalized_plan == {
        'worker': 'conversation_search_agent',
        'reason': parsed.get('reason', ''),
        'requires_person_ref': False,
    }
