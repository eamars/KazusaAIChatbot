from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from string import Template
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.rag.conversation_evidence import (
    selector as conversation_evidence_agent,
)
from kazusa_ai_chatbot.rag.conversation_evidence.workers import (
    aggregate as conversation_aggregate_agent,
)
from kazusa_ai_chatbot.rag.conversation_evidence.workers import (
    filter as conversation_filter_agent,
)
from kazusa_ai_chatbot.rag.conversation_evidence.workers import (
    keyword as conversation_keyword_agent,
)
from kazusa_ai_chatbot.rag.conversation_evidence.workers import (
    search as conversation_search_agent,
)
from kazusa_ai_chatbot.rag.live_context import selector as live_context_agent
from kazusa_ai_chatbot.rag.memory_evidence import selector as memory_evidence_agent
from kazusa_ai_chatbot.rag.memory_evidence.workers import (
    persistent_keyword as persistent_memory_keyword_agent,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers import (
    persistent_search as persistent_memory_search_agent,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers import (
    user_memory as user_memory_evidence_agent,
)
from kazusa_ai_chatbot.rag.person_context import selector as person_context_agent
from kazusa_ai_chatbot.rag.person_context.workers import list as user_list_agent
from kazusa_ai_chatbot.rag.person_context.workers import lookup as user_lookup_agent
from kazusa_ai_chatbot.rag.person_context.workers import (
    relationship as relationship_agent,
)
from kazusa_ai_chatbot.rag.recall import review as recall_agent

BASELINE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "rag_agent_package_prompt_baseline.json"
)


class _CapturingLLM:
    """Record LangChain message payloads and return deterministic JSON."""

    def __init__(self, response_payload: dict[str, Any]) -> None:
        self.response = SimpleNamespace(
            content=json.dumps(response_payload, ensure_ascii=False)
        )
        self.calls: list[list[dict[str, Any]]] = []

    async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
        """Record one LLM call and return the configured response."""
        self.calls.append([_message_record(message) for message in messages])
        return_value = self.response
        return return_value


def _base_context() -> dict[str, Any]:
    """Build stable runtime context for representative LLM payloads."""
    context = {
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "Tester",
        "character_global_user_id": "character-1",
        "current_timestamp_utc": "2026-05-02T00:00:00+00:00",
        "current_slot": "selector-only neutral slot",
        "original_query": "selector stability audit",
        "known_facts": [
            {
                "slot": "prior slot",
                "summary": "Prior evidence.",
                "raw_result": {
                    "rows": [
                        {
                            "timestamp": "2026-05-02T20:00:00+00:00",
                            "body_text": "Prior evidence text.",
                        }
                    ],
                    "updated_at": "2026-05-02T20:01:00+00:00",
                },
            }
        ],
        "local_time_context": {
            "current_local_datetime": "2026-05-03 12:00",
            "current_local_date": "2026-05-03",
            "current_local_weekday": "Sunday",
        },
    }
    return context


def _prompt_text(value: object) -> str:
    """Return a stable prompt text for strings and string.Template prompts."""
    if isinstance(value, Template):
        prompt = value.template
        return prompt
    if isinstance(value, str):
        return value
    prompt = str(value)
    return prompt


def _message_record(message: object) -> dict[str, Any]:
    """Project a LangChain message into a stable JSON-serializable record."""
    raw_content = getattr(message, "content")
    content = _prompt_text(raw_content)
    try:
        parsed_json = json.loads(content)
    except json.JSONDecodeError:
        parsed_json = None
    record = {
        "role": type(message).__name__,
        "content": content,
        "json": parsed_json,
    }
    return record


async def _capture_call(
    monkeypatch: pytest.MonkeyPatch,
    *,
    module: object,
    llm_attr: str,
    response_payload: dict[str, Any],
    call_factory: Callable[[], Awaitable[object]],
) -> list[dict[str, Any]]:
    """Patch one LLM object, run its handler, and return captured messages."""
    llm = _CapturingLLM(response_payload)
    monkeypatch.setattr(module, llm_attr, llm)
    await call_factory()
    assert len(llm.calls) == 1
    messages = llm.calls[0]
    return messages


async def _prompt_payload_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Collect prompt constants and representative LLM message payloads."""
    context = _base_context()
    prompts = {
        "conversation_evidence.selector": _prompt_text(
            conversation_evidence_agent._SELECTOR_PROMPT
        ),
        "conversation_search.generator": _prompt_text(
            conversation_search_agent._GENERATOR_PROMPT
        ),
        "conversation_search.judge": _prompt_text(
            conversation_search_agent._JUDGE_PROMPT
        ),
        "conversation_filter.generator": _prompt_text(
            conversation_filter_agent._GENERATOR_PROMPT
        ),
        "conversation_filter.judge": _prompt_text(
            conversation_filter_agent._JUDGE_PROMPT
        ),
        "conversation_keyword.generator": _prompt_text(
            conversation_keyword_agent._GENERATOR_PROMPT
        ),
        "conversation_keyword.judge": _prompt_text(
            conversation_keyword_agent._JUDGE_PROMPT
        ),
        "conversation_aggregate.extractor": _prompt_text(
            conversation_aggregate_agent._EXTRACTOR_PROMPT
        ),
        "memory_evidence.selector": _prompt_text(
            memory_evidence_agent._SELECTOR_PROMPT
        ),
        "persistent_memory_search.generator": _prompt_text(
            persistent_memory_search_agent._GENERATOR_PROMPT
        ),
        "persistent_memory_search.judge": _prompt_text(
            persistent_memory_search_agent._JUDGE_PROMPT
        ),
        "persistent_memory_keyword.generator": _prompt_text(
            persistent_memory_keyword_agent._GENERATOR_PROMPT
        ),
        "persistent_memory_keyword.judge": _prompt_text(
            persistent_memory_keyword_agent._JUDGE_PROMPT
        ),
        "user_memory_evidence.review": _prompt_text(
            user_memory_evidence_agent._REVIEW_PROMPT
        ),
        "person_context.selector": _prompt_text(
            person_context_agent._SELECTOR_PROMPT
        ),
        "user_lookup.extractor": _prompt_text(user_lookup_agent._EXTRACTOR_PROMPT),
        "user_lookup.picker": _prompt_text(user_lookup_agent._PICKER_PROMPT),
        "user_list.extractor": _prompt_text(user_list_agent._EXTRACTOR_PROMPT),
        "relationship.extractor": _prompt_text(
            relationship_agent._EXTRACTOR_PROMPT
        ),
        "live_context.selector": _prompt_text(
            live_context_agent._EXTERNAL_LIVE_SELECTOR_PROMPT
        ),
        "recall.review": _prompt_text(recall_agent._RECALL_REVIEW_PROMPT),
    }

    payloads: dict[str, list[dict[str, Any]]] = {}
    monkeypatch.setattr(
        conversation_evidence_agent,
        "_deterministic_plan",
        lambda task: None,
    )
    payloads["conversation_evidence.selector"] = await _capture_call(
        monkeypatch,
        module=conversation_evidence_agent,
        llm_attr="_selector_llm",
        response_payload={
            "worker": "conversation_search_agent",
            "reason": "audit",
        },
        call_factory=lambda: conversation_evidence_agent._select_plan(
            "selector-only neutral slot",
            context,
        ),
    )
    payloads["conversation_search.generator"] = await _capture_call(
        monkeypatch,
        module=conversation_search_agent,
        llm_attr="_generator_llm",
        response_payload={
            "search_query": "benchmark discussion",
            "literal_anchors": ["benchmark"],
            "top_k": 5,
        },
        call_factory=lambda: conversation_search_agent._generator(
            "Conversation-evidence: retrieve benchmark discussion",
            context,
            "",
        ),
    )
    payloads["conversation_search.judge"] = await _capture_call(
        monkeypatch,
        module=conversation_search_agent,
        llm_attr="_judge_llm",
        response_payload={"resolved": True, "feedback": ""},
        call_factory=lambda: conversation_search_agent._judge(
            "Conversation-evidence: retrieve benchmark discussion",
            [{"body_text": "benchmark discussion"}],
            context,
        ),
    )
    payloads["conversation_filter.generator"] = await _capture_call(
        monkeypatch,
        module=conversation_filter_agent,
        llm_attr="_generator_llm",
        response_payload={"platform_channel_id": "channel-1", "top_k": 5},
        call_factory=lambda: conversation_filter_agent._generator(
            "Conversation-filter: list recent messages",
            context,
            "",
        ),
    )
    payloads["conversation_filter.judge"] = await _capture_call(
        monkeypatch,
        module=conversation_filter_agent,
        llm_attr="_judge_llm",
        response_payload={"resolved": True, "feedback": ""},
        call_factory=lambda: conversation_filter_agent._judge(
            "Conversation-filter: list recent messages",
            [{"body_text": "recent message"}],
        ),
    )
    payloads["conversation_keyword.generator"] = await _capture_call(
        monkeypatch,
        module=conversation_keyword_agent,
        llm_attr="_generator_llm",
        response_payload={"keyword": "benchmark", "top_k": 5},
        call_factory=lambda: conversation_keyword_agent._generator(
            "Conversation-keyword: find benchmark",
            context,
            "",
        ),
    )
    payloads["conversation_keyword.judge"] = await _capture_call(
        monkeypatch,
        module=conversation_keyword_agent,
        llm_attr="_judge_llm",
        response_payload={"resolved": True, "feedback": ""},
        call_factory=lambda: conversation_keyword_agent._judge(
            "Conversation-keyword: find benchmark",
            [{"body_text": "benchmark"}],
        ),
    )
    payloads["conversation_aggregate.extractor"] = await _capture_call(
        monkeypatch,
        module=conversation_aggregate_agent,
        llm_attr="_extractor_llm",
        response_payload={"metric": "count", "group_by": "", "limit": 5},
        call_factory=lambda: conversation_aggregate_agent._extract_aggregate_args(
            "Conversation-aggregate: count recent messages",
            context,
        ),
    )

    monkeypatch.setattr(
        memory_evidence_agent,
        "_deterministic_plan",
        lambda task, context=None: None,
    )
    payloads["memory_evidence.selector"] = await _capture_call(
        monkeypatch,
        module=memory_evidence_agent,
        llm_attr="_selector_llm",
        response_payload={
            "worker": "persistent_memory_search_agent",
            "reason": "audit",
        },
        call_factory=lambda: memory_evidence_agent._select_plan(
            "selector-only neutral slot",
            context,
        ),
    )
    payloads["persistent_memory_search.generator"] = await _capture_call(
        monkeypatch,
        module=persistent_memory_search_agent,
        llm_attr="_generator_llm",
        response_payload={
            "search_query": "benchmark memory",
            "literal_anchors": ["benchmark"],
            "top_k": 5,
        },
        call_factory=lambda: persistent_memory_search_agent._generator(
            "Memory-search: retrieve benchmark memory",
            context,
            "",
        ),
    )
    payloads["persistent_memory_search.judge"] = await _capture_call(
        monkeypatch,
        module=persistent_memory_search_agent,
        llm_attr="_judge_llm",
        response_payload={"resolved": True, "feedback": ""},
        call_factory=lambda: persistent_memory_search_agent._judge(
            "Memory-search: retrieve benchmark memory",
            [{"content": "benchmark memory"}],
        ),
    )
    payloads["persistent_memory_keyword.generator"] = await _capture_call(
        monkeypatch,
        module=persistent_memory_keyword_agent,
        llm_attr="_generator_llm",
        response_payload={"keyword": "benchmark", "top_k": 5},
        call_factory=lambda: persistent_memory_keyword_agent._generator(
            "Memory-keyword: retrieve benchmark",
            context,
            "",
        ),
    )
    payloads["persistent_memory_keyword.judge"] = await _capture_call(
        monkeypatch,
        module=persistent_memory_keyword_agent,
        llm_attr="_judge_llm",
        response_payload={"resolved": True, "feedback": ""},
        call_factory=lambda: persistent_memory_keyword_agent._judge(
            "Memory-keyword: retrieve benchmark",
            [{"content": "benchmark memory"}],
        ),
    )
    payloads["user_memory_evidence.review"] = await _capture_call(
        monkeypatch,
        module=user_memory_evidence_agent,
        llm_attr="_review_llm",
        response_payload={
            "confirmed_unit_ids": ["unit-1"],
            "nearby_unit_ids": [],
            "summary": "matching row",
            "uncertainty": "",
        },
        call_factory=lambda: user_memory_evidence_agent._review_user_memory_rows(
            "Memory-evidence: retrieve current user preference",
            [
                {
                    "unit_id": "unit-1",
                    "fact": "User prefers benchmark notes.",
                    "updated_at": "2026-05-02T20:00:00+00:00",
                }
            ],
        ),
    )

    monkeypatch.setattr(
        person_context_agent,
        "_deterministic_plan",
        lambda task: None,
    )
    payloads["person_context.selector"] = await _capture_call(
        monkeypatch,
        module=person_context_agent,
        llm_attr="_selector_llm",
        response_payload={
            "mode": "lookup",
            "target": "display_name",
            "reason": "audit",
        },
        call_factory=lambda: person_context_agent._select_plan(
            "selector-only neutral slot",
            context,
        ),
    )
    payloads["user_lookup.extractor"] = await _capture_call(
        monkeypatch,
        module=user_lookup_agent,
        llm_attr="_extractor_llm",
        response_payload={"display_name": "Tester"},
        call_factory=lambda: user_lookup_agent._extract_display_name_with_llm(
            "Identity: resolve Tester",
            context,
        ),
    )
    payloads["user_lookup.picker"] = await _capture_call(
        monkeypatch,
        module=user_lookup_agent,
        llm_attr="_picker_llm",
        response_payload={"global_user_id": "user-2"},
        call_factory=lambda: user_lookup_agent._pick_best_candidate_with_llm(
            "Tester",
            [
                {
                    "global_user_id": "user-1",
                    "display_name": "Tester one",
                    "platform": "qq",
                },
                {
                    "global_user_id": "user-2",
                    "display_name": "Tester two",
                    "platform": "qq",
                },
            ],
        ),
    )
    payloads["user_list.extractor"] = await _capture_call(
        monkeypatch,
        module=user_list_agent,
        llm_attr="_extractor_llm",
        response_payload={"display_name_suffix": "er", "limit": 5},
        call_factory=lambda: user_list_agent._extract_user_list_args(
            "User-list: users ending with er",
            context,
        ),
    )
    payloads["relationship.extractor"] = await _capture_call(
        monkeypatch,
        module=relationship_agent,
        llm_attr="_extractor_llm",
        response_payload={"rank_by": "affinity", "direction": "top", "limit": 5},
        call_factory=lambda: relationship_agent._extract_relationship_args(
            "Relationship: top affinity users",
            context,
        ),
    )

    payloads["live_context.selector"] = await _capture_call(
        monkeypatch,
        module=live_context_agent,
        llm_attr="_external_live_selector_llm",
        response_payload={
            "fact_type": "weather",
            "target_source": "explicit",
            "target": "Auckland",
            "missing_context": [],
        },
        call_factory=lambda: live_context_agent._select_external_live_plan(
            "Live-context: current weather in Auckland",
            context,
        ),
    )
    payloads["recall.review"] = await _capture_call(
        monkeypatch,
        module=recall_agent,
        llm_attr="_recall_review_llm",
        response_payload={
            "confirmed_candidate_indexes": [0],
            "nearby_candidate_indexes": [],
            "summary": "confirmed",
            "uncertainty": "",
            "source_hints": [],
        },
        call_factory=lambda: recall_agent._review_recall_candidates(
            task="Recall: current agreement about benchmark notes",
            mode="active_episode_agreement",
            candidates=[
                {
                    "source": "conversation_progress",
                    "claim": "User and character agreed to review benchmark notes.",
                    "temporal_scope": "active",
                    "lifecycle_status": "active",
                    "evidence_time": "2026-05-02 20:00",
                    "authority": "conversation_progress",
                }
            ],
        ),
    )

    snapshot = {
        "version": 1,
        "prompts": prompts,
        "payloads": payloads,
    }
    return snapshot


@pytest.mark.asyncio
async def test_rag_agent_package_prompts_and_payloads_are_stable(
    monkeypatch,
) -> None:
    """Compare moved RAG agent prompts and payloads against the stable baseline."""
    snapshot = await _prompt_payload_snapshot(monkeypatch)
    assert BASELINE_PATH.exists(), f"Missing prompt baseline: {BASELINE_PATH}"
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert snapshot == baseline
