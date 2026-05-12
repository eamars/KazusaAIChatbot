from __future__ import annotations

import pytest

from kazusa_ai_chatbot.config import RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR
from kazusa_ai_chatbot.rag import conversation_search_agent
from kazusa_ai_chatbot.rag import persistent_memory_search_agent


class _FakeTool:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, args: dict[str, object]) -> object:
        self.calls.append(dict(args))
        return self.result


@pytest.mark.asyncio
async def test_conversation_search_tool_fuses_semantic_and_keyword_rows(
    monkeypatch,
) -> None:
    """Semantic conversation search should run exact anchors when available."""

    semantic_tool = _FakeTool([
        (
            0.82,
            {
                "platform_message_id": "message-cross",
                "body_text": "GPU market-share topic",
                "timestamp": "2026-05-11T09:00:00+00:00",
            },
        ),
        (
            0.94,
            {
                "platform_message_id": "message-drift",
                "body_text": "broad hardware echo",
                "timestamp": "2026-05-11T09:01:00+00:00",
            },
        ),
    ])
    keyword_tool = _FakeTool([
        {
            "platform_message_id": "message-cross",
            "body_text": "GPU market-share topic",
            "timestamp": "2026-05-11T09:00:00+00:00",
        }
    ])
    monkeypatch.setattr(conversation_search_agent, "search_conversation", semantic_tool)
    monkeypatch.setattr(
        conversation_search_agent,
        "search_conversation_keyword",
        keyword_tool,
    )
    monkeypatch.setattr(
        conversation_search_agent,
        "_conversation_neighbor_rows",
        _async_empty_neighbors,
    )

    result = await conversation_search_agent._tool(
        {
            "search_query": "GPU market-share discussion",
            "literal_anchors": ["GPU"],
            "platform_channel_id": "channel-1",
            "top_k": 20,
        }
    )

    assert semantic_tool.calls[0]["search_query"] == "GPU market-share discussion"
    assert "literal_anchors" not in semantic_tool.calls[0]
    assert keyword_tool.calls[0]["keyword"] == "GPU"
    assert result[0]["platform_message_id"] == "message-cross"
    assert result[0]["methods"] == ["semantic", "keyword:GPU"]
    assert result[0]["matched_anchors"] == ["GPU"]


@pytest.mark.asyncio
async def test_persistent_memory_search_tool_fuses_semantic_and_keyword_rows(
    monkeypatch,
) -> None:
    """Persistent-memory semantic search should support literal anchors."""

    semantic_tool = _FakeTool([
        {
            "memory_name": "hardware",
            "content": "GPU market-share topic",
            "timestamp": "2026-05-11T09:00:00+00:00",
            "cosine_similarity": 0.78,
        }
    ])
    keyword_tool = _FakeTool([
        {
            "memory_name": "hardware",
            "content": "GPU market-share topic",
            "timestamp": "2026-05-11T09:00:00+00:00",
        }
    ])
    monkeypatch.setattr(
        persistent_memory_search_agent,
        "search_persistent_memory",
        semantic_tool,
    )
    monkeypatch.setattr(
        persistent_memory_search_agent,
        "search_persistent_memory_keyword",
        keyword_tool,
    )

    result = await persistent_memory_search_agent._tool(
        {
            "search_query": "GPU market-share memory",
            "literal_anchors": ["GPU"],
            "top_k": 20,
        }
    )

    assert semantic_tool.calls[0]["search_query"] == "GPU market-share memory"
    assert "literal_anchors" not in semantic_tool.calls[0]
    assert keyword_tool.calls[0]["keyword"] == "GPU"
    assert result[0]["memory_name"] == "hardware"
    assert result[0]["methods"] == ["semantic", "keyword:GPU"]


def test_conversation_search_reapplies_scope_time_and_literal_anchors() -> None:
    """Trusted runtime scope and local relative-day bounds override LLM args."""

    args = conversation_search_agent._apply_runtime_constraints(
        {
            "search_query": "GPU market share",
            "literal_anchors": [],
            "platform": "discord",
            "platform_channel_id": "wrong-channel",
            "global_user_id": "wrong-user",
            "from_timestamp": "2026-05-11T00:00:00+00:00",
            "to_timestamp": "2026-05-11T23:59:59+00:00",
            "top_k": 20,
        },
        "Conversation-evidence: retrieve messages from yesterday mentioning '显卡'",
        {
            "platform": "qq",
            "platform_channel_id": "905393941",
            "global_user_id": "user-1",
            "conversation_user_scope": "current_user",
            "time_context": {
                "current_local_datetime": "2026-05-12 08:40",
                "current_local_weekday": "Tuesday",
            },
        },
    )

    assert args["platform"] == "qq"
    assert args["platform_channel_id"] == "905393941"
    assert args["global_user_id"] == "user-1"
    assert args["from_timestamp"] == "2026-05-10T12:00:00+00:00"
    assert args["to_timestamp"] == "2026-05-11T11:59:59+00:00"
    assert args["literal_anchors"] == ["显卡"]


def test_conversation_search_does_not_inherit_unscoped_current_user() -> None:
    """Unscoped group searches should not become current-user-only searches."""

    args = conversation_search_agent._apply_runtime_constraints(
        {
            "search_query": "GPU market share",
            "global_user_id": "llm-invented-user",
            "top_k": 20,
        },
        "Conversation-evidence: retrieve recent messages about GPU",
        {
            "platform": "qq",
            "platform_channel_id": "905393941",
            "global_user_id": "current-user",
        },
    )

    assert args["platform"] == "qq"
    assert args["platform_channel_id"] == "905393941"
    assert "global_user_id" not in args


def test_conversation_search_applies_one_sided_trusted_time_bound() -> None:
    """A single trusted time bound should override that generated field."""

    args = conversation_search_agent._apply_runtime_constraints(
        {
            "search_query": "GPU market share",
            "from_timestamp": "2026-05-01T00:00:00+00:00",
            "to_timestamp": "2026-05-12T00:00:00+00:00",
            "top_k": 20,
        },
        "Conversation-evidence: retrieve recent messages about GPU",
        {
            "from_timestamp": "2026-05-10T12:00:00+00:00",
        },
    )

    assert args["from_timestamp"] == "2026-05-10T12:00:00+00:00"
    assert args["to_timestamp"] == "2026-05-12T00:00:00+00:00"


def test_persistent_memory_search_rejects_untrusted_source_filter() -> None:
    """Generated source filters should not survive without trusted context."""

    args = persistent_memory_search_agent._apply_runtime_constraints(
        {
            "search_query": "GPU memory",
            "source_global_user_id": "llm-invented-user",
            "top_k": 20,
        },
        context={},
    )

    assert args == {"search_query": "GPU memory", "top_k": 20}


def test_persistent_memory_search_reapplies_trusted_source_filter() -> None:
    """Trusted source filters should override generated source filters."""

    args = persistent_memory_search_agent._apply_runtime_constraints(
        {
            "search_query": "GPU memory",
            "source_global_user_id": "llm-invented-user",
            "top_k": 20,
        },
        context={"source_global_user_id": "trusted-user"},
    )

    assert args["source_global_user_id"] == "trusted-user"


@pytest.mark.asyncio
async def test_conversation_neighbor_rows_fetches_nearest_sides(
    monkeypatch,
) -> None:
    """Neighbor expansion should fetch bounded rows before and after the seed."""

    calls: list[dict[str, object]] = []

    async def fake_get_conversation_history(**kwargs: object) -> list[dict[str, object]]:
        calls.append(dict(kwargs))
        sort_direction = kwargs["sort_direction"]
        if sort_direction == -1:
            return [
                {
                    "platform_message_id": "before",
                    "body_text": "before seed",
                    "timestamp": "2026-05-11T08:59:59+00:00",
                }
            ]
        return [
            {
                "platform_message_id": "after",
                "body_text": "after seed",
                "timestamp": "2026-05-11T09:00:01+00:00",
            }
        ]

    monkeypatch.setattr(
        conversation_search_agent,
        "get_conversation_history",
        fake_get_conversation_history,
    )
    candidates = conversation_search_agent.merge_hybrid_candidates(
        [
            {
                "platform_message_id": "seed",
                "body_text": "GPU seed",
                "timestamp": "2026-05-11T09:00:00+00:00",
                "score": RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
            }
        ],
        [],
        semantic_only_floor=RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
        selected_limit=20,
        source="conversation",
    )

    rows = await conversation_search_agent._conversation_neighbor_rows(
        candidates,
        {"platform": "qq", "platform_channel_id": "905393941"},
        semantic_only_floor=RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
        seed_limit=8,
        message_limit=3,
        window_minutes=3,
    )

    assert [call["sort_direction"] for call in calls] == [-1, 1]
    assert [row["platform_message_id"] for row in rows] == ["before", "after"]


async def _async_empty_neighbors(
    candidates: object,
    args: dict[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return []
