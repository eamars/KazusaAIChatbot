from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.config import RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR
from kazusa_ai_chatbot.rag.conversation_evidence.workers import (
    search as conversation_search_agent,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers import (
    persistent_search as persistent_memory_search_agent,
)


class _FakeTool:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def ainvoke(self, args: dict[str, object]) -> object:
        self.calls.append(dict(args))
        return self.result


class _FakeObjectId:
    """ObjectId-like value that plain JSON serialization cannot encode."""

    def __str__(self) -> str:
        """Return the stable string form used by Mongo row identity."""

        return_value = "row-object-id"
        return return_value


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeJudgeLLM:
    def __init__(self) -> None:
        self.messages: list[object] = []

    async def ainvoke(self, messages: list[object]) -> _FakeResponse:
        self.messages = list(messages)
        response = _FakeResponse('{"resolved": true, "feedback": ""}')
        return response


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


@pytest.mark.asyncio
async def test_conversation_search_judge_serializes_mongo_rows(
    monkeypatch,
) -> None:
    """Conversation judge payloads should strip raw Mongo row identifiers."""

    judge_llm = _FakeJudgeLLM()
    monkeypatch.setattr(conversation_search_agent, "_judge_llm", judge_llm)

    resolved, feedback = await conversation_search_agent._judge(
        "Conversation-evidence: retrieve GPU discussion",
        [
            {
                "_id": _FakeObjectId(),
                "conversation_row_id": "row-message",
                "body_text": "GPU market-share topic",
                "embedding": [0.1, 0.2, 0.3],
            },
        ],
        {
            "original_query": "What did people say around the GPU screenshot?",
            "current_slot": "Conversation-evidence: retrieve GPU discussion",
        },
    )

    human_message = judge_llm.messages[-1]
    payload = json.loads(human_message.content)
    assert resolved is True
    assert feedback == ""
    assert payload["context"]["original_query"] == (
        "What did people say around the GPU screenshot?"
    )
    assert "_id" not in payload["result"][0]
    assert "conversation_row_id" not in payload["result"][0]
    assert "embedding" not in payload["result"][0]


@pytest.mark.asyncio
async def test_persistent_memory_search_judge_serializes_mongo_rows(
    monkeypatch,
) -> None:
    """Persistent-memory judge payloads should strip raw Mongo row IDs."""

    judge_llm = _FakeJudgeLLM()
    monkeypatch.setattr(persistent_memory_search_agent, "_judge_llm", judge_llm)

    resolved, feedback = await persistent_memory_search_agent._judge(
        "Memory-evidence: retrieve GPU memory",
        [
            {
                "_id": _FakeObjectId(),
                "memory_name": "hardware",
                "content": "GPU market-share memory",
            },
        ],
    )

    human_message = judge_llm.messages[-1]
    payload = json.loads(human_message.content)
    assert resolved is True
    assert feedback == ""
    assert "_id" not in payload["result"][0]


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
            "local_time_context": {
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


def test_persistent_memory_search_prompt_preserves_chinese_attribute_anchors() -> None:
    """Memory search prompt should preserve source-language attribute handles."""

    prompt = persistent_memory_search_agent._GENERATOR_PROMPT

    assert '不要把中文查询翻译成英文' in prompt
    assert '具体属性名' in prompt
    assert '主体名和被询问的具体属性名作为 literal anchors' in prompt


def test_conversation_result_row_preserves_prompt_safe_local_timestamp() -> None:
    """Already projected conversation rows should keep local prompt timestamps."""

    row = conversation_search_agent._conversation_result_row(
        {
            "platform_message_id": "seed",
            "body_text": "Google Drive 又不是第一次这样了",
            "timestamp": "2026-05-23 08:26:45",
        }
    )

    assert row["timestamp"] == "2026-05-23 08:26:45"


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
    assert [row["relation_to_seed"] for row in rows] == [
        "previous_message",
        "next_message",
    ]
    assert [row["seed_platform_message_id"] for row in rows] == [
        "seed",
        "seed",
    ]


@pytest.mark.asyncio
async def test_conversation_neighbor_rows_normalizes_local_seed_time(
    monkeypatch,
) -> None:
    """Neighbor expansion should query storage UTC for local prompt timestamps."""

    calls: list[dict[str, object]] = []

    async def fake_get_conversation_history(**kwargs: object) -> list[dict[str, object]]:
        calls.append(dict(kwargs))
        return []

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
                "timestamp": "2026-05-11 21:00:00",
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

    assert rows == []
    assert calls == [
        {
            "platform": "qq",
            "platform_channel_id": "905393941",
            "limit": 3,
            "from_timestamp": "2026-05-11T08:57:00+00:00",
            "to_timestamp": "2026-05-11T09:00:00+00:00",
            "sort_direction": -1,
        },
        {
            "platform": "qq",
            "platform_channel_id": "905393941",
            "limit": 3,
            "from_timestamp": "2026-05-11T09:00:00+00:00",
            "to_timestamp": "2026-05-11T09:03:00+00:00",
            "sort_direction": 1,
        },
    ]


@pytest.mark.asyncio
async def test_conversation_neighbor_rows_keep_each_seed_window(
    monkeypatch,
) -> None:
    """Neighbor expansion should preserve bounded rows from each seed."""

    call_index = 0

    async def fake_get_conversation_history(**kwargs: object) -> list[dict[str, object]]:
        nonlocal call_index
        call_index += 1
        sort_direction = kwargs["sort_direction"]
        suffix = "before" if sort_direction == -1 else "after"
        return [
            {
                "platform_message_id": f"{suffix}-{call_index}",
                "body_text": f"{suffix} seed context",
                "timestamp": f"2026-05-11T09:00:0{call_index}+00:00",
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
                "platform_message_id": "seed-1",
                "body_text": "first seed",
                "timestamp": "2026-05-11T09:00:00+00:00",
                "score": RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
            },
            {
                "platform_message_id": "seed-2",
                "body_text": "second seed",
                "timestamp": "2026-05-11T09:10:00+00:00",
                "score": RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
            },
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
        seed_limit=2,
        message_limit=1,
        window_minutes=3,
    )

    assert [row["body_text"] for row in rows] == [
        "before seed context",
        "after seed context",
        "before seed context",
        "after seed context",
    ]


@pytest.mark.asyncio
async def test_conversation_neighbor_rows_strip_raw_storage_fields(
    monkeypatch,
) -> None:
    """Neighbor expansion should not pass Mongo storage fields into prompts."""

    async def fake_get_conversation_history(**_: object) -> list[dict[str, object]]:
        return [
            {
                "_id": _FakeObjectId(),
                "platform_message_id": "neighbor",
                "body_text": "neighbor text",
                "raw_wire_text": "[CQ:large-storage-value]",
                "timestamp": "2026-05-11T09:00:01+00:00",
                "embedding": [0.1, 0.2, 0.3],
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

    assert len(rows) == 1
    assert rows[0]["conversation_row_id"] == "row-object-id"
    assert "embedding" not in rows[0]
    assert "raw_wire_text" not in rows[0]


def test_conversation_result_row_strips_malformed_raw_storage_fields() -> None:
    """The search result contract should stay compact even for odd rows."""

    row = conversation_search_agent._conversation_result_row(
        {
            "_id": _FakeObjectId(),
            "error": "unexpected storage-shaped row",
            "embedding": [0.1, 0.2, 0.3],
            "raw_wire_text": "raw text",
            "base64_data": "inline-bytes",
        }
    )

    assert row == {
        "conversation_row_id": "row-object-id",
        "error": "unexpected storage-shaped row",
    }


async def _async_empty_neighbors(
    candidates: object,
    args: dict[str, object],
    **_: object,
) -> list[dict[str, object]]:
    return []
