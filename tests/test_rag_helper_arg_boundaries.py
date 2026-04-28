from __future__ import annotations

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.rag import conversation_aggregate_agent
from kazusa_ai_chatbot.rag import conversation_filter_agent
from kazusa_ai_chatbot.rag import conversation_keyword_agent
from kazusa_ai_chatbot.rag import conversation_search_agent
from kazusa_ai_chatbot.rag import persistent_memory_keyword_agent
from kazusa_ai_chatbot.rag import persistent_memory_search_agent
from kazusa_ai_chatbot.rag import relationship_agent
from kazusa_ai_chatbot.rag import user_list_agent


def test_conversation_search_args_do_not_stringify_container_fields() -> None:
    """Semantic conversation-search args should ignore non-string scalar fields."""

    args = conversation_search_agent._normalize_args({
        "search_query": {"bad": "query"},
        "global_user_id": ["bad-user"],
        "platform": {"bad": "platform"},
        "platform_channel_id": " channel-1 ",
        "from_timestamp": {"bad": "from"},
        "to_timestamp": ["bad-to"],
        "top_k": 3,
    })

    assert args == {"platform_channel_id": "channel-1", "top_k": 3}


def test_conversation_search_args_accept_string_fields() -> None:
    """Semantic conversation-search args preserve valid string fields."""

    args = conversation_search_agent._normalize_args({
        "search_query": " tea preference ",
        "global_user_id": " user-1 ",
        "platform": " qq ",
        "platform_channel_id": " channel-1 ",
        "from_timestamp": " 2026-04-28T00:00:00+00:00 ",
        "to_timestamp": " 2026-04-29T00:00:00+00:00 ",
        "top_k": 3,
    })

    assert args == {
        "search_query": "tea preference",
        "global_user_id": "user-1",
        "platform": "qq",
        "platform_channel_id": "channel-1",
        "from_timestamp": "2026-04-28T00:00:00+00:00",
        "to_timestamp": "2026-04-29T00:00:00+00:00",
        "top_k": 3,
    }


def test_conversation_keyword_args_do_not_stringify_container_fields() -> None:
    """Keyword-search args should ignore non-string scalar fields."""

    args = conversation_keyword_agent._normalize_args({
        "keyword": {"bad": "keyword"},
        "global_user_id": ["bad-user"],
        "platform": {"bad": "platform"},
        "platform_channel_id": " channel-1 ",
        "from_timestamp": {"bad": "from"},
        "to_timestamp": ["bad-to"],
        "top_k": 3,
    })

    assert args == {"platform_channel_id": "channel-1", "top_k": 3}


def test_conversation_filter_args_do_not_stringify_container_fields() -> None:
    """Conversation-filter args should ignore non-string scalar fields."""

    args = conversation_filter_agent._normalize_args({
        "platform": {"bad": "platform"},
        "platform_channel_id": " channel-1 ",
        "global_user_id": ["bad-user"],
        "display_name": {"bad": "name"},
        "from_timestamp": ["bad-from"],
        "to_timestamp": {"bad": "to"},
        "limit": 4,
    })

    assert args == {"platform_channel_id": "channel-1", "limit": 4}


def test_conversation_aggregate_args_do_not_stringify_container_fields() -> None:
    """Aggregate args should ignore non-string keyword/window values."""

    args = conversation_aggregate_agent._normalize_args({
        "keyword": {"bad": "keyword"},
        "time_window": ["bad-window"],
        "limit": 4,
    })

    assert args == {
        "aggregate": "message_count_by_user",
        "keyword": "",
        "time_window": "recent",
        "limit": 4,
    }


def test_persistent_memory_search_args_do_not_stringify_container_fields() -> None:
    """Persistent-memory search args should ignore non-string scalar fields."""

    args = persistent_memory_search_agent._normalize_args({
        "search_query": {"bad": "query"},
        "source_global_user_id": ["bad-user"],
        "source_kind": ["bad-kind"],
        "status": {"bad": "status"},
        "expiry_before": ["bad-before"],
        "expiry_after": {"bad": "after"},
        "top_k": 3,
    })

    assert args == {"top_k": 3}


def test_persistent_memory_search_args_accept_string_fields() -> None:
    """Persistent-memory search args preserve valid supported string fields."""

    args = persistent_memory_search_agent._normalize_args({
        "search_query": " tea preference ",
        "source_global_user_id": " user-1 ",
        "source_kind": " conversation_extracted ",
        "status": " active ",
        "expiry_before": " 2026-04-29T00:00:00+00:00 ",
        "expiry_after": " 2026-04-28T00:00:00+00:00 ",
        "top_k": 3,
    })

    assert args == {
        "search_query": "tea preference",
        "source_global_user_id": "user-1",
        "status": "active",
        "expiry_before": "2026-04-29T00:00:00+00:00",
        "expiry_after": "2026-04-28T00:00:00+00:00",
        "top_k": 3,
    }


def test_persistent_memory_search_args_erase_character_source_id() -> None:
    """Persistent-memory semantic args treat the character source ID as generic."""

    args = persistent_memory_search_agent._normalize_args({
        "search_query": "Kazusa exclusive weapon",
        "source_global_user_id": CHARACTER_GLOBAL_USER_ID,
        "top_k": 3,
    })

    assert args == {
        "search_query": "Kazusa exclusive weapon",
        "top_k": 3,
    }


def test_persistent_memory_keyword_args_do_not_stringify_container_fields() -> None:
    """Persistent-memory keyword args should ignore non-string scalar fields."""

    args = persistent_memory_keyword_agent._normalize_args({
        "keyword": {"bad": "keyword"},
        "source_global_user_id": ["bad-user"],
        "source_kind": ["bad-kind"],
        "status": {"bad": "status"},
        "expiry_before": ["bad-before"],
        "expiry_after": {"bad": "after"},
        "top_k": 3,
    })

    assert args == {"top_k": 3}


def test_persistent_memory_keyword_args_erase_character_source_id() -> None:
    """Persistent-memory keyword args treat the character source ID as generic."""

    args = persistent_memory_keyword_agent._normalize_args({
        "keyword": "专属武器",
        "source_global_user_id": CHARACTER_GLOBAL_USER_ID,
        "top_k": 3,
    })

    assert args == {"keyword": "专属武器", "top_k": 3}


def test_user_list_args_do_not_stringify_container_fields() -> None:
    """User-list args should not stringify malformed enum/text fields."""

    args = user_list_agent._normalize_args({
        "source": {"bad": "source"},
        "display_name_operator": ["bad-op"],
        "display_name_value": {"bad": "name"},
        "limit": 4,
    })

    assert args == {
        "source": "user_profiles",
        "display_name_operator": "contains",
        "display_name_value": "",
        "limit": 4,
    }


def test_user_list_args_accept_string_fields() -> None:
    """User-list args preserve valid string fields."""

    args = user_list_agent._normalize_args({
        "source": " conversation_participants ",
        "display_name_operator": " ends_with ",
        "display_name_value": " 子 ",
        "limit": 4,
    })

    assert args == {
        "source": "conversation_participants",
        "display_name_operator": "ends_with",
        "display_name_value": "子",
        "limit": 4,
    }


def test_relationship_args_do_not_stringify_container_fields() -> None:
    """Relationship args should not stringify malformed enum fields."""

    args = relationship_agent._normalize_relationship_args({
        "mode": {"bad": "mode"},
        "rank_order": ["bad-order"],
        "limit": 4,
    })

    assert args is None


def test_relationship_args_accept_string_fields() -> None:
    """Relationship args preserve valid string fields."""

    args = relationship_agent._normalize_relationship_args({
        "mode": " n ",
        "rank_order": " top ",
        "limit": 4,
    })

    assert args == {"mode": "n", "rank_order": "top", "limit": 4}
