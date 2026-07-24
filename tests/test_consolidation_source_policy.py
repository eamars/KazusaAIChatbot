"""Focused tests for consolidator lane source-policy contracts."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest

from kazusa_ai_chatbot.consolidation.origin import ConsolidationOriginMetadata
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc


def _source_policy_module() -> Any:
    """Import the planned source-policy module with a clear failure."""

    try:
        module = importlib.import_module(
            "kazusa_ai_chatbot.consolidation.source_policy"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Missing consolidation.source_policy module required by the "
            "lane-router bigbang plan."
        )
        raise exc
    return module


def _origin() -> ConsolidationOriginMetadata:
    """Build a current-turn user-message consolidation origin."""

    origin: ConsolidationOriginMetadata = {
        "episode_id": "episode-source-policy-1",
        "trigger_source": "user_message",
        "input_sources": ["dialog_text"],
        "output_mode": "visible_reply",
        "timestamp": "2026-07-03T00:00:00+00:00",
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "platform": "qq",
        "platform_channel_id": "private-1",
        "channel_type": "private",
        "platform_message_id": "message-1",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": ["conversation-row-1"],
        "current_platform_user_id": "platform-user-1",
        "current_global_user_id": "global-user-1",
        "current_display_name": "Test User",
    }
    return origin


def _state() -> dict[str, Any]:
    """Build a minimal source-view state."""

    turn_clock = build_turn_clock_from_storage_utc("2026-07-03T00:00:00+00:00")
    state: dict[str, Any] = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "decontextualized_input": "I now work in Auckland.",
        "final_dialog": ["Kazusa acknowledges the user's update."],
        "internal_monologue": "The user gave a concrete personal fact.",
        "chat_history_recent": [
            {
                "role": "user",
                "content": "I now work in Auckland.",
                "timestamp": "2026-07-03T00:00:00+00:00",
            }
        ],
        "rag_result": {
            "memory_evidence": [],
            "conversation_evidence": [],
            "external_evidence": [],
            "recall_evidence": [],
        },
        "consolidation_origin": _origin(),
    }
    return state


def _source(source_key: str, source_kind: str) -> dict[str, Any]:
    """Build one source-view row for source-policy tests."""

    source_view = {
        "source_key": source_key,
        "source_kind": source_kind,
        "summary": "same words should not drive deterministic lane choice",
        "source_refs": [
            {
                "source": source_kind,
                "timestamp": "2026-07-03T00:00:00+00:00",
                "conversation_row_id": f"{source_key}-row",
            }
        ],
    }
    return source_view


def test_source_views_include_current_turn_refs() -> None:
    """Current chat and final dialog source views must carry source refs."""

    module = _source_policy_module()

    source_views = module.build_consolidation_source_views(_state())
    views_by_key = {
        source_view["source_key"]: source_view
        for source_view in source_views
    }

    assert views_by_key["current_turn_user_message"]["source_kind"] == (
        "user_message"
    )
    assert views_by_key["current_turn_user_message"]["source_refs"]
    assert views_by_key["assistant_final_dialog"]["source_kind"] == (
        "assistant_final_dialog"
    )
    assert views_by_key["assistant_final_dialog"]["source_refs"]


def test_source_views_do_not_invent_blank_assistant_acceptance() -> None:
    """A blank final dialog must not become acceptance evidence."""

    module = _source_policy_module()
    state = _state()
    state["final_dialog"] = []

    source_views = module.build_consolidation_source_views(state)
    source_keys = {
        source_view["source_key"]
        for source_view in source_views
    }

    assert "current_turn_user_message" in source_keys
    assert "assistant_final_dialog" not in source_keys


def test_user_memory_rejects_internal_only_source() -> None:
    """Objective user memory cannot be created from internal thought alone."""

    module = _source_policy_module()

    result = module.validate_lane_source_policy(
        "user_memory_units",
        [_source("internal", "internal_thought")],
    )

    assert result["accepted"] is False
    assert result["reason"] == "source_class_not_allowed"


def test_active_commitment_requires_assistant_acceptance_source() -> None:
    """Commitments require user request plus assistant acceptance evidence."""

    module = _source_policy_module()

    user_only = module.validate_lane_source_policy(
        "active_commitment",
        [_source("current_turn_user_message", "user_message")],
    )
    accepted = module.validate_lane_source_policy(
        "active_commitment",
        [
            _source("current_turn_user_message", "user_message"),
            _source("assistant_final_dialog", "assistant_final_dialog"),
        ],
    )

    assert user_only["accepted"] is False
    assert accepted["accepted"] is True


def test_character_self_guidance_requires_acceptance_source() -> None:
    """Accepted global behavior guidance needs assistant acceptance evidence."""

    module = _source_policy_module()

    user_only = module.validate_lane_source_policy(
        "character_self_guidance",
        [_source("current_turn_user_message", "user_message")],
    )
    accepted = module.validate_lane_source_policy(
        "character_self_guidance",
        [
            _source("current_turn_user_message", "user_message"),
            _source("assistant_final_dialog", "assistant_final_dialog"),
        ],
    )

    assert user_only["accepted"] is False
    assert accepted["accepted"] is True


def test_shared_memory_promotion_rejects_ordinary_chat_sources() -> None:
    """Ordinary chat cannot promote generic shared/world memory."""

    module = _source_policy_module()

    ordinary_chat = module.validate_lane_source_policy(
        "shared_memory_promotion",
        [
            _source("current_turn_user_message", "user_message"),
            _source("assistant_final_dialog", "assistant_final_dialog"),
        ],
    )
    promoted_reflection = module.validate_lane_source_policy(
        "shared_memory_promotion",
        [_source("reflection", "reflection_run")],
        privacy_review={
            "user_details_removed": True,
            "private_detail_risk": "low",
            "boundary_assessment": "global project memory",
        },
    )

    assert ordinary_chat["accepted"] is False
    assert promoted_reflection["accepted"] is True


def test_reflection_rag_evidence_projects_reflection_source_view() -> None:
    """Reflection-shaped RAG evidence should be a reflection source class."""

    module = _source_policy_module()
    state = _state()
    state["rag_result"] = {
        "memory_evidence": [
            {
                "source_kind": "reflection_run",
                "summary": "Approved project-wide lore.",
                "evidence_refs": [{"reflection_run_id": "reflection-1"}],
                "privacy_review": {
                    "user_details_removed": True,
                    "private_detail_risk": "low",
                    "boundary_assessment": "global project memory",
                },
            }
        ],
        "conversation_evidence": [],
        "external_evidence": [],
        "recall_evidence": [],
    }

    source_views = module.build_consolidation_source_views(state)
    views_by_key = {
        source_view["source_key"]: source_view
        for source_view in source_views
    }
    reflection_view = views_by_key["reflection_run"]
    policy = module.validate_lane_source_policy(
        "shared_memory_promotion",
        [reflection_view],
        privacy_review=reflection_view["privacy_review"],
    )

    assert reflection_view["source_kind"] == "reflection_run"
    assert reflection_view["source_refs"]
    assert policy["accepted"] is True


def test_reflection_user_style_signal_projects_style_source_view() -> None:
    """Reflection style signals should feed style lanes, not user memory."""

    module = _source_policy_module()
    state = _state()
    state["rag_result"] = {
        "memory_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "recall_evidence": [],
        "user_style_signal": {
            "source_reflection_run_ids": ["reflection-style-1"],
            "summary": "The user prefers concise first answers.",
        },
    }

    source_views = module.build_consolidation_source_views(state)
    reflection_view = next(
        source_view
        for source_view in source_views
        if source_view["source_key"] == "reflection_user_style_signal"
    )
    style_policy = module.validate_lane_source_policy(
        "interaction_style_image",
        [reflection_view],
    )
    user_memory_policy = module.validate_lane_source_policy(
        "user_memory_units",
        [reflection_view],
    )

    assert reflection_view["source_role"] == "user_style_signal"
    assert reflection_view["source_refs"] == [
        {
            "source": "reflection_run",
            "reflection_run_id": "reflection-style-1",
        }
    ]
    assert style_policy["accepted"] is True
    assert user_memory_policy["accepted"] is False


def test_reflection_promotion_and_style_sources_stay_separate() -> None:
    """Mixed reflection payloads must not share privacy/source-role metadata."""

    module = _source_policy_module()
    state = _state()
    state["rag_result"] = {
        "memory_evidence": [
            {
                "source_kind": "reflection_run",
                "summary": "Approved project-wide lore.",
                "evidence_refs": [{"reflection_run_id": "reflection-1"}],
                "privacy_review": {
                    "user_details_removed": True,
                    "private_detail_risk": "low",
                    "boundary_assessment": "global project memory",
                },
            }
        ],
        "conversation_evidence": [],
        "external_evidence": [],
        "recall_evidence": [],
        "user_style_signal": {
            "source_reflection_run_ids": ["reflection-style-1"],
            "summary": "The user prefers concise first answers.",
        },
    }

    source_views = module.build_consolidation_source_views(state)
    views_by_key = {
        source_view["source_key"]: source_view
        for source_view in source_views
    }
    promotion_view = views_by_key["reflection_run"]
    style_view = views_by_key["reflection_user_style_signal"]
    promotion_policy = module.validate_lane_source_policy(
        "shared_memory_promotion",
        [promotion_view],
        privacy_review=promotion_view["privacy_review"],
    )
    style_promotion_policy = module.validate_lane_source_policy(
        "shared_memory_promotion",
        [style_view],
        privacy_review=style_view.get("privacy_review"),
    )
    style_policy = module.validate_lane_source_policy(
        "interaction_style_image",
        [style_view],
    )

    assert promotion_view["privacy_review"]["user_details_removed"] is True
    assert "source_role" not in promotion_view
    assert style_view["source_role"] == "user_style_signal"
    assert "privacy_review" not in style_view
    assert promotion_policy["accepted"] is True
    assert style_promotion_policy["accepted"] is False
    assert style_policy["accepted"] is True


def test_source_policy_does_not_make_text_semantic_decisions() -> None:
    """Source policy may inspect source classes, not user text meaning."""

    module = _source_policy_module()
    source_text = inspect.getsource(module.validate_lane_source_policy)

    forbidden_fragments = (
        'get("summary")',
        "['summary']",
        "decontextualized_input",
        "decontextualized_input",
        "final_dialog",
        "喜欢",
        "提醒",
        "以后",
        "preference",
        "remind",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source_text
