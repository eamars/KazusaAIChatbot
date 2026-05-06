"""Tests for memory-writer prompt payload projection."""

from __future__ import annotations

import copy

import pytest

from kazusa_ai_chatbot.memory_writer_prompt_projection import (
    project_character_image_prompt_payload,
    project_memory_unit_extractor_prompt_payload,
    project_memory_unit_rewrite_prompt_payload,
    project_reflection_promotion_prompt_payload,
    project_relationship_prompt_payload,
)


CHARACTER_NAME = "杏山千纱 (Kyōyama Kazusa)"


def test_memory_unit_extractor_projects_speaker_metadata_without_text_changes() -> None:
    """Chat text should remain raw while speaker metadata uses profile names."""

    payload = {
        "chat_history_recent": [
            {
                "role": "user",
                "display_name": "测试用户",
                "body_text": "我觉得你刚才有点冷淡。",
            },
            {
                "role": "assistant",
                "display_name": "旧的显示名",
                "body_text": "我不是故意冷淡。",
            },
        ],
        "decontextualized_input": "我觉得你刚才有点冷淡。",
    }
    original_payload = copy.deepcopy(payload)

    projected = project_memory_unit_extractor_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )

    assert payload == original_payload
    assert projected is not payload
    user_row = projected["chat_history_recent"][0]
    character_row = projected["chat_history_recent"][1]
    assert user_row["speaker_name"] == "测试用户"
    assert user_row["body_text"] == "我觉得你刚才有点冷淡。"
    assert "role" not in user_row
    assert "display_name" not in user_row
    assert character_row["speaker_name"] == CHARACTER_NAME
    assert character_row["body_text"] == "我不是故意冷淡。"
    assert "role" not in character_row
    assert "display_name" not in character_row
    assert "active_character" not in str(projected)
    assert "speaker_kind" not in str(projected)


def test_reflection_promotion_projection_renames_source_utterance() -> None:
    """Promotion evidence should avoid model-facing active-character labels."""

    payload = {
        "evidence_cards": [
            {
                "active_character_utterance": "这条规则可以记。",
                "sanitized_observation": "公共规则可记。",
            },
        ],
    }

    projected = project_reflection_promotion_prompt_payload(
        payload,
        character_name=CHARACTER_NAME,
    )

    assert projected["evidence_cards"][0]["source_utterance"] == "这条规则可以记。"
    assert "active_character_utterance" in payload["evidence_cards"][0]
    assert "active_character" not in str(projected)


def test_noop_projection_functions_return_isolated_copies() -> None:
    """Projection surfaces without speaker rows should still isolate payloads."""

    payload = {"fact": {"text": "用户说我会继续检查。"}}
    projectors = [
        project_memory_unit_rewrite_prompt_payload,
        project_relationship_prompt_payload,
        project_character_image_prompt_payload,
        project_reflection_promotion_prompt_payload,
    ]

    for projector in projectors:
        projected = projector(payload, character_name=CHARACTER_NAME)
        projected["fact"]["text"] = "changed"
        assert payload["fact"]["text"] == "用户说我会继续检查。"


def test_projection_rejects_missing_character_name() -> None:
    """Prompt metadata must receive the profile-derived character name."""

    with pytest.raises(ValueError, match="character_name"):
        project_memory_unit_extractor_prompt_payload(
            {"chat_history_recent": []},
            character_name="",
        )
