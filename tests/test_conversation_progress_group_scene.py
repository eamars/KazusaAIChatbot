"""Deterministic tests for the transient public group-scene projection."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.conversation_progress import (
    build_group_scene_context,
    project_group_scene_prompt,
)
from kazusa_ai_chatbot.conversation_progress.policy import (
    GROUP_SCENE_MAX_ADDRESSED_NAMES,
    GROUP_SCENE_MAX_NAME_CHARS,
    GROUP_SCENE_MAX_RENDERED_CHARS,
    GROUP_SCENE_MAX_TURN_TEXT_CHARS,
    GROUP_SCENE_MAX_TURNS,
    GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS,
)


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "qq_group_public_scene_interleaving.json"
)


def _fixture() -> dict[str, Any]:
    """Load the sanitized interleaving fixture with explicit UTF-8 decoding."""

    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return fixture


def _build_from_fixture(
    fixture: dict[str, Any],
) -> dict[str, Any]:
    """Build one public scene using the fixed trigger and roster inputs."""

    trigger = fixture["trigger"]
    context = build_group_scene_context(
        ambient_logical_turns=fixture["ambient_logical_turns"],
        trigger_occurred_at=trigger["occurred_at"],
        trigger_speaker_name=trigger["speaker_name"],
        trigger_body_text=trigger["body_text"],
        trigger_addressed_global_user_ids=(
            trigger["addressed_global_user_ids"]
        ),
        trigger_reply_to_display_name=trigger["reply_to_display_name"],
        scope_users=fixture["scope_users"],
    )
    return context


def test_group_scene_merges_trigger_and_labels_relative_order() -> None:
    """The trigger is inserted chronologically and survives newest-turn caps."""

    context = _build_from_fixture(_fixture())

    assert [turn["scene_position"] for turn in context["turns"]] == [
        "before_trigger",
        "before_trigger",
        "trigger",
        "after_trigger",
        "after_trigger",
        "after_trigger",
    ]
    assert context["turns"][2]["text"] == "Which plan should we do first?"
    assert context["turns"][2]["speaker_name"] == "C"
    assert context["turns"][2]["addressed_names"] == ["Asuna"]
    assert context["omitted_turn_count"] == 2
    assert len(context["turns"]) == GROUP_SCENE_MAX_TURNS


def test_group_scene_redacts_ids_and_resolves_visible_names() -> None:
    """Only bounded visible names survive address projection."""

    fixture = _fixture()
    fixture["ambient_logical_turns"] = [
        deepcopy(fixture["ambient_logical_turns"][0])
    ]
    fixture["ambient_logical_turns"][0][
        "addressed_to_global_user_ids"
    ] = ["user-a", "private-global-id"]

    context = _build_from_fixture(fixture)
    turn = context["turns"][0]

    assert turn["addressed_names"] == ["A"]
    assert "private-global-id" not in json.dumps(context)
    assert "row-a1" not in json.dumps(context)
    assert "global_user_id" not in json.dumps(context)
    assert "platform_user_id" not in json.dumps(context)


def test_group_scene_applies_field_and_participant_caps() -> None:
    """Names, addresses, text, participants, and turns use exact caps."""

    fixture = _fixture()
    long_scope = [
        {
            "global_user_id": f"user-{index}",
            "display_name": "N" * (GROUP_SCENE_MAX_NAME_CHARS + 20),
        }
        for index in range(GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS + 8)
    ]
    long_turns = []
    for index in range(GROUP_SCENE_MAX_TURNS + 4):
        long_turn = deepcopy(fixture["ambient_logical_turns"][0])
        long_turn["turn_id"] = f"long-{index}"
        long_turn["occurred_at"] = f"2026-07-14T00:01:{index:02d}Z"
        long_turn["display_name"] = "S" * (GROUP_SCENE_MAX_NAME_CHARS + 20)
        long_turn["global_user_id"] = f"user-{index}"
        long_turn["addressed_to_global_user_ids"] = [
            f"user-{address_index}"
            for address_index in range(GROUP_SCENE_MAX_ADDRESSED_NAMES + 4)
        ]
        long_turn["fragments"] = [
            "T" * (GROUP_SCENE_MAX_TURN_TEXT_CHARS + 80)
        ]
        long_turns.append(long_turn)
    fixture["ambient_logical_turns"] = long_turns
    fixture["scope_users"] = long_scope

    context = _build_from_fixture(fixture)

    assert len(context["turns"]) <= GROUP_SCENE_MAX_TURNS
    assert len(context["visible_participants"]) <= (
        GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS
    )
    for turn in context["turns"]:
        assert len(turn["speaker_name"]) <= GROUP_SCENE_MAX_NAME_CHARS
        assert len(turn["reply_to_name"]) <= GROUP_SCENE_MAX_NAME_CHARS
        assert len(turn["text"]) <= GROUP_SCENE_MAX_TURN_TEXT_CHARS
        assert len(turn["addressed_names"]) <= GROUP_SCENE_MAX_ADDRESSED_NAMES
        assert all(
            len(name) <= GROUP_SCENE_MAX_NAME_CHARS
            for name in turn["addressed_names"]
        )


def test_group_scene_prompt_drops_old_turns_and_keeps_trigger() -> None:
    """Rendered fitting drops oldest context while retaining the trigger."""

    fixture = _fixture()
    fixture["trigger"]["body_text"] = "TRIGGER_MUST_SURVIVE"
    for turn in fixture["ambient_logical_turns"]:
        turn["fragments"] = ["X" * 360]

    context = _build_from_fixture(fixture)
    rendered = project_group_scene_prompt(context)

    assert len(rendered) <= GROUP_SCENE_MAX_RENDERED_CHARS
    assert "At trigger:" in rendered
    assert "TRIGGER_MUST_SURVIVE" in rendered
    assert "Before trigger:" in rendered or "After trigger:" in rendered


def test_group_scene_prompt_uses_semantic_labels_without_metadata() -> None:
    """The prompt is human-readable and contains no schema or storage terms."""

    rendered = project_group_scene_prompt(_build_from_fixture(_fixture()))

    assert "Before trigger:" in rendered
    assert "At trigger:" in rendered
    assert "After trigger:" in rendered
    for forbidden in (
        "schema_version",
        "conversation_row_ids",
        "global_user_id",
        "platform_user_id",
        "row-a1",
        "trace-a2",
        "timestamp",
    ):
        assert forbidden not in rendered


def test_group_scene_skips_malformed_ambient_rows_and_keeps_trigger() -> None:
    """A malformed ambient row degrades the scene without losing the trigger."""

    fixture = _fixture()
    del fixture['ambient_logical_turns'][0]['display_name']

    context = _build_from_fixture(fixture)
    rendered = project_group_scene_prompt(context)

    assert any(
        turn['scene_position'] == 'trigger'
        and turn['text'] == 'Which plan should we do first?'
        for turn in context['turns']
    )
    assert len(rendered) <= GROUP_SCENE_MAX_RENDERED_CHARS


def test_group_scene_render_cap_is_non_fatal_for_oversized_context() -> None:
    """An oversized transient context is rendered within the hard cap."""

    context = _build_from_fixture(_fixture())
    context['visible_participants'] = [
        'N' * (GROUP_SCENE_MAX_NAME_CHARS + 80)
        for _ in range(GROUP_SCENE_MAX_VISIBLE_PARTICIPANTS + 8)
    ]
    for turn in context['turns']:
        turn['speaker_name'] = 'S' * (GROUP_SCENE_MAX_NAME_CHARS + 80)
        turn['reply_to_name'] = 'R' * (GROUP_SCENE_MAX_NAME_CHARS + 80)
        turn['addressed_names'] = [
            'A' * (GROUP_SCENE_MAX_NAME_CHARS + 80)
            for _ in range(GROUP_SCENE_MAX_ADDRESSED_NAMES + 4)
        ]
        turn['text'] = 'T' * (GROUP_SCENE_MAX_TURN_TEXT_CHARS + 80)

    rendered = project_group_scene_prompt(context)

    assert len(rendered) <= GROUP_SCENE_MAX_RENDERED_CHARS


def test_group_scene_renderer_enforces_turn_cap_for_shaped_context() -> None:
    """The renderer retains the trigger and newest turns under the turn cap."""

    context = _build_from_fixture(_fixture())
    template = deepcopy(context['turns'][0])
    for index in range(GROUP_SCENE_MAX_TURNS + 10):
        extra_turn = deepcopy(template)
        extra_turn['scene_position'] = 'after_trigger'
        extra_turn['text'] = f'EXTRA_CONTEXT_{index}'
        context['turns'].append(extra_turn)

    rendered = project_group_scene_prompt(context)

    assert 'Which plan should we do first?' in rendered
    assert 'EXTRA_CONTEXT_0' not in rendered
    assert 'EXTRA_CONTEXT_15' in rendered


def test_group_scene_renderer_degrades_redundant_trigger_rows() -> None:
    """Redundant trigger labels do not make rendering fail or exceed caps."""

    context = _build_from_fixture(_fixture())
    template = deepcopy(context['turns'][0])
    template['scene_position'] = 'trigger'
    duplicate_markers = []
    for index in range(GROUP_SCENE_MAX_TURNS + 10):
        duplicate_marker = f'DUPLICATE_TRIGGER_{index}'
        duplicate_markers.append(duplicate_marker)
        extra_turn = deepcopy(template)
        extra_turn['text'] = duplicate_marker
        context['turns'].append(extra_turn)

    rendered = project_group_scene_prompt(context)

    assert len(rendered) <= GROUP_SCENE_MAX_RENDERED_CHARS
    assert 'Which plan should we do first?' in rendered
    assert all(marker not in rendered for marker in duplicate_markers)
