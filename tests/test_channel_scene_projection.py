"""Tests for prompt-safe channel scene projection helpers."""

from __future__ import annotations

from kazusa_ai_chatbot.channel_scene_projection import (
    project_channel_topic_text,
    project_group_review_instruction_preamble,
    usable_channel_label,
)


def test_usable_channel_label_rejects_non_group_and_synthetic_labels() -> None:
    """Only human-readable group labels should be usable scene hints."""

    assert usable_channel_label('private', '动画讨论群') == ''
    assert usable_channel_label('system', '动画讨论群') == ''
    assert usable_channel_label('group', '') == ''
    assert usable_channel_label('group', 'Private') == ''
    assert usable_channel_label('group', 'Group 227608960') == ''
    assert usable_channel_label('group', 'QQ group 227608960') == ''
    assert usable_channel_label('group', '227608960') == ''
    assert usable_channel_label('group', '动画讨论群\n备用') == ''


def test_usable_channel_label_bounds_human_readable_group_label() -> None:
    """A usable label should be trimmed before prompt rendering."""

    label = usable_channel_label('group', ' 动画讨论群 ' + '甲' * 80)

    assert label.startswith('动画讨论群')
    assert len(label) <= 40


def test_project_channel_topic_text_integrates_label_into_existing_topic() -> None:
    """Group name should be folded into channel_topic text, not a new field."""

    projected = project_channel_topic_text(
        channel_type='group',
        channel_name='动画讨论群',
        channel_topic='新番角色和剧情走向',
    )

    assert projected == '“动画讨论群”群聊中正在讨论：新番角色和剧情走向'


def test_project_channel_topic_text_preserves_topic_without_usable_label() -> None:
    """Synthetic labels must not appear in prompt-facing scene text."""

    projected = project_channel_topic_text(
        channel_type='group',
        channel_name='Group 227608960',
        channel_topic='HPE 网络研讨会截图',
    )

    assert projected == 'HPE 网络研讨会截图'


def test_project_channel_topic_text_renders_named_group_without_topic() -> None:
    """A usable group label can still frame a topicless current scene."""

    projected = project_channel_topic_text(
        channel_type='group',
        channel_name='动画讨论群',
        channel_topic='',
    )

    assert projected == '“动画讨论群”群聊里刚出现这条消息，但还没有形成明确连续话题。'


def test_project_group_review_instruction_preamble_uses_existing_sentence() -> None:
    """Self-cognition should integrate the label into the source instruction."""

    assert project_group_review_instruction_preamble('动画讨论群') == (
        '下面是我在“动画讨论群”群聊里看到的现场观察资料。'
    )
    assert project_group_review_instruction_preamble('Group 227608960') == (
        '下面是已选中的群聊现场观察资料。'
    )
