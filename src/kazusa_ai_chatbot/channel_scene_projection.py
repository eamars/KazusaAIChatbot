"""Prompt-safe projection helpers for channel scene text."""

from __future__ import annotations

import re

_MAX_CHANNEL_NAME_CHARS = 40
_SYNTHETIC_GROUP_RE = re.compile(
    r'^(?:qq\s+)?group(?:\s*[:#_-]?\s*\d+)?$',
    re.IGNORECASE,
)
_PRIVATE_OR_GENERIC_NAMES = frozenset({
    'private',
    'private chat',
    'direct',
    'direct message',
    'dm',
    '私聊',
    '私人聊天',
})


def usable_channel_label(channel_type: str, channel_name: str) -> str:
    """Return a prompt-safe human group label, or empty string."""

    if str(channel_type or '').strip().lower() != 'group':
        return ''

    label = _single_line_text(channel_name)
    if not label:
        return ''
    if label.lower() in _PRIVATE_OR_GENERIC_NAMES:
        return ''
    if _SYNTHETIC_GROUP_RE.fullmatch(label):
        return ''
    if _is_pure_identifier(label):
        return ''

    return label[:_MAX_CHANNEL_NAME_CHARS].strip()


def project_channel_topic_text(
    channel_type: str,
    channel_name: str,
    channel_topic: str,
) -> str:
    """Render existing LLM-facing scene text for live chat prompts."""

    topic = str(channel_topic or '').strip()
    label = usable_channel_label(
        channel_type=channel_type,
        channel_name=channel_name,
    )
    if not label:
        return topic
    if topic:
        return f'“{label}”群聊中正在讨论：{topic}'
    return f'“{label}”群聊里刚出现这条消息，但还没有形成明确连续话题。'


def project_group_review_instruction_preamble(channel_name: str) -> str:
    """Render the group-review source-instruction preamble."""

    label = usable_channel_label(
        channel_type='group',
        channel_name=channel_name,
    )
    if not label:
        return '下面是已选中的群聊现场观察资料。'
    return f'下面是我在“{label}”群聊里看到的现场观察资料。'


def _single_line_text(value: str) -> str:
    """Return normalized single-line text, rejecting multiline input."""

    raw_value = str(value or '')
    if '\n' in raw_value or '\r' in raw_value:
        return ''
    return ' '.join(raw_value.strip().split())


def _is_pure_identifier(label: str) -> bool:
    """Return whether a label is just a numeric platform identifier."""

    compact = re.sub(r'[\s#:_-]+', '', label)
    return bool(compact) and compact.isdigit()
