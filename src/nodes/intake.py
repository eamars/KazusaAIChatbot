"""Stage 1 — Message Intake (no LLM).

Normalises a raw Discord message into the BotState fields needed by
downstream nodes.  This node is the entry point of the graph.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from state import BotState

_MENTION_RE = re.compile(r"<@!?(\d+)>")


def _is_directed_elsewhere(text: str, bot_id: str) -> bool:
    """Return True if the message mentions other users but NOT the bot.

    If the message contains no mentions at all, return False (it's a
    normal message).  If the bot is among the mentioned users, return
    False (the user is talking to us, possibly alongside others).
    """
    mention_ids = set(_MENTION_RE.findall(text))
    if not mention_ids:
        return False
    # If the bot is mentioned, the message is (at least partly) for us
    if bot_id and bot_id in mention_ids:
        return False
    # Only other users are mentioned — not for us
    return True


def intake(state: BotState) -> BotState:
    """Pass-through that ensures required fields are present and clean.

    When called from the Discord bot, the state is already populated with
    user_id, channel_id, etc.  This node does light normalisation:
      - strips Discord mention markup
      - filters out messages clearly directed at other users
      - sets should_respond
    """
    raw_text = state.get("message_text", "")
    bot_id = state.get("bot_id", "")

    # Check if message is directed at someone else BEFORE stripping mentions
    directed_elsewhere = _is_directed_elsewhere(raw_text, bot_id)

    # Strip Discord mention markup like <@12345> or <@!12345>
    text = _MENTION_RE.sub("", raw_text).strip()

    return {
        **state,
        "message_text": text,
        "should_respond": bool(text) and not directed_elsewhere,
        "timestamp": state.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }
