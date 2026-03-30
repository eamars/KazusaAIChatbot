"""Stage 1 — Message Intake (no LLM).

Normalises a raw Discord message into the BotState fields needed by
downstream nodes.  This node is the entry point of the graph.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from bot.state import BotState


def intake(state: BotState) -> BotState:
    """Pass-through that ensures required fields are present and clean.

    When called from the Discord bot, the state is already populated with
    user_id, channel_id, etc.  This node does light normalisation:
      - strips Discord mention markup
      - sets should_respond (always True here; the Discord bot layer
        already filters irrelevant messages)
    """
    text = state.get("message_text", "")

    # Strip Discord mention markup like <@12345> or <@!12345>
    text = re.sub(r"<@!?\d+>", "", text).strip()

    return {
        **state,
        "message_text": text,
        "should_respond": bool(text),
        "timestamp": state.get("timestamp", datetime.now(timezone.utc).isoformat()),
    }
