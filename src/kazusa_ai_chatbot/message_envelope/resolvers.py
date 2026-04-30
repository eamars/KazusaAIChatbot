"""Default mention resolvers for adapter-produced message envelopes.

This module owns platform-neutral resolver implementations. Inputs are raw
mention fragments emitted by platform normalizers; outputs are resolved mention
TypedDicts. Future deployments can extend this slot with a profile-backed
resolver without changing normalizers or service intake code.
"""

from __future__ import annotations

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.message_envelope.types import RawMention, ResolvedMention


class PassthroughMentionResolver:
    """Resolve mentions with the data already present at the adapter seam."""

    def resolve(self, raw_mention: RawMention) -> ResolvedMention:
        """Resolve one raw mention without profile-store lookup.

        Args:
            raw_mention: Mention fragment produced by a platform normalizer.

        Returns:
            Resolved mention payload. Bot mentions receive the configured
            character global id; other mentions keep an empty global id until a
            profile-backed resolver is introduced.
        """

        entity_kind = raw_mention.get("entity_kind", "unknown")
        global_user_id = ""
        if entity_kind == "bot":
            global_user_id = CHARACTER_GLOBAL_USER_ID

        resolved: ResolvedMention = {
            "platform": raw_mention.get("platform", ""),
            "platform_user_id": raw_mention.get("platform_user_id", ""),
            "global_user_id": global_user_id,
            "display_name": raw_mention.get("display_name", ""),
            "entity_kind": entity_kind,
            "raw_text": raw_mention.get("raw_text", ""),
        }
        return resolved
