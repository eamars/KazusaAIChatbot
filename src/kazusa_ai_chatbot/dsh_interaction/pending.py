"""Durable-style relay pending state and exact reply lineage matching."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kazusa_ai_chatbot.dsh_interaction.contracts import (
    PENDING_SCHEMA_VERSION,
    DshInteractionPendingV1,
)


class PendingInteractionStore:
    """In-memory pending owner injected only by deterministic tests."""

    def __init__(self) -> None:
        self._rows: dict[str, DshInteractionPendingV1] = {}

    def create(
        self,
        *,
        interaction_id: str,
        request_digest: str = "sha256:request",
        resolution_thread_id: str,
        segment_id: str,
        brain_conversation_ref: str = "brain",
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        delivered_platform_message_id: str | None = None,
        response_goal: str = "Await the user's response.",
        relay_mode: str = "question",
        created_at: str = "2026-01-01T00:00:00Z",
        expires_at: str,
        grant: Any = None,
        request_identity: dict[str, Any] | None = None,
        decision: dict[str, Any] | None = None,
        delivery_receipt: dict[str, Any] | None = None,
        reply_result: dict[str, Any] | None = None,
    ) -> DshInteractionPendingV1:
        """Create or return one relay row by immutable interaction identity."""

        existing = self._rows.get(interaction_id)
        if existing is not None:
            return existing
        row = DshInteractionPendingV1(
            schema_version=PENDING_SCHEMA_VERSION,
            interaction_id=interaction_id,
            request_digest=request_digest,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            brain_conversation_ref=brain_conversation_ref,
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            status="delivered" if delivered_platform_message_id else "pending",
            response_goal=response_goal,
            relay_mode=relay_mode,
            created_at=created_at,
            expires_at=expires_at,
            delivered_platform_message_id=delivered_platform_message_id,
            delivery_receipt=delivery_receipt,
            replied_at=None,
            reply_platform_message_id=None,
            request_identity=request_identity or {"interaction_id": interaction_id},
            decision=decision,
            reply_result=reply_result,
            grant=grant,
        )
        self._rows[interaction_id] = row
        return row

    def mark_delivered(
        self,
        interaction_id: str,
        platform_message_id: str,
    ) -> DshInteractionPendingV1:
        """Record the exact adapter delivery lineage."""

        row = self._required(interaction_id)
        updated = DshInteractionPendingV1(
            **{
                **row.to_dict(),
                "delivered_platform_message_id": platform_message_id,
                "status": "delivered",
                "delivery_receipt": {
                    "platform_message_id": platform_message_id,
                    "recorded": True,
                },
            }
        )
        self._rows[interaction_id] = updated
        return updated

    def match_reply(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        reply_to_platform_message_id: str,
        now: str,
    ) -> DshInteractionPendingV1 | None:
        """Find an unexpired pending row by every reply lineage field."""

        current = _parse(now)
        for row in self._rows.values():
            if row.status not in {"pending", "delivered"}:
                continue
            if _parse(row.expires_at) <= current:
                continue
            if (
                row.platform == platform
                and row.platform_channel_id == platform_channel_id
                and row.global_user_id == global_user_id
                and row.delivered_platform_message_id == reply_to_platform_message_id
            ):
                return row
        return None

    def mark_replied(
        self,
        interaction_id: str,
        reply_platform_message_id: str,
        replied_at: str,
    ) -> DshInteractionPendingV1:
        """Close one pending row after an exact reply match."""

        row = self._required(interaction_id)
        updated = DshInteractionPendingV1(
            **{
                **row.to_dict(),
                "status": "replied",
                "replied_at": replied_at,
                "reply_platform_message_id": reply_platform_message_id,
            }
        )
        self._rows[interaction_id] = updated
        return updated

    def expire(self, *, now: str) -> int:
        """Mark all expired open relay rows closed."""

        current = _parse(now)
        changed = 0
        for key, row in list(self._rows.items()):
            if row.status in {"pending", "delivered"} and _parse(row.expires_at) <= current:
                self._rows[key] = DshInteractionPendingV1(
                    **{**row.to_dict(), "status": "expired"}
                )
                changed += 1
        return changed

    def get(self, interaction_id: str) -> DshInteractionPendingV1 | None:
        """Return one pending row."""

        return self._rows.get(interaction_id)

    def _required(self, interaction_id: str) -> DshInteractionPendingV1:
        """Return one existing pending row."""

        row = self._rows.get(interaction_id)
        if row is None:
            raise ValueError("pending interaction does not exist")
        return row


def _parse(value: str) -> datetime:
    """Parse a timezone-aware ISO timestamp."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("pending timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("pending timestamp requires timezone")
    return parsed.astimezone(UTC)
