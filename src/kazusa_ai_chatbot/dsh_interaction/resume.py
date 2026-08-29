"""Same-thread continuation and one-shot Brain grant enactment."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any

from kazusa_ai_chatbot.db.dsh_interactions import InteractionRepository
from kazusa_ai_chatbot.dsh_interaction.contracts import (
    INTERACTION_SCHEMA_VERSION,
    DshOneShotGrantV1,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json


class InteractionResumer:
    """Consume matching approval grants and continue one exact segment."""

    def __init__(
        self,
        *,
        interaction_store: InteractionRepository,
        continue_resolution: Callable[..., Awaitable[Mapping[str, Any]]],
        issue_continuation_authority: Callable[..., Awaitable[str] | str] | None = None,
    ) -> None:
        self._interaction_store = interaction_store
        self._continue_resolution = continue_resolution
        self._issue_continuation_authority = issue_continuation_authority

    def issue_grant(
        self,
        *,
        interaction_id: str,
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
        tool_name: str,
        arguments_digest: str,
        workspace_fingerprint: str,
        scope_fingerprint: str,
        policy_epoch: str,
    ) -> DshOneShotGrantV1:
        """Create a ten-minute grant bound to one semantic retry."""

        if not isinstance(activation_id, str) or not activation_id.strip():
            raise ValueError("grant activation id is required")
        if (
            isinstance(lease_epoch, bool)
            or not isinstance(lease_epoch, int)
            or lease_epoch < 1
        ):
            raise ValueError("grant lease epoch must be positive")
        now = datetime.now(UTC)
        return DshOneShotGrantV1(
            schema_version=INTERACTION_SCHEMA_VERSION,
            interaction_id=interaction_id,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            activation_id=activation_id,
            lease_epoch=lease_epoch,
            tool_name=tool_name,
            arguments_digest=arguments_digest,
            workspace_fingerprint=workspace_fingerprint,
            scope_fingerprint=scope_fingerprint,
            policy_epoch=policy_epoch,
            grant_status="available",
            issued_at=now.isoformat().replace("+00:00", "Z"),
            expires_at=(now + timedelta(minutes=10)).isoformat().replace("+00:00", "Z"),
        )

    async def resume(
        self,
        *,
        grant: DshOneShotGrantV1,
        reply_decision: Mapping[str, Any],
        resolution_thread_id: str,
        segment_id: str,
        activation_id: str,
        lease_epoch: int,
    ) -> Mapping[str, Any]:
        """Consume a matching grant and schedule exactly one continuation."""

        if grant.grant_status != "available":
            raise ValueError("grant is not available")
        if grant.resolution_thread_id != resolution_thread_id or grant.segment_id != segment_id:
            raise ValueError("grant thread or segment does not match")
        if grant.activation_id != activation_id or grant.lease_epoch != lease_epoch:
            raise ValueError("grant activation or lease does not match")
        if _parse(grant.expires_at) <= datetime.now(UTC):
            raise ValueError("grant has expired")
        typed_decision = _typed_reply_decision(reply_decision)
        if typed_decision["decision"] != "allow_once":
            raise ValueError("grant continuation requires allow_once")
        issuer = self._issue_continuation_authority
        if issuer is None:
            raise ValueError("canonical continuation authority issuer is required")
        token = issuer(
            grant=grant,
            reply_decision=typed_decision,
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
        )
        if inspect.isawaitable(token):
            token = await token
        if not isinstance(token, str) or not token.strip():
            raise ValueError("continuation authority issuer returned no token")
        consumed = await self._interaction_store.consume_grant(
            resolution_thread_id=grant.resolution_thread_id,
            segment_id=grant.segment_id,
            activation_id=grant.activation_id,
            lease_epoch=grant.lease_epoch,
            tool_name=grant.tool_name,
            arguments_digest=grant.arguments_digest,
            workspace_fingerprint=grant.workspace_fingerprint,
            scope_fingerprint=grant.scope_fingerprint,
            policy_epoch=grant.policy_epoch,
            now=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        if not consumed:
            raise ValueError("grant is unavailable or has already been consumed")
        result = await self._continue_resolution(
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
            grant=grant,
            decision=typed_decision,
            continuation_id=_continuation_id(grant, typed_decision),
            continuation_authority_token=token,
        )
        return result


def _typed_reply_decision(value: Mapping[str, Any]) -> dict[str, Any]:
    """Accept only the semantic reply result emitted by cognition."""

    if not isinstance(value, Mapping):
        raise ValueError("reply decision must be an object")
    decision = value.get("decision")
    if decision not in {"answer", "allow_once", "reject", "continue_waiting"}:
        raise ValueError("reply decision is unsupported")
    answer = value.get("answer")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ValueError("reply decision answer must be non-empty text")
    reason = value.get("reason")
    if reason is not None and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("reply decision reason must be non-empty text")
    return {
        "decision": decision,
        "answer": answer,
        "reason": reason,
    }


def _continuation_id(
    grant: DshOneShotGrantV1,
    decision: Mapping[str, Any],
) -> str:
    """Derive a stable continuation identity from the consumed grant lineage."""

    payload = {
        "interaction_id": grant.interaction_id,
        "resolution_thread_id": grant.resolution_thread_id,
        "segment_id": grant.segment_id,
        "activation_id": grant.activation_id,
        "lease_epoch": grant.lease_epoch,
        "tool_name": grant.tool_name,
        "arguments_digest": grant.arguments_digest,
        "workspace_fingerprint": grant.workspace_fingerprint,
        "scope_fingerprint": grant.scope_fingerprint,
        "policy_epoch": grant.policy_epoch,
        "decision": dict(decision),
    }
    return f"continuation_{sha256(canonical_json(payload)).hexdigest()}"


def _parse(value: str) -> datetime:
    """Parse a timezone-aware ISO timestamp."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("grant timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("grant timestamp requires timezone")
    return parsed.astimezone(UTC)
