"""Brain-owned semantic judgment and deterministic decision enactment."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.dsh_interaction.contracts import (
    INTERACTION_SCHEMA_VERSION,
    DshBrainInteractionDecisionV1,
    DshBrainInteractionRequestV1,
    DshOneShotGrantV1,
)


class BrainDecisionEngine:
    """Pass interaction meaning to a Brain judge and validate its result."""

    def __init__(
        self,
        *,
        judge: Callable[[DshBrainInteractionRequestV1, Mapping[str, Any]], Awaitable[Mapping[str, Any] | DshBrainInteractionDecisionV1] | Mapping[str, Any] | DshBrainInteractionDecisionV1],
    ) -> None:
        self._judge = judge

    async def decide(
        self,
        request: DshBrainInteractionRequestV1,
        *,
        context: Mapping[str, Any],
    ) -> DshBrainInteractionDecisionV1:
        """Return the Brain's kind-compatible semantic decision."""

        candidate = self._judge(request, context)
        if hasattr(candidate, "__await__"):
            candidate = await candidate  # type: ignore[assignment]
        if isinstance(candidate, DshBrainInteractionDecisionV1):
            decision = candidate
        else:
            if not isinstance(candidate, Mapping):
                raise ValueError("Brain judge returned a non-object decision")
            decision = DshBrainInteractionDecisionV1.from_mapping(dict(candidate))
        if decision.interaction_id != request.interaction_id:
            raise ValueError("Brain decision interaction identity mismatch")
        if decision.request_digest != request.request_digest:
            raise ValueError("Brain decision request digest mismatch")
        if decision.kind != request.kind:
            raise ValueError("Brain decision kind mismatch")
        return decision


def enact_decision(
    request: DshBrainInteractionRequestV1,
    decision: DshBrainInteractionDecisionV1,
    *,
    context: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Project a validated Brain decision into a bounded sidecar result."""

    if request.interaction_id != decision.interaction_id:
        raise ValueError("interaction decision identity mismatch")
    result: dict[str, Any] = {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": decision.decision,
        "reason": decision.reason,
    }
    if decision.answer is not None:
        result["answer"] = decision.answer
    if decision.response_goal is not None:
        result["response_goal"] = decision.response_goal
    if decision.relay_mode is not None:
        result["relay_mode"] = decision.relay_mode
    if decision.decision == "allow_once":
        result["grant"] = _build_grant(request, context or {}, now=now).to_dict()
    if decision.decision == "relay_to_user":
        result["checkpoint_required"] = True
    else:
        result["checkpoint_required"] = False
    return result


def _build_grant(
    request: DshBrainInteractionRequestV1,
    context: Mapping[str, Any],
    *,
    now: str | None = None,
) -> DshOneShotGrantV1:
    """Bind an approval grant to the exact request and deterministic context."""

    workspace = context.get("workspace_fingerprint", request.workspace_fingerprint)
    policy = context.get("policy_epoch", request.policy_epoch)
    if not isinstance(workspace, str) or not workspace:
        raise ValueError("workspace fingerprint is required for an approval grant")
    if not isinstance(policy, str) or not policy:
        raise ValueError("policy epoch is required for an approval grant")
    if request.tool_name is None:
        raise ValueError("tool name is required for an approval grant")
    current = _parse(now) if now is not None else datetime.now(UTC)
    request_expiry = _parse(request.expires_at)
    expires_at = min(
        request_expiry,
        current + timedelta(seconds=10 * 60),
    )
    if expires_at <= current:
        raise ValueError("approval grant lifetime is expired")
    issued = current.isoformat().replace("+00:00", "Z")
    expires = expires_at.isoformat().replace("+00:00", "Z")
    return DshOneShotGrantV1(
        schema_version=INTERACTION_SCHEMA_VERSION,
        interaction_id=request.interaction_id,
        resolution_thread_id=request.resolution_thread_id,
        segment_id=request.segment_id,
        activation_id=request.activation_id,
        lease_epoch=request.lease_epoch,
        tool_name=request.tool_name,
        arguments_digest=request.arguments_digest,
        workspace_fingerprint=workspace,
        scope_fingerprint=request.scope_fingerprint,
        policy_epoch=policy,
        grant_status="available",
        issued_at=issued,
        expires_at=expires,
    )


def _parse(value: str) -> datetime:
    """Parse a timezone-aware timestamp used for deterministic grant bounds."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("grant timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("grant timestamp requires timezone")
    return parsed.astimezone(UTC)
