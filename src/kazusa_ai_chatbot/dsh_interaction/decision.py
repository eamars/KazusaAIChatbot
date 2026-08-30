"""Brain-owned semantic judgment and deterministic decision enactment."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.dsh_interaction.contracts import (
    GRANT_SECONDS,
    INTERACTION_SCHEMA_VERSION,
    DshBrainInteractionDecisionV2,
    DshBrainInteractionRequestV2,
    DshOneShotGrantV2,
)

DecisionCandidate = (
    Mapping[str, Any]
    | DshBrainInteractionDecisionV2
)
DecisionJudge = Callable[
    [DshBrainInteractionRequestV2, Mapping[str, Any]],
    Awaitable[DecisionCandidate] | DecisionCandidate,
]


class BrainDecisionEngine:
    """Pass interaction meaning to the existing cognition-owned judge."""

    def __init__(self, *, judge: DecisionJudge) -> None:
        self._judge = judge

    async def decide(
        self,
        request: DshBrainInteractionRequestV2,
        *,
        context: Mapping[str, Any],
    ) -> DshBrainInteractionDecisionV2:
        """Return the judge's validated kind-compatible decision."""

        candidate = self._judge(request, context)
        if inspect.isawaitable(candidate):
            candidate = await candidate
        if isinstance(candidate, DshBrainInteractionDecisionV2):
            decision = candidate
        else:
            if not isinstance(candidate, Mapping):
                raise TypeError("Brain judge returned a non-object decision")
            decision = DshBrainInteractionDecisionV2.from_mapping(dict(candidate))
        if decision.interaction_id != request.interaction_id:
            raise ValueError("Brain decision interaction identity mismatch")
        if decision.request_digest != request.request_digest:
            raise ValueError("Brain decision request digest mismatch")
        if decision.kind != request.kind:
            raise ValueError("Brain decision kind mismatch")
        return decision


def enact_decision(
    request: DshBrainInteractionRequestV2,
    decision: DshBrainInteractionDecisionV2,
    *,
    context: Mapping[str, Any] | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Project one validated decision into the internal response envelope."""

    if request.interaction_id != decision.interaction_id:
        raise ValueError("interaction decision identity mismatch")
    if request.request_digest != decision.request_digest:
        raise ValueError("interaction decision request digest mismatch")
    result: dict[str, Any] = {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": decision.decision,
        "answer": decision.answer,
        "reason": decision.reason,
    }
    if decision.decision == "allow_once":
        grant = _build_grant(request, context or {}, now=now)
        result["grant"] = grant.to_dict()
    return result


def _build_grant(
    request: DshBrainInteractionRequestV2,
    context: Mapping[str, Any],
    *,
    now: str | None = None,
) -> DshOneShotGrantV2:
    """Bind a one-shot grant to the exact request and policy context."""

    workspace = context.get("workspace_fingerprint", request.workspace_fingerprint)
    policy = context.get("policy_epoch", request.policy_epoch)
    if not isinstance(workspace, str) or not workspace.strip():
        raise ValueError("workspace fingerprint is required for an approval grant")
    if not isinstance(policy, str) or not policy.strip():
        raise ValueError("policy epoch is required for an approval grant")
    if request.tool_name is None:
        raise ValueError("tool name is required for an approval grant")
    current = _parse(now) if now is not None else datetime.now(UTC)
    request_expiry = _parse(request.expires_at)
    expires_at = min(
        request_expiry,
        current + timedelta(seconds=GRANT_SECONDS),
    )
    if expires_at <= current:
        raise ValueError("approval grant lifetime is expired")
    issued = current.isoformat().replace("+00:00", "Z")
    expires = expires_at.isoformat().replace("+00:00", "Z")
    return DshOneShotGrantV2(
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


def parse_timestamp(value: str) -> datetime:
    """Parse one timezone-aware timestamp used for grant bounds."""

    return _parse(value)


def _parse(value: str) -> datetime:
    """Parse one timezone-aware UTC timestamp."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("interaction timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("interaction timestamp requires timezone")
    return parsed.astimezone(UTC)
