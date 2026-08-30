"""Authenticated Brain boundary for character-owned DSH interactions."""

from __future__ import annotations

import hmac
import inspect
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from kazusa_ai_chatbot.db.dsh_interactions import (
    InteractionRepository,
    MongoInteractionStore,
)
from kazusa_ai_chatbot.dsh_interaction.auth import validate_request
from kazusa_ai_chatbot.dsh_interaction.contracts import (
    INTERACTION_SCHEMA_VERSION,
    MAX_INTERACTION_BODY_BYTES,
    DshBrainInteractionRequestV2,
    DshOneShotGrantV2,
)
from kazusa_ai_chatbot.dsh_interaction.decision import (
    BrainDecisionEngine,
    enact_decision,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json


class BrainInteractionService:
    """Authenticate, judge, persist, and enact one internal interaction."""

    def __init__(
        self,
        *,
        secret: bytes,
        judge: BrainDecisionEngine,
        interaction_store: InteractionRepository,
        context_provider: Callable[
            [DshBrainInteractionRequestV2],
            Awaitable[Mapping[str, Any]] | Mapping[str, Any],
        ]
        | None = None,
    ) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("Brain interaction secret is required")
        self._secret = secret
        self._judge = judge
        self._interaction_store = interaction_store
        self._context_provider = context_provider

    @classmethod
    def from_environment(
        cls,
        *,
        judge: BrainDecisionEngine | None = None,
        context_provider: Callable[..., Any] | None = None,
    ) -> BrainInteractionService:
        """Build the production service from explicit injected owners."""

        secret_text = os.environ.get("KAZUSA_DSH_BRAIN_SHARED_SECRET", "").strip()
        if not secret_text:
            raise ValueError("KAZUSA_DSH_BRAIN_SHARED_SECRET is required")
        if not os.environ.get("KAZUSA_DSH_TOOL_GATEWAY_SECRET", "").strip():
            raise ValueError("KAZUSA_DSH_TOOL_GATEWAY_SECRET is required")
        if judge is None:
            raise ValueError("canonical cognition judge injection is required")
        return cls(
            secret=secret_text.encode("utf-8"),
            judge=judge,
            interaction_store=MongoInteractionStore(),
            context_provider=context_provider,
        )

    def accepts_bearer(self, supplied: object) -> bool:
        """Return whether one bearer credential matches this service secret."""

        if not isinstance(supplied, str) or not supplied:
            return False
        try:
            configured = self._secret.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return bool(configured) and hmac.compare_digest(supplied, configured)

    async def handle_mapping(self, value: object) -> dict[str, Any]:
        """Parse and handle one signed request mapping."""

        request = DshBrainInteractionRequestV2.from_mapping(value)
        return await self.handle_signed(request)

    async def handle_signed(
        self,
        request: DshBrainInteractionRequestV2,
    ) -> dict[str, Any]:
        """Authenticate, durably claim, judge, and enact one interaction."""

        body_size = len(canonical_json(request.to_dict()))
        if body_size > MAX_INTERACTION_BODY_BYTES:
            raise ValueError("interaction body exceeds the bound")
        validate_request(request, secret=self._secret)
        prior = await self._interaction_store.get(request.interaction_id)
        if prior is not None:
            self._assert_same_request(prior, request)
            prior_result = prior.get("result")
            if isinstance(prior_result, Mapping):
                return dict(prior_result)
        else:
            await self._interaction_store.create(_initial_row(request))

        try:
            await self._interaction_store.consume_nonce(request.issuer, request.nonce)
        except ValueError:
            prior = await self._interaction_store.get(request.interaction_id)
            if prior is not None and isinstance(prior.get("result"), Mapping):
                return dict(prior["result"])
            raise

        granted = await self._consume_matching_grant(request)
        if granted is not None:
            return granted

        context: Mapping[str, Any] = {}
        if self._context_provider is not None:
            candidate = self._context_provider(request)
            if inspect.isawaitable(candidate):
                candidate = await candidate
            if not isinstance(candidate, Mapping):
                raise ValueError("interaction context provider returned a non-object")
            context = candidate
            context_fields = {
                key: context[key]
                for key in ("workspace_fingerprint", "policy_epoch")
                if isinstance(context.get(key), str) and context[key].strip()
            }
            if context_fields:
                await self._interaction_store.update(
                    request.interaction_id,
                    context_fields,
                )

        decision = await self._judge.decide(request, context=context)
        result = enact_decision(request, decision, context=context)
        decision_fields: dict[str, Any] = {
            "decision": decision.to_dict(),
            "decision_state": decision.decision,
            "result": result,
        }
        raw_grant = result.get("grant")
        if isinstance(raw_grant, Mapping):
            grant = DshOneShotGrantV2.from_mapping(raw_grant)
            decision_fields.update(_grant_fields(grant))
        await self._interaction_store.update(
            request.interaction_id,
            decision_fields,
        )
        if decision.decision == "allow_once":
            reconciled = await self._reconcile_immediate_grant(
                request,
                decision_fields,
            )
            if reconciled is None:
                result = _unavailable_allow_once_result(request)
                await self._interaction_store.update(
                    request.interaction_id,
                    {
                        "grant_status": "expired",
                        "status": "decided",
                        "result": result,
                    },
                )
            else:
                result = reconciled
                await self._interaction_store.update(
                    request.interaction_id,
                    {"status": "decided", "result": result},
                )
        else:
            await self._interaction_store.update(
                request.interaction_id,
                {"result": result, "status": "decided"},
            )
        return result

    async def _consume_matching_grant(
        self,
        request: DshBrainInteractionRequestV2,
    ) -> dict[str, Any] | None:
        """Consume one exact available grant for a fresh native retry."""

        if request.kind != "approval" or request.tool_name is None:
            return None
        now = _now()
        consumed = await self._interaction_store.consume_grant(
            resolution_thread_id=request.resolution_thread_id,
            segment_id=request.segment_id,
            activation_id=request.activation_id,
            lease_epoch=request.lease_epoch,
            tool_name=request.tool_name,
            arguments_digest=request.arguments_digest,
            workspace_fingerprint=request.workspace_fingerprint,
            scope_fingerprint=request.scope_fingerprint,
            policy_epoch=request.policy_epoch,
            now=now,
        )
        if consumed is None:
            return None
        raw_grant = consumed.get("grant")
        if not isinstance(raw_grant, Mapping):
            raise TypeError("consumed approval grant is unavailable")
        grant = DshOneShotGrantV2.from_mapping(raw_grant)
        if grant.grant_status != "consumed":
            raise ValueError("consumed approval grant status is invalid")
        reason = "matching approval grant authorized the native retry"
        decision = {
            "schema_version": INTERACTION_SCHEMA_VERSION,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "allow_once",
            "answer": None,
            "reason": reason,
        }
        result = {
            **decision,
            "grant": grant.to_dict(),
        }
        consumed_at = consumed.get("grant_consumed_at")
        if not isinstance(consumed_at, str) or not consumed_at.strip():
            consumed_at = now
        await self._interaction_store.update(
            request.interaction_id,
            {
                "decision": decision,
                "decision_state": "allow_once",
                "grant": grant.to_dict(),
                "grant_status": "consumed",
                "grant_consumed_at": consumed_at,
                "status": "decided",
                "result": result,
            },
        )
        return result

    async def _reconcile_immediate_grant(
        self,
        request: DshBrainInteractionRequestV2,
        row: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Atomically consume an immediate approval before acknowledging it."""

        raw_grant = row.get("grant")
        if not isinstance(raw_grant, Mapping):
            return None
        grant = DshOneShotGrantV2.from_mapping(raw_grant)
        prior_result = row.get("result")
        if not isinstance(prior_result, Mapping):
            return None
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
            now=_now(),
        )
        if consumed is None:
            return None
        consumed_grant = consumed.get("grant")
        grant_value = (
            {**dict(consumed_grant), "grant_status": "consumed"}
            if isinstance(consumed_grant, Mapping)
            else {**grant.to_dict(), "grant_status": "consumed"}
        )
        result = {**dict(prior_result), "grant": grant_value}
        await self._interaction_store.update(
            request.interaction_id,
            {
                "grant_status": "consumed",
                "grant": grant_value,
                "grant_consumed_at": consumed.get("grant_consumed_at", _now()),
                "result": result,
            },
        )
        return result

    @staticmethod
    def _assert_same_request(
        row: Mapping[str, Any],
        request: DshBrainInteractionRequestV2,
    ) -> None:
        """Reject an interaction id reused for a different signed identity."""

        if row.get("request_digest") != request.request_digest:
            raise ValueError("interaction id was reused with a different request")


def _initial_row(request: DshBrainInteractionRequestV2) -> dict[str, Any]:
    """Create the durable immutable identity and audit row."""

    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "issuer": request.issuer,
        "nonce": request.nonce,
        "request_digest": request.request_digest,
        "request_identity": request.to_dict(),
        "resolution_thread_id": request.resolution_thread_id,
        "segment_id": request.segment_id,
        "activation_id": request.activation_id,
        "lease_epoch": request.lease_epoch,
        "tool_name": request.tool_name,
        "operation_id": request.operation_id,
        "operation_payload_digest": request.operation_payload_digest,
        "arguments_digest": request.arguments_digest,
        "scope_fingerprint": request.scope_fingerprint,
        "audience_fingerprint": request.audience_fingerprint,
        "brain_conversation_ref": request.brain_conversation_ref,
        "profile_version": request.profile_version,
        "catalog_digest": request.catalog_digest,
        "model_route_digest": request.model_route_digest,
        "workspace_fingerprint": request.workspace_fingerprint,
        "policy_epoch": request.policy_epoch,
        "issued_reference_digest": request.issued_reference_digest,
        "platform": request.platform,
        "platform_channel_id": request.platform_channel_id,
        "global_user_id": request.global_user_id,
        "status": "processing",
        "decision": None,
        "decision_state": None,
        "result": None,
        "grant": None,
        "grant_status": None,
    }


def _grant_fields(grant: DshOneShotGrantV2) -> dict[str, Any]:
    """Project exact grant identity into atomic lookup fields."""

    return {
        "grant": grant.to_dict(),
        "grant_status": grant.grant_status,
        "resolution_thread_id": grant.resolution_thread_id,
        "segment_id": grant.segment_id,
        "activation_id": grant.activation_id,
        "lease_epoch": grant.lease_epoch,
        "tool_name": grant.tool_name,
        "arguments_digest": grant.arguments_digest,
        "workspace_fingerprint": grant.workspace_fingerprint,
        "scope_fingerprint": grant.scope_fingerprint,
        "policy_epoch": grant.policy_epoch,
        "expires_at": grant.expires_at,
    }


def _replayed_allow_once_result(
    request: DshBrainInteractionRequestV2,
) -> dict[str, Any]:
    """Return a non-reusable result for a consumed approval grant."""

    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "reject",
        "answer": None,
        "reason": "approval grant was already consumed",
    }


def _unavailable_allow_once_result(
    request: DshBrainInteractionRequestV2,
) -> dict[str, Any]:
    """Return a fixed safe result when an approval grant cannot be consumed."""

    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "reject",
        "answer": None,
        "reason": "approval grant was unavailable or expired",
    }


def _now() -> str:
    """Return one canonical current UTC timestamp."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
