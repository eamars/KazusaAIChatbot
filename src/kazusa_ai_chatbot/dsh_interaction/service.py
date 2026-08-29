"""Brain interaction boundary coordinating auth, cognition, persistence, and relay."""

from __future__ import annotations

import hmac
import inspect
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

from kazusa_ai_chatbot.db.dsh_interactions import (
    InteractionRepository,
    MongoInteractionStore,
)
from kazusa_ai_chatbot.dsh_interaction.auth import validate_request
from kazusa_ai_chatbot.dsh_interaction.contracts import (
    INTERACTION_SCHEMA_VERSION,
    MAX_INTERACTION_BODY_BYTES,
    PENDING_SCHEMA_VERSION,
    DshBrainInteractionDecisionV1,
    DshBrainInteractionRequestV1,
    DshBrainReplyDecisionV1,
    DshInteractionPendingV1,
    DshOneShotGrantV1,
)
from kazusa_ai_chatbot.dsh_interaction.decision import (
    BrainDecisionEngine,
    enact_decision,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json

ReplyJudge = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Awaitable[Mapping[str, Any] | DshBrainInteractionDecisionV1]
    | Mapping[str, Any]
    | DshBrainInteractionDecisionV1,
]

ContinuationAuthorityIssuer = Callable[
    [DshBrainInteractionRequestV1, Mapping[str, Any], DshOneShotGrantV1 | None],
    Awaitable[str] | str,
]


class BrainInteractionService:
    """Handle one signed sidecar interaction at the Brain boundary."""

    def __init__(
        self,
        *,
        secret: bytes,
        judge: BrainDecisionEngine,
        interaction_store: InteractionRepository,
        reply_judge: ReplyJudge | None = None,
        deliver: Callable[
            [Mapping[str, Any], DshBrainInteractionRequestV1],
            Awaitable[str | Mapping[str, Any] | None]
            | str
            | Mapping[str, Any]
            | None,
        ]
        | None = None,
        context_provider: Callable[
            [DshBrainInteractionRequestV1],
            Awaitable[Mapping[str, Any]] | Mapping[str, Any],
        ]
        | None = None,
        continue_resolution: Callable[..., Awaitable[Mapping[str, Any]]]
        | None = None,
        issue_continuation_authority: ContinuationAuthorityIssuer | None = None,
    ) -> None:
        if not isinstance(secret, bytes) or not secret:
            raise ValueError("Brain interaction secret is required")
        self._secret = secret
        self._judge = judge
        self._interaction_store = interaction_store
        self._reply_judge = reply_judge
        self._deliver = deliver
        self._context_provider = context_provider
        self._continue_resolution = continue_resolution
        self._issue_continuation_authority = issue_continuation_authority

    @classmethod
    def from_environment(
        cls,
        *,
        judge: BrainDecisionEngine | None = None,
        reply_judge: ReplyJudge | None = None,
        deliver: Callable[..., Any] | None = None,
        context_provider: Callable[..., Any] | None = None,
        continue_resolution: Callable[..., Any] | None = None,
        issue_continuation_authority: ContinuationAuthorityIssuer | None = None,
    ) -> BrainInteractionService:
        """Build the production service from explicit injected owners.

        Cognition, dispatcher delivery, and resolver continuation remain
        injected composition seams.  The persistence owner and both bridge
        secrets come from the deployment environment and have no fallback.
        """

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
            reply_judge=reply_judge,
            deliver=deliver,
            context_provider=context_provider,
            continue_resolution=continue_resolution,
            issue_continuation_authority=issue_continuation_authority,
        )

    def accepts_bearer(self, supplied: object) -> bool:
        """Return whether one loopback bearer matches this service secret."""

        if not isinstance(supplied, str) or not supplied:
            return False
        try:
            configured = self._secret.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return bool(configured) and hmac.compare_digest(supplied, configured)

    async def handle_mapping(self, value: object) -> dict[str, Any]:
        """Parse and handle one signed request mapping."""

        request = DshBrainInteractionRequestV1.from_mapping(value)
        return await self.handle_signed(request)

    async def handle_signed(
        self,
        request: DshBrainInteractionRequestV1,
    ) -> dict[str, Any]:
        """Authenticate, durably claim, judge, and enact one interaction."""

        if len(canonical_json(request.to_dict())) > MAX_INTERACTION_BODY_BYTES:
            raise ValueError("interaction body exceeds the bound")
        validate_request(request, secret=self._secret)
        prior = await self._interaction_store.get(request.interaction_id)
        if prior is not None:
            self._assert_same_request(prior, request)
            prior_result = prior.get("result")
            if isinstance(prior_result, Mapping):
                if prior.get("decision_state") == "allow_once":
                    if prior.get("grant_status") == "consumed":
                        return _replayed_allow_once_result(request)
                    if prior.get("grant_status") == "available":
                        reconciled = await self._reconcile_immediate_grant(
                            request,
                            prior,
                        )
                        if reconciled is not None:
                            return reconciled
                        return _replayed_allow_once_result(request)
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
        if isinstance(result.get("grant"), Mapping):
            grant = DshOneShotGrantV1.from_mapping(result["grant"])
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
        elif decision.decision == "relay_to_user":
            result = await self._relay(request, decision, result)
        else:
            await self._interaction_store.update(
                request.interaction_id,
                {"result": result, "status": "decided"},
            )
        return result

    async def _consume_matching_grant(
        self,
        request: DshBrainInteractionRequestV1,
    ) -> dict[str, Any] | None:
        """Consume a relayed approval only for the exact fresh native request."""

        if request.kind != "approval" or request.tool_name is None:
            return None
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
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
            raise ValueError("consumed approval grant is unavailable")
        grant = DshOneShotGrantV1.from_mapping(raw_grant)
        if grant.grant_status != "consumed":
            raise ValueError("consumed approval grant status is invalid")
        reason = "matching approval grant authorized the native retry"
        decision = {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "allow_once",
            "answer": None,
            "response_goal": None,
            "relay_mode": None,
            "reason": reason,
        }
        result = {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "allow_once",
            "reason": reason,
            "grant": grant.to_dict(),
            "checkpoint_required": False,
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
        request: DshBrainInteractionRequestV1,
        row: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """Atomically consume an immediate approval before acknowledging it."""

        raw_grant = row.get("grant")
        if not isinstance(raw_grant, Mapping):
            return None
        grant = DshOneShotGrantV1.from_mapping(raw_grant)
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
            now=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )
        if consumed is None:
            return None
        consumed_grant = consumed.get("grant")
        if isinstance(consumed_grant, Mapping):
            grant_value = {
                **dict(consumed_grant),
                "grant_status": "consumed",
            }
        else:
            grant_value = {**grant.to_dict(), "grant_status": "consumed"}
        result = {
            **dict(prior_result),
            "grant": grant_value,
            "checkpoint_required": False,
        }
        await self._interaction_store.update(
            request.interaction_id,
            {
                "grant_status": "consumed",
                "grant": grant_value,
                "grant_consumed_at": consumed.get(
                    "grant_consumed_at",
                    datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                ),
                "result": result,
            },
        )
        return result

    async def handle_checkpoint(
        self,
        request: DshBrainInteractionRequestV1,
        *,
        response_goal: str,
        relay_mode: str,
    ) -> dict[str, Any]:
        """Return a durable relay checkpoint after exact request validation.

        The sidecar may repeat this call after a transport failure.  The
        interaction row, rather than the sidecar process, is the replay owner;
        a checkpoint is acknowledged only when the durable delivery receipt is
        present.
        """

        validate_request(request, secret=self._secret)
        if not isinstance(response_goal, str) or not response_goal.strip():
            raise ValueError("checkpoint response_goal is required")
        if relay_mode not in {"question", "approval", "plan_review"}:
            raise ValueError("checkpoint relay_mode is unsupported")
        row = await self._interaction_store.get(request.interaction_id)
        if row is None:
            raise ValueError("checkpoint interaction is unavailable")
        self._assert_same_request(row, request)
        result = row.get("result")
        receipt = row.get("delivery_receipt")
        message_id = row.get("delivered_platform_message_id")
        if (
            row.get("status") != "delivered"
            or not isinstance(result, Mapping)
            or not isinstance(receipt, Mapping)
            or not isinstance(message_id, str)
            or not message_id.strip()
        ):
            return {
                "schema_version": INTERACTION_SCHEMA_VERSION,
                "interaction_id": request.interaction_id,
                "request_digest": request.request_digest,
                "kind": request.kind,
                "decision": "relay_to_user",
                "reason": "delivery receipt is not durable",
                "response_goal": response_goal,
                "relay_mode": relay_mode,
                "checkpoint_required": False,
                "delivery_status": "failed",
            }
        stored_goal = row.get("response_goal")
        stored_mode = row.get("relay_mode")
        if stored_goal != response_goal or stored_mode != relay_mode:
            raise ValueError("checkpoint relay identity mismatch")
        return dict(result)

    async def _relay(
        self,
        request: DshBrainInteractionRequestV1,
        decision: DshBrainInteractionDecisionV1,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist a typed relay goal before delivery and checkpointing."""

        if decision.response_goal is None or decision.relay_mode is None:
            raise ValueError("relay decision must carry response_goal and relay_mode")
        now = datetime.now(UTC)
        created_at = now.isoformat().replace("+00:00", "Z")
        expires_at = (now + timedelta(hours=24)).isoformat().replace("+00:00", "Z")
        await self._interaction_store.update(
            request.interaction_id,
            {
                "status": "pending",
                "response_goal": decision.response_goal,
                "relay_mode": decision.relay_mode,
                "created_at": created_at,
                "expires_at": expires_at,
                "delivery_receipt": None,
                "delivered_platform_message_id": None,
            },
        )
        message_id: str | None = None
        receipt: dict[str, Any] | None = None
        if self._deliver is not None:
            candidate = self._deliver(
                {
                    "response_goal": decision.response_goal,
                    "relay_mode": decision.relay_mode,
                },
                request,
            )
            if inspect.isawaitable(candidate):
                candidate = await candidate
            message_id, receipt = _delivery_result(candidate)
        if message_id is None or receipt is None:
            failed = {
                **result,
                "checkpoint_required": False,
                "delivery_status": "failed",
            }
            await self._interaction_store.update(
                request.interaction_id,
                {
                    "status": "failed",
                    "delivery_receipt": None,
                    "result": failed,
                },
            )
            return failed
        checkpoint = {
            **result,
            "checkpoint_required": True,
            "pending_interaction_id": request.interaction_id,
            "delivered_platform_message_id": message_id,
        }
        await self._interaction_store.update(
            request.interaction_id,
            {
                "status": "delivered",
                "delivered_platform_message_id": message_id,
                "delivery_receipt": receipt,
                "result": checkpoint,
            },
        )
        return checkpoint

    async def handle_user_reply(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        reply_to_platform_message_id: str,
        reply_platform_message_id: str,
        reply_text: str,
        now: str | None = None,
    ) -> dict[str, Any]:
        """Judge an exact user reply and continue only from its typed result."""

        _required_identity(
            platform,
            platform_channel_id,
            global_user_id,
            reply_to_platform_message_id,
            reply_platform_message_id,
        )
        if not isinstance(reply_text, str) or not reply_text.strip():
            raise ValueError("reply_text is required")
        current = now or datetime.now(UTC).isoformat().replace("+00:00", "Z")
        row = await self._interaction_store.find_pending(
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            reply_to_platform_message_id=reply_to_platform_message_id,
            now=current,
        )
        if row is None:
            return {"status": "rejected", "reason": "pending interaction was not found"}
        request_identity = row.get("request_identity")
        if not isinstance(request_identity, Mapping):
            raise ValueError("pending request identity is unavailable")
        DshBrainInteractionRequestV1.from_mapping(request_identity)
        if self._reply_judge is None:
            raise ValueError("reply cognition judge is required")
        reply_context = {
            "platform": platform,
            "platform_channel_id": platform_channel_id,
            "global_user_id": global_user_id,
            "reply_to_platform_message_id": reply_to_platform_message_id,
            "reply_platform_message_id": reply_platform_message_id,
            "reply_text": reply_text,
        }
        candidate = self._reply_judge(row, reply_context)
        if inspect.isawaitable(candidate):
            candidate = await candidate
        decision = await _reply_decision(candidate, row)
        return await self.enact_typed_reply(
            row=row,
            decision=decision,
            reply_platform_message_id=reply_platform_message_id,
            current=current,
        )

    async def enact_typed_reply(
        self,
        *,
        row: Mapping[str, Any],
        decision: DshBrainReplyDecisionV1,
        reply_platform_message_id: str,
        current: str,
    ) -> dict[str, Any]:
        """Enact one cognition-owned reply result after normal chat commit.

        This method accepts only the typed P-stage result.  It deliberately has
        no reply-text parameter, so the raw user wording cannot become a
        continuation command or a grant rule.
        """

        if not isinstance(row, Mapping):
            raise ValueError("pending interaction row is required")
        interaction_id = row.get("interaction_id")
        if not isinstance(interaction_id, str) or not interaction_id.strip():
            raise ValueError("pending interaction id is required")
        durable_row = await self._interaction_store.get(interaction_id)
        if durable_row is not None:
            row = durable_row
        if (
            row.get("grant_status") == "consumed"
            and isinstance(row.get("continuation_authority_token"), str)
            and row.get("continuation_authority_token", "").strip()
        ):
            if decision.decision != "allow_once":
                return {
                    "status": "rejected",
                    "reason": "durable approval continuation identity mismatches",
                }
            return await self._reconcile_pending_continuation(
                row=row,
                decision=decision,
                reply_platform_message_id=reply_platform_message_id,
            )
        if row.get("status") == "replied":
            stored_result = row.get("result")
            if isinstance(stored_result, Mapping):
                return dict(stored_result)
            return {
                "status": "rejected",
                "reason": "pending interaction result is unavailable",
            }
        if row.get("status") == "continuation_pending":
            return await self._reconcile_pending_continuation(
                row=row,
                decision=decision,
                reply_platform_message_id=reply_platform_message_id,
            )
        if row.get("status") not in {"pending", "delivered"}:
            return {
                "status": "rejected",
                "reason": "pending interaction is no longer replyable",
            }
        request_identity = row.get("request_identity")
        if not isinstance(request_identity, Mapping):
            raise ValueError("pending request identity is unavailable")
        pending_request = DshBrainInteractionRequestV1.from_mapping(request_identity)
        if decision.interaction_id != pending_request.interaction_id:
            raise ValueError("reply decision interaction identity mismatch")
        if decision.request_digest != pending_request.request_digest:
            raise ValueError("reply decision request digest mismatch")
        _required_identity(reply_platform_message_id)
        reply_result = {
            "decision": decision.decision,
            "answer": decision.answer,
            "reason": decision.reason,
        }
        fields: dict[str, Any] = {
            "reply_result": reply_result,
            "replied_at": current,
            "reply_platform_message_id": reply_platform_message_id,
        }
        grant: DshOneShotGrantV1 | None = None
        continuation_authority_token: str | None = None
        if (
            decision.decision in {"answer", "allow_once"}
            and self._continue_resolution is None
        ):
            raise ValueError("resolution continuation owner is required")
        if decision.decision == "allow_once":
            grant = _build_grant_from_row(pending_request, row, now=current)
            # Obtain the canonical authority before writing the grant, so an
            # unavailable issuer cannot create a continuation without it.
            continuation_authority_token = await self._continuation_authority(
                pending_request,
                row,
                grant,
            )
            fields.update({
                **_grant_fields(grant),
                # Persist the canonical continuation token with the available
                # grant so a process loss can retry the same continuation.
                "continuation_authority_token": continuation_authority_token,
                "status": "continuation_pending",
                "result": {
                    "status": "continuation_pending",
                    **reply_result,
                },
            })
        elif decision.decision == "answer":
            continuation_authority_token = await self._continuation_authority(
                pending_request,
                row,
                None,
            )
        if decision.decision == "continue_waiting":
            waiting = {"status": "waiting", **reply_result}
            fields.update({"status": "pending", "result": waiting})
            await self._interaction_store.update(row["interaction_id"], fields)
            return waiting
        if decision.decision == "reject":
            rejected = {"status": "rejected", **reply_result}
            fields.update({"status": "replied", "result": rejected})
            await self._interaction_store.update(row["interaction_id"], fields)
            return rejected
        if continuation_authority_token is None:
            raise ValueError("continuation authority token is required")
        continuation_pending_fields = {
            **fields,
            "status": "continuation_pending",
            "continuation_authority_token": continuation_authority_token,
            "result": {
                "status": "continuation_pending",
                **reply_result,
            },
        }
        await self._interaction_store.update(
            row["interaction_id"],
            continuation_pending_fields,
        )
        output = await self._call_continuation(
            row=row,
            pending_request=pending_request,
            reply_result=reply_result,
            grant=grant,
            continuation_authority_token=continuation_authority_token,
        )
        await self._interaction_store.update(
            row["interaction_id"],
            {"status": "replied", "result": output},
        )
        return output

    async def _reconcile_pending_continuation(
        self,
        *,
        row: Mapping[str, Any],
        decision: DshBrainReplyDecisionV1,
        reply_platform_message_id: str,
    ) -> dict[str, Any]:
        """Retry one durable continuation operation after transport loss."""

        request_identity = row.get("request_identity")
        if not isinstance(request_identity, Mapping):
            raise ValueError("pending request identity is unavailable")
        pending_request = DshBrainInteractionRequestV1.from_mapping(request_identity)
        if decision.interaction_id != pending_request.interaction_id:
            raise ValueError("reply decision interaction identity mismatch")
        if decision.request_digest != pending_request.request_digest:
            raise ValueError("reply decision request digest mismatch")
        stored_reply = row.get("reply_result")
        expected_reply = {
            "decision": decision.decision,
            "answer": decision.answer,
            "reason": decision.reason,
        }
        if not isinstance(stored_reply, Mapping) or dict(stored_reply) != expected_reply:
            return {
                "status": "rejected",
                "reason": "continuation decision does not match durable reply",
            }
        stored_reply_id = row.get("reply_platform_message_id")
        if stored_reply_id != reply_platform_message_id:
            raise ValueError("reply message identity does not match durable reply")
        token = row.get("continuation_authority_token")
        if not isinstance(token, str) or not token.strip():
            raise ValueError("durable continuation authority token is unavailable")
        raw_grant = row.get("grant")
        grant = (
            DshOneShotGrantV1.from_mapping(raw_grant)
            if isinstance(raw_grant, Mapping)
            else None
        )
        output = await self._call_continuation(
            row=row,
            pending_request=pending_request,
            reply_result=expected_reply,
            grant=grant,
            continuation_authority_token=token,
        )
        await self._interaction_store.update(
            pending_request.interaction_id,
            {"status": "replied", "result": output},
        )
        return output

    async def _call_continuation(
        self,
        *,
        row: Mapping[str, Any],
        pending_request: DshBrainInteractionRequestV1,
        reply_result: Mapping[str, Any],
        grant: DshOneShotGrantV1 | None,
        continuation_authority_token: str,
    ) -> dict[str, Any]:
        """Invoke the injected resolver owner with typed, hidden authority."""

        if self._continue_resolution is None:
            raise ValueError("resolution continuation owner is required")
        continuation_fields = {
            "resolution_thread_id": row["resolution_thread_id"],
            "segment_id": row["segment_id"],
            "activation_id": row["activation_id"],
            "lease_epoch": row["lease_epoch"],
            "interaction_id": row["interaction_id"],
            "decision": dict(reply_result),
            "grant": grant,
            "continuation_authority_token": continuation_authority_token,
            "brain_conversation_ref": row["brain_conversation_ref"],
            "platform": row["platform"],
            "platform_channel_id": row["platform_channel_id"],
            "global_user_id": row["global_user_id"],
            "scope_fingerprint": row["scope_fingerprint"],
            "audience_fingerprint": row["audience_fingerprint"],
            "workspace_fingerprint": row["workspace_fingerprint"],
            "model_route_digest": row["model_route_digest"],
            "catalog_digest": row["catalog_digest"],
            "profile_version": row["profile_version"],
            "policy_epoch": row["policy_epoch"],
            "issued_reference_digest": row["issued_reference_digest"],
        }
        continuation = self._continue_resolution(**continuation_fields)
        if inspect.isawaitable(continuation):
            continuation = await continuation
        if not isinstance(continuation, Mapping):
            raise ValueError("resolution continuation returned a non-object")
        lineage = {
            "resolution_thread_id": row["resolution_thread_id"],
            "segment_id": row["segment_id"],
            "activation_id": row["activation_id"],
            "lease_epoch": row["lease_epoch"],
        }
        for field_name, expected in lineage.items():
            if field_name in continuation and continuation[field_name] != expected:
                raise ValueError(
                    f"continuation lineage mismatch: {field_name}"
                )
        return {**dict(continuation), **lineage}

    async def _continuation_authority(
        self,
        request: DshBrainInteractionRequestV1,
        row: Mapping[str, Any],
        grant: DshOneShotGrantV1 | None,
    ) -> str:
        """Issue a fresh authority token for one consumed approval grant."""

        issuer = self._issue_continuation_authority
        if issuer is None:
            raise ValueError(
                "canonical continuation authority issuer is required"
            )
        candidate = issuer(request, row, grant)
        if inspect.isawaitable(candidate):
            candidate = await candidate
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError("continuation authority issuer returned no token")
        return candidate

    async def pending_for_reply(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        reply_to_platform_message_id: str,
        now: str,
    ) -> DshInteractionPendingV1 | None:
        """Load one pending interaction by exact adapter reply lineage."""

        row = await self._interaction_store.find_pending(
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            reply_to_platform_message_id=reply_to_platform_message_id,
            now=now,
        )
        return (
            None
            if row is None
            else DshInteractionPendingV1.from_mapping(
                _pending_public_mapping(row),
            )
        )

    async def pending_row_for_reply(
        self,
        *,
        platform: str,
        platform_channel_id: str,
        global_user_id: str,
        reply_to_platform_message_id: str,
        now: str,
    ) -> dict[str, Any] | None:
        """Load the durable internal row for normal-chat reply projection."""

        _required_identity(
            platform,
            platform_channel_id,
            global_user_id,
            reply_to_platform_message_id,
        )
        row = await self._interaction_store.find_pending(
            platform=platform,
            platform_channel_id=platform_channel_id,
            global_user_id=global_user_id,
            reply_to_platform_message_id=reply_to_platform_message_id,
            now=now,
        )
        return None if row is None else dict(row)

    @staticmethod
    def _assert_same_request(
        row: Mapping[str, Any],
        request: DshBrainInteractionRequestV1,
    ) -> None:
        """Reject an interaction id reused for a different signed identity."""

        stored_digest = row.get("request_digest")
        if stored_digest != request.request_digest:
            raise ValueError("interaction id was reused with a different request")


def _initial_row(request: DshBrainInteractionRequestV1) -> dict[str, Any]:
    """Create the durable immutable request identity row."""

    return {
        "schema_version": PENDING_SCHEMA_VERSION,
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
        "status": "processing",
        "decision": None,
        "decision_state": None,
        "result": None,
        "response_goal": "Awaiting Brain decision.",
        "relay_mode": "question",
        "created_at": request.issued_at,
        "expires_at": request.expires_at,
        "platform": request.platform,
        "platform_channel_id": request.platform_channel_id,
        "global_user_id": request.global_user_id,
        "delivered_platform_message_id": None,
        "delivery_receipt": None,
        "replied_at": None,
        "reply_platform_message_id": None,
        "reply_result": None,
        "grant": None,
        "grant_status": None,
        "continuation_authority_token": None,
    }


def _pending_public_mapping(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one durable row into the public pending DTO shape."""

    fields = {
        "schema_version", "interaction_id", "request_digest",
        "resolution_thread_id", "segment_id", "brain_conversation_ref",
        "platform", "platform_channel_id", "global_user_id", "status",
        "response_goal", "relay_mode", "created_at", "expires_at",
        "delivered_platform_message_id", "delivery_receipt", "replied_at",
        "reply_platform_message_id", "request_identity", "decision",
        "reply_result", "grant",
    }
    return {key: row[key] for key in fields if key in row}


def _required_identity(*values: str) -> None:
    """Require every adapter identity component at the boundary."""

    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("exact platform, channel, user, and message identity is required")


def _delivery_result(
    value: str | Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    """Normalize an adapter's actual delivery receipt without fabricating ids."""

    if isinstance(value, Mapping):
        message_id = value.get("platform_message_id")
        if isinstance(message_id, str) and message_id.strip():
            receipt = dict(value)
            if not isinstance(receipt.get("delivered_at"), str) or not receipt["delivered_at"].strip():
                return None, None
            return message_id, receipt
    return None, None


def _grant_fields(grant: DshOneShotGrantV1) -> dict[str, Any]:
    """Project grant identity into the durable atomic-lookup fields."""

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
    request: DshBrainInteractionRequestV1,
) -> dict[str, Any]:
    """Return a non-reusable result for a replayed immediate approval."""

    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "reject",
        "reason": "approval grant was already consumed",
        "checkpoint_required": False,
    }


def _unavailable_allow_once_result(
    request: DshBrainInteractionRequestV1,
) -> dict[str, Any]:
    """Return a fixed safe result when an immediate grant cannot be consumed."""

    return {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "interaction_id": request.interaction_id,
        "request_digest": request.request_digest,
        "kind": request.kind,
        "decision": "reject",
        "reason": "approval grant was unavailable or expired",
        "checkpoint_required": False,
    }


async def _reply_decision(
    candidate: object,
    row: Mapping[str, Any],
) -> DshBrainReplyDecisionV1:
    """Validate the cognition-owned reply result as one typed decision."""

    if not isinstance(candidate, Mapping):
        raise ValueError("reply cognition returned a non-object decision")
    request_identity = row.get("request_identity")
    if not isinstance(request_identity, Mapping):
        raise ValueError("pending request identity is unavailable")
    request = DshBrainInteractionRequestV1.from_mapping(request_identity)
    decision = DshBrainReplyDecisionV1.from_mapping(dict(candidate))
    if decision.interaction_id != request.interaction_id:
        raise ValueError("reply decision interaction identity mismatch")
    if decision.request_digest != request.request_digest:
        raise ValueError("reply decision request digest mismatch")
    return decision


def _build_grant_from_row(
    request: DshBrainInteractionRequestV1,
    row: Mapping[str, Any],
    *,
    now: str | None = None,
) -> DshOneShotGrantV1:
    """Build one grant from immutable pending request and policy context."""

    workspace = row.get("workspace_fingerprint")
    policy = row.get("policy_epoch")
    if not isinstance(workspace, str) or not workspace:
        raise ValueError("workspace fingerprint is required for an approval grant")
    if not isinstance(policy, str) or not policy:
        raise ValueError("policy epoch is required for an approval grant")
    if request.tool_name is None:
        raise ValueError("tool name is required for an approval grant")
    current = _parse_time(now) if now is not None else datetime.now(UTC)
    expires_at = min(
        _parse_time(str(row.get("expires_at", request.expires_at))),
        current + timedelta(minutes=10),
    )
    if expires_at <= current:
        raise ValueError("approval grant lifetime is expired")
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
        issued_at=current.isoformat().replace("+00:00", "Z"),
        expires_at=expires_at.isoformat().replace("+00:00", "Z"),
    )


def _parse_time(value: str) -> datetime:
    """Parse a timezone-aware timestamp for grant bounds."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("interaction timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("interaction timestamp requires timezone")
    return parsed.astimezone(UTC)
