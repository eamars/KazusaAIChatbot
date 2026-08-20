"""Process-local chain-session registry for Cognition V3 resolver recurrence."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from uuid import uuid4

SESSION_SCHEMA = "chain_session.v1"

_IMMUTABLE_FIELDS = (
    "schema_version",
    "episode",
    "state_scope",
    "character_constraints",
    "character_identity_context",
    "character_operational_context",
    "relationship_context",
    "direct_facts",
    "available_actions",
    "available_resolver_capabilities",
    "runtime_capability_limits",
    "current_turn_relational_willingness",
    "scene_context",
    "private_continuity_context",
    "past_dialog_cognition_context",
    "group_engagement_action_context",
)
_CYCLE_FIELDS = (
    "mutable_state",
    "resolver_context",
    "resolver_goal_progress",
    "required_resolver_evidence_dependency",
    "resolver_cycle_index",
    "pending_resolver_resume",
)


class SessionContractError(RuntimeError):
    """Fail-closed chain-session contract violation."""


@dataclass(frozen=True)
class ReattachmentDecision:
    """Whether a cold input may continue an existing chain session."""

    reattached: bool
    divergent_field: str = ""


@dataclass
class ChainSessionV1:
    """One owner-scoped in-memory performance carrier for resolver cycles."""

    session_key_digest: str
    episode_id_digest: str
    scope: str
    immutable_input_digest: str
    original_evidence_digest: str
    expected_mutable_state_digest: str
    expected_mutable_state: Mapping[str, object]
    expected_willingness_digest: str
    expected_relational_willingness: Mapping[str, object] | None
    expected_cycle_index: int
    accepted_messages: tuple[tuple[str, str], ...]
    accepted_products: tuple[Mapping[str, object], ...]
    accepted_evidence: tuple[Mapping[str, object], ...]
    current_roster: tuple[str, ...]
    attempt_ledger: Mapping[str, int]
    token_ledger: Mapping[str, int]
    last_cycle_delta_digest: str = ""
    reanchor_used: bool = False
    last_output: Mapping[str, object] | None = None
    created_monotonic: float = field(default_factory=time.monotonic)
    last_used_monotonic: float = field(default_factory=time.monotonic)
    expires_monotonic: float = field(default_factory=time.monotonic)
    owner_token: str = ""


@dataclass(frozen=True)
class SessionClaim:
    """One bounded ownership outcome for a live chain session."""

    session: ChainSessionV1 | None
    claim_token: str = ""
    disposition: str = ""


class ChainSessionRegistry:
    """Bounded process-local session registry with LRU/TTL eviction."""

    def __init__(self, *, capacity: int = 256) -> None:
        """Create a bounded process-local session registry."""

        if capacity <= 0:
            raise ValueError("session capacity must be positive")
        self.capacity = capacity
        self._sessions: dict[str, ChainSessionV1] = {}

    def get(self, key: str) -> ChainSessionV1 | None:
        """Return a live session and refresh its last-use timestamp."""

        session = self._sessions.get(key)
        if session is None:
            return None
        if time.monotonic() >= session.expires_monotonic:
            self._sessions.pop(key, None)
            return None
        session.last_used_monotonic = time.monotonic()
        return session

    def put(self, session: ChainSessionV1) -> None:
        """Store one session, evicting expired and least-recently-used entries."""

        key = session.session_key_digest
        self._sessions[key] = session
        self._evict(key)

    def claim(self, key: str) -> SessionClaim:
        """Claim one live session for its single reattachment owner."""

        session = self.get(key)
        if session is None:
            return SessionClaim(session=None, disposition="session_miss")
        if session.owner_token:
            return SessionClaim(
                session=session,
                disposition="session_rebuilt_concurrent_owner",
            )
        claim_token = uuid4().hex
        session.owner_token = claim_token
        return SessionClaim(session=session, claim_token=claim_token)

    def release(self, key: str, claim_token: str) -> None:
        """Release a matching owner claim after its bounded continuation."""

        if not claim_token:
            return
        session = self.get(key)
        if session is not None and session.owner_token == claim_token:
            session.owner_token = ""

    def _evict(self, protected_key: str) -> None:
        """Evict expired then least-recently-used sessions to capacity."""

        now = time.monotonic()
        for key, session in list(self._sessions.items()):
            if key != protected_key and now >= session.expires_monotonic:
                self._sessions.pop(key, None)
        while len(self._sessions) > self.capacity:
            oldest = min(
                (
                    key
                    for key in self._sessions
                    if key != protected_key
                ),
                key=lambda candidate: self._sessions[
                    candidate
                ].last_used_monotonic,
                default=None,
            )
            if oldest is None:
                return
            self._sessions.pop(oldest, None)


def canonical_json_digest(value: object) -> str:
    """Return the canonical UTF-8 JSON SHA-256 digest of one value."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _presence(
    payload: Mapping[str, object],
    field_name: str,
) -> dict[str, object]:
    """Project one input field as explicit presence plus value."""

    return {
        "present": field_name in payload,
        "value": payload.get(field_name),
    }


def build_session_key(
    *,
    episode_id: str,
    state_scope: str,
    owner_identity: str,
) -> str:
    """Build the hashed session key without retaining raw identity fields."""

    if not episode_id.strip() or not state_scope.strip():
        raise SessionContractError(
            "session key requires non-empty episode id and scope"
        )
    if not owner_identity.strip():
        raise SessionContractError(
            "session key requires non-empty owner identity"
        )
    material = {
        "episode_id": episode_id,
        "state_scope": state_scope,
        "owner_identity": owner_identity,
    }
    return canonical_json_digest(material)


def build_immutable_digest(
    payload: Mapping[str, object],
    *,
    original_evidence_digest: str | None = None,
) -> str:
    """Build the immutable projection digest in the fixed field order."""

    projection = {
        field_name: _presence(payload, field_name)
        for field_name in _IMMUTABLE_FIELDS
    }
    if original_evidence_digest is not None:
        projection["original_evidence_digest"] = original_evidence_digest
    else:
        evidence = payload.get("evidence")
        if isinstance(evidence, Sequence) and not isinstance(
            evidence,
            (str, bytes, bytearray),
        ):
            projection["original_evidence_digest"] = canonical_json_digest(
                list(evidence)
            )
        else:
            projection["original_evidence_digest"] = ""
    return canonical_json_digest(projection)


def _cold_cycle_index(payload: Mapping[str, object]) -> int:
    """Return the normalized cold resolver-cycle index."""

    value = payload.get("resolver_cycle_index")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionContractError(
            "resolver_cycle_index must be an integer or absent"
        )
    if value < 0:
        raise SessionContractError(
            "resolver_cycle_index cannot be negative"
        )
    return value


def create_cold_session(
    *,
    payload: Mapping[str, object],
    episode_id: str,
    owner_identity: str,
    accepted_messages: Sequence[tuple[str, str]] = (),
    accepted_products: Sequence[Mapping[str, object]] = (),
    current_roster: Sequence[str] = (),
    attempt_ledger: Mapping[str, int] | None = None,
    token_ledger: Mapping[str, int] | None = None,
    last_output: Mapping[str, object] | None = None,
    expected_relational_willingness: Mapping[str, object] | None = None,
    ttl_seconds: float = 3600.0,
) -> ChainSessionV1:
    """Create one cold session from a validated input."""

    scope = payload["state_scope"]
    if not isinstance(scope, str) or not scope.strip():
        raise SessionContractError("session scope must be non-empty")
    evidence = payload["evidence"]
    if not isinstance(evidence, list):
        raise SessionContractError("evidence must be a list")
    if ttl_seconds <= 0:
        raise SessionContractError("session TTL must be positive")
    immutable_payload = dict(payload)
    if expected_relational_willingness is not None:
        immutable_payload["current_turn_relational_willingness"] = dict(
            expected_relational_willingness
        )
    willingness = immutable_payload.get("current_turn_relational_willingness")
    willingness_digest = (
        canonical_json_digest(willingness)
        if willingness is not None
        else ""
    )
    expected_state = payload["mutable_state"]
    if last_output is not None:
        state_update = last_output.get("state_update")
        if not isinstance(state_update, Mapping):
            raise SessionContractError("output state_update must be a mapping")
        if state_update.get("expected_previous_state") != expected_state:
            raise SessionContractError(
                "output expected_previous_state does not match cold mutable_state"
            )
        replacement = state_update.get("replacement_state")
        if not isinstance(replacement, Mapping):
            raise SessionContractError("replacement state must be a mapping")
        expected_state = replacement
    now = time.monotonic()
    return ChainSessionV1(
        session_key_digest=build_session_key(
            episode_id=episode_id,
            state_scope=scope,
            owner_identity=owner_identity,
        ),
        episode_id_digest=canonical_json_digest(episode_id),
        scope=scope,
        immutable_input_digest=build_immutable_digest(immutable_payload),
        original_evidence_digest=canonical_json_digest(evidence),
        expected_mutable_state_digest=canonical_json_digest(expected_state),
        expected_mutable_state=dict(expected_state),
        expected_willingness_digest=willingness_digest,
        expected_relational_willingness=(
            dict(willingness)
            if isinstance(willingness, Mapping)
            else None
        ),
        expected_cycle_index=_cold_cycle_index(payload) + 1,
        accepted_messages=tuple(accepted_messages),
        accepted_products=tuple(accepted_products),
        accepted_evidence=tuple(evidence),
        current_roster=tuple(current_roster),
        attempt_ledger=dict(attempt_ledger or {}),
        token_ledger=dict(token_ledger or {}),
        last_output=dict(last_output) if last_output is not None else None,
        created_monotonic=now,
        last_used_monotonic=now,
        expires_monotonic=now + ttl_seconds,
        owner_token="",
    )


def reattach_or_rebuild(
    *,
    session: ChainSessionV1,
    payload: Mapping[str, object],
) -> ReattachmentDecision:
    """Admit a cold session only when every immutable field still matches."""

    if build_immutable_digest(
        payload,
        original_evidence_digest=session.original_evidence_digest,
    ) != session.immutable_input_digest:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="immutable_projection",
        )
    incoming_index = _cold_cycle_index(payload)
    if incoming_index != session.expected_cycle_index:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="resolver_cycle_index",
        )
    if canonical_json_digest(payload["mutable_state"]) != (
        session.expected_mutable_state_digest
    ):
        return ReattachmentDecision(
            reattached=False,
            divergent_field="mutable_state",
        )
    if payload["mutable_state"] != session.expected_mutable_state:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="mutable_state",
        )
    evidence = payload["evidence"]
    if not isinstance(evidence, list):
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence",
        )
    accepted = list(session.accepted_evidence)
    if evidence[: len(accepted)] != accepted:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence_prefix",
        )
    if len(evidence) != len(accepted) + 1:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence_append_count",
        )
    new_row = evidence[-1]
    if not isinstance(new_row, Mapping):
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence_row_type",
        )
    evidence_ref = new_row.get("evidence_ref")
    if not isinstance(evidence_ref, Mapping) or evidence_ref.get(
        "source_kind"
    ) != "resolver_observation":
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence_source_kind",
        )
    new_handle = new_row.get("evidence_handle")
    if not isinstance(new_handle, str) or not new_handle.strip():
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence_handle",
        )
    accepted_handles = {
        row.get("evidence_handle")
        for row in accepted
        if isinstance(row, Mapping)
        and isinstance(row.get("evidence_handle"), str)
    }
    if new_handle in accepted_handles:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="evidence_handle",
        )
    willingness = payload.get("current_turn_relational_willingness")
    willingness_digest = (
        canonical_json_digest(willingness)
        if willingness is not None
        else ""
    )
    if willingness_digest != session.expected_willingness_digest:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="current_turn_relational_willingness",
        )
    return ReattachmentDecision(reattached=True)


def build_cycle_delta(
    *,
    session: ChainSessionV1,
    payload: Mapping[str, object],
) -> str:
    """Build the accepted cycle-delta projection after a valid reattachment."""

    evidence = payload["evidence"]
    if not isinstance(evidence, list) or len(evidence) != len(
        session.accepted_evidence
    ) + 1:
        raise SessionContractError("cycle delta requires exactly one new row")
    projection: dict[str, object] = {
        "new_evidence": dict(evidence[-1]),
    }
    projection.update({
        field_name: _presence(payload, field_name)
        for field_name in _CYCLE_FIELDS
    })
    return canonical_json_digest(projection)


def advance_session_after_output(
    *,
    session: ChainSessionV1,
    payload: Mapping[str, object],
    output: Mapping[str, object],
    accepted_messages: Sequence[tuple[str, str]] | None = None,
    accepted_products: Sequence[Mapping[str, object]] | None = None,
    current_roster: Sequence[str] | None = None,
    attempt_ledger: Mapping[str, int] | None = None,
    token_ledger: Mapping[str, int] | None = None,
    reanchor_used: bool | None = None,
) -> ChainSessionV1:
    """Advance state, evidence, willingness, and cycle index after output."""

    state_update = output["state_update"]
    if not isinstance(state_update, Mapping):
        raise SessionContractError("output state_update must be a mapping")
    expected_previous = state_update["expected_previous_state"]
    replacement = state_update["replacement_state"]
    if expected_previous != payload["mutable_state"]:
        raise SessionContractError(
            "output expected_previous_state does not match incoming mutable_state"
        )
    if not isinstance(replacement, Mapping):
        raise SessionContractError("replacement state must be a mapping")
    evidence = payload["evidence"]
    if not isinstance(evidence, list):
        raise SessionContractError("evidence must be a list")
    willingness = payload.get("current_turn_relational_willingness")
    now = time.monotonic()
    return ChainSessionV1(
        session_key_digest=session.session_key_digest,
        episode_id_digest=session.episode_id_digest,
        scope=session.scope,
        immutable_input_digest=session.immutable_input_digest,
        original_evidence_digest=session.original_evidence_digest,
        expected_mutable_state_digest=canonical_json_digest(replacement),
        expected_mutable_state=dict(replacement),
        expected_willingness_digest=(
            canonical_json_digest(willingness)
            if willingness is not None
            else ""
        ),
        expected_relational_willingness=(
            dict(willingness)
            if isinstance(willingness, Mapping)
            else None
        ),
        expected_cycle_index=_cold_cycle_index(payload) + 1,
        accepted_messages=(
            tuple(accepted_messages)
            if accepted_messages is not None
            else session.accepted_messages
        ),
        accepted_products=(
            tuple(accepted_products)
            if accepted_products is not None
            else session.accepted_products
        ),
        accepted_evidence=tuple(evidence),
        current_roster=(
            tuple(current_roster)
            if current_roster is not None
            else session.current_roster
        ),
        attempt_ledger=(
            dict(attempt_ledger)
            if attempt_ledger is not None
            else session.attempt_ledger
        ),
        token_ledger=(
            dict(token_ledger)
            if token_ledger is not None
            else session.token_ledger
        ),
        last_cycle_delta_digest=(
            build_cycle_delta(session=session, payload=payload)
            if isinstance(evidence, list)
            and len(evidence) == len(session.accepted_evidence) + 1
            else ""
        ),
        reanchor_used=(
            reanchor_used
            if reanchor_used is not None
            else session.reanchor_used
        ),
        last_output=dict(output),
        created_monotonic=session.created_monotonic,
        last_used_monotonic=now,
        expires_monotonic=session.expires_monotonic,
        owner_token=session.owner_token,
    )


__all__ = [
    "SESSION_SCHEMA",
    "ChainSessionRegistry",
    "ChainSessionV1",
    "ReattachmentDecision",
    "SessionClaim",
    "SessionContractError",
    "advance_session_after_output",
    "build_cycle_delta",
    "build_immutable_digest",
    "build_session_key",
    "canonical_json_digest",
    "create_cold_session",
    "reattach_or_rebuild",
]
