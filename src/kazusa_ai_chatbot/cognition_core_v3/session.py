"""Process-local chain-session registry for Cognition V3 resolver recurrence."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Self
from uuid import uuid4

from kazusa_ai_chatbot.cognition_core_v2.output_projection import (
    build_state_update,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    ResolverValidationError,
    validate_current_turn_relational_willingness,
    validate_required_resolver_evidence_dependency,
    validate_resolver_capability_request,
    validate_resolver_goal_progress,
    validate_resolver_pending_resume,
)

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
_CANONICAL_STATE_FIELDS = frozenset(
    {
        "goals",
        "threats",
        "active_events",
        "knowledge_gaps",
        "affect_activations",
    }
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

    schema_version: str
    session_key_digest: str
    episode_id_digest: str
    scope: str
    immutable_input_digest: str
    original_evidence_digest: str
    expected_mutable_state_digest: str
    expected_willingness_digest: str
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

    def __post_init__(self) -> None:
        """Validate the sealed process-local carrier version."""

        if self.schema_version != SESSION_SCHEMA:
            raise SessionContractError(
                "session schema_version must be chain_session.v1"
            )


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


class _ImmutableDigest(str):
    """String digest carrying the process-local projection for diagnostics."""

    __slots__ = ("projection",)

    def __new__(
        cls,
        digest: str,
        projection: Mapping[str, object],
    ) -> Self:
        """Build a string-compatible digest with an immutable snapshot."""

        value = super().__new__(cls, digest)
        value.projection = projection
        return value


def _canonical_json_value(value: object, *, path: str = "$") -> object:
    """Validate and copy one value admitted by the canonical JSON contract."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise SessionContractError(
            f"canonical JSON rejects floating-point value at {path}"
        )
    if isinstance(value, Mapping):
        normalized_mapping: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SessionContractError(
                    f"canonical JSON requires string mapping keys at {path}"
                )
            normalized_mapping[key] = _canonical_json_value(
                item,
                path=f"{path}.{key}",
            )
        return_value = normalized_mapping
        return return_value
    if isinstance(value, list):
        normalized_list = [
            _canonical_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
        return_value = normalized_list
        return return_value
    raise SessionContractError(
        f"canonical JSON received unsupported value at {path}"
    )


def _canonical_json_bytes(value: object) -> bytes:
    """Return validated compact UTF-8 JSON bytes for one value."""

    normalized = _canonical_json_value(value)
    try:
        serialized = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        encoded = serialized.encode("utf-8")
    except (TypeError, ValueError, OverflowError, UnicodeError) as exc:
        raise SessionContractError(
            "canonical JSON serialization failed"
        ) from exc
    return_value = encoded
    return return_value


def _identity_projection_contains_float(value: object) -> bool:
    """Return whether an identity projection contains any float value."""

    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        contains_float = any(
            _identity_projection_contains_float(item)
            for item in value.values()
        )
        return contains_float
    if isinstance(value, list):
        contains_float = any(
            _identity_projection_contains_float(item)
            for item in value
        )
        return contains_float
    return False


def _canonical_identity_projection_value(
    value: object,
    *,
    path: str = "$",
) -> object:
    """Encode a validated identity projection without float collisions.

    The strict digest primitive rejects every float. When the validated
    character identity context contains finite scalar floats, the complete
    identity value is encoded with explicit type tags so a user mapping or
    string cannot collide with a float representation. This conversion is
    intentionally limited to that immutable field; all other fields use the
    strict canonical JSON validator directly.
    """

    if value is None:
        return_value = ["null"]
        return return_value
    if isinstance(value, bool):
        return_value = ["bool", value]
        return return_value
    if isinstance(value, int):
        return_value = ["int", str(value)]
        return return_value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SessionContractError(
                f"canonical JSON rejects non-finite value at {path}"
            )
        return_value = ["float", value.hex()]
        return return_value
    if isinstance(value, str):
        return_value = ["str", value]
        return return_value
    if isinstance(value, Mapping):
        encoded_items: list[list[object]] = []
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise SessionContractError(
                f"canonical JSON requires string mapping keys at {path}"
            )
        for key in sorted(keys):
            encoded_items.append([
                key,
                _canonical_identity_projection_value(
                    value[key],
                    path=f"{path}.{key}",
                ),
            ])
        return_value = ["map", encoded_items]
        return return_value
    if isinstance(value, list):
        encoded_items = [
            _canonical_identity_projection_value(
                item,
                path=f"{path}[{index}]",
            )
            for index, item in enumerate(value)
        ]
        return_value = ["list", encoded_items]
        return return_value
    raise SessionContractError(
        f"canonical JSON received unsupported value at {path}"
    )


def canonical_json_digest(value: object) -> str:
    """Return the canonical UTF-8 JSON SHA-256 digest of one value."""

    serialized = _canonical_json_bytes(value)
    digest = hashlib.sha256(serialized).hexdigest()
    return_value = digest
    return return_value


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

    projection = _build_immutable_projection(
        payload,
        original_evidence_digest=original_evidence_digest,
    )
    digest = canonical_json_digest(projection)
    return_value = digest
    return return_value


def _build_immutable_projection(
    payload: Mapping[str, object],
    *,
    original_evidence_digest: str | None = None,
) -> dict[str, object]:
    """Build the exact immutable projection used for digest and diagnostics."""

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
    normalized_projection: dict[str, object] = {}
    for field_name, field_projection in projection.items():
        if field_name == "original_evidence_digest":
            normalized_projection[field_name] = _canonical_json_value(
                field_projection,
                path=f"$.{field_name}",
            )
            continue
        if not isinstance(field_projection, Mapping):
            raise SessionContractError(
                f"immutable projection field {field_name} is invalid"
            )
        field_value = field_projection["value"]
        if (
            field_name == "character_identity_context"
            and _identity_projection_contains_float(field_value)
        ):
            normalized_field = {
                "present": field_projection["present"],
                "value": _canonical_identity_projection_value(
                    field_value,
                    path=f"$.{field_name}",
                ),
            }
        else:
            normalized_field = _canonical_json_value(
                field_projection,
                path=f"$.{field_name}",
            )
        normalized_projection[field_name] = normalized_field
    return_value = normalized_projection
    return return_value


def _immutable_digest_with_projection(
    payload: Mapping[str, object],
    *,
    original_evidence_digest: str | None = None,
) -> _ImmutableDigest:
    """Build a digest and retain its independent projection for field reports."""

    projection = _build_immutable_projection(
        payload,
        original_evidence_digest=original_evidence_digest,
    )
    digest = canonical_json_digest(projection)
    return_value = _ImmutableDigest(digest, projection)
    return return_value


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


def _canonicalize_mutable_state(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Project mutable state through the public V2 output owner.

    V2 output construction defines the ordering of entity and affect rows.
    Session CAS checks use that projection on both sides so transport ordering
    does not become a semantic state difference.
    """

    if not isinstance(state, Mapping):
        raise SessionContractError("mutable state must be a mapping")
    if not any(field_name in state for field_name in _CANONICAL_STATE_FIELDS):
        return dict(state)
    state_update = build_state_update(state, state)
    canonical_state = state_update["expected_previous_state"]
    if not isinstance(canonical_state, Mapping):
        raise SessionContractError(
            "canonical mutable state projection must be a mapping"
        )
    return dict(canonical_state)


def _episode_id_from_payload(payload: Mapping[str, object]) -> str:
    """Return the required episode id used by resolver carrier validators."""

    episode = payload.get("episode")
    if not isinstance(episode, Mapping):
        raise SessionContractError("episode must be a mapping")
    episode_id = episode.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id.strip():
        raise SessionContractError("episode.episode_id must be non-empty")
    return_value = episode_id
    return return_value


def _validated_willingness(
    value: object,
    payload: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate and return the canonical relational willingness carrier."""

    try:
        validated = validate_current_turn_relational_willingness(
            value,
            episode_id=_episode_id_from_payload(payload),
        )
    except ResolverValidationError as exc:
        raise SessionContractError(
            f"current_turn_relational_willingness is invalid: {exc}"
        ) from exc
    return_value = dict(validated)
    return return_value


def _validate_last_output_carriers(
    output: Mapping[str, object],
    payload: Mapping[str, object],
) -> None:
    """Validate resolver carriers present on an accepted terminal output."""

    goal_progress = output.get("resolver_goal_progress")
    if goal_progress is not None:
        try:
            validate_resolver_goal_progress(goal_progress)
        except ResolverValidationError as exc:
            raise SessionContractError(
                f"resolver_goal_progress is invalid: {exc}"
            ) from exc
    willingness = payload.get("current_turn_relational_willingness")
    if willingness is not None:
        _validated_willingness(willingness, payload)


def _expected_state_from_last_output(
    session: ChainSessionV1,
) -> dict[str, object]:
    """Derive the only authorized next state from the accepted output."""

    if session.last_output is None:
        raise SessionContractError(
            "session has no authoritative last_output state"
        )
    state_update = session.last_output.get("state_update")
    if not isinstance(state_update, Mapping):
        raise SessionContractError("session last_output state_update is invalid")
    replacement = state_update.get("replacement_state")
    if not isinstance(replacement, Mapping):
        raise SessionContractError(
            "session last_output replacement_state is invalid"
        )
    expected_state = _canonicalize_mutable_state(replacement)
    expected_digest = canonical_json_digest(expected_state)
    if expected_digest != session.expected_mutable_state_digest:
        raise SessionContractError(
            "session expected mutable-state digest disagrees with last_output"
        )
    return_value = expected_state
    return return_value


def _immutable_divergent_field(
    session: ChainSessionV1,
    payload: Mapping[str, object],
) -> str | None:
    """Return the first exact immutable field that differs canonically."""

    expected_projection = getattr(
        session.immutable_input_digest,
        "projection",
        None,
    )
    if not isinstance(expected_projection, Mapping):
        if payload.get("state_scope") != session.scope:
            return_value = "state_scope"
            return return_value
        try:
            incoming_episode_digest = canonical_json_digest(
                _episode_id_from_payload(payload)
            )
        except SessionContractError:
            return_value = "episode"
            return return_value
        if incoming_episode_digest != session.episode_id_digest:
            return_value = "episode"
            return return_value
        return_value = "immutable_input_digest"
        return return_value

    incoming_projection = _build_immutable_projection(
        payload,
        original_evidence_digest=session.original_evidence_digest,
    )
    for field_name in _IMMUTABLE_FIELDS:
        expected_field = expected_projection.get(field_name)
        incoming_field = incoming_projection.get(field_name)
        if _canonical_json_bytes(expected_field) != _canonical_json_bytes(
            incoming_field
        ):
            return_value = field_name
            return return_value
    return_value = None
    return return_value


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
    ttl_seconds: float,
) -> ChainSessionV1:
    """Create one cold session from a validated input."""

    if not isinstance(payload, Mapping):
        raise SessionContractError("session payload must be a mapping")
    scope = payload["state_scope"]
    if not isinstance(scope, str) or not scope.strip():
        raise SessionContractError("session scope must be non-empty")
    evidence = payload["evidence"]
    if not isinstance(evidence, list):
        raise SessionContractError("evidence must be a list")
    if (
        isinstance(ttl_seconds, bool)
        or not isinstance(ttl_seconds, (int, float))
        or not math.isfinite(ttl_seconds)
        or ttl_seconds <= 0
    ):
        raise SessionContractError("session TTL must be positive")
    immutable_payload = dict(payload)
    if expected_relational_willingness is not None:
        immutable_payload["current_turn_relational_willingness"] = (
            _validated_willingness(
                expected_relational_willingness,
                payload,
            )
        )
    willingness = immutable_payload.get("current_turn_relational_willingness")
    if willingness is not None:
        willingness = _validated_willingness(willingness, payload)
        immutable_payload["current_turn_relational_willingness"] = willingness
    willingness_digest = (
        canonical_json_digest(willingness)
        if willingness is not None
        else ""
    )
    incoming_state = _canonicalize_mutable_state(payload["mutable_state"])
    expected_state = incoming_state
    if last_output is not None:
        if not isinstance(last_output, Mapping):
            raise SessionContractError("last_output must be a mapping")
        state_update = last_output.get("state_update")
        if not isinstance(state_update, Mapping):
            raise SessionContractError("output state_update must be a mapping")
        expected_previous = state_update.get("expected_previous_state")
        if not isinstance(expected_previous, Mapping):
            raise SessionContractError(
                "output expected_previous_state must be a mapping"
            )
        canonical_expected_previous = _canonicalize_mutable_state(
            expected_previous
        )
        if canonical_expected_previous != incoming_state:
            raise SessionContractError(
                "output expected_previous_state does not match cold mutable_state"
            )
        replacement = state_update.get("replacement_state")
        if not isinstance(replacement, Mapping):
            raise SessionContractError("replacement state must be a mapping")
        expected_state = _canonicalize_mutable_state(replacement)
        _validate_last_output_carriers(last_output, payload)
    original_evidence_digest = canonical_json_digest(evidence)
    immutable_input_digest = _immutable_digest_with_projection(
        immutable_payload,
        original_evidence_digest=original_evidence_digest,
    )
    now = time.monotonic()
    session = ChainSessionV1(
        schema_version=SESSION_SCHEMA,
        session_key_digest=build_session_key(
            episode_id=episode_id,
            state_scope=scope,
            owner_identity=owner_identity,
        ),
        episode_id_digest=canonical_json_digest(episode_id),
        scope=scope,
        immutable_input_digest=immutable_input_digest,
        original_evidence_digest=original_evidence_digest,
        expected_mutable_state_digest=canonical_json_digest(expected_state),
        expected_willingness_digest=willingness_digest,
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
    return_value = session
    return return_value


def _evidence_source_id(row: Mapping[str, object]) -> str | None:
    """Return one evidence row's source id when its shape exposes one."""

    evidence_ref = row.get("evidence_ref")
    if not isinstance(evidence_ref, Mapping):
        return_value = None
        return return_value
    source_id = evidence_ref.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        return_value = None
        return return_value
    return_value = source_id
    return return_value


def _evidence_divergent_field(
    session: ChainSessionV1,
    payload: Mapping[str, object],
) -> tuple[str | None, Mapping[str, object] | None]:
    """Validate the append-only evidence prefix and return its new row."""

    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        return_value = ("evidence", None)
        return return_value
    accepted = list(session.accepted_evidence)
    if len(evidence) < len(accepted):
        return_value = ("evidence_prefix", None)
        return return_value
    if not _evidence_prefix_matches(accepted, evidence):
        return_value = ("evidence_prefix", None)
        return return_value
    if len(evidence) != len(accepted) + 1:
        return_value = ("evidence_append_count", None)
        return return_value
    new_row = evidence[-1]
    if not isinstance(new_row, Mapping):
        return_value = ("evidence_row_type", None)
        return return_value
    evidence_ref = new_row.get("evidence_ref")
    if not isinstance(evidence_ref, Mapping):
        return_value = ("evidence_source_kind", None)
        return return_value
    if evidence_ref.get("source_kind") != "resolver_observation":
        return_value = ("evidence_source_kind", None)
        return return_value
    source_id = evidence_ref.get("source_id")
    if not isinstance(source_id, str) or not source_id.strip():
        return_value = ("evidence_source_id", None)
        return return_value
    accepted_source_ids = {
        accepted_source_id
        for accepted_row in accepted
        if isinstance(accepted_row, Mapping)
        for accepted_source_id in [_evidence_source_id(accepted_row)]
        if accepted_source_id is not None
    }
    if source_id in accepted_source_ids:
        return_value = ("evidence_source_id", None)
        return return_value
    expected_handle = f"e{len(accepted) + 1}"
    if new_row.get("evidence_handle") != expected_handle:
        return_value = ("evidence_handle", None)
        return return_value
    return_value = (None, new_row)
    return return_value


def _evidence_prefix_matches(
    accepted: Sequence[Mapping[str, object]],
    incoming: Sequence[object],
) -> bool:
    """Compare accepted evidence rows by canonical bytes in order."""

    for index, accepted_row in enumerate(accepted):
        incoming_row = incoming[index]
        try:
            accepted_bytes = _canonical_json_bytes(accepted_row)
            incoming_bytes = _canonical_json_bytes(incoming_row)
        except SessionContractError:
            return_value = False
            return return_value
        if incoming_bytes != accepted_bytes:
            return_value = False
            return return_value
    return_value = True
    return return_value


def _goal_progress_divergent_field(
    session: ChainSessionV1,
    payload: Mapping[str, object],
) -> str | None:
    """Require the incoming goal-progress carrier to equal last output.

    The output contract always includes a nullable field; ``None`` represents
    the absence of the optional carrier, while a mapping is compared exactly.
    """

    if session.last_output is None:
        return_value = "last_output"
        return return_value
    prior_value = session.last_output.get("resolver_goal_progress")
    incoming_value = payload.get("resolver_goal_progress")
    prior_present = (
        "resolver_goal_progress" in session.last_output
        and prior_value is not None
    )
    incoming_present = (
        "resolver_goal_progress" in payload
        and incoming_value is not None
    )
    if prior_present != incoming_present:
        return_value = "resolver_goal_progress"
        return return_value
    if not prior_present:
        return_value = None
        return return_value
    try:
        normalized_prior = validate_resolver_goal_progress(prior_value)
        normalized_incoming = validate_resolver_goal_progress(incoming_value)
    except ResolverValidationError:
        return_value = "resolver_goal_progress"
        return return_value
    if _canonical_json_bytes(normalized_prior) != _canonical_json_bytes(
        normalized_incoming
    ):
        return_value = "resolver_goal_progress"
        return return_value
    return_value = None
    return return_value


def _dependency_divergent_field(
    session: ChainSessionV1,
    payload: Mapping[str, object],
    new_row: Mapping[str, object],
    incoming_index: int,
) -> str | None:
    """Validate dependency structure and bind it to the prior request/row."""

    if "required_resolver_evidence_dependency" not in payload:
        return_value = None
        return return_value
    dependency = payload.get("required_resolver_evidence_dependency")
    if dependency is None:
        return_value = "required_resolver_evidence_dependency"
        return return_value
    try:
        normalized_dependency = validate_required_resolver_evidence_dependency(
            dependency
        )
    except ResolverValidationError:
        return_value = "required_resolver_evidence_dependency"
        return return_value
    source_id = _evidence_source_id(new_row)
    if source_id is None or normalized_dependency["observation_id"] != source_id:
        return_value = "required_resolver_evidence_dependency"
        return return_value
    if session.last_output is None:
        return_value = "last_output"
        return return_value
    raw_requests = session.last_output.get("resolver_requests")
    if not isinstance(raw_requests, list):
        return_value = "required_resolver_evidence_dependency"
        return return_value
    matching_request_index: int | None = None
    matching_request: Mapping[str, object] | None = None
    for index, raw_request in enumerate(raw_requests, start=1):
        if not isinstance(raw_request, Mapping):
            continue
        try:
            request = validate_resolver_capability_request(raw_request)
        except ResolverValidationError:
            continue
        if request["capability_kind"] != "task_resolution_request":
            continue
        if _canonical_json_bytes(request["goal_continuation_ref"]) == (
            _canonical_json_bytes(normalized_dependency["goal_continuation_ref"])
        ):
            matching_request_index = index
            matching_request = request
            break
    if matching_request_index is None or matching_request is None:
        return_value = "required_resolver_evidence_dependency"
        return return_value
    prior_cycle_index = incoming_index - 1
    expected_request_handle = (
        f"resolver_request_{prior_cycle_index}_{matching_request_index}"
    )
    expected_observation_handle = (
        f"resolver_observation_{prior_cycle_index}_{matching_request_index}"
    )
    if normalized_dependency["accepted_request_handle"] != expected_request_handle:
        return_value = "required_resolver_evidence_dependency"
        return return_value
    if (
        normalized_dependency["prompt_safe_observation_handle"]
        != expected_observation_handle
    ):
        return_value = "required_resolver_evidence_dependency"
        return return_value
    return_value = None
    return return_value


def _reference_from_mapping(
    value: Mapping[str, object],
    field_names: Sequence[str],
) -> str | None:
    """Read an optional observation/resume reference without inventing one."""

    for field_name in field_names:
        if field_name not in value:
            continue
        raw_reference = value[field_name]
        if isinstance(raw_reference, Mapping):
            for nested_name in (
                "observation_id",
                "resolver_observation_id",
                "pending_resume_id",
                "resume_id",
                "source_id",
            ):
                nested_value = raw_reference.get(nested_name)
                if isinstance(nested_value, str) and nested_value.strip():
                    return_value = nested_value
                    return return_value
            return_value = None
            return return_value
        if isinstance(raw_reference, str) and raw_reference.strip():
            return_value = raw_reference
            return return_value
        return_value = None
        return return_value
    return_value = None
    return return_value


def _pending_resume_divergent_field(
    payload: Mapping[str, object],
    new_row: Mapping[str, object],
) -> str | None:
    """Validate pending resume and bind any explicit row reference."""

    if "pending_resolver_resume" not in payload:
        return_value = None
        return return_value
    pending = payload.get("pending_resolver_resume")
    if pending is None:
        return_value = "pending_resolver_resume"
        return return_value
    try:
        normalized_pending = validate_resolver_pending_resume(pending)
    except ResolverValidationError:
        return_value = "pending_resolver_resume"
        return return_value
    source_id = _evidence_source_id(new_row)
    pending_observation_id = _reference_from_mapping(
        pending,
        (
            "observation_id",
            "resolver_observation_id",
            "observation_ref",
            "resolver_observation_ref",
        ),
    )
    if pending_observation_id is not None and pending_observation_id != source_id:
        return_value = "pending_resolver_resume"
        return return_value
    pending_resume_id = _reference_from_mapping(
        pending,
        ("pending_resume_id", "resume_reference", "resume_ref"),
    )
    row_resume_id = _reference_from_mapping(
        new_row,
        ("pending_resume_id", "resume_reference", "resume_ref"),
    )
    evidence_ref = new_row.get("evidence_ref")
    if isinstance(evidence_ref, Mapping):
        evidence_ref_resume_id = _reference_from_mapping(
            evidence_ref,
            ("pending_resume_id", "resume_reference", "resume_ref"),
        )
        if row_resume_id is None:
            row_resume_id = evidence_ref_resume_id
    if row_resume_id is None and pending_resume_id == source_id:
        row_resume_id = source_id
    if pending_resume_id is not None and row_resume_id != pending_resume_id:
        return_value = "pending_resolver_resume"
        return return_value
    if row_resume_id is not None and row_resume_id != normalized_pending[
        "resume_id"
    ]:
        return_value = "pending_resolver_resume"
        return return_value
    return_value = None
    return return_value


def reattach_or_rebuild(
    *,
    session: ChainSessionV1,
    payload: Mapping[str, object],
) -> ReattachmentDecision:
    """Admit a cold session only when every immutable field still matches."""

    if session.schema_version != SESSION_SCHEMA:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="schema_version",
        )
    try:
        immutable_field = _immutable_divergent_field(session, payload)
    except SessionContractError:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="immutable_input_digest",
        )
    if immutable_field is not None:
        return ReattachmentDecision(
            reattached=False,
            divergent_field=immutable_field,
        )
    incoming_index = _cold_cycle_index(payload)
    if incoming_index != session.expected_cycle_index:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="resolver_cycle_index",
        )
    try:
        expected_state = _expected_state_from_last_output(session)
    except SessionContractError:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="last_output",
        )
    try:
        incoming_state = _canonicalize_mutable_state(payload["mutable_state"])
    except (KeyError, SessionContractError):
        return ReattachmentDecision(
            reattached=False,
            divergent_field="mutable_state",
        )
    try:
        incoming_state_digest = canonical_json_digest(incoming_state)
        expected_state_digest = canonical_json_digest(expected_state)
        states_match = (
            incoming_state_digest == session.expected_mutable_state_digest
            and expected_state_digest == session.expected_mutable_state_digest
            and _canonical_json_bytes(incoming_state)
            == _canonical_json_bytes(expected_state)
        )
    except SessionContractError:
        states_match = False
    if not states_match:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="mutable_state",
        )
    evidence_field, new_row = _evidence_divergent_field(session, payload)
    if evidence_field is not None or new_row is None:
        return ReattachmentDecision(
            reattached=False,
            divergent_field=evidence_field or "evidence",
        )
    goal_progress_field = _goal_progress_divergent_field(session, payload)
    if goal_progress_field is not None:
        return ReattachmentDecision(
            reattached=False,
            divergent_field=goal_progress_field,
        )
    dependency_field = _dependency_divergent_field(
        session,
        payload,
        new_row,
        incoming_index,
    )
    if dependency_field is not None:
        return ReattachmentDecision(
            reattached=False,
            divergent_field=dependency_field,
        )
    pending_field = _pending_resume_divergent_field(payload, new_row)
    if pending_field is not None:
        return ReattachmentDecision(
            reattached=False,
            divergent_field=pending_field,
        )
    willingness = payload.get("current_turn_relational_willingness")
    if willingness is not None:
        try:
            willingness = _validated_willingness(willingness, payload)
        except SessionContractError:
            return ReattachmentDecision(
                reattached=False,
                divergent_field="current_turn_relational_willingness",
            )
    try:
        willingness_digest = (
            canonical_json_digest(willingness)
            if willingness is not None
            else ""
        )
    except SessionContractError:
        return ReattachmentDecision(
            reattached=False,
            divergent_field="current_turn_relational_willingness",
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

    evidence_field, new_row = _evidence_divergent_field(session, payload)
    if evidence_field is not None or new_row is None:
        raise SessionContractError(
            f"cycle delta evidence is invalid: {evidence_field or 'evidence'}"
        )
    new_evidence_projection = {
        "present": True,
        "value": dict(new_row),
    }
    projection: dict[str, object] = {
        "new_evidence": new_evidence_projection,
    }
    projection.update({
        field_name: _presence(payload, field_name)
        for field_name in _CYCLE_FIELDS
    })
    digest = canonical_json_digest(projection)
    return_value = digest
    return return_value


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

    if session.schema_version != SESSION_SCHEMA:
        raise SessionContractError("session schema_version is invalid")
    if not isinstance(payload, Mapping) or not isinstance(output, Mapping):
        raise SessionContractError("session payload and output must be mappings")
    state_update = output.get("state_update")
    if not isinstance(state_update, Mapping):
        raise SessionContractError("output state_update must be a mapping")
    expected_previous = state_update.get("expected_previous_state")
    replacement = state_update.get("replacement_state")
    if not isinstance(expected_previous, Mapping):
        raise SessionContractError(
            "output expected_previous_state must be a mapping"
        )
    if not isinstance(replacement, Mapping):
        raise SessionContractError("replacement state must be a mapping")
    try:
        incoming_state = _canonicalize_mutable_state(payload["mutable_state"])
    except (KeyError, SessionContractError) as exc:
        raise SessionContractError("incoming mutable_state is invalid") from exc
    incoming_index = _cold_cycle_index(payload)
    if (
        session.last_output is not None
        and incoming_index != session.expected_cycle_index
    ):
        raise SessionContractError(
            "incoming resolver_cycle_index is not the next admissible index"
        )
    if session.last_output is not None:
        try:
            expected_state = _expected_state_from_last_output(session)
        except SessionContractError as exc:
            raise SessionContractError(
                "session last_output does not authorize mutable_state"
            ) from exc
        if _canonical_json_bytes(expected_state) != _canonical_json_bytes(
            incoming_state
        ):
            raise SessionContractError(
                "incoming mutable_state does not match session last_output"
            )
    elif canonical_json_digest(incoming_state) != (
        session.expected_mutable_state_digest
    ):
        raise SessionContractError(
            "incoming mutable_state does not match cold session state"
        )
    canonical_expected_previous = _canonicalize_mutable_state(
        expected_previous
    )
    if canonical_expected_previous != incoming_state:
        raise SessionContractError(
            "output expected_previous_state does not match incoming mutable_state"
        )
    canonical_replacement = _canonicalize_mutable_state(replacement)
    _validate_last_output_carriers(output, payload)
    willingness = payload.get("current_turn_relational_willingness")
    if willingness is not None:
        willingness = _validated_willingness(willingness, payload)
    willingness_digest = (
        canonical_json_digest(willingness)
        if willingness is not None
        else ""
    )
    if willingness_digest != session.expected_willingness_digest:
        raise SessionContractError(
            "current_turn_relational_willingness diverged from session"
        )
    evidence = payload.get("evidence")
    if not isinstance(evidence, list):
        raise SessionContractError("evidence must be a list")
    next_cycle_index = incoming_index + 1
    if len(evidence) == len(session.accepted_evidence) + 1:
        cycle_delta_digest = build_cycle_delta(
            session=session,
            payload=payload,
        )
    elif len(evidence) == len(session.accepted_evidence):
        if not _evidence_prefix_matches(session.accepted_evidence, evidence):
            raise SessionContractError(
                "session advance evidence prefix diverged"
            )
        cycle_delta_digest = ""
    else:
        raise SessionContractError(
            "session advance requires the accepted evidence prefix and at most one row"
        )
    now = time.monotonic()
    advanced = ChainSessionV1(
        schema_version=SESSION_SCHEMA,
        session_key_digest=session.session_key_digest,
        episode_id_digest=session.episode_id_digest,
        scope=session.scope,
        immutable_input_digest=session.immutable_input_digest,
        original_evidence_digest=session.original_evidence_digest,
        expected_mutable_state_digest=canonical_json_digest(
            canonical_replacement
        ),
        expected_willingness_digest=willingness_digest,
        expected_cycle_index=next_cycle_index,
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
        last_cycle_delta_digest=cycle_delta_digest,
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
    return_value = advanced
    return return_value


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
