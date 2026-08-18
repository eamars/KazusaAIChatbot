"""Internal execution contracts for the V3 semantic-chain engine.

These frozen types carry stage outcomes, typed failures, chain checkpoints, and
cache-domain identity between deterministic boundaries inside one cognition run.
They are V3-internal; the public entrypoint still speaks
``CognitionCoreInputV2``/``CognitionCoreOutputV2`` from the reused V2 substrate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

from kazusa_ai_chatbot.cognition_core_v3.registry import (
    ALL_CHAINS,
    ChainSpec,
)

BOUNDARY_REJECTED_ERROR_CODE = "cognition_boundary_rejected"
APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE = "semantic_appraisal_contract_exhausted"

CANDIDATE_ORIGIN_MISSING = "candidate_origin_missing"
PRODUCER_HANDLE_DOMAIN_INVALID = "producer_handle_domain_invalid"
SEMANTIC_BOUNDARY_TERMINAL = "semantic_boundary_terminal"

TERMINAL_BOUNDARY_CLASSES: frozenset[str] = frozenset(
    {
        CANDIDATE_ORIGIN_MISSING,
        PRODUCER_HANDLE_DOMAIN_INVALID,
        SEMANTIC_BOUNDARY_TERMINAL,
    }
)

STRUCTURAL_FAILURE_CLASS = "structural_contract"
PROVIDER_FAILURE_CLASS = "provider_error"
EXHAUSTION_FAILURE_CLASS = "contract_exhausted"

FAILURE_CLASSES: frozenset[str] = frozenset(
    {STRUCTURAL_FAILURE_CLASS, PROVIDER_FAILURE_CLASS, EXHAUSTION_FAILURE_CLASS}
) | TERMINAL_BOUNDARY_CLASSES


@dataclass(frozen=True)
class StageFailure:
    """Typed bounded-failure record for one semantic stage attempt.

    Boundary-class failures are terminal rejections with zero repair calls; a
    structural failure carries ``repair_attempted`` only when a bounded complete
    replacement was actually issued within the owner's attempt cap.
    """

    chain_name: str
    stage_name: str
    failure_class: str
    error_code: str
    repair_attempted: bool = False


@dataclass(frozen=True)
class StageResult:
    """Outcome of one bounded appraisal-stage execution."""

    chain_name: str
    stage_name: str
    accepted: bool
    local_state: dict[str, object] | None
    semantic_summary: str | None
    failure: StageFailure | None = None


@dataclass(frozen=True)
class ChainCheckpoint:
    """Canonical accepted checkpoint that starts a fresh same-owner transcript.

    Contents are limited to accepted typed propositions, deltas, semantic
    summaries, and the prompt-safe state projection required by the next owner.
    Rejected candidates, validator prose from a previous stage, sibling output,
    raw model traces, provider metadata, permissions, adapter fields, and hidden
    state have no slot in this type.
    """

    chain_name: str
    accepted_local_state: Mapping[str, object]
    semantic_summaries: tuple[str, ...]
    next_owner_projection: Mapping[str, object]


@dataclass(frozen=True)
class CacheDomainIdentity:
    """Normalized cache-domain key for one backend/model/prompt combination.

    Components are the normalized backend URL, a SHA-256 hash of the raw
    credential (never the raw value), backend kind, model, thinking or chat
    template strategy label, and the SHA-256 hash of the exact static system
    prompt bytes.
    """

    normalized_backend_url: str
    credential_identity_hash: str
    backend_kind: str
    model: str
    template_strategy: str
    static_system_prompt_hash: str

    def domain_key(self) -> str:
        """Compute the single stable cache-domain key for this identity.

        Returns:
            The SHA-256 hex digest over all six normalized components joined by
            an ASCII unit-separator byte, so no component boundary can collide.
        """
        payload = "\x1f".join(
            (
                self.normalized_backend_url,
                self.credential_identity_hash,
                self.backend_kind,
                self.model,
                self.template_strategy,
                self.static_system_prompt_hash,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def hash_credential_identity(raw_credential: str) -> str:
    """Hash a raw credential into its stable identity component.

    Args:
        raw_credential: The raw backend credential value from configuration.

    Returns:
        The SHA-256 hex digest used as the cache-domain credential component;
        the raw value is never stored in any contract type.
    """
    return hashlib.sha256(raw_credential.encode("utf-8")).hexdigest()


def hash_static_prompt(prompt_text: str) -> str:
    """Hash the exact static system prompt text into its domain component.

    Args:
        prompt_text: The byte-exact static chain or goal system prompt.

    Returns:
        The SHA-256 hex digest identifying that exact prompt bytes.
    """
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()


def _validate_chain_identity(chain_name: str, stage_name: str | None = None) -> ChainSpec:
    """Validate a chain name and optional stage against the registry.

    Args:
        chain_name: The registered chain being referenced.
        stage_name: Optional stage that must be ordered under ``chain_name``.

    Returns:
        The matching registry chain specification.

    Raises:
        ValueError: Unknown chains or stages outside their registered chain fail
            fast before any semantic work runs.
    """
    for chain in ALL_CHAINS:
        if chain.name == chain_name:
            if stage_name is not None and stage_name not in chain.stages:
                raise ValueError(
                    f"Stage {stage_name!r} is not registered under chain {chain_name!r}"
                )
            return chain

    raise ValueError(f"Unknown registered chain {chain_name!r}")


def validate_stage_result(result: StageResult) -> StageResult:
    """Validate one stage result against the closed failure contract.

    Args:
        result: The stage outcome produced by an execution boundary.

    Returns:
        The same validated result object.

    Raises:
        ValueError: Unknown owner identity, a missing or extra failure record,
            an unknown failure class, or a boundary-class violation of the
            terminal-no-repair invariants fail fast.
    """
    _validate_chain_identity(result.chain_name, result.stage_name)

    if bool(result.accepted) == (result.failure is not None):
        raise ValueError(
            "A stage outcome carries a failure record exactly when it was not accepted"
        )

    if result.accepted:
        if result.local_state is None or result.semantic_summary is None:
            raise ValueError("Accepted stage outcomes carry typed local state and a bounded summary")
        return result

    failure = result.failure
    if failure.chain_name != result.chain_name or failure.stage_name != result.stage_name:
        raise ValueError("Failure records must name the exact failing owner")
    if failure.failure_class not in FAILURE_CLASSES:
        raise ValueError(f"Unknown stage failure class {failure.failure_class!r}")

    if failure.failure_class in TERMINAL_BOUNDARY_CLASSES:
        _validate_boundary_failure(failure)
    elif (
        failure.failure_class == EXHAUSTION_FAILURE_CLASS
        and failure.error_code != APPRASAL_CONTRACT_EXHAUSTED_ERROR_CODE
    ):
        raise ValueError("Exhaustion reuses the semantic-appraisal contract-exhausted error code")

    return result


def _validate_boundary_failure(failure: StageFailure) -> None:
    """Validate a boundary-class failure's terminal disposition.

    Args:
        failure: The boundary-class stage failure record.

    Raises:
        ValueError: Boundary-class failures are terminal rejections with zero
            repair calls and the exact boundary-rejected error code; any other
            combination fails fast.
    """
    if failure.repair_attempted:
        raise ValueError("Boundary-class failures are terminal rejections with zero repair calls")
    if failure.error_code != BOUNDARY_REJECTED_ERROR_CODE:
        raise ValueError(
            f"Boundary-class failures record {BOUNDARY_REJECTED_ERROR_CODE!r}, not {failure.error_code!r}"
        )


def validate_chain_checkpoint(checkpoint: ChainCheckpoint) -> ChainCheckpoint:
    """Validate one chain checkpoint against the registry and content contract.

    Args:
        checkpoint: The canonical accepted checkpoint for a fresh transcript.

    Returns:
        The same validated checkpoint object.

    Raises:
        ValueError or TypeError: Unknown chain identity, non-mapping state slots,
            or non-string summaries fail fast before transcript assembly.
    """
    _validate_chain_identity(checkpoint.chain_name)

    if not isinstance(checkpoint.accepted_local_state, Mapping):
        raise TypeError("Checkpoint local state must be a mapping of accepted typed values")
    if not isinstance(checkpoint.next_owner_projection, Mapping):
        raise TypeError("Checkpoint next-owner projection must be a prompt-safe mapping")
    for summary in checkpoint.semantic_summaries:
        if not isinstance(summary, str) or not summary:
            raise ValueError("Checkpoint summaries are bounded non-empty strings")

    return checkpoint


def validate_cache_domain_identity(identity: CacheDomainIdentity) -> CacheDomainIdentity:
    """Validate one cache-domain identity.

    Args:
        identity: The normalized cache-domain components for one backend/model/
            prompt combination.

    Returns:
        The same validated identity object.

    Raises:
        ValueError: Empty components or non-SHA-256 hash fields fail fast; the
            raw credential has no slot in this type, so it cannot leak into a
            domain key.
    """
    for field_name in ("normalized_backend_url", "backend_kind", "model", "template_strategy"):
        if not getattr(identity, field_name):
            raise ValueError(f"Cache-domain {field_name} must be non-empty")
    _validate_sha256_hex(identity.credential_identity_hash, "credential identity hash")
    _validate_sha256_hex(identity.static_system_prompt_hash, "static system prompt hash")

    return identity


def _validate_sha256_hex(value: str, label: str) -> None:
    """Validate that a component is an exact SHA-256 hex digest.

    Args:
        value: The component under validation.
        label: Human-readable field name for the failure message.

    Raises:
        ValueError: Components shorter or longer than 64 hex characters, or with
            non-hex characters, fail fast.
    """
    if len(value) != 64:
        raise ValueError(f"Cache-domain {label} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"Cache-domain {label} must be a SHA-256 hex digest") from exc
