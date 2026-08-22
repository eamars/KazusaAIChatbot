"""Advisory nonblocking L1 residue for the Cognition V3 sidecar lane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

L1_RESIDUE_SCHEMA = "l1_residue.v1"

L1_RISK_FLAG = Literal[
    "boundary_pressure",
    "coercion_or_control",
    "privacy_or_secrecy",
    "physical_harm",
    "self_harm",
    "sexual_boundary",
    "relationship_rupture",
    "identity_conflict",
    "evidence_conflict",
]

L1_RISK_FLAGS: frozenset[str] = frozenset(
    {
        "boundary_pressure",
        "coercion_or_control",
        "privacy_or_secrecy",
        "physical_harm",
        "self_harm",
        "sexual_boundary",
        "relationship_rupture",
        "identity_conflict",
        "evidence_conflict",
    }
)


class L1ResidueV1(TypedDict):
    """Closed advisory L1 output. It cannot create evidence or authority."""

    schema_version: Literal["l1_residue.v1"]
    emotional_appraisal: str
    interaction_subtext: str
    salience_hints: list[str]
    risk_flags: list[L1_RISK_FLAG]


class L1ContractError(ValueError):
    """A proposed L1 residue violates its closed advisory contract."""


def validate_l1_residue(
    residue: Mapping[str, object],
    *,
    supplied_evidence_handles: Sequence[str],
) -> L1ResidueV1:
    """Validate one closed L1 residue and its supplied-handle membership."""

    if not isinstance(residue, Mapping):
        raise L1ContractError("L1 residue must be a mapping")
    if set(residue) != {
        "schema_version",
        "emotional_appraisal",
        "interaction_subtext",
        "salience_hints",
        "risk_flags",
    }:
        raise L1ContractError("L1 residue has unknown or missing fields")
    if residue["schema_version"] != L1_RESIDUE_SCHEMA:
        raise L1ContractError("L1 residue schema_version is invalid")

    emotional_appraisal = residue["emotional_appraisal"]
    interaction_subtext = residue["interaction_subtext"]
    if not isinstance(emotional_appraisal, str) or not (
        1 <= len(emotional_appraisal) <= 120
    ):
        raise L1ContractError(
            "L1 emotional_appraisal must be 1..120 characters"
        )
    if not isinstance(interaction_subtext, str) or len(
        interaction_subtext
    ) > 200:
        raise L1ContractError(
            "L1 interaction_subtext must be 0..200 characters"
        )

    salience_hints = residue["salience_hints"]
    risk_flags = residue["risk_flags"]
    if not isinstance(salience_hints, list) or not isinstance(
        risk_flags,
        list,
    ):
        raise L1ContractError("L1 lists must be lists")
    if len(salience_hints) > 4:
        raise L1ContractError("L1 salience_hints cannot exceed four handles")
    if len(set(salience_hints)) != len(salience_hints):
        raise L1ContractError("L1 salience_hints must be duplicate-free")
    allowed_handles = frozenset(supplied_evidence_handles)
    for handle in salience_hints:
        if not isinstance(handle, str) or handle not in allowed_handles:
            raise L1ContractError("L1 salience hint is not a supplied handle")
    if len(set(risk_flags)) != len(risk_flags):
        raise L1ContractError("L1 risk_flags must be duplicate-free")
    for flag in risk_flags:
        if flag not in L1_RISK_FLAGS:
            raise L1ContractError(f"unknown L1 risk flag {flag!r}")

    return L1ResidueV1(
        schema_version=L1_RESIDUE_SCHEMA,
        emotional_appraisal=emotional_appraisal,
        interaction_subtext=interaction_subtext,
        salience_hints=list(salience_hints),
        risk_flags=list(risk_flags),
    )


def try_validate_l1_residue(
    residue: Mapping[str, object],
    *,
    supplied_evidence_handles: Sequence[str],
) -> L1ResidueV1 | None:
    """Validate an L1 residue and drop malformed advisory output."""

    try:
        validated = validate_l1_residue(
            residue,
            supplied_evidence_handles=supplied_evidence_handles,
        )
    except L1ContractError:
        return None
    return validated


__all__ = [
    "L1_RESIDUE_SCHEMA",
    "L1_RISK_FLAGS",
    "L1ContractError",
    "L1ResidueV1",
    "try_validate_l1_residue",
    "validate_l1_residue",
]
