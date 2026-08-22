"""Canonical four-stage semantic cognition engine.

The package exposes one public entrypoint, ``run_cognition``.  The caller-owned
workspace is projected into one A1, one A2, one goal, and one ordinary/self
response product.  Model-facing products contain semantic meaning only; state
identity and effect authorization remain deterministic caller concerns.

The protected replay capture API is exposed alongside it, scoped per trace so
replay harnesses can bind one record scope and read the exact stage attempts
recorded during that run.

"""

from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_APPRAISAL_FAMILIES,
    CANONICAL_COGNITION_INPUT_SCHEMA,
    CANONICAL_COGNITION_OUTPUT_SCHEMA,
    CANONICAL_FAMILY_AXES,
    CanonicalAppraisal,
    CanonicalCognitionOutput,
    CanonicalGoal,
    CanonicalResponsePlan,
    CanonicalTurnWorkspace,
)
from kazusa_ai_chatbot.cognition_core_v3.facade import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    run_cognition,
    snapshot_protected_chain_records,
)

__all__ = [
    "CANONICAL_A1_FAMILIES",
    "CANONICAL_A2_FAMILIES",
    "CANONICAL_APPRAISAL_FAMILIES",
    "CANONICAL_COGNITION_INPUT_SCHEMA",
    "CANONICAL_COGNITION_OUTPUT_SCHEMA",
    "CANONICAL_FAMILY_AXES",
    "CanonicalAppraisal",
    "CanonicalCognitionOutput",
    "CanonicalGoal",
    "CanonicalResponsePlan",
    "CanonicalTurnWorkspace",
    "bind_protected_chain_records",
    "reset_protected_chain_records",
    "run_cognition",
    "snapshot_protected_chain_records",
]
