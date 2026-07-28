"""Public pure-domain helpers for character identity growth."""

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    apply_identity_patches,
    candidate_transition_allowed,
    dedupe_evidence_refs,
    diff_effective_identities,
    derive_growth_health_state,
    evidence_counts,
)
from kazusa_ai_chatbot.character_identity_growth.policy import (
    evaluate_identity_growth_policy,
)
from kazusa_ai_chatbot.character_identity_growth.projection import (
    build_identity_proposal_input,
    build_identity_review_input,
    identity_projection_digest,
    project_candidate_for_growth_prompt,
    project_candidate_for_console,
    project_growth_health_for_console,
    project_growth_run_for_console,
    project_identity_for_cognition,
    project_identity_for_console,
    project_identity_for_growth_prompt,
    project_identity_for_surface,
    projected_identity_consumer_kinds,
)
from kazusa_ai_chatbot.character_identity_growth.validation import (
    validate_effective_identity,
    validate_evidence_ref,
    validate_identity_evidence_card,
    validate_identity_patch,
    validate_identity_proposal_decision,
    validate_identity_review_decision,
)

__all__ = [
    "apply_identity_patches",
    "build_identity_proposal_input",
    "build_identity_review_input",
    "candidate_transition_allowed",
    "dedupe_evidence_refs",
    "diff_effective_identities",
    "derive_growth_health_state",
    "evaluate_identity_growth_policy",
    "evidence_counts",
    "identity_projection_digest",
    "models",
    "project_candidate_for_console",
    "project_candidate_for_growth_prompt",
    "project_growth_health_for_console",
    "project_growth_run_for_console",
    "project_identity_for_cognition",
    "project_identity_for_console",
    "project_identity_for_growth_prompt",
    "project_identity_for_surface",
    "projected_identity_consumer_kinds",
    "validate_effective_identity",
    "validate_evidence_ref",
    "validate_identity_evidence_card",
    "validate_identity_patch",
    "validate_identity_proposal_decision",
    "validate_identity_review_decision",
]
