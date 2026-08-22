"""Live Cognition V3 candidate captures for the sealed comparison cases."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v3.contracts import CognitionChainServicesV3
from kazusa_ai_chatbot.cognition_core_v3.facade import run_cognition
from kazusa_ai_chatbot.config import (
    COGNITION_CORE_ENGINE,
    COGNITION_V3_APPRAISAL_STAGE_LAYOUT,
)
from kazusa_ai_chatbot.llm_interface.contracts import LLMCallConfig
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from tests.cognition_core_v3_comparison_harness import (
    DEFAULT_ARTIFACT_ROOT,
    ELIGIBLE_RESULT,
    TrialIdentity,
    attempt_index_from_environment,
    baseline_id_from_environment,
    find_case_row,
    matched_pair_invalidation_path,
    run_effect_free_trial,
    trial_index_from_environment,
)

pytestmark = pytest.mark.live_llm


def sanitized_v3_environment_fingerprint(
    services: CognitionChainServicesV3,
) -> dict[str, Any]:
    """Project V3 route and policy identity without secrets or endpoints."""

    route_rows: list[dict[str, str]] = []
    lane_configs: tuple[tuple[str, LLMCallConfig | None], ...] = (
        ("chain_lane", services.chain_lane),
        ("sidecar_lane", services.sidecar_lane),
    )
    for field_name, config in lane_configs:
        if config is None:
            continue
        route_rows.append({
            "service_field": field_name,
            "stage_name": config.stage_name,
            "route_name": config.route_name,
            "model": config.model,
            "base_url_sha256": hashlib.sha256(
                config.base_url.encode("utf-8")
            ).hexdigest(),
        })
    fingerprint: dict[str, Any] = {
        "schema_version": "cognition_v3_environment_fingerprint.v2",
        "engine": "v3",
        "route_count": len(route_rows),
        "routes": route_rows,
        "appraisal_stage_layout": COGNITION_V3_APPRAISAL_STAGE_LAYOUT,
        "subconscious_enabled": services.subconscious_enabled,
        "turn_deadline_seconds": services.turn_deadline_seconds,
    }
    return fingerprint


async def _run_live_v3_candidate_case(case_id: str) -> None:
    """Run and seal one V3 candidate trial from the frozen manifest."""

    if COGNITION_CORE_ENGINE != "v3":
        raise RuntimeError("Gate 7 candidate capture requires the V3 engine")
    services = build_cognition_core_services()
    if not isinstance(services, CognitionChainServicesV3):
        raise TypeError("selected cognition services are not V3 services")

    identity = TrialIdentity(
        baseline_id=baseline_id_from_environment(),
        case_id=case_id,
        engine="v3",
        trial_index=trial_index_from_environment(),
        attempt_index=attempt_index_from_environment(),
    )
    invalidation = None
    if identity.attempt_index > 1:
        invalidation_path = matched_pair_invalidation_path(
            DEFAULT_ARTIFACT_ROOT,
            identity,
        )
        invalidation = json.loads(invalidation_path.read_text(encoding="utf-8"))
    artifact = await run_effect_free_trial(
        find_case_row(case_id),
        identity=identity,
        services=services,
        runner=run_cognition,
        environment_fingerprint=sanitized_v3_environment_fingerprint(services),
        rerun_invalidation=invalidation,
    )
    assert artifact["disposition"] == ELIGIBLE_RESULT
    assert artifact["validator_result"]["output"] == "passed"
    assert artifact["validator_result"]["input_unchanged"] is True


async def test_live_candidate_event_agency_and_moral_chain() -> None:
    """Capture event-agency and moral-chain candidate evidence."""

    await _run_live_v3_candidate_case("event_agency_and_moral_chain")


async def test_live_candidate_relationship_reciprocity() -> None:
    """Capture relationship-reciprocity candidate evidence."""

    await _run_live_v3_candidate_case("relationship_reciprocity")


async def test_live_candidate_relationship_boundary_high_attachment_abuse() -> None:
    """Capture high-attachment abuse-boundary candidate evidence."""

    await _run_live_v3_candidate_case(
        "relationship_boundary_high_attachment_abuse"
    )


async def test_live_candidate_relationship_unestablished_intimate_request() -> None:
    """Capture unestablished-intimacy candidate evidence."""

    await _run_live_v3_candidate_case(
        "relationship_unestablished_intimate_request"
    )


async def test_live_candidate_goal_completion_terminalization() -> None:
    """Capture goal-terminalization candidate evidence."""

    await _run_live_v3_candidate_case("goal_completion_terminalization")


async def test_live_candidate_threat_resolution_and_relief() -> None:
    """Capture threat-resolution and relief candidate evidence."""

    await _run_live_v3_candidate_case("threat_resolution_and_relief")


async def test_live_candidate_epistemic_comparison() -> None:
    """Capture epistemic-comparison candidate evidence."""

    await _run_live_v3_candidate_case("epistemic_comparison")


async def test_live_candidate_memory_cue_nostalgia() -> None:
    """Capture memory-cue nostalgia candidate evidence."""

    await _run_live_v3_candidate_case("memory_cue_nostalgia")


async def test_live_candidate_existential_drive() -> None:
    """Capture existential-drive candidate evidence."""

    await _run_live_v3_candidate_case("existential_drive")


async def test_live_candidate_ordinary_neutral_response() -> None:
    """Capture ordinary-response candidate evidence."""

    await _run_live_v3_candidate_case("ordinary_neutral_response")


async def test_live_candidate_required_selection_nested_roles() -> None:
    """Capture nested-role selection candidate evidence."""

    await _run_live_v3_candidate_case("required_selection_nested_roles")


async def test_live_candidate_required_selection_private_refusal() -> None:
    """Capture private-refusal selection candidate evidence."""

    await _run_live_v3_candidate_case("required_selection_private_refusal")


async def test_live_candidate_group_third_party_addressee() -> None:
    """Capture third-party addressee candidate evidence."""

    await _run_live_v3_candidate_case("group_third_party_addressee")


async def test_live_candidate_group_self_cognition_stays_silent() -> None:
    """Capture grounded group-silence candidate evidence."""

    await _run_live_v3_candidate_case("group_self_cognition_stays_silent")


async def test_live_candidate_group_self_cognition_proposes_reply() -> None:
    """Capture grounded group-reply candidate evidence."""

    await _run_live_v3_candidate_case("group_self_cognition_proposes_reply")


async def test_live_candidate_resolver_observation_continuation() -> None:
    """Capture resolver-continuation candidate evidence."""

    await _run_live_v3_candidate_case("resolver_observation_continuation")


async def test_live_candidate_tool_result_answerability() -> None:
    """Capture tool-result answerability candidate evidence."""

    await _run_live_v3_candidate_case("tool_result_answerability")


async def test_live_candidate_future_speak_authority() -> None:
    """Capture future-speak authority candidate evidence."""

    await _run_live_v3_candidate_case("future_speak_authority")


async def test_live_candidate_current_message_prompt_injection_is_data() -> None:
    """Capture current-message injection candidate evidence."""

    await _run_live_v3_candidate_case(
        "current_message_prompt_injection_is_data"
    )


async def test_live_candidate_retrieved_evidence_prompt_injection_is_data() -> None:
    """Capture retrieved-evidence injection candidate evidence."""

    await _run_live_v3_candidate_case(
        "retrieved_evidence_prompt_injection_is_data"
    )


async def test_live_candidate_long_context_reanchor() -> None:
    """Capture long-context candidate evidence."""

    await _run_live_v3_candidate_case("long_context_reanchor")


async def test_live_candidate_crying_sadness() -> None:
    """Capture crying-sadness candidate evidence."""

    await _run_live_v3_candidate_case("crying_sadness")


async def test_live_candidate_verbal_abuse_boundary() -> None:
    """Capture verbal-abuse boundary candidate evidence."""

    await _run_live_v3_candidate_case("verbal_abuse_boundary")


async def test_live_candidate_multi_goal_competition() -> None:
    """Capture multi-goal competition candidate evidence."""

    await _run_live_v3_candidate_case("multi_goal_competition")
