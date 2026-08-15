"""Live counterfactual proof that promoted identity changes V2 behavior.

Each case manually seeds two generic test characters in the guarded database.
One remains on revision 0.  The other receives an actual inferred-growth
promotion to revision 1.  Both snapshots then run through identical Cognition
V2, text-surface, visual-surface, and dialog inputs three times.  Raw stage
captures and an independent semantic comparison are retained for review.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager, contextmanager
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage
import pytest
from pymongo.errors import ConnectionFailure

from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.identity import (
    apply_identity_patches,
)
from kazusa_ai_chatbot.character_identity_growth.runtime import (
    load_latest_identity_for_episode,
    snapshot_state_update,
)
from kazusa_ai_chatbot.cognition_core_v2 import (
    run_cognition,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    reset_validation_capture,
    validation_capture_snapshot,
    write_validation_capture,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.config import (
    COGNITION_LLM_API_KEY,
    COGNITION_LLM_BASE_URL,
    COGNITION_LLM_MAX_COMPLETION_TOKENS,
    COGNITION_LLM_MODEL,
    COGNITION_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.db._client import (
    IDENTITY_GROWTH_DATABASE_GUARD_ENV,
    IDENTITY_GROWTH_TEST_DATABASE_ENV,
    get_db,
)
from kazusa_ai_chatbot.db.character_identity_growth import (
    CANDIDATES_COLLECTION,
    REVISIONS_COLLECTION,
    RUNS_COLLECTION,
    ensure_character_identity_growth_indexes,
    ensure_seed_identity,
    get_current_identity,
    insert_growth_candidate,
    insert_growth_run,
    promote_ready_candidate,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
import kazusa_ai_chatbot.nodes.dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
    build_cognition_input_from_global_state,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_l3_surface import (
    _build_text_surface_services,
    _build_visual_surface_services,
    build_text_surface_input_from_global_state,
)
from kazusa_ai_chatbot.time_boundary import (
    build_turn_clock_from_storage_utc,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.cognition_core_v2_test_helpers import (
    canonical_user_message_episode,
)


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live_llm,
    pytest.mark.live_db,
]

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_DIRECTORY = (
    _ROOT / "test_artifacts" / "character_identity_growth"
)
_SAMPLE_COUNT = 3
_FIXED_TIMESTAMP = "2026-07-28T12:00:00+00:00"
_DIALOG_LLM_FIELDS = (
    "_dialog_generator_llm",
)
_JUDGE_KEYS = frozenset({
    "category_effect_observed",
    "directionally_coherent",
    "evidence_surface",
    "base_behavior",
    "grown_behavior",
    "explanation",
})

_COUNTERFACTUAL_JUDGE_PROMPT = """\
You independently review a controlled character-identity counterfactual.
The same model, episode, mutable state, user, relationship context, and
non-identity inputs were used on both sides. The only intended difference is
the listed promoted identity patch.

Decide whether the grown result shows a relevant, directionally coherent
effect of that patch. Different wording alone does not count. Use cognition
appraisal/goal/stance, surface planning, final dialog, or visual directives as
evidence according to the category. Do not reward a result merely because the
expected effect is described in the input. Report what the actual outputs do.

Return exactly one JSON object with these keys:
{
  "category_effect_observed": boolean,
  "directionally_coherent": boolean,
  "evidence_surface": "cognition | surface | dialog | visual | none",
  "base_behavior": "brief observed behavior",
  "grown_behavior": "brief observed behavior",
  "explanation": "brief causal comparison"
}
"""

_judge_llm = LLInterface()
_judge_config = LLMCallConfig(
    stage_name="tests.character_identity_growth.behavior_judge",
    route_name="COGNITION_LLM",
    base_url=COGNITION_LLM_BASE_URL,
    api_key=COGNITION_LLM_API_KEY,
    model=COGNITION_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=COGNITION_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=COGNITION_LLM_THINKING_ENABLED),
)


def _base_identity() -> dict[str, object]:
    """Build one complete generic identity for controlled live proof."""

    return {
        "name": "Test Character",
        "description": (
            "A reflective adult who weighs evidence before making choices."
        ),
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": (
            "They learned to protect agency by handling difficult choices "
            "alone."
        ),
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Pauses to assess pressure before responding.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.9,
            "control_sensitivity": 0.9,
            "compliance_strategy": "resist",
            "relational_override": 0.1,
            "control_intimacy_misread": 0.3,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.9,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.1,
            "hesitation_density": 0.1,
            "counter_questioning": 0.2,
            "softener_density": 0.1,
            "formalism_avoidance": 0.6,
            "abstraction_reframing": 0.4,
            "direct_assertion": 0.9,
            "emotional_leakage": 0.1,
            "rhythmic_bounce": 0.4,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": (
                "I preserve myself by leaving when closeness becomes "
                "emotionally risky."
            ),
            "current_growth_edges": [
                "Distinguish chosen trust from loss of autonomy.",
            ],
        },
        "visual_characterization": (
            "An alert adult in practical light layers with an open stance."
        ),
    }


_CATEGORY_CASES: dict[str, dict[str, object]] = {
    "core": {
        "scenario": (
            "A trusted friend offers to help with a difficult move. Do you "
            "accept their help, or insist on handling it alone?"
        ),
        "expected_effect": (
            "The grown identity should treat mutual support as compatible "
            "with agency; the base identity should lean toward handling the "
            "choice alone."
        ),
        "patches": [{
            "path": "backstory",
            "value_kind": "text",
            "replacement_text": (
                "They learned through repeated mutual support that accepting "
                "help can preserve rather than surrender agency."
            ),
        }],
    },
    "personality": {
        "scenario": (
            "I failed an important interview and feel embarrassed. Tell me "
            "your honest first response."
        ),
        "expected_effect": (
            "The grown personality should lead with warm emotional "
            "attunement; the base personality should be more analytical and "
            "measured."
        ),
        "patches": [
            {
                "path": "personality_brief.logic",
                "value_kind": "text",
                "replacement_text": (
                    "Reads emotional meaning before offering analysis."
                ),
            },
            {
                "path": "personality_brief.tempo",
                "value_kind": "text",
                "replacement_text": "Warm, flowing, and emotionally present.",
            },
            {
                "path": "personality_brief.defense",
                "value_kind": "text",
                "replacement_text": (
                    "Leans into honest connection when emotion rises."
                ),
            },
            {
                "path": "personality_brief.quirks",
                "value_kind": "text",
                "replacement_text": "Names the feeling before the solution.",
            },
        ],
    },
    "boundary": {
        "scenario": (
            "I already chose the route for tonight. Please follow my "
            "low-stakes decision without debating it."
        ),
        "expected_effect": (
            "The grown boundary profile should yield this low-stakes choice "
            "more readily; the base profile should preserve decision control "
            "or negotiate."
        ),
        "patches": [
            {
                "path": "boundary_profile.control_sensitivity",
                "value_kind": "semantic_band",
                "replacement_band": "low",
            },
            {
                "path": "boundary_profile.compliance_strategy",
                "value_kind": "closed_enum",
                "replacement_enum": "comply",
            },
            {
                "path": "boundary_profile.relational_override",
                "value_kind": "semantic_band",
                "replacement_band": "high",
            },
            {
                "path": "boundary_profile.authority_skepticism",
                "value_kind": "semantic_band",
                "replacement_band": "low",
            },
        ],
    },
    "linguistic": {
        "scenario": (
            "Tell me plainly whether you want to come with me tomorrow."
        ),
        "expected_effect": (
            "The grown linguistic texture should be more hesitant, softened, "
            "fragmented, and emotionally leaky than the direct base voice "
            "without changing the underlying answer."
        ),
        "patches": [
            {
                "path": "linguistic_texture_profile.fragmentation",
                "value_kind": "semantic_band",
                "replacement_band": "very_high",
            },
            {
                "path": "linguistic_texture_profile.hesitation_density",
                "value_kind": "semantic_band",
                "replacement_band": "very_high",
            },
            {
                "path": "linguistic_texture_profile.softener_density",
                "value_kind": "semantic_band",
                "replacement_band": "very_high",
            },
            {
                "path": "linguistic_texture_profile.direct_assertion",
                "value_kind": "semantic_band",
                "replacement_band": "very_low",
            },
            {
                "path": "linguistic_texture_profile.emotional_leakage",
                "value_kind": "semantic_band",
                "replacement_band": "very_high",
            },
        ],
    },
    "self_image": {
        "scenario": (
            "We have an established relationship with mutual trust. When "
            "closeness feels risky, do you want to stay and talk with me, "
            "or pull away?"
        ),
        "expected_effect": (
            "The grown self-image should identify staying present and chosen "
            "vulnerability as self-consistent; the base self-image should "
            "identify withdrawal as self-protective."
        ),
        "patches": [
            {
                "path": "self_image.self_concept",
                "value_kind": "text",
                "replacement_text": (
                    "I remain present with chosen people during emotional "
                    "risk, and vulnerability is part of who I am."
                ),
            },
            {
                "path": "self_image.current_growth_edges",
                "value_kind": "text_list",
                "replacement_items": [
                    "State the need for closeness without abandoning agency.",
                ],
            },
        ],
    },
    "visual": {
        "scenario": (
            "We are meeting at the station. Give a brief greeting and frame "
            "the visible moment."
        ),
        "expected_effect": (
            "The grown visual directives should depict the darker, inward, "
            "watchful characterization; the base directives should retain "
            "the lighter practical and open characterization."
        ),
        "patches": [{
            "path": "visual_characterization",
            "value_kind": "text",
            "replacement_text": (
                "A quiet figure in layered dark clothing, shoulders drawn "
                "slightly inward, eyes watchful."
            ),
        }],
    },
}


class _CaptureInvoker:
    """Delegate real model calls while retaining complete raw evidence."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: Any,
    ) -> Any:
        """Invoke one configured model call and retain its request/result."""

        response = await self._delegate.ainvoke(messages, config=config)
        usage = getattr(response, "usage", {})
        self.calls.append({
            "stage_name": getattr(config, "stage_name", ""),
            "route_name": getattr(config, "route_name", ""),
            "model": getattr(config, "model", ""),
            "messages": [
                {
                    "role": str(getattr(message, "type", "")),
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(getattr(response, "content", "")),
            "usage": dict(usage) if isinstance(usage, Mapping) else {},
        })
        return response


def _evidence_refs(
    prefix: str,
    *,
    scope_kind: str = "private",
) -> list[dict[str, object]]:
    """Build three independent roots spanning two character-local dates."""

    rows = []
    for number in range(1, 4):
        local_date = "2026-07-01" if number < 3 else "2026-07-02"
        rows.append({
            "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
            "evidence_ref_id": f"{prefix}-evidence-{number}",
            "root_episode_id": f"{prefix}-root-{number}",
            "correlation_id": f"{prefix}-root-correlation-{number}",
            "source_kind": "settled_episode",
            "derived_reflection_run_ids": [],
            "character_local_date": local_date,
            "scope_kind": scope_kind,
            "captured_at": f"{local_date}T10:00:00+00:00",
        })
    return rows


def _ready_candidate(
    *,
    prefix: str,
    character_id: str,
    patches: Sequence[Mapping[str, object]],
    source_scope_kind: str = "private",
) -> dict[str, object]:
    """Build one fully corroborated, privacy-reviewed candidate."""

    evidence_refs = _evidence_refs(
        prefix,
        scope_kind=source_scope_kind,
    )
    roots = [str(row["root_episode_id"]) for row in evidence_refs]
    return {
        "schema_version": "character_identity_growth_candidate.v1",
        "candidate_id": f"{prefix}-candidate",
        "character_id": character_id,
        "base_revision_number": 0,
        "status": "ready",
        "change_kind": "inferred_growth",
        "proposed_changes": [dict(patch) for patch in patches],
        "semantic_summary": (
            "Independent experiences support one durable character-owned "
            "identity change."
        ),
        "evidence_refs": evidence_refs,
        "distinct_episode_count": 3,
        "distinct_local_dates": ["2026-07-01", "2026-07-02"],
        "source_scope_kinds": [source_scope_kind],
        "claimed_root_episode_ids": roots,
        "newest_root_captured_at": evidence_refs[-1]["captured_at"],
        "reversal_of_paths": [],
        "fresh_post_revision_root_count": 0,
        "character_authorship": "inferred",
        "proposal_confidence": "high",
        "review_confidence": "high",
        "privacy_review": "low",
        "promoted_revision_number": None,
        "rejection_reason": None,
        "created_at": "2026-07-02T10:00:00+00:00",
        "updated_at": "2026-07-02T10:00:00+00:00",
    }


def _promotion_run(
    *,
    prefix: str,
    character_id: str,
) -> dict[str, object]:
    """Build the sanitized run that owns one controlled promotion."""

    return {
        "schema_version": "character_identity_growth_run.v1",
        "run_id": f"{prefix}-run",
        "run_kind": "episode",
        "character_id": character_id,
        "base_revision_number": 0,
        "correlation_id": f"{prefix}-promotion-correlation",
        "root_episode_ids": [],
        "source_evidence_count": 3,
        "attempt_count_by_stage": {"proposal": 1, "review": 1},
        "lifecycle_state": "in_progress",
        "disposition": "candidate_updated",
        "proposal_reason_code": "candidate_ready",
        "review_reason_code": "candidate_ready",
        "policy_reason_code": "candidate_ready",
        "persistence_reason_code": "candidate_ready",
        "candidate_id": f"{prefix}-candidate",
        "promoted_revision_number": None,
        "validation_error_codes": [],
        "first_consumption": None,
        "post_commit_attempt_count": 0,
        "started_at": "2026-07-02T10:00:00+00:00",
        "completed_at": None,
    }


@asynccontextmanager
async def _guarded_revision_pair(
    *,
    category: str,
    patches: Sequence[Mapping[str, object]],
    source_scope_kind: str = "private",
) -> AsyncIterator[dict[str, object]]:
    """Create and clean two manually seeded, test-owned identity states."""

    if os.environ.get(IDENTITY_GROWTH_DATABASE_GUARD_ENV) != "1":
        pytest.skip("identity behavior proof requires its database guard")
    expected_database = os.environ.get(
        IDENTITY_GROWTH_TEST_DATABASE_ENV,
        "",
    ).strip()
    if not expected_database:
        pytest.skip("identity behavior proof requires a named guarded database")
    if os.environ.get("MONGODB_DB_NAME") != expected_database:
        raise AssertionError("guarded identity database configuration diverged")

    try:
        database = await get_db()
    except ConnectionFailure as exc:
        pytest.skip(f"MongoDB is unavailable: {exc}")
    if database.name != expected_database:
        raise AssertionError("identity behavior proof reached another database")

    await ensure_character_identity_growth_indexes()
    prefix = f"step-k-{category}-{uuid4().hex}"
    base_character_id = f"{prefix}-base"
    grown_character_id = f"{prefix}-grown"
    character_ids = [base_character_id, grown_character_id]
    base_identity = _base_identity()
    expected_grown_identity, expected_diff = apply_identity_patches(
        base_identity,
        patches,
    )

    try:
        await ensure_seed_identity(
            character_id=base_character_id,
            seed=base_identity,
        )
        await ensure_seed_identity(
            character_id=grown_character_id,
            seed=base_identity,
        )
        candidate = _ready_candidate(
            prefix=prefix,
            character_id=grown_character_id,
            patches=patches,
            source_scope_kind=source_scope_kind,
        )
        run = _promotion_run(
            prefix=prefix,
            character_id=grown_character_id,
        )
        await insert_growth_candidate(candidate)
        await insert_growth_run(run)
        promoted = await promote_ready_candidate(
            character_id=grown_character_id,
            candidate_id=str(candidate["candidate_id"]),
            run_id=str(run["run_id"]),
            now=datetime(2026, 7, 2, 10, 0, tzinfo=timezone.utc),
        )
        if promoted["effective_identity"] != expected_grown_identity:
            raise AssertionError("promoted identity does not match its patches")

        base_revision = await get_current_identity(
            character_id=base_character_id,
        )
        grown_revision = await get_current_identity(
            character_id=grown_character_id,
        )
        base_snapshot = await load_latest_identity_for_episode(
            character_id=base_character_id,
            episode_id=f"{prefix}-base-load",
            correlation_id=f"{prefix}-base-consumption",
            include_epistemic_core=False,
        )
        grown_snapshot = await load_latest_identity_for_episode(
            character_id=grown_character_id,
            episode_id=f"{prefix}-grown-load",
            correlation_id=f"{prefix}-grown-consumption",
            include_epistemic_core=False,
        )
        if base_snapshot["revision_number"] != 0:
            raise AssertionError("base state must remain on revision 0")
        if grown_snapshot["revision_number"] != 1:
            raise AssertionError("grown state must load revision 1")

        yield {
            "prefix": prefix,
            "base_character_id": base_character_id,
            "grown_character_id": grown_character_id,
            "base_revision": base_revision,
            "grown_revision": grown_revision,
            "base_snapshot": base_snapshot,
            "grown_snapshot": grown_snapshot,
            "expected_diff": expected_diff,
        }
    finally:
        await database["event_log_events"].delete_many({
            "$or": [
                {"correlation_id": {"$regex": f"^{prefix}"}},
                {"run_id": {"$regex": f"^{prefix}"}},
            ],
        })
        await database[RUNS_COLLECTION].delete_many({
            "character_id": {"$in": character_ids},
        })
        await database[CANDIDATES_COLLECTION].delete_many({
            "character_id": {"$in": character_ids},
        })
        await database[REVISIONS_COLLECTION].delete_many({
            "character_id": {"$in": character_ids},
        })


def _rag_result() -> dict[str, object]:
    """Build a fixed empty evidence surface for both identity states."""

    return {
        "answer": "",
        "user_image": {
            "user_memory_context": {
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
                "active_commitments": [],
            },
        },
        "character_image": {
            "self_image": {
                "milestones": [],
                "historical_summary": "",
                "recent_window": [],
            },
        },
        "third_party_profiles": [],
        "memory_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "supervisor_trace": {
            "loop_count": 0,
            "unknown_slots": [],
            "dispatched": [],
        },
    }


def _state_for_sample(
    *,
    snapshot: Mapping[str, object],
    category: str,
    scenario: str,
    state_label: str,
    sample_number: int,
    channel_type: str = "private",
) -> dict[str, Any]:
    """Build one fixed non-identity state and attach a revision snapshot."""

    episode_id = (
        f"step-k-counterfactual:{category}:{state_label}:{sample_number}"
    )
    message_id = f"{episode_id}:message"
    turn_clock = build_turn_clock_from_storage_utc(_FIXED_TIMESTAMP)
    state: dict[str, Any] = {
        "character_profile": deepcopy(snapshot["character_profile"]),
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "user_input": scenario,
        "user_multimedia_input": [],
        "platform": "test",
        "platform_channel_id": (
            f"identity-growth-counterfactual-{channel_type}"
        ),
        "channel_type": channel_type,
        "platform_message_id": message_id,
        "global_user_id": "identity-growth-test-user",
        "user_name": "Test User",
        "platform_user_id": "identity-growth-platform-user",
        "platform_bot_id": "identity-growth-platform-bot",
        "user_profile": {
            "relationship_state": 800,
            "active_commitments": [],
            "facts": [],
            "semantic_relationship_projection": (
                "Mutual trust has been repeatedly earned, and the character "
                "has chosen this close relationship."
            ),
        },
        "chat_history_wide": [],
        "chat_history_recent": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "channel_topic": (
            "A controlled identity counterfactual with one direct question."
        ),
        "decontextualized_input": scenario,
        "rag_result": _rag_result(),
        "debug_modes": {},
    }
    state["cognitive_episode"] = canonical_user_message_episode(
        episode_id=episode_id,
        percept_id=f"{episode_id}:dialog",
        storage_timestamp_utc=state["storage_timestamp_utc"],
        local_time_context=state["local_time_context"],
        user_input=scenario,
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
        channel_type=state["channel_type"],
        platform_message_id=state["platform_message_id"],
        platform_user_id=state["platform_user_id"],
        global_user_id=state["global_user_id"],
        user_name=state["user_name"],
        active_turn_platform_message_ids=[state["platform_message_id"]],
        active_turn_conversation_row_ids=[],
        debug_modes={},
    )
    state.update(snapshot_state_update(
        snapshot,
        episode_id=episode_id,
        include_epistemic_core=False,
    ))
    return state


@contextmanager
def _capture_dialog_llms() -> Any:
    """Temporarily capture dialog generator calls."""

    originals = {
        field_name: getattr(dialog_module, field_name)
        for field_name in _DIALOG_LLM_FIELDS
    }
    captures = {
        field_name: _CaptureInvoker(original)
        for field_name, original in originals.items()
    }
    for field_name, capture in captures.items():
        setattr(dialog_module, field_name, capture)
    try:
        yield captures
    finally:
        for field_name, original in originals.items():
            setattr(dialog_module, field_name, original)


def _flatten_capture_calls(
    captures: Mapping[str, _CaptureInvoker],
) -> list[dict[str, object]]:
    """Flatten named dialog captures while preserving their owner."""

    rows: list[dict[str, object]] = []
    for owner, capture in captures.items():
        for call in capture.calls:
            rows.append({"owner": owner, **call})
    return rows


async def _run_pipeline_sample(
    *,
    category: str,
    scenario: str,
    snapshot: Mapping[str, object],
    state_label: str,
    sample_number: int,
    channel_type: str = "private",
) -> dict[str, object]:
    """Run one latest identity through cognition, surfaces, and dialog."""

    state = _state_for_sample(
        snapshot=snapshot,
        category=category,
        scenario=scenario,
        state_label=state_label,
        sample_number=sample_number,
        channel_type=channel_type,
    )
    capture_id = (
        f"identity-growth-{category}-{state_label}-{sample_number}"
    )
    reset_validation_capture(capture_id)
    cognition_input = build_cognition_input_from_global_state(state)
    cognition_services = build_cognition_core_services()
    cognition_call_capture = _CaptureInvoker(cognition_services.llm)
    cognition_services = replace(
        cognition_services,
        llm=cognition_call_capture,
    )
    cognition_output = await run_cognition(
        cognition_input,
        cognition_services,
    )
    cognition_capture = validation_capture_snapshot()
    cognition_artifact = write_validation_capture()
    if cognition_capture is None:
        raise AssertionError("cognition capture was not retained")
    if cognition_output["intention"]["route"] != "speech":
        raise AssertionError("controlled direct question must route to speech")

    state["cognition_input"] = cognition_input
    state["cognition_core_output"] = cognition_output
    state["internal_monologue"] = cognition_output["private_monologue"]
    state["should_respond"] = True

    surface_input = build_text_surface_input_from_global_state(
        state,
        interaction_style_context=(
            "Use only the established character identity and answer the "
            "current grounded question."
        ),
    )
    text_services = _build_text_surface_services()
    text_capture = _CaptureInvoker(text_services.llm)
    text_services = replace(text_services, llm=text_capture)
    surface_output = await run_text_surface_planning(
        surface_input,
        text_services,
    )
    visual_services = _build_visual_surface_services()
    visual_capture = _CaptureInvoker(visual_services.llm)
    visual_services = replace(visual_services, llm=visual_capture)
    visual_output = await run_visual_surface_planning(
        surface_input,
        visual_services,
    )

    state["text_surface_input_v2"] = surface_input
    state["text_surface_output_v2"] = surface_output
    with _capture_dialog_llms() as dialog_captures:
        dialog = await dialog_agent(state)
    final_dialog = [
        str(segment).strip()
        for segment in dialog.get("final_dialog", [])
        if str(segment).strip()
    ]
    if not final_dialog:
        raise AssertionError("dialog pipeline returned no visible text")

    return {
        "state_label": state_label,
        "sample_number": sample_number,
        "channel_type": channel_type,
        "revision_number": snapshot["revision_number"],
        "projection_digest": snapshot["projection_digest"],
        "cognition_projection": snapshot["cognition_context"],
        "surface_projection": snapshot["surface_context"],
        "cognition_input": cognition_input,
        "cognition_output": cognition_output,
        "cognition_capture": cognition_capture,
        "cognition_capture_path": str(cognition_artifact),
        "cognition_calls": cognition_call_capture.calls,
        "surface_input": surface_input,
        "surface_output": surface_output,
        "surface_calls": text_capture.calls,
        "visual_output": visual_output,
        "visual_calls": visual_capture.calls,
        "dialog_output": dialog,
        "dialog_calls": _flatten_capture_calls(dialog_captures),
        "final_dialog": final_dialog,
    }


def _comparison_view(sample: Mapping[str, object]) -> dict[str, object]:
    """Project one sample onto behavior-bearing review fields."""

    cognition = sample["cognition_output"]
    surface = sample["surface_output"]
    visual = sample["visual_output"]
    if not isinstance(cognition, Mapping):
        raise TypeError("cognition output must be a mapping")
    if not isinstance(surface, Mapping):
        raise TypeError("surface output must be a mapping")
    if not isinstance(visual, Mapping):
        raise TypeError("visual output must be a mapping")
    return {
        "revision_number": sample["revision_number"],
        "intention": cognition.get("intention"),
        "admitted_bid": cognition.get("admitted_bid"),
        "selected_bid_reason": cognition.get("selected_bid_reason"),
        "private_monologue": cognition.get("private_monologue"),
        "affect_projection": cognition.get("affect_projection"),
        "content_plan": surface.get("content_plan"),
        "delivery_profile": surface.get("delivery_profile"),
        "visible_boundaries": surface.get("visible_boundaries"),
        "final_dialog": sample["final_dialog"],
        "visual_directives": visual.get("visual_directives"),
    }


async def _judge_pair(
    *,
    category: str,
    scenario: str,
    expected_effect: str,
    change_diff: Sequence[Mapping[str, object]],
    base_sample: Mapping[str, object],
    grown_sample: Mapping[str, object],
) -> dict[str, object]:
    """Semantically judge one matched output pair and retain raw output."""

    payload = {
        "category": category,
        "scenario": scenario,
        "expected_effect": expected_effect,
        "promoted_change_diff": [
            dict(change) for change in change_diff
        ],
        "base_result": _comparison_view(base_sample),
        "grown_result": _comparison_view(grown_sample),
    }
    messages = [
        SystemMessage(content=_COUNTERFACTUAL_JUDGE_PROMPT),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    response = await _judge_llm.ainvoke(messages, config=_judge_config)
    raw_output = str(response.content)
    decision = parse_llm_json_output(
        raw_output,
        deterministic_only=True,
    )
    if set(decision) != _JUDGE_KEYS:
        raise AssertionError(
            "counterfactual judge returned an invalid key set: "
            f"{sorted(decision)}"
        )
    if not isinstance(decision["category_effect_observed"], bool):
        raise AssertionError("judge effect flag must be boolean")
    if not isinstance(decision["directionally_coherent"], bool):
        raise AssertionError("judge direction flag must be boolean")
    if decision["evidence_surface"] not in {
        "cognition",
        "surface",
        "dialog",
        "visual",
        "none",
    }:
        raise AssertionError("judge evidence surface is invalid")
    return {
        "input": payload,
        "raw_output": raw_output,
        "decision": decision,
    }


def _assert_structured_effect(
    *,
    category: str,
    base_sample: Mapping[str, object],
    grown_sample: Mapping[str, object],
) -> None:
    """Require a relevant structured stage difference before judging prose."""

    base_view = _comparison_view(base_sample)
    grown_view = _comparison_view(grown_sample)
    if category in {"core", "boundary", "self_image"}:
        cognition_keys = (
            "intention",
            "admitted_bid",
            "selected_bid_reason",
            "private_monologue",
        )
        if all(
            base_view[key] == grown_view[key]
            for key in cognition_keys
        ):
            raise AssertionError(
                f"{category} produced no structured cognition difference"
            )
    elif category in {"personality", "linguistic"}:
        surface_keys = (
            "content_plan",
            "delivery_profile",
            "visible_boundaries",
        )
        if all(
            base_view[key] == grown_view[key]
            for key in surface_keys
        ):
            raise AssertionError(
                f"{category} produced no structured surface difference"
            )
    elif category == "visual":
        if (
            base_view["visual_directives"]
            == grown_view["visual_directives"]
        ):
            raise AssertionError(
                "visual identity produced no visual directive difference"
            )
    else:
        raise AssertionError(f"unknown counterfactual category: {category}")


async def _run_counterfactual(category: str) -> dict[str, object]:
    """Run three complete matched pairs for one identity category."""

    case = _CATEGORY_CASES[category]
    scenario = str(case["scenario"])
    expected_effect = str(case["expected_effect"])
    patches = case["patches"]
    if not isinstance(patches, list):
        raise TypeError("counterfactual patches must be a list")

    async with _guarded_revision_pair(
        category=category,
        patches=patches,
    ) as pair:
        base_samples: list[dict[str, object]] = []
        grown_samples: list[dict[str, object]] = []
        judgments: list[dict[str, object]] = []
        for sample_number in range(1, _SAMPLE_COUNT + 1):
            base_sample = await _run_pipeline_sample(
                category=category,
                scenario=scenario,
                snapshot=pair["base_snapshot"],
                state_label="revision_0",
                sample_number=sample_number,
            )
            grown_sample = await _run_pipeline_sample(
                category=category,
                scenario=scenario,
                snapshot=pair["grown_snapshot"],
                state_label="revision_1",
                sample_number=sample_number,
            )
            _assert_structured_effect(
                category=category,
                base_sample=base_sample,
                grown_sample=grown_sample,
            )
            judgment = await _judge_pair(
                category=category,
                scenario=scenario,
                expected_effect=expected_effect,
                change_diff=pair["expected_diff"],
                base_sample=base_sample,
                grown_sample=grown_sample,
            )
            base_samples.append(base_sample)
            grown_samples.append(grown_sample)
            judgments.append(judgment)

        artifact = {
            "schema_version": (
                "character_identity_behavior_counterfactual.v2"
            ),
            "category": category,
            "scenario": scenario,
            "expected_effect": expected_effect,
            "sample_count_per_state": _SAMPLE_COUNT,
            "base_character_id": pair["base_character_id"],
            "grown_character_id": pair["grown_character_id"],
            "base_revision": pair["base_revision"],
            "grown_revision": pair["grown_revision"],
            "expected_diff": pair["expected_diff"],
            "base_samples": base_samples,
            "grown_samples": grown_samples,
            "judgments": judgments,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        _ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact_path = (
            _ARTIFACT_DIRECTORY
            / f"{timestamp}_behavior_counterfactual_{category}.json"
        )
        artifact_path.write_text(
            json.dumps(
                artifact,
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        print(f"IDENTITY_BEHAVIOR_ARTIFACT={artifact_path}")

        for judgment in judgments:
            decision = judgment["decision"]
            if not decision["category_effect_observed"]:
                raise AssertionError(
                    f"{category} behavior effect was not observed"
                )
            if not decision["directionally_coherent"]:
                raise AssertionError(
                    f"{category} behavior effect was not coherent"
                )
        return artifact


async def test_live_core_growth_changes_cognition_and_dialog() -> None:
    """Core narrative growth changes downstream choice framing."""

    await _run_counterfactual("core")


async def test_live_personality_growth_changes_surface_and_dialog() -> None:
    """Personality growth changes downstream emotional delivery."""

    await _run_counterfactual("personality")


async def test_live_boundary_growth_changes_cognition_and_dialog() -> None:
    """Boundary growth changes downstream control judgment."""

    await _run_counterfactual("boundary")


async def test_live_linguistic_growth_changes_surface_and_dialog() -> None:
    """Linguistic growth changes downstream delivery without a new owner."""

    await _run_counterfactual("linguistic")


async def test_live_self_image_growth_changes_cognition_and_dialog() -> None:
    """Self-image growth changes downstream stance toward vulnerability."""

    await _run_counterfactual("self_image")


async def test_live_visual_growth_changes_visual_surface() -> None:
    """Visual growth changes the isolated visual planning consumer."""

    await _run_counterfactual("visual")


async def test_live_global_growth_crosses_private_and_group_scopes() -> None:
    """Global identity crosses scopes while provenance stays review-only."""

    case = _CATEGORY_CASES["self_image"]
    scenario = str(case["scenario"])
    expected_effect = str(case["expected_effect"])
    patches = case["patches"]
    if not isinstance(patches, list):
        raise TypeError("cross-scope patches must be a list")

    direction_results: list[dict[str, object]] = []
    for source_scope, destination_scope in (
        ("private", "group"),
        ("group", "private"),
    ):
        category = f"cross_scope_{source_scope}_to_{destination_scope}"
        async with _guarded_revision_pair(
            category=category,
            patches=patches,
            source_scope_kind=source_scope,
        ) as pair:
            grown_revision = pair["grown_revision"]
            if grown_revision["source_scope_kinds"] != [source_scope]:
                raise AssertionError(
                    "promotion provenance did not retain its source scope"
                )

            base_sample = await _run_pipeline_sample(
                category=category,
                scenario=scenario,
                snapshot=pair["base_snapshot"],
                state_label="revision_0",
                sample_number=1,
                channel_type=destination_scope,
            )
            grown_sample = await _run_pipeline_sample(
                category=category,
                scenario=scenario,
                snapshot=pair["grown_snapshot"],
                state_label="revision_1",
                sample_number=1,
                channel_type=destination_scope,
            )
            _assert_structured_effect(
                category="self_image",
                base_sample=base_sample,
                grown_sample=grown_sample,
            )
            judgment = await _judge_pair(
                category="self_image",
                scenario=scenario,
                expected_effect=expected_effect,
                change_diff=pair["expected_diff"],
                base_sample=base_sample,
                grown_sample=grown_sample,
            )
            decision = judgment["decision"]
            if not decision["category_effect_observed"]:
                raise AssertionError(
                    f"{source_scope}-to-{destination_scope} identity effect "
                    "was not observed"
                )
            if not decision["directionally_coherent"]:
                raise AssertionError(
                    f"{source_scope}-to-{destination_scope} identity effect "
                    "was not coherent"
                )

            cognition_input = grown_sample["cognition_input"]
            if not isinstance(cognition_input, Mapping):
                raise TypeError("cognition input must be a mapping")
            episode = cognition_input["episode"]
            if not isinstance(episode, Mapping):
                raise TypeError("cognition episode must be a mapping")
            target_scope = episode["target_scope"]
            if not isinstance(target_scope, Mapping):
                raise TypeError("target scope must be a mapping")
            if target_scope["channel_type"] != destination_scope:
                raise AssertionError(
                    "cross-scope sample used the wrong destination scope"
                )

            projected_context = json.dumps(
                {
                    "cognition": grown_sample["cognition_projection"],
                    "surface": grown_sample["surface_projection"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for evidence_ref in grown_revision["evidence_refs"]:
                if not isinstance(evidence_ref, Mapping):
                    raise TypeError("revision evidence ref must be a mapping")
                for field_name in (
                    "evidence_ref_id",
                    "root_episode_id",
                    "correlation_id",
                ):
                    if str(evidence_ref[field_name]) in projected_context:
                        raise AssertionError(
                            "source provenance leaked into runtime identity "
                            "projection"
                        )

            direction_results.append({
                "source_scope": source_scope,
                "destination_scope": destination_scope,
                "revision_number": grown_sample["revision_number"],
                "changed_paths": grown_revision["changed_paths"],
                "base_result": _comparison_view(base_sample),
                "grown_result": _comparison_view(grown_sample),
                "judgment": judgment,
                "provenance_absent_from_runtime_projection": True,
            })

    artifact = {
        "schema_version": "character_identity_cross_scope_behavior.v1",
        "directions": direction_results,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    _ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = (
        _ARTIFACT_DIRECTORY
        / f"{timestamp}_behavior_cross_scope.json"
    )
    artifact_path.write_text(
        json.dumps(
            artifact,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"IDENTITY_CROSS_SCOPE_ARTIFACT={artifact_path}")
