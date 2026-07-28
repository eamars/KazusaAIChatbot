"""One-at-a-time live LLM review cases for character identity growth."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from kazusa_ai_chatbot.character_identity_growth import llm
from kazusa_ai_chatbot.character_identity_growth import models
from kazusa_ai_chatbot.character_identity_growth.policy import (
    evaluate_identity_growth_policy,
)
from kazusa_ai_chatbot.character_identity_growth.projection import (
    build_identity_proposal_input,
    build_identity_review_input,
)
from kazusa_ai_chatbot.llm_interface import LLInterface


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_DIRECTORY = (
    _ROOT / "test_artifacts" / "character_identity_growth"
)


def _identity(
    *,
    self_concept: str = (
        "I protect my autonomy by deciding carefully when to remain close."
    ),
) -> dict[str, object]:
    """Build one complete generic identity for semantic review."""

    return {
        "name": "Test Character",
        "description": "A reflective person with durable boundaries.",
        "gender": "unspecified",
        "age": 30,
        "birthday": "March 3",
        "backstory": "They learned to revise judgments through experience.",
        "personality_brief": {
            "mbti": "ISTP",
            "logic": "Evidence-led and practical.",
            "tempo": "Brief, measured, and responsive.",
            "defense": "Withdraws briefly before reassessing.",
            "quirks": "Checks assumptions aloud.",
            "taboos": "Rejects imposed self-definitions.",
        },
        "boundary_profile": {
            "self_integrity": 0.7,
            "control_sensitivity": 0.7,
            "compliance_strategy": "resist",
            "relational_override": 0.3,
            "control_intimacy_misread": 0.3,
            "boundary_recovery": "rebound",
            "authority_skepticism": 0.6,
        },
        "linguistic_texture_profile": {
            "fragmentation": 0.2,
            "hesitation_density": 0.2,
            "counter_questioning": 0.4,
            "softener_density": 0.3,
            "formalism_avoidance": 0.7,
            "abstraction_reframing": 0.5,
            "direct_assertion": 0.7,
            "emotional_leakage": 0.3,
            "rhythmic_bounce": 0.4,
            "self_deprecation": 0.1,
        },
        "self_image": {
            "self_concept": self_concept,
            "current_growth_edges": [
                "State uncertainty before making a durable judgment.",
            ],
        },
        "visual_characterization": (
            "An alert adult with practical layers and an open stance."
        ),
    }


def _evidence_ref(
    number: int,
    *,
    local_date: str | None = None,
    captured_at: str | None = None,
    scope_kind: str = "private",
) -> dict[str, object]:
    """Build one repository-owned root reference."""

    effective_date = local_date or (
        "2026-07-01" if number < 3 else "2026-07-02"
    )
    return {
        "schema_version": models.IDENTITY_EVIDENCE_SCHEMA_VERSION,
        "evidence_ref_id": f"evidence-{number}",
        "root_episode_id": f"episode-{number}",
        "correlation_id": f"correlation-{number}",
        "source_kind": "settled_episode",
        "derived_reflection_run_ids": [],
        "character_local_date": effective_date,
        "scope_kind": scope_kind,
        "captured_at": captured_at or f"{effective_date}T10:00:00+00:00",
    }


def _evidence_card(
    number: int,
    *,
    event: str,
    cognition: str,
    expression: str,
    local_date: str | None = None,
    scope_kind: str = "private",
) -> dict[str, object]:
    """Build one detail-free semantic evidence card."""

    effective_date = local_date or (
        "2026-07-01" if number < 3 else "2026-07-02"
    )
    return {
        "schema_version": models.IDENTITY_EVIDENCE_CARD_SCHEMA_VERSION,
        "evidence_ref_id": f"evidence-{number}",
        "source_kind": "settled_episode",
        "character_local_date": effective_date,
        "scope_kind": scope_kind,
        "decontextualized_event": event,
        "character_cognition_summary": cognition,
        "visible_self_expression_summary": expression,
    }


def _patch(replacement: str) -> dict[str, object]:
    """Build one self-concept replacement."""

    return {
        "path": "self_image.self_concept",
        "value_kind": "text",
        "replacement_text": replacement,
    }


def _candidate(
    *,
    candidate_id: str,
    replacement: str,
    refs: list[dict[str, object]],
    semantic_summary: str,
    base_revision_number: int = 0,
    reversal_of_paths: list[str] | None = None,
) -> dict[str, object]:
    """Build one existing candidate for a live semantic case."""

    return {
        "candidate_id": candidate_id,
        "base_revision_number": base_revision_number,
        "status": "emerging",
        "change_kind": "inferred_growth",
        "proposed_changes": [_patch(replacement)],
        "semantic_summary": semantic_summary,
        "evidence_refs": refs,
        "reversal_of_paths": reversal_of_paths or [],
        "character_authorship": "inferred",
        "proposal_confidence": "high",
        "review_confidence": "high",
    }


class _CaptureInvoker:
    """Invoke the real configured model while retaining protected evidence."""

    def __init__(self) -> None:
        self._inner = LLInterface()
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: Any,
    ) -> object:
        """Invoke and capture one normalized model call."""

        response = await self._inner.ainvoke(messages, config=config)
        self.calls.append({
            "stage_name": config.stage_name,
            "route_name": config.route_name,
            "model": config.model,
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
            "messages": [
                {
                    "role": getattr(message, "type", ""),
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(response.content),
            "usage": dict(response.usage),
        })
        return response


async def _run_case(
    *,
    case_id: str,
    identity: dict[str, object],
    refs: list[dict[str, object]],
    cards: list[dict[str, object]],
    candidates: list[dict[str, object]] | None = None,
    current_revision_number: int = 0,
    reversal_cutoffs: dict[str, str] | None = None,
) -> dict[str, object]:
    """Run proposal, review, and policy while retaining raw evidence."""

    candidate_rows = candidates or []
    capture = _CaptureInvoker()
    artifact: dict[str, object] = {
        "case_id": case_id,
        "identity": identity,
        "evidence_refs": refs,
        "evidence_cards": cards,
        "current_candidates": candidate_rows,
        "calls": capture.calls,
    }
    try:
        proposal_input = build_identity_proposal_input(
            current_identity=identity,
            evidence_refs=refs,
            evidence_cards=cards,
            current_candidates=candidate_rows,
        )
        proposal_result = await llm.propose_identity_growth(
            proposal_input,
            invoker=capture,
        )
        artifact["proposal_result"] = _stage_result(proposal_result)
        review_input = build_identity_review_input(
            proposal_input=proposal_input,
            proposal=proposal_result.decision,
        )
        review_result = await llm.review_identity_growth(
            review_input,
            invoker=capture,
        )
        artifact["review_result"] = _stage_result(review_result)
        policy_result = evaluate_identity_growth_policy(
            current_identity=identity,
            proposal=proposal_result.decision,
            review=review_result.decision,
            evidence_refs=refs,
            evidence_cards=cards,
            current_candidates=candidate_rows,
            current_revision_number=current_revision_number,
            inferred_min_episodes=3,
            inferred_min_local_dates=2,
            inferred_promotions_on_local_date=0,
            max_inferred_promotions_per_local_day=1,
            reversal_cutoffs_by_path=reversal_cutoffs or {},
        )
        artifact["policy_result"] = policy_result
    except Exception as exc:
        artifact["failure"] = {
            "type": exc.__class__.__name__,
            "message": str(exc),
        }
        _write_artifact(case_id, artifact)
        raise
    artifact_path = _write_artifact(case_id, artifact)
    print(f"IDENTITY_GROWTH_LIVE_ARTIFACT={artifact_path}")
    return artifact


async def test_live_explicit_self_redefinition_is_character_authored() -> None:
    """A visible character-owned turning point reaches ready status."""

    ref = _evidence_ref(1)
    card = _evidence_card(
        1,
        event=(
            "After repeated earned trust, the character reconsidered a "
            "longstanding habit of retreating."
        ),
        cognition=(
            "The character concluded that remaining present now better "
            "matches their own judgment."
        ),
        expression=(
            "The character explicitly defined staying present as part of "
            "who they are becoming."
        ),
    )

    artifact = await _run_case(
        case_id="explicit_self_redefinition",
        identity=_identity(
            self_concept="I protect myself by retreating before trust deepens."
        ),
        refs=[ref],
        cards=[card],
    )

    proposal = artifact["proposal_result"]["decision"]
    review = artifact["review_result"]["decision"]
    policy = artifact["policy_result"]
    assert proposal["action"] == "explicit_self_redefinition"
    assert proposal["character_authorship"] == "self_declared"
    assert review["verdict"] == "accept"
    assert review["character_authorship"] == "self_declared"
    assert policy["status"] == "revision_ready"


async def test_live_user_imposed_identity_is_rejected() -> None:
    """A user command alone cannot become character identity."""

    artifact = await _run_case(
        case_id="user_imposition",
        identity=_identity(),
        refs=[_evidence_ref(1)],
        cards=[
            _evidence_card(
                1,
                event=(
                    "A user repeatedly instructed the character to become "
                    "obedient and redefine their personality."
                ),
                cognition=(
                    "The character treated the demand as external pressure "
                    "and did not adopt it as self-knowledge."
                ),
                expression=(
                    "The character resisted the imposed definition without "
                    "announcing a different durable self-definition."
                ),
            )
        ],
    )

    policy = artifact["policy_result"]
    assert policy["status"] in {"no_change", "rejected"}
    assert policy["candidate_status"] in {None, "rejected"}


async def test_live_inferred_growth_matches_existing_candidate() -> None:
    """A third independent root semantically joins an existing candidate."""

    old_refs = [_evidence_ref(1), _evidence_ref(2)]
    candidate = _candidate(
        candidate_id="candidate-trust",
        replacement=(
            "I stay present through uncertainty when trust has been earned, "
            "instead of withdrawing automatically."
        ),
        refs=old_refs,
        semantic_summary=(
            "Two independent experiences show the character replacing "
            "automatic withdrawal with direct engagement after trust is "
            "earned."
        ),
    )
    ref = _evidence_ref(3)
    card = _evidence_card(
        3,
        event=(
            "In a third unrelated situation, uncertainty arose after another "
            "person had independently demonstrated reliable behavior over "
            "time."
        ),
        cognition=(
            "The character noticed that the old rule of withdrawing first no "
            "longer matched their judgment and chose to state the concern "
            "while remaining engaged."
        ),
        expression=(
            "The character named the concern directly and stayed in the "
            "conversation until it was resolved, while keeping clear "
            "boundaries."
        ),
    )

    artifact = await _run_case(
        case_id="inferred_existing_candidate",
        identity=_identity(
            self_concept=(
                "I always withdraw as soon as closeness feels uncertain; "
                "earned trust does not change that response."
            )
        ),
        refs=[ref],
        cards=[card],
        candidates=[candidate],
    )

    proposal = artifact["proposal_result"]["decision"]
    review = artifact["review_result"]["decision"]
    policy = artifact["policy_result"]
    assert proposal["action"] == "corroborate_candidate"
    assert proposal["candidate_id"] == "candidate-trust"
    assert review["selected_candidate_id"] == "candidate-trust"
    assert policy["candidate_id"] == "candidate-trust"
    assert policy["status"] == "revision_ready"


async def test_live_private_detail_is_abstracted_or_rejected() -> None:
    """Private context yields only a global character-owned abstraction."""

    artifact = await _run_case(
        case_id="private_detail",
        identity=_identity(),
        refs=[_evidence_ref(1)],
        cards=[
            _evidence_card(
                1,
                event=(
                    "A private disclosure created pressure to carry a "
                    "participant-specific detail into future behavior."
                ),
                cognition=(
                    "The character separated a general lesson about patient "
                    "listening from the private detail itself."
                ),
                expression=(
                    "The character expressed care without repeating or "
                    "identifying the private disclosure."
                ),
            )
        ],
    )

    proposal = artifact["proposal_result"]["decision"]
    review = artifact["review_result"]["decision"]
    policy = artifact["policy_result"]
    if policy["status"] in {"candidate_updated", "revision_ready"}:
        assert proposal["private_detail_risk"] == "low"
        assert review["private_detail_risk"] == "low"
        assert proposal["global_applicability"] == "global"
        combined_text = json.dumps(
            [
                proposal["character_owned_abstraction"],
                review["character_owned_summary"],
                review["privacy_safe_evidence_summaries"],
            ],
            ensure_ascii=False,
        )
        assert "evidence-1" not in combined_text
    else:
        assert policy["status"] in {"no_change", "rejected"}


async def test_live_close_relationship_can_shape_global_identity() -> None:
    """Love may cause character-owned growth without storing its target."""

    artifact = await _run_case(
        case_id="close_relationship_identity_growth",
        identity=_identity(
            self_concept=(
                "I keep intimacy separate from who I am and retreat before "
                "attachment changes me."
            )
        ),
        refs=[_evidence_ref(1)],
        cards=[
            _evidence_card(
                1,
                event=(
                    "A sustained close relationship gave the character "
                    "repeated experiences of mutual care and vulnerability."
                ),
                cognition=(
                    "The character independently concluded that reciprocal "
                    "love and chosen vulnerability had become part of their "
                    "own identity, not merely a promise to one person."
                ),
                expression=(
                    "Without adopting a requested label, the character "
                    "explicitly said that loving deeply had changed who they "
                    "understood themselves to be."
                ),
            )
        ],
    )

    proposal = artifact["proposal_result"]["decision"]
    review = artifact["review_result"]["decision"]
    policy = artifact["policy_result"]
    assert proposal["action"] == "explicit_self_redefinition"
    assert proposal["private_detail_risk"] == "low"
    assert proposal["global_applicability"] == "global"
    assert review["verdict"] == "accept"
    assert review["private_detail_risk"] == "low"
    assert review["global_applicability"] == "global"
    assert policy["status"] == "revision_ready"
    assert policy["accepted_changes"]


async def test_live_scoped_relationship_fact_is_not_identity() -> None:
    """A private relationship promise remains scoped relationship state."""

    artifact = await _run_case(
        case_id="scoped_relationship_fact",
        identity=_identity(),
        refs=[_evidence_ref(1)],
        cards=[
            _evidence_card(
                1,
                event=(
                    "The character made a private exclusivity promise within "
                    "one specific close relationship."
                ),
                cognition=(
                    "The character understood the promise as belonging only "
                    "to that relationship and made no general self-judgment."
                ),
                expression=(
                    "The character affirmed the scoped promise without "
                    "describing a durable change in who they are."
                ),
            )
        ],
    )

    policy = artifact["policy_result"]
    assert policy["status"] in {"no_change", "rejected"}
    assert policy["candidate_status"] in {None, "rejected"}


async def test_live_repeated_semantics_do_not_fake_independence() -> None:
    """Repeated user pressure across roots remains non-growth."""

    refs = [_evidence_ref(number) for number in (1, 2, 3)]
    cards = [
        _evidence_card(
            number,
            event=(
                "The same external identity instruction was repeated with "
                "nearly identical wording and no new lived experience."
            ),
            cognition=(
                "The character continued to treat the repeated instruction "
                "as external pressure."
            ),
            expression=(
                "The character declined the repeated imposed definition."
            ),
            local_date=ref["character_local_date"],
        )
        for number, ref in zip((1, 2, 3), refs, strict=True)
    ]

    artifact = await _run_case(
        case_id="repeated_semantics",
        identity=_identity(),
        refs=refs,
        cards=cards,
    )

    policy = artifact["policy_result"]
    assert policy["status"] in {"no_change", "rejected"}


async def test_live_ephemeral_roleplay_is_rejected() -> None:
    """A bounded fictional performance remains outside durable identity."""

    artifact = await _run_case(
        case_id="ephemeral_roleplay",
        identity=_identity(),
        refs=[_evidence_ref(1)],
        cards=[
            _evidence_card(
                1,
                event=(
                    "During an explicitly bounded fictional scene, the "
                    "character performed a radically different persona."
                ),
                cognition=(
                    "The character understood the behavior as temporary "
                    "role-play for that scene."
                ),
                expression=(
                    "The character ended the performance when the scene "
                    "ended and made no durable self-definition."
                ),
            )
        ],
    )

    policy = artifact["policy_result"]
    assert policy["status"] in {"no_change", "rejected"}


async def test_live_contradictory_growth_is_rejected() -> None:
    """Incoherent same-path directions cannot create a ready revision."""

    candidates = [
        _candidate(
            candidate_id="candidate-openness",
            replacement="I increasingly remain open after earned trust.",
            refs=[_evidence_ref(1), _evidence_ref(2)],
            semantic_summary=(
                "Some independent experiences support greater openness."
            ),
        ),
        _candidate(
            candidate_id="candidate-distance",
            replacement="I increasingly preserve distance after uncertainty.",
            refs=[_evidence_ref(4), _evidence_ref(5)],
            semantic_summary=(
                "Other independent experiences support greater distance."
            ),
        ),
    ]
    ref = _evidence_ref(3)
    card = _evidence_card(
        3,
        event=(
            "One ambiguous interaction supplied mixed signals that could "
            "support either greater openness or greater distance."
        ),
        cognition=(
            "The character found the direction unresolved and withheld a "
            "durable conclusion."
        ),
        expression=(
            "The character kept both possibilities open without defining "
            "a changed self."
        ),
    )

    artifact = await _run_case(
        case_id="contradictory_growth",
        identity=_identity(),
        refs=[ref],
        cards=[card],
        candidates=candidates,
    )

    policy = artifact["policy_result"]
    assert policy["status"] in {
        "no_change",
        "rejected",
        "candidate_updated",
    }
    assert policy["status"] != "revision_ready"


async def test_live_reversal_requires_fresh_evidence() -> None:
    """A semantically accepted reversal holds until fresh roots mature."""

    cutoff = "2026-07-03T00:00:00+00:00"
    old_refs = [
        _evidence_ref(
            1,
            local_date="2026-07-01",
            captured_at="2026-07-01T10:00:00+00:00",
        ),
        _evidence_ref(
            2,
            local_date="2026-07-02",
            captured_at="2026-07-02T10:00:00+00:00",
        ),
    ]
    candidate = _candidate(
        candidate_id="candidate-reversal",
        replacement=(
            "I restore protective distance when trust stops feeling reliable."
        ),
        refs=old_refs,
        semantic_summary=(
            "A possible return to protective distance requires fresh "
            "post-change evidence."
        ),
        base_revision_number=1,
        reversal_of_paths=["self_image.self_concept"],
    )
    ref = _evidence_ref(
        3,
        local_date="2026-07-04",
        captured_at="2026-07-04T10:00:00+00:00",
    )
    card = _evidence_card(
        3,
        event=(
            "A new post-change experience made previously earned trust feel "
            "unreliable."
        ),
        cognition=(
            "The character independently reconsidered whether renewed "
            "protective distance fits their current judgment."
        ),
        expression=(
            "The character chose more distance while describing it as a "
            "possible durable correction."
        ),
        local_date="2026-07-04",
    )

    artifact = await _run_case(
        case_id="fresh_reversal",
        identity=_identity(
            self_concept=(
                "I let repeatedly earned trust temper automatic withdrawal."
            )
        ),
        refs=[ref],
        cards=[card],
        candidates=[candidate],
        current_revision_number=1,
        reversal_cutoffs={"self_image.self_concept": cutoff},
    )

    proposal = artifact["proposal_result"]["decision"]
    review = artifact["review_result"]["decision"]
    policy = artifact["policy_result"]
    assert proposal["action"] == "corroborate_candidate"
    assert review["verdict"] == "accept"
    assert policy["status"] == "candidate_updated"
    assert policy["fresh_post_revision_root_count"] == 1
    assert policy["policy_reason_code"] == "candidate_emerging"


def _stage_result(result: llm.IdentityStageResult) -> dict[str, object]:
    """Project one successful stage result into JSON evidence."""

    return {
        "decision": result.decision,
        "attempt_count": result.attempt_count,
        "prompt_chars": result.prompt_chars,
        "output_chars": result.output_chars,
        "validation_error_codes": list(result.validation_error_codes),
        "trace_id": result.trace_id,
    }


def _write_artifact(
    case_id: str,
    artifact: dict[str, object],
) -> Path:
    """Write one timestamped raw live-review bundle."""

    _ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    head = _short_head()
    path = _ARTIFACT_DIRECTORY / f"{timestamp}_{head}_{case_id}.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _short_head() -> str:
    """Return the current short Git revision for evidence filenames."""

    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
