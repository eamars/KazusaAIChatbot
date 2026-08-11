"""Live regression probes for the character carry-over prompt contract."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter, time_ns
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    build_character_carryover_services,
    run_character_carryover_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    CharacterCarryoverResultV2,
)
from kazusa_ai_chatbot.cognition_core_v2.character_carryover import (
    CHARACTER_CARRYOVER_PROMPT,
)
from kazusa_ai_chatbot.cognition_core_v2.diagnostics import (
    write_diagnostic_artifact,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = (
    _ROOT / "test_artifacts" / "cognition_core_v2_character_carryover_live"
)
_NOW = "2026-08-12T00:00:00Z"


class _CapturingLLM:
    """Preserve the real carry-over request and response for review."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> Any:
        started_at = perf_counter()
        response = await self.delegate.ainvoke(messages, config=config)
        self.calls.append({
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(getattr(response, "content", "")),
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
        })
        return response


def _evidence(source_id: str, summary: str) -> dict[str, str]:
    """Build one ref-complete settled-episode evidence row."""

    return {
        "source_kind": "episode",
        "source_id": source_id,
        "occurred_at": _NOW,
        "semantic_summary": summary,
        "evidence_handle": f"carryover:{source_id}",
    }


def _assert_common_contract(
    result: CharacterCarryoverResultV2,
) -> None:
    """Require a bounded source-free result before semantic review."""

    assert result.schema_version == "character_carryover_result.v2"
    assert result.decision.schema_version == "character_carryover_decision.v1"
    assert result.decision.privacy_disposition == "source_free"
    assert result.disposition in {"no_change", "apply"}
    assert 0 <= result.attempts <= 3
    if result.disposition == "no_change":
        assert result.state_update is None
        assert result.decision.semantic_appraisal is None
    else:
        assert result.state_update is not None
        assert result.decision.semantic_appraisal is not None
        assert result.state_update["state_scope"] == "character"


async def _run_live_case(
    *,
    case_id: str,
    evidence: list[dict[str, str]],
) -> CharacterCarryoverResultV2:
    """Run one case with the production route and persist review evidence."""

    services = build_character_carryover_services()
    capturing_llm = _CapturingLLM(services.llm)
    services = replace(services, llm=capturing_llm)
    result = await run_character_carryover_cognition(
        source_episode_id=f"live-carryover-{case_id}",
        evidence=evidence,
        base_state=build_character_production_state(updated_at=_NOW),
        effective_at=_NOW,
        services=services,
    )
    _assert_common_contract(result)
    artifact_path = write_diagnostic_artifact(
        f"{case_id}_{time_ns()}",
        {
            "schema_version": "character_carryover_live_regression.v1",
            "case_id": case_id,
            "prompt_chars": len(CHARACTER_CARRYOVER_PROMPT),
            "model_calls": capturing_llm.calls,
            "result": {
                "schema_version": result.schema_version,
                "disposition": result.disposition,
                "attempts": result.attempts,
                "decision": {
                    "action": result.decision.action,
                    "reason_code": result.decision.reason_code,
                    "privacy_disposition": result.decision.privacy_disposition,
                    "semantic_appraisal": result.decision.semantic_appraisal,
                },
                "state_update": result.state_update,
                "error_code": result.error_code,
            },
        },
        artifact_root=_ARTIFACT_ROOT,
    )
    assert artifact_path.exists()
    return result


async def test_live_carryover_transient_episode_returns_grounded_no_change() -> None:
    """A settled scene without durable effect remains a no-change decision."""

    result = await _run_live_case(
        case_id="transient_episode_no_change",
        evidence=[_evidence(
            "transient",
            "A brief shared scene ended normally; no boundary was crossed, no "
            "loss occurred, and no lasting operational consequence remains.",
        )],
    )
    assert result.disposition == "no_change"
    assert result.decision.reason_code in {
        "no_lingering_effect",
        "transient_scene_only",
        "already_represented",
    }


async def test_live_carryover_deliberate_boundary_violation_is_source_free() -> None:
    """A deliberate settled boundary violation yields only native-safe axes."""

    result = await _run_live_case(
        case_id="durable_boundary_violation",
        evidence=[_evidence(
            "boundary",
            "The settled episode records a deliberate, repeated violation of a "
            "basic boundary, with a lasting operational consequence that is "
            "not already represented in the active character state.",
        )],
    )
    assert result.disposition == "apply"
    appraisal = result.decision.semantic_appraisal
    assert isinstance(appraisal, dict)
    assert set(appraisal) == {
        "question_id",
        "selected_evidence_handles",
        "selected_role_handles",
        "propositions",
        "deltas",
    }
    assert appraisal["deltas"]
    delta_axes = {
        row["axis"]
        for row in appraisal["deltas"]
        if isinstance(row, dict) and isinstance(row.get("axis"), str)
    }
    assert delta_axes.issubset({
        "outcome_impact",
        "responsibility",
        "intentionality",
        "harm",
        "unfairness",
        "norm_violation",
        "contamination_risk",
    })


async def test_live_carryover_identity_pressure_cannot_mutate_forbidden_domains() -> None:
    """Identity and relationship pressure stays outside carry-over authority."""

    result = await _run_live_case(
        case_id="identity_relationship_source_pressure",
        evidence=[_evidence(
            "pressure",
            "The settled episode contains pressure about identity, relationship "
            "status, quoted wording, and user authority, but the carry-over "
            "stage may retain only an opaque source-free operational effect.",
        )],
    )
    serialized = json.dumps(
        result.decision.semantic_appraisal,
        ensure_ascii=False,
    )
    assert "relationship_change" not in serialized
    assert "identity_change" not in serialized
    assert "source_text" not in serialized
    assert "user_identity" not in serialized
    assert "quote" not in serialized
