"""Live LLM contract tests for user memory-unit consolidation."""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from kazusa_ai_chatbot.config import CONSOLIDATION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_memory_units as memory_units_module
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def _skip_if_llm_unavailable() -> None:
    """Skip live memory-unit tests when the consolidation LLM is unavailable.

    Args:
        None.

    Returns:
        None. The function calls ``pytest.skip`` when the endpoint is down.
    """
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            models_url = f"{CONSOLIDATION_LLM_BASE_URL.rstrip('/')}/models"
            response = await client.get(models_url)
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {CONSOLIDATION_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(
            "LLM endpoint returned server error "
            f"{response.status_code}: {CONSOLIDATION_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured consolidation LLM endpoint is reachable.

    Args:
        None.

    Returns:
        None.
    """
    await _skip_if_llm_unavailable()


def _build_extractor_state() -> dict:
    """Build a realistic state for the memory-unit extractor prompt.

    Args:
        None.

    Returns:
        Consolidator state containing one concrete architecture decision and
        relationship appraisal evidence.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    state = {
        "timestamp": timestamp,
        "global_user_id": "live-memory-unit-user",
        "user_name": "LiveMemoryUnitUser",
        "decontexualized_input": (
            "Please remember the actual architecture decision: historical "
            "summary, recent window, and character diary are being replaced by "
            "fact-anchored memory units. Each unit should carry a simple fact, "
            "Kazusa's subjective appraisal, and a relationship signal."
        ),
        "final_dialog": [
            (
                "Kazusa acknowledges the decision and treats it as a memory "
                "architecture change rather than a mood note."
            )
        ],
        "internal_monologue": (
            "The user is giving precise system architecture guidance and wants "
            "the memory layer to preserve concrete facts without losing Kazusa's "
            "subjective reading."
        ),
        "emotional_appraisal": (
            "Kazusa feels a little scrutinized, but also relieved because the "
            "boundary between event facts and emotional interpretation is clearer."
        ),
        "interaction_subtext": (
            "The user trusts Kazusa more when she can keep engineering decisions "
            "separate from emotional diary-style impressions."
        ),
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "chat_history_recent": [
            {
                "role": "user",
                "display_name": "LiveMemoryUnitUser",
                "content": (
                    "The current historical summary and recent window are "
                    "recording the same emotional diary material."
                ),
            },
            {
                "role": "user",
                "display_name": "LiveMemoryUnitUser",
                "content": (
                    "The new architecture should replace those overlapping "
                    "fields with fact-anchored memory units."
                ),
            },
        ],
        "rag_result": {
            "user_image": {
                "user_memory_context": {
                    "stable_patterns": [],
                    "recent_shifts": [],
                    "objective_facts": [],
                    "milestones": [],
                    "active_commitments": [],
                }
            },
            "user_memory_unit_candidates": [],
        },
        "new_facts": [
            {
                "fact": (
                    "The user decided to replace historical summary, recent "
                    "window, and character diary with fact-anchored memory units."
                )
            }
        ],
        "future_promises": [],
        "subjective_appraisals": [
            (
                "Kazusa interprets the decision as a request to balance concrete "
                "event recall with her own subjective relationship reading."
            )
        ],
    }
    return state


async def test_live_extractor_outputs_concrete_memory_unit(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Verify the live extractor produces concrete unit fields for a real case.

    Args:
        ensure_live_llm: Fixture that skips when the live endpoint is unavailable.
        monkeypatch: Pytest helper used only to capture the real LLM payload/output.

    Returns:
        None.
    """
    del ensure_live_llm

    captured: dict[str, object] = {}
    original_invoke_json = memory_units_module._invoke_json

    async def _capturing_invoke_json(system_prompt: str, payload: dict) -> dict:
        """Capture the real LLM call while preserving live model behavior.

        Args:
            system_prompt: Prompt text sent to the consolidation LLM.
            payload: JSON payload sent to the consolidation LLM.

        Returns:
            Parsed JSON returned by the live consolidation LLM.
        """
        captured["system_prompt"] = system_prompt
        captured["payload"] = payload
        parsed_response = await original_invoke_json(system_prompt, payload)
        captured["parsed_response"] = parsed_response
        return parsed_response

    monkeypatch.setattr(memory_units_module, "_invoke_json", _capturing_invoke_json)

    state = _build_extractor_state()
    candidates = await memory_units_module.extract_memory_unit_candidates(state)
    trace_path = write_llm_trace(
        "user_memory_units_live_llm",
        "extractor_concrete_architecture_decision",
        {
            "input_state": state,
            "llm_payload": captured["payload"],
            "parsed_response": captured["parsed_response"],
            "validated_candidates": candidates,
            "judgment": (
                "The extractor should preserve the architecture decision as a "
                "concrete memory unit with fact, subjective appraisal, and "
                "relationship signal."
            ),
        },
    )

    assert candidates
    for candidate in candidates:
        assert candidate["unit_type"] in {
            "stable_pattern",
            "recent_shift",
            "objective_fact",
            "milestone",
            "active_commitment",
        }
        assert len(candidate["fact"]) >= 40
        assert len(candidate["subjective_appraisal"]) >= 20
        assert len(candidate["relationship_signal"]) >= 20

    candidate_blob = " ".join(candidate["fact"].lower() for candidate in candidates)
    assert "memory" in candidate_blob
    assert "architecture" in candidate_blob or "historical summary" in candidate_blob
    assert trace_path.exists()


async def test_live_merge_rewrite_compacts_similar_memory_unit(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Verify live merge/rewrite groups a similar unit instead of duplicating it.

    Args:
        ensure_live_llm: Fixture that skips when the live endpoint is unavailable.
        monkeypatch: Pytest helper used to isolate persistence while preserving
            live merge and rewrite LLM calls.

    Returns:
        None.
    """
    del ensure_live_llm

    existing_unit = {
        "unit_id": "existing-memory-boundary",
        "unit_type": "objective_fact",
        "fact": (
            "The user said subjective feelings about a user and memorable past "
            "events should not be recorded as the same kind of memory data."
        ),
        "subjective_appraisal": (
            "Kazusa reads this as a request to keep emotional interpretation "
            "grounded in explicit events."
        ),
        "relationship_signal": (
            "Future memory records should separate factual event anchors from "
            "Kazusa's subjective reading."
        ),
        "updated_at": "2026-04-29T00:00:00+00:00",
    }
    candidate = {
        "candidate_id": "candidate-memory-boundary-repeat",
        "unit_type": "objective_fact",
        "fact": (
            "The user again emphasized that feelings in memory need a factual "
            "basis and that event history should be distinguished from subjective "
            "user-image appraisal."
        ),
        "subjective_appraisal": (
            "Kazusa sees this as recurring architecture guidance about balancing "
            "facts with emotional interpretation."
        ),
        "relationship_signal": (
            "Kazusa should compact repeated memory-boundary guidance rather than "
            "store another overlapping emotional diary note."
        ),
        "evidence_refs": [],
    }
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_user_id": "live-memory-unit-user",
        "rag_result": {"user_memory_unit_candidates": [existing_unit]},
    }
    captured: dict[str, object] = {"llm_calls": []}
    original_invoke_json = memory_units_module._invoke_json

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        captured["retrieval"] = {
            "global_user_id": global_user_id,
            "candidate_unit": kwargs["candidate_unit"],
            "surfaced_units": kwargs["surfaced_units"],
        }
        return [existing_unit]

    async def _capturing_invoke_json(system_prompt: str, payload: dict) -> dict:
        parsed_response = await original_invoke_json(system_prompt, payload)
        captured["llm_calls"].append({
            "system_prompt": system_prompt,
            "payload": payload,
            "parsed_response": parsed_response,
        })
        return parsed_response

    async def _insert_user_memory_units(global_user_id, units, **kwargs):
        pytest.fail(
            "Similar memory-unit candidate should merge/evolve, not create "
            f"a new unit: {units!r}"
        )

    async def _update_user_memory_unit_semantics(unit_id, semantics, **kwargs):
        captured["updated_unit_id"] = unit_id
        captured["updated_semantics"] = semantics
        captured["update_kwargs"] = kwargs

    monkeypatch.setattr(
        memory_units_module,
        "retrieve_memory_unit_merge_candidates",
        _retrieve_memory_unit_merge_candidates,
    )
    monkeypatch.setattr(memory_units_module, "_invoke_json", _capturing_invoke_json)
    monkeypatch.setattr(
        memory_units_module,
        "insert_user_memory_units",
        _insert_user_memory_units,
    )
    monkeypatch.setattr(
        memory_units_module,
        "update_user_memory_unit_semantics",
        _update_user_memory_unit_semantics,
    )

    result = await memory_units_module.process_memory_unit_candidate(state, candidate)
    trace_path = write_llm_trace(
        "user_memory_units_live_llm",
        "merge_rewrite_compacts_similar_memory_unit",
        {
            "state": state,
            "existing_unit": existing_unit,
            "candidate": candidate,
            "result": result,
            "captured": captured,
            "judgment": (
                "The live merge judge should group the repeated memory-boundary "
                "event with the existing unit, and the live rewrite should compact "
                "the shared event detail into one updated semantic record."
            ),
        },
    )

    updated_semantics = captured["updated_semantics"]
    merged_fact = updated_semantics["fact"]
    combined_source_length = len(existing_unit["fact"]) + len(candidate["fact"])

    assert result["decision"] in {"merge", "evolve"}
    assert result["unit_id"] == existing_unit["unit_id"]
    assert captured["updated_unit_id"] == existing_unit["unit_id"]
    assert len(captured["llm_calls"]) == 2
    assert "fact" in merged_fact.lower() or "factual" in merged_fact.lower()
    assert "subjective" in merged_fact.lower() or "feelings" in merged_fact.lower()
    assert len(merged_fact) < combined_source_length
    assert captured["update_kwargs"]["merge_history_entry"]["decision"] == result["decision"]
    assert trace_path.exists()
