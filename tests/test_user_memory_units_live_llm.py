"""Live LLM contract tests for user memory-unit consolidation."""

from __future__ import annotations

from datetime import datetime, timezone
import json

import httpx
import pytest

from kazusa_ai_chatbot.config import CONSOLIDATION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import persona_supervisor2_consolidator_memory_units as memory_units_module
from kazusa_ai_chatbot.time_context import build_character_time_context
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


class _CapturingAsyncLLM:
    """Capture LLM calls while delegating to the real live model."""

    def __init__(self, wrapped_llm, calls: list[dict]) -> None:
        self._wrapped_llm = wrapped_llm
        self._calls = calls

    async def ainvoke(self, messages):
        response = await self._wrapped_llm.ainvoke(messages)
        payload = json.loads(messages[1].content)
        parsed_response = memory_units_module.parse_llm_json_output(response.content)
        self._calls.append({
            "system_prompt": messages[0].content,
            "payload": payload,
            "raw_response": response.content,
            "parsed_response": parsed_response,
        })
        return response


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
        "time_context": build_character_time_context(timestamp),
        "global_user_id": "live-memory-unit-user",
        "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
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


def _build_dated_commitment_extractor_state() -> dict:
    """Build a state that reproduces the relative-date commitment bug.

    Args:
        None.

    Returns:
        Consolidator state where ``明天`` must resolve to 2026-05-07.
    """
    timestamp = "2026-05-06T07:31:00+00:00"
    state = {
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
        "global_user_id": "live-memory-unit-user",
        "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
        "user_name": "LiveMemoryUnitUser",
        "decontexualized_input": (
            "用户接受了千纱提出的条件：明天要先完成展示，展示合格后"
            "才能继续推进更亲密的互动。"
        ),
        "final_dialog": [
            (
                "明天的展示先合格再说，表现特别好的话，千纱才会考虑给一点甜头。"
            )
        ],
        "internal_monologue": (
            "千纱把这件事当作一个必须按日期兑现的条件，不允许用户跳过。"
        ),
        "emotional_appraisal": (
            "千纱觉得用户心急，但她仍然坚持展示合格这个前提。"
        ),
        "interaction_subtext": (
            "这是一项有到期日的互动承诺，后续回复需要知道日期已经锚定。"
        ),
        "logical_stance": "CONFIRM",
        "character_intent": "SET_BOUNDARY",
        "chat_history_recent": [
            {
                "role": "user",
                "display_name": "LiveMemoryUnitUser",
                "timestamp": "2026-05-06T07:30:00+00:00",
                "content": "那明天展示合格后是不是就可以继续？",
            },
            {
                "role": "assistant",
                "display_name": "杏山千纱 (Kyōyama Kazusa)",
                "timestamp": "2026-05-06T07:31:00+00:00",
                "content": (
                    "明天先展示，合格以后我才考虑下一步，表现特别好再说甜头。"
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
        "new_facts": [],
        "future_promises": [
            {
                "promise": (
                    "用户需要在 2026-05-07 完成展示并达到合格标准。"
                )
            }
        ],
        "subjective_appraisals": [
            "千纱认为这项展示条件必须按日期兑现，不能被明天这种相对词拖延。",
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

    llm_calls: list[dict] = []
    extractor_llm = _CapturingAsyncLLM(memory_units_module._extractor_llm, llm_calls)
    monkeypatch.setattr(memory_units_module, "_extractor_llm", extractor_llm)

    state = _build_extractor_state()
    candidates = await memory_units_module.extract_memory_unit_candidates(state)
    trace_path = write_llm_trace(
        "user_memory_units_live_llm",
        "extractor_concrete_architecture_decision",
        {
            "input_state": state,
            "llm_payload": llm_calls[0]["payload"],
            "raw_response": llm_calls[0]["raw_response"],
            "parsed_response": llm_calls[0]["parsed_response"],
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


async def test_live_extractor_anchors_tomorrow_commitment_due_at(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Verify live extraction anchors ``明天`` into an absolute due date.

    Args:
        ensure_live_llm: Fixture that skips when the live endpoint is unavailable.
        monkeypatch: Pytest helper used only to capture the real LLM payload/output.

    Returns:
        None.
    """
    del ensure_live_llm

    llm_calls: list[dict] = []
    extractor_llm = _CapturingAsyncLLM(memory_units_module._extractor_llm, llm_calls)
    monkeypatch.setattr(memory_units_module, "_extractor_llm", extractor_llm)

    state = _build_dated_commitment_extractor_state()
    candidates = await memory_units_module.extract_memory_unit_candidates(state)
    active_commitments = [
        candidate
        for candidate in candidates
        if candidate["unit_type"] == "active_commitment"
    ]
    trace_path = write_llm_trace(
        "user_memory_units_live_llm",
        "extractor_anchors_tomorrow_commitment_due_at",
        {
            "input_state": state,
            "llm_payload": llm_calls[0]["payload"],
            "raw_response": llm_calls[0]["raw_response"],
            "parsed_response": llm_calls[0]["parsed_response"],
            "validated_candidates": candidates,
            "active_commitments": active_commitments,
            "judgment": (
                "The extractor should convert 明天 to the local date "
                "2026-05-07 and persist due_at as UTC midnight for that date."
            ),
        },
    )

    assert active_commitments
    dated_commitment = active_commitments[0]
    assert dated_commitment["due_at"] == "2026-05-06T12:00:00+00:00"

    semantic_blob = " ".join(
        [
            dated_commitment["fact"],
            dated_commitment["subjective_appraisal"],
            dated_commitment["relationship_signal"],
        ]
    )
    assert "2026-05-07" in semantic_blob
    assert "明天" not in semantic_blob
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
    timestamp = datetime.now(timezone.utc).isoformat()
    state = {
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
        "global_user_id": "live-memory-unit-user",
        "rag_result": {"user_memory_unit_candidates": [existing_unit]},
    }
    captured: dict[str, object] = {"llm_calls": []}

    async def _retrieve_memory_unit_merge_candidates(global_user_id, **kwargs):
        captured["retrieval"] = {
            "global_user_id": global_user_id,
            "candidate_unit": kwargs["candidate_unit"],
            "surfaced_units": kwargs["surfaced_units"],
        }
        return [existing_unit]

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
    merge_judge_llm = _CapturingAsyncLLM(
        memory_units_module._merge_judge_llm,
        captured["llm_calls"],
    )
    rewrite_llm = _CapturingAsyncLLM(
        memory_units_module._rewrite_llm,
        captured["llm_calls"],
    )
    monkeypatch.setattr(memory_units_module, "_merge_judge_llm", merge_judge_llm)
    monkeypatch.setattr(memory_units_module, "_rewrite_llm", rewrite_llm)
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
    merged_fact_lower = merged_fact.lower()
    combined_source_length = len(existing_unit["fact"]) + len(candidate["fact"])

    assert result["decision"] in {"merge", "evolve"}
    assert result["unit_id"] == existing_unit["unit_id"]
    assert captured["updated_unit_id"] == existing_unit["unit_id"]
    assert len(captured["llm_calls"]) == 2
    assert any(
        term in merged_fact_lower
        for term in ("fact", "factual", "\u4e8b\u5b9e")
    )
    assert any(
        term in merged_fact_lower
        for term in ("subjective", "feelings", "\u4e3b\u89c2", "\u60c5\u611f")
    )
    assert len(merged_fact) < combined_source_length
    assert captured["update_kwargs"]["merge_history_entry"]["decision"] == result["decision"]
    assert trace_path.exists()


async def test_live_stability_judge_uses_session_spread_evidence(
    ensure_live_llm,
    monkeypatch,
) -> None:
    """Verify the live stability judge receives enough evidence to classify.

    Args:
        ensure_live_llm: Fixture that skips when the live endpoint is unavailable.
        monkeypatch: Pytest helper used to capture the live stability call.

    Returns:
        None.
    """
    del ensure_live_llm

    existing_unit = {
        "unit_id": "existing-stability-pattern",
        "unit_type": "recent_shift",
        "fact": (
            "The user repeatedly asks Kazusa to keep memory records fact-based "
            "instead of diary-like emotional summaries."
        ),
        "subjective_appraisal": (
            "Kazusa reads this as recurring architecture guidance rather than "
            "a one-off wording preference."
        ),
        "relationship_signal": (
            "Kazusa should continue grounding memory in concrete events."
        ),
        "count": 3,
        "source_refs": [
            {
                "source": "chat",
                "timestamp": "2026-04-27T10:00:00+12:00",
                "message_id": "stability-1",
            },
            {
                "source": "chat",
                "timestamp": "2026-04-28T10:00:00+12:00",
                "message_id": "stability-2",
            },
        ],
        "updated_at": "2026-04-28T10:00:00+12:00",
        "last_seen_at": "2026-04-28T10:00:00+12:00",
    }
    candidate = {
        "candidate_id": "candidate-stability-repeat",
        "unit_type": "recent_shift",
        "fact": (
            "The user again rejects overlapping emotional memory surfaces and "
            "asks for factual event anchors."
        ),
        "subjective_appraisal": (
            "Kazusa sees this as another instance of the same memory-quality "
            "guidance."
        ),
        "relationship_signal": (
            "Future consolidation should preserve factual anchors first."
        ),
        "evidence_refs": [
            {
                "source": "chat",
                "timestamp": "2026-04-29T10:00:00+12:00",
                "message_id": "stability-3",
            }
        ],
    }
    merge_result = {
        "candidate_id": "candidate-stability-repeat",
        "decision": "evolve",
        "cluster_id": "existing-stability-pattern",
        "reason": "same recurring memory architecture guidance",
    }
    timestamp = "2026-04-29T10:00:00+12:00"
    state = {
        "timestamp": timestamp,
        "time_context": build_character_time_context(timestamp),
        "global_user_id": "live-memory-unit-user",
    }
    payload = memory_units_module._stability_payload(
        state,
        unit_id="existing-stability-pattern",
        candidate=candidate,
        merge_result=merge_result,
        candidate_clusters=[existing_unit],
    )

    llm_calls: list[dict] = []
    stability_llm = _CapturingAsyncLLM(
        memory_units_module._stability_llm,
        llm_calls,
    )
    monkeypatch.setattr(memory_units_module, "_stability_llm", stability_llm)
    stability_result = await memory_units_module._judge_memory_unit_stability(
        state,
        unit_id="existing-stability-pattern",
        candidate=candidate,
        merge_result=merge_result,
        candidate_clusters=[existing_unit],
    )
    trace_path = write_llm_trace(
        "user_memory_units_live_llm",
        "stability_judge_session_spread_evidence",
        {
            "payload": payload,
            "raw_response": llm_calls[0]["raw_response"],
            "parsed_response": llm_calls[0]["parsed_response"],
            "validated_result": stability_result,
            "judgment": (
                "The live stability judge should see multiple-session evidence "
                "and classify the recurring pattern as stable."
            ),
        },
    )

    assert stability_result["window"] == "stable"
    assert trace_path.exists()
