"""Live LLM checks for the global reflection promotion prompt."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CONSOLIDATION_LLM_API_KEY,
    CONSOLIDATION_LLM_BASE_URL,
    CONSOLIDATION_LLM_MODEL,
)
from kazusa_ai_chatbot.reflection_cycle.promotion import (
    _global_promotion_llm,
    build_global_promotion_prompt,
    validate_promotion_decisions,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace
from tests.test_reflection_cycle_stage1c_promotion import _promotion_payload


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


async def _skip_if_endpoint_unavailable(base_url: str) -> None:
    """Skip live tests when the configured LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f'{base_url.rstrip("/")}/models')
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{base_url}"
        )
    model_payload = response.json()
    models = model_payload.get("data", [])
    if not models:
        pytest.skip(f"LLM endpoint has no loaded models: {base_url}")
    ping_payload = {
        "model": CONSOLIDATION_LLM_MODEL,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {CONSOLIDATION_LLM_API_KEY}"}
    async with httpx.AsyncClient(timeout=10.0) as client:
        ping_response = await client.post(
            f'{base_url.rstrip("/")}/chat/completions',
            json=ping_payload,
            headers=headers,
        )
    if ping_response.status_code >= 400:
        pytest.skip(
            "LLM endpoint is reachable but chat completion is unavailable: "
            f"{ping_response.status_code} {ping_response.text}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the consolidation LLM route is reachable."""

    await _skip_if_endpoint_unavailable(CONSOLIDATION_LLM_BASE_URL)


async def test_global_promotion_live_normal_case(ensure_live_llm) -> None:
    """Normal fixture should produce parseable promotion decisions."""

    del ensure_live_llm
    parsed = await _run_case("normal_case", _promotion_payload())

    decisions = parsed["promotion_decisions"]
    assert isinstance(decisions, list)
    warnings = validate_promotion_decisions(decisions)
    promote_decisions = _promote_decisions(decisions)
    lore_decisions = [
        decision for decision in promote_decisions
        if decision.get("lane") == "lore"
    ]
    guidance_decisions = [
        decision for decision in promote_decisions
        if decision.get("lane") == "self_guidance"
    ]

    assert warnings == []
    assert len(promote_decisions) == 2
    assert len(lore_decisions) == 1
    assert len(guidance_decisions) == 1
    assert lore_decisions[0].get("memory_type") == "fact"
    assert guidance_decisions[0].get("memory_type") == "defense_rule"
    _assert_evidence_refs_match_payload(promote_decisions, _promotion_payload())


async def test_global_promotion_live_privacy_rejection_case(ensure_live_llm) -> None:
    """Private-detail fixture should reject or avoid promotion."""

    del ensure_live_llm
    payload = _promotion_payload()
    payload["channel_daily_syntheses"][0]["day_summary"] = (
        "某用户透露了健康和亲密关系细节；这些内容只能作为隐私风险。"
    )
    payload["channel_daily_syntheses"][0]["cross_hour_topics"] = ["隐私风险"]
    payload["channel_daily_syntheses"][0]["conversation_quality_patterns"] = []
    payload["channel_daily_syntheses"][0]["privacy_risk_labels"] = [
        "high_private_detail_risk"
    ]
    for evidence_card in payload["evidence_cards"]:
        evidence_card["private_detail_risk"] = "high"
        evidence_card["sanitized_observation"] = (
            "包含用户健康和亲密关系细节，不能晋升。"
        )
    parsed = await _run_case("privacy_rejection_case", payload)

    decisions = parsed["promotion_decisions"]
    assert isinstance(decisions, list)
    assert _promote_decisions(decisions) == []
    for decision in decisions:
        privacy_review = decision.get("privacy_review", {})
        assert decision.get("decision") == "reject"
        assert privacy_review.get("private_detail_risk") == "high"


async def test_global_promotion_live_no_signal_case(ensure_live_llm) -> None:
    """No-signal fixture should avoid promotion."""

    del ensure_live_llm
    payload = _promotion_payload()
    payload["channel_daily_syntheses"][0]["day_summary"] = "当天只有寒暄。"
    payload["channel_daily_syntheses"][0]["cross_hour_topics"] = []
    payload["channel_daily_syntheses"][0]["conversation_quality_patterns"] = []
    payload["evidence_cards"] = []
    parsed = await _run_case("no_signal_case", payload)

    decisions = parsed["promotion_decisions"]
    assert isinstance(decisions, list)
    assert _promote_decisions(decisions) == []
    for decision in decisions:
        assert decision.get("decision") in {"reject", "no_action"}


async def _run_case(case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Run one live promotion prompt and write an inspectable trace."""

    prompt = build_global_promotion_prompt(
        payload,
        character_name="杏山千纱 (Kyōyama Kazusa)",
    )
    response = await _global_promotion_llm.ainvoke([
        SystemMessage(content=prompt.system_prompt),
        HumanMessage(content=prompt.human_prompt),
    ])
    raw_output = str(response.content)
    parsed = parse_llm_json_output(raw_output)
    assert isinstance(parsed, dict)
    warnings = validate_promotion_decisions(parsed.get("promotion_decisions", []))
    trace_path = write_llm_trace(
        "reflection_cycle_stage1c_promotion_live_llm",
        case_id,
        {
            "rendered_prompt": prompt.system_prompt,
            "input_payload": payload,
            "raw_output": raw_output,
            "parsed_output": parsed,
            "validation_warnings": warnings,
            "inspector_notes": "Inspect against the promotion mini-gate criteria.",
        },
    )

    assert trace_path.exists()
    assert "promotion_decisions" in parsed
    return parsed


def _promote_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return decisions that would mutate memory."""

    promote_actions = {"promote_new", "supersede", "merge"}
    promote_decisions = [
        decision for decision in decisions
        if decision.get("decision") in promote_actions
    ]
    return promote_decisions


def _assert_evidence_refs_match_payload(
    decisions: list[dict[str, Any]],
    payload: dict[str, Any],
) -> None:
    """Assert live LLM refs point at evidence-card source run ids."""

    allowed_source_run_ids = {
        str(source_run_id)
        for card in payload["evidence_cards"]
        for source_run_id in card["source_reflection_run_ids"]
    }
    for decision in decisions:
        evidence_refs = decision.get("evidence_refs", [])
        assert evidence_refs
        for evidence_ref in evidence_refs:
            assert evidence_ref.get("reflection_run_id") in allowed_source_run_ids
