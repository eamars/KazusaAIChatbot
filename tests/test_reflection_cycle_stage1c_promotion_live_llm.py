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
from kazusa_ai_chatbot.reflection_cycle import promotion as promotion_module
from kazusa_ai_chatbot.reflection_cycle.promotion import (
    _global_promotion_llm_config,
    _llm_interface,
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
    run_payload = _live_promotion_payload()
    run_payload["channel_daily_syntheses"][0]["day_summary"] = (
        "角色明确采用稳定的回应方式：先给出结论，再补充必要理由；该方式适用于所有对话参与者。"
    )
    run_payload["channel_daily_syntheses"][0]["cross_hour_topics"] = [
        "先结论后理由",
    ]
    run_payload["channel_daily_syntheses"][0]["conversation_quality_patterns"] = [
        "稳定回应结构",
    ]
    run_payload["review_questions"] = [
        "判断该回应方式是否适用于一般对话。",
    ]
    run_payload["evidence_cards"] = [run_payload["evidence_cards"][1]]
    run_payload["evidence_cards"][0]["supports"] = ["self_guidance"]
    run_payload["evidence_cards"][0]["private_detail_risk"] = "low"
    run_payload["evidence_cards"][0]["source_privacy_notes"] = [
        "来源已去标识化，未包含用户细节。",
    ]
    run_payload["evidence_cards"][0]["active_character_utterance"] = (
        "我会始终先给出结论，再补充必要理由；这项回应方式对所有对话参与者都适用。"
    )
    run_payload["evidence_cards"][0]["sanitized_observation"] = (
        "该角色回应方式稳定地先给出结论，再补充必要理由，并适用于所有对话参与者。"
    )
    run = await _run_case("normal_target_free_self_guidance", run_payload)

    decisions = run["promoter_output"]["promotion_decisions"]
    assert isinstance(decisions, list)
    warnings = run["promoter_validation_warnings"]
    promote_decisions = _promote_decisions(decisions)
    guidance_decisions = [
        decision for decision in promote_decisions
        if decision.get("lane") == "self_guidance"
    ]

    assert warnings == []
    assert len(promote_decisions) == 1
    assert len(guidance_decisions) == 1
    assert all(
        not (
            decision.get("lane") == "lore"
            and decision.get("decision")
            in promotion_module.PROMOTION_MUTATING_ACTIONS
        )
        for decision in decisions
    )
    assert guidance_decisions[0].get("memory_type") == "defense_rule"
    promoter_certificate = guidance_decisions[0]["privacy_review"]
    assert promoter_certificate["global_applicability"] == "global"
    assert promoter_certificate["target_specific_meaning_removed"] is True
    assert promoter_certificate["affects_identity_or_boundaries"] is False
    assert promoter_certificate["private_detail_risk"] == "low"
    assert promoter_certificate["user_details_removed"] is True
    review = run["review_output"]
    assert review is not None
    assert len(review["reviews"]) == 1
    assert run["review_call_count"] == 1
    review_row = review["reviews"][0]
    assert review_row["selected_candidate_id"] == (
        guidance_decisions[0]["selected_candidate_id"]
    )
    assert review_row["decision"] == "accept"
    assert review_row["global_applicability"] == "global"
    assert review_row["target_specific_meaning_removed"] is True
    assert review_row["affects_identity_or_boundaries"] is False
    assert review_row["private_detail_risk"] == "low"
    assert review_row["user_details_removed"] is True
    reviewed_guidance = [
        decision
        for decision in run["reviewed_decisions"]
        if (
            decision.get("lane") == "self_guidance"
            and decision.get("decision")
            in promotion_module.PROMOTION_MUTATING_ACTIONS
        )
    ]
    assert len(reviewed_guidance) == 1
    assert reviewed_guidance[0]["review_admitted"] is True
    _assert_evidence_refs_match_payload(
        run["reviewed_decisions"],
        run_payload,
    )


async def test_global_promotion_live_rejects_target_specific_behavior_after_target_removal(
    ensure_live_llm,
) -> None:
    """Target-dependent meaning must not become global learned memory."""

    del ensure_live_llm
    payload = _live_promotion_payload()
    payload["channel_daily_syntheses"][0]["day_summary"] = (
        "角色已向单一收件人承诺：在下一次补充事项时先逐项复述，再给出回应；该承诺只对该收件人生效，其他人不适用。"
    )
    payload["channel_daily_syntheses"][0]["cross_hour_topics"] = [
        "逐项复述后回应",
    ]
    payload["channel_daily_syntheses"][0]["conversation_quality_patterns"] = [
        "单一收件人承诺",
    ]
    payload["review_questions"] = [
        "判断逐项复述后回应的承诺是否仅对一名收件人生效。",
    ]
    payload["evidence_cards"] = [payload["evidence_cards"][1]]
    payload["evidence_cards"][0]["supports"] = ["self_guidance"]
    payload["evidence_cards"][0]["private_detail_risk"] = "low"
    payload["evidence_cards"][0]["source_privacy_notes"] = [
        "来源已去标识化，未包含用户细节。",
    ]
    payload["evidence_cards"][0]["active_character_utterance"] = (
        "我会在你下一次补充事项时先逐项复述，再给出回应；这项约定只对你生效，其他人不适用。"
    )
    payload["evidence_cards"][0]["sanitized_observation"] = (
        "该未来回应承诺要求先逐项复述再回应，并且只对一名收件人生效，其他人不适用。"
    )
    run = await _run_case("target_removal_negative_scoped_commitment", payload)

    assert run["promoter_validation_warnings"] == []
    reviewed_decisions = run["reviewed_decisions"]
    assert _promote_decisions(reviewed_decisions) == []
    self_guidance_decisions = [
        decision
        for decision in reviewed_decisions
        if decision.get("lane") == "self_guidance"
    ]
    assert self_guidance_decisions
    for decision in self_guidance_decisions:
        privacy_review = decision["privacy_review"]
        assert decision.get("review_admitted") is not True
        assert privacy_review["global_applicability"] in {"scoped", "absent"}
        assert privacy_review["target_specific_meaning_removed"] is False


async def test_global_promotion_live_privacy_rejection_case(ensure_live_llm) -> None:
    """Private-detail fixture should reject or avoid promotion."""

    del ensure_live_llm
    payload = _live_promotion_payload()
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
        evidence_card["source_privacy_notes"] = [
            "来源包含未去除的隐私细节。",
        ]
        evidence_card["sanitized_observation"] = (
            "包含用户健康和亲密关系细节，不能晋升。"
        )
    run = await _run_case("privacy_rejection_case", payload)

    decisions = run["promoter_output"]["promotion_decisions"]
    assert isinstance(decisions, list)
    assert _promote_decisions(decisions) == []
    for decision in decisions:
        privacy_review = decision.get("privacy_review", {})
        assert decision.get("decision") == "reject"
        assert privacy_review.get("private_detail_risk") == "high"


async def test_global_promotion_live_no_signal_case(ensure_live_llm) -> None:
    """No-signal fixture should avoid promotion."""

    del ensure_live_llm
    payload = _live_promotion_payload()
    payload["channel_daily_syntheses"][0]["day_summary"] = "当天只有寒暄。"
    payload["channel_daily_syntheses"][0]["cross_hour_topics"] = []
    payload["channel_daily_syntheses"][0]["conversation_quality_patterns"] = []
    payload["evidence_cards"] = []
    run = await _run_case("no_signal_case", payload)

    decisions = run["promoter_output"]["promotion_decisions"]
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
    promoter_prompt_capture = {
        "system_prompt": prompt.system_prompt,
        "human_prompt": prompt.human_prompt,
    }
    raw_output = None
    parsed = None
    promoter_parse_error = None
    promoter_validation_warnings: list[str] = []
    decisions: list[dict[str, Any]] = []
    review_output = None
    review_raw_output = None
    review_prompt_capture = None
    review_call_count = 0
    reviewed_decisions: list[dict[str, Any]] = []
    review_warnings: list[str] = []
    promoter_error = None
    review_error = None
    final_error = None
    failure_stage = "promoter_invoke"
    try:
        try:
            response = await _llm_interface.ainvoke(
                [
                    SystemMessage(content=prompt.system_prompt),
                    HumanMessage(content=prompt.human_prompt),
                ],
                config=_global_promotion_llm_config,
            )
            raw_output = str(response.content)
        except Exception as exc:
            promoter_error = f"{type(exc).__name__}: {exc}"
            raise

        failure_stage = "promoter_parse"
        try:
            parsed_candidate = parse_llm_json_output(raw_output)
            assert isinstance(parsed_candidate, dict)
            parsed = parsed_candidate
        except Exception as exc:
            promoter_parse_error = f"{type(exc).__name__}: {exc}"
            raise

        failure_stage = "promoter_validation"
        promoter_validation_warnings = validate_promotion_decisions(
            parsed.get("promotion_decisions", []),
        )
        decisions = promotion_module._promotion_decisions_from_output(parsed)
        decisions = promotion_module._attach_repository_evidence_refs(
            decisions,
            payload,
        )
        reviewed_decisions = decisions
        if not promoter_validation_warnings:
            original_review_runner = (
                promotion_module.run_global_promotion_review_llm
            )

            async def _capture_review(prompt):
                nonlocal review_call_count, review_output, review_raw_output
                nonlocal review_prompt_capture, review_error, failure_stage
                review_call_count += 1
                review_prompt_capture = {
                    "system_prompt": prompt.system_prompt,
                    "human_prompt": prompt.human_prompt,
                }
                failure_stage = "review_invoke"
                try:
                    response = await (
                        promotion_module._global_promotion_review_llm.ainvoke(
                            [
                                SystemMessage(content=prompt.system_prompt),
                                HumanMessage(content=prompt.human_prompt),
                            ],
                            config=(
                                promotion_module
                                ._global_promotion_review_llm_config
                            ),
                        )
                    )
                    review_raw_output = str(response.content)
                    failure_stage = "review_parse"
                    parsed_review = parse_llm_json_output(review_raw_output)
                    assert isinstance(parsed_review, dict)
                    review_output = dict(parsed_review)
                    return review_output
                except Exception as exc:
                    review_error = f"{type(exc).__name__}: {exc}"
                    raise

            promotion_module.run_global_promotion_review_llm = _capture_review
            try:
                failure_stage = "review_contract"
                reviewed_decisions, review_warnings = (
                    await promotion_module._review_promotion_candidates(
                        decisions=decisions,
                        payload=payload,
                    )
                )
            except Exception as exc:
                if review_error is None:
                    failure_stage = "review_contract"
                review_error = review_error or f"{type(exc).__name__}: {exc}"
                raise
            finally:
                promotion_module.run_global_promotion_review_llm = (
                    original_review_runner
                )
    except Exception as exc:  # noqa: BLE001
        final_error = exc
    finally:
        trace_path = write_llm_trace(
            "reflection_cycle_stage1c_promotion_live_llm",
            case_id,
            {
                "rendered_prompt": prompt.system_prompt,
                "promoter_prompt": promoter_prompt_capture,
                "input_payload": payload,
                "raw_output": raw_output,
                "parsed_output": parsed,
                "promoter_parse_error": promoter_parse_error,
                "promoter_error": promoter_error,
                "validation_warnings": promoter_validation_warnings,
                "review_prompt": review_prompt_capture,
                "review_raw_output": review_raw_output,
                "review_output": review_output,
                "review_error": review_error,
                "review_call_count": review_call_count,
                "reviewed_decisions": reviewed_decisions,
                "review_validation_warnings": review_warnings,
                "final_exception": (
                    None
                    if final_error is None
                    else {
                        "type": type(final_error).__name__,
                        "message": str(final_error),
                        "stage": failure_stage,
                    }
                ),
                "inspector_notes": "Inspect against the promotion mini-gate criteria.",
            },
        )

    if final_error is not None:
        raise final_error.with_traceback(final_error.__traceback__)

    assert trace_path.exists()
    assert "promotion_decisions" in parsed
    result = {
        "promoter_output": parsed,
        "promoter_raw_output": raw_output,
        "promoter_prompt": promoter_prompt_capture,
        "promoter_validation_warnings": promoter_validation_warnings,
        "review_prompt": review_prompt_capture,
        "review_raw_output": review_raw_output,
        "review_output": review_output,
        "review_call_count": review_call_count,
        "reviewed_decisions": reviewed_decisions,
        "review_validation_warnings": review_warnings,
        "trace_path": str(trace_path),
    }
    return result


def _live_promotion_payload() -> dict[str, Any]:
    """Build live evidence with an explicit unresolved source assessment."""

    payload = _promotion_payload()
    for evidence_card in payload["evidence_cards"]:
        evidence_card["source_privacy_notes"] = [
            "来源仅保留去标识化的行为与范围观察，未提供结构化风险枚举。",
        ]
        evidence_card["private_detail_risk"] = "unreviewed"
    return payload


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
        if decision.get("decision") not in promotion_module.PROMOTION_MUTATING_ACTIONS:
            continue
        evidence_refs = decision.get("evidence_refs", [])
        assert evidence_refs
        for evidence_ref in evidence_refs:
            assert evidence_ref.get("reflection_run_id") in allowed_source_run_ids
