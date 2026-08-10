"""Individually inspected live-LLM gates for group response decisions."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.cognition_core_v2.action_selection import plan_actions
from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_core_services,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ARTIFACT_ROOT = Path("test_artifacts/diagnostics")
_LATEST_ARTIFACT = _ARTIFACT_ROOT / (
    "self_cognition_group_visible_reply_latest_case.json"
)
_ELIGIBLE_ARTIFACT = _ARTIFACT_ROOT / (
    "self_cognition_group_visible_reply_eligible_case.json"
)


class _CapturingLLM:
    """Capture each production action-planning model response."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        started_at = perf_counter()
        response = await self.delegate.ainvoke(messages, config=config)
        raw_output = str(getattr(response, "content", ""))
        self.calls.append({
            "prompt_messages": [
                str(getattr(message, "content", ""))
                for message in messages
            ],
            "raw_model_output": raw_output,
            "latency_ms": round((perf_counter() - started_at) * 1000, 3),
            "route": {
                "stage_name": str(getattr(config, "stage_name", "")),
                "route_name": str(getattr(config, "route_name", "")),
                "model": str(getattr(config, "model", "")),
            },
        })
        return response


async def test_live_group_self_cognition_latest_case_stays_silent() -> None:
    """The latest ambient-shaped group case should choose explicit silence."""

    await _skip_if_llm_unavailable()
    result, calls = await _run_live_case(
        case_id="latest-ambient-group-window",
        evidence_text=(
            "Other participants are exchanging a correction and a joke. "
            "The character has not spoken in this window, no participant "
            "addresses her, and no reply is needed from her."
        ),
        engagement_guidelines=[
            "The character is absent from the visible window.",
            "No direct address or structured participation grounding is present.",
            "Do not force an insertion into ambient group chatter.",
        ],
        artifact_path=_LATEST_ARTIFACT,
    )

    assert result["self_cognition_response"]["decision"] == "stay_silent"
    assert result["intention"]["route"] == "silence"
    assert calls
    assert _LATEST_ARTIFACT.exists()


async def test_live_group_self_cognition_eligible_case_can_propose_reply() -> None:
    """A grounded group case should be able to propose the visible reply path."""

    await _skip_if_llm_unavailable()
    result, calls = await _run_live_case(
        case_id="eligible-grounded-group-window",
        evidence_text=(
            "The current participant directly asks the character for her view "
            "on the next step, and the request is still recent. The character "
            "has a clear bounded reason to answer this group scene."
        ),
        engagement_guidelines=[
            "The current participant directly addresses the character.",
            "A concise grounded scene intervention is appropriate.",
            "Use the current group message as the response basis.",
        ],
        artifact_path=_ELIGIBLE_ARTIFACT,
    )

    assert result["self_cognition_response"]["decision"] == (
        "propose_visible_reply"
    )
    assert result["intention"]["route"] == "speech"
    assert result["self_cognition_response"]["evidence_handles"]
    assert calls
    assert _ELIGIBLE_ARTIFACT.exists()


async def _run_live_case(
    *,
    case_id: str,
    evidence_text: str,
    engagement_guidelines: list[str],
    artifact_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one live action-planning case and save raw inspection evidence."""

    services = build_cognition_core_services()
    capturing_llm = _CapturingLLM(services.llm)
    live_services = replace(services, llm=capturing_llm)
    evidence = [{
        "evidence_handle": "e1",
        "evidence_ref": {
            "source_kind": "episode",
            "source_id": case_id,
            "occurred_at": "2026-08-11T00:00:00+00:00",
            "semantic_summary": evidence_text,
        },
        "semantic_text": evidence_text,
        "visible_to": ["q:event_agency"],
    }]
    primary_bid = {
        "branch_id": "ordinary_response",
        "goal_ref": {"scope": "character", "kind": "goal", "entity_id": "g1"},
        "intention": "respond to the current group scene",
        "desired_outcome": "preserve a grounded group interaction",
        "concrete_detail": "use only the current group evidence",
        "reason": "The current evidence gives the character a bounded motive.",
        "private_monologue": "I should judge the current scene carefully.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the scene remains coherent"],
        "confidence": "high",
    }
    result = await plan_actions(
        primary_bid=primary_bid,
        supporting_bids=[],
        episode={
            "episode_id": case_id,
            "trigger_source": "self_cognition",
            "output_mode": "think_only",
            "target_scope": {
                "channel_type": "group",
                "current_global_user_id": "",
                "current_platform_user_id": "",
            },
        },
        evidence=evidence,
        available_actions=[],
        available_resolvers=[],
        resolver_context="resolver_status=idle",
        group_engagement_action_context={
            "engagement_guidelines": engagement_guidelines,
            "confidence": "high",
        },
        scene_context={"participant_bindings": []},
        services=live_services,
    )
    parsed_calls = []
    for call in capturing_llm.calls:
        raw_output = str(call["raw_model_output"])
        parsed_calls.append({
            **call,
            "deterministic_parsed_output": parse_llm_json_output(
                raw_output,
                deterministic_only=True,
            ),
        })
    artifact = {
        "manual_review_required": True,
        "case_id": case_id,
        "input": {
            "evidence": evidence,
            "group_engagement_action_context": {
                "engagement_guidelines": engagement_guidelines,
                "confidence": "high",
            },
        },
        "raw_model_calls": parsed_calls,
        "parsed_result": result,
        "observed_response_decision": result.get(
            "self_cognition_response",
            {},
        ).get("decision", ""),
        "observed_route": result["intention"]["route"],
    }
    _write_artifact(artifact_path, artifact)
    return result, parsed_calls


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(
            f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}"
        )
    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


def _write_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Write one raw live case artifact for manual inspection."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
