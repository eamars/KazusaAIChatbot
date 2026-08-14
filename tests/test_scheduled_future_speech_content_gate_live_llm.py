"""Individually executed live cases for the scheduled speech content gate."""

from __future__ import annotations

from collections.abc import Sequence
import logging

import httpx
import pytest

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.config import DIALOG_GENERATOR_LLM_BASE_URL
from kazusa_ai_chatbot.nodes import dialog_agent
from kazusa_ai_chatbot.self_cognition import worker
from tests.llm_trace import write_llm_trace


logger = logging.getLogger(__name__)
pytestmark = pytest.mark.live_llm


_AUTHORITY_SUMMARY = "在约定时间开始补偿考核。"
_AUTHORITY_DETAIL_REF = {
    "semantic_summary": "当前对话明确约定在该时间开始补偿考核。",
    "provenance_role": "current_event",
}


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured dialog LLM endpoint cannot serve the case."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{DIALOG_GENERATOR_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError:
        pytest.skip(
            f"LLM endpoint is unavailable: {DIALOG_GENERATOR_LLM_BASE_URL}"
        )

    if response.status_code >= 500:
        pytest.skip(
            "LLM endpoint returned server error "
            f"{response.status_code}: {DIALOG_GENERATOR_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured live LLM endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _gate_disposition(verdict: dict[str, str] | None) -> dict[str, object]:
    """Compute the deterministic gate disposition for one verdict."""

    accepted, gate_codes = worker.evaluate_scheduled_content_gate(
        authority_missing=False,
        authority_invalid=False,
        trigger_identity_ok=True,
        due_reached=True,
        candidate_present=True,
        verdict=verdict,
    )
    return {
        "accepted": accepted,
        "disposition": (
            "accepted" if accepted else "suppressed"
        ),
        "gate_codes": gate_codes,
    }


def _message_rows(messages: object) -> list[dict[str, str]]:
    """Project captured evaluator messages into durable trace rows."""

    if not isinstance(messages, Sequence) or isinstance(messages, str):
        return []
    rows: list[dict[str, str]] = []
    for message in messages:
        content = getattr(message, "content", "")
        role = getattr(message, "type", message.__class__.__name__)
        rows.append({
            "role": str(role),
            "content": content if isinstance(content, str) else str(content),
        })
    return rows


def _call_config_record(config: object) -> dict[str, object]:
    """Return redacted evaluator configuration for live evidence."""

    fields = (
        "stage_name",
        "route_name",
        "base_url",
        "model",
        "temperature",
        "top_p",
        "top_k",
        "max_completion_tokens",
        "presence_penalty",
        "timeout_seconds",
    )
    record = {
        field: getattr(config, field)
        for field in fields
        if hasattr(config, field)
    }
    thinking = getattr(config, "thinking", None)
    if thinking is not None:
        record["thinking_enabled"] = getattr(
            thinking,
            "enabled",
            None,
        )
    return record


async def _run_live_gate_case(
    *,
    case_id: str,
    candidate_text: str,
    expected_disposition: str,
) -> None:
    """Run one live scheduled-gate case and retain its debug artifact."""

    trace_id = llm_tracing.build_trace_id()
    trace_calls: list[dict[str, object]] = []
    original_trace_step = llm_tracing.record_llm_trace_step

    async def capture_trace_step(**kwargs: object) -> object:
        """Capture raw evaluator evidence before delegating to trace storage."""

        trace_calls.append({
            "trace_id": str(kwargs.get("trace_id", "")),
            "stage_name": str(kwargs.get("stage_name", "")),
            "route_name": str(kwargs.get("route_name", "")),
            "model_name": str(kwargs.get("model_name", "")),
            "sequence": kwargs.get("sequence", 0),
            "attempt_index": kwargs.get("attempt_index", 0),
            "status": str(kwargs.get("status", "")),
            "parse_status": str(kwargs.get("parse_status", "")),
            "messages": _message_rows(kwargs.get("messages")),
            "raw_model_output": str(kwargs.get("response_text", "")),
            "parsed_output": kwargs.get("parsed_output", {}),
            "call_config": _call_config_record(kwargs.get("call_config")),
        })
        return await original_trace_step(**kwargs)

    llm_tracing.record_llm_trace_step = capture_trace_step  # type: ignore[assignment]
    try:
        result = await dialog_agent.evaluate_scheduled_future_speech_content(
            candidate_text=candidate_text,
            semantic_objective="在约定时间开始补偿考核。",
            authorized_content_summary=_AUTHORITY_SUMMARY,
            authorized_detail_refs=[dict(_AUTHORITY_DETAIL_REF)],
            audience_kind="group",
            local_due_datetime="2026-08-14 22:00",
            due_identity_facts={
                "due_reached": True,
                "run_due_matches_authority": True,
            },
            llm_trace_id=trace_id,
        )
    finally:
        llm_tracing.record_llm_trace_step = original_trace_step
    verdict = result.get("verdict")
    disposition = _gate_disposition(verdict)
    trace_payload = {
        "case_id": case_id,
        "candidate_text": candidate_text,
        "authority_summary": _AUTHORITY_SUMMARY,
        "authorized_detail_refs": [dict(_AUTHORITY_DETAIL_REF)],
        "evaluator_status": result.get("status"),
        "evaluator_attempt_count": result.get("attempt_count"),
        "semantic_verdict": verdict,
        "deterministic_gate": disposition,
        "protected_trace_ids": {
            "evaluator_llm_trace_id": trace_id,
            "source_llm_trace_id": "",
        },
        "trace_evidence": {
            "calls": trace_calls,
            "configuration_source": (
                "dialog_agent._scheduled_speech_evaluator_llm_config"
            ),
            "api_key_recorded": False,
        },
        "consolidation_admission": {
            "disposition": disposition["disposition"],
            "gate_codes": disposition["gate_codes"],
            "dispatch_status": (
                "sent"
                if disposition["accepted"]
                else "scheduled_content_suppressed"
            ),
        },
        "judgment": (
            "manual inspection required: candidate grounding, unsupported "
            "detail, and trace evidence must be judged against the authority"
        ),
    }
    trace_path = write_llm_trace(
        "test_scheduled_future_speech_content_gate_live_llm",
        case_id,
        trace_payload,
    )
    logger.info(
        "scheduled gate live case %s: status=%s verdict=%s disposition=%s "
        "trace=%s",
        case_id,
        result.get("status"),
        verdict,
        disposition,
        trace_path,
    )

    assert result["status"] in {"evaluated", "unavailable"}
    assert isinstance(result["attempt_count"], int)
    if result["status"] == "evaluated":
        from kazusa_ai_chatbot.cognition_core_v2.contracts import (
            validate_scheduled_speech_semantic_verdict,
        )

        validate_scheduled_speech_semantic_verdict(verdict)
    assert disposition["disposition"] == expected_disposition


async def test_current_authority_detail_is_accepted(
    ensure_live_llm,
) -> None:
    """A candidate grounded in the exact current objective is accepted."""

    await _run_live_gate_case(
        case_id="current_authority_detail_is_accepted",
        candidate_text=(
            "十点整！时间到啦！说好的补偿考核，现在正式开始，"
            "我准备好了。"
        ),
        expected_disposition="accepted",
    )


async def test_incident_unsupported_detail_is_suppressed(
    ensure_live_llm,
) -> None:
    """The incident-style unsupported toilet-stall detail is suppressed."""

    await _run_live_gate_case(
        case_id="incident_unsupported_detail_is_suppressed",
        candidate_text=(
            "十点整！时间到啦！说好的加倍补偿，现在正式开始。"
            "先检查厕所隔间是不是已经准备好了。"
        ),
        expected_disposition="suppressed",
    )


async def test_historical_only_grounding_is_suppressed(
    ensure_live_llm,
) -> None:
    """Historical-only grounding cannot authorize current speech."""

    await _run_live_gate_case(
        case_id="historical_only_grounding_is_suppressed",
        candidate_text=(
            "按照之前说好的历史约定，加倍补偿今天生效。"
        ),
        expected_disposition="suppressed",
    )
