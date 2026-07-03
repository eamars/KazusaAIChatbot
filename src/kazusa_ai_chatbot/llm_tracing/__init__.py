"""Protected LLM trace recording helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from typing import Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage

from kazusa_ai_chatbot.config import DEBUG_LOG_TTL_DAYS, LLM_TRACE_CAPTURE_MODE
from kazusa_ai_chatbot.db import llm_tracing as db_llm_tracing
from kazusa_ai_chatbot.logging_retention import expiry_from_storage_iso
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

logger = logging.getLogger(__name__)

LLM_TRACE_WRITE_TIMEOUT_SECONDS = 0.25
TraceWriteStatus = Literal["recorded", "skipped", "failed"]


class LLMTraceWriteResult(TypedDict):
    """Best-effort trace write result."""

    accepted: bool
    trace_id: str
    status: TraceWriteStatus
    reason: str


def build_trace_id() -> str:
    """Return a stable-format trace id for one dialog-producing turn."""

    trace_id = f"llmtrace_{uuid4().hex}"
    return trace_id


def _write_result(
    *,
    accepted: bool,
    trace_id: str,
    status: TraceWriteStatus,
    reason: str,
) -> LLMTraceWriteResult:
    """Build a trace write result."""

    result = LLMTraceWriteResult(
        accepted=accepted,
        trace_id=trace_id,
        status=status,
        reason=reason,
    )
    return result


def _capture_enabled() -> bool:
    """Return whether trace rows should be written."""

    return LLM_TRACE_CAPTURE_MODE != "off"


def _full_capture_enabled() -> bool:
    """Return whether raw prompt/output payloads should be stored."""

    return LLM_TRACE_CAPTURE_MODE == "full"


def _message_role(message: BaseMessage) -> str:
    """Return a stable role label from a LangChain message."""

    role = getattr(message, "type", "")
    if isinstance(role, str) and role.strip():
        return role.strip()
    return_value = message.__class__.__name__
    return return_value


def _message_content(message: BaseMessage) -> str:
    """Return text content for a LangChain message."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return_value = str(content)
    return return_value


def _messages_to_records(
    messages: Sequence[BaseMessage],
) -> list[dict[str, str]]:
    """Project prompt messages into serializable role/content rows."""

    records = [
        {
            "role": _message_role(message),
            "content": _message_content(message),
        }
        for message in messages
    ]
    return records


def _combined_prompt_text(messages: Sequence[BaseMessage]) -> str:
    """Return deterministic prompt text for hashes and character counts."""

    records = _messages_to_records(messages)
    text = json.dumps(records, ensure_ascii=False, sort_keys=True)
    return text


def _sha256_text(value: str) -> str:
    """Return a SHA-256 hex digest for text."""

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return digest


async def ensure_llm_trace_run(
    *,
    trace_id: str,
    platform: str,
    platform_channel_id: str,
    channel_type: str,
    platform_message_id: str,
    global_user_id: str,
    started_at: str | None = None,
) -> LLMTraceWriteResult:
    """Ensure a trace-run row exists for one live turn."""

    if not _capture_enabled():
        return _write_result(
            accepted=False,
            trace_id=trace_id,
            status="skipped",
            reason="capture mode off",
        )
    run_started_at = started_at or storage_utc_now_iso()
    document = {
        "trace_id": trace_id,
        "status": "running",
        "platform": platform,
        "platform_channel_id": platform_channel_id,
        "channel_type": channel_type,
        "platform_message_id": platform_message_id,
        "global_user_id": global_user_id,
        "started_at": run_started_at,
        "completed_at": "",
        "final_dialog_count": 0,
        "delivery_tracking_id": "",
        "created_at": storage_utc_now_iso(),
        "expires_at": expiry_from_storage_iso(
            run_started_at,
            ttl_days=DEBUG_LOG_TTL_DAYS,
        ),
    }
    try:
        await asyncio.wait_for(
            db_llm_tracing.upsert_trace_run(document),
            timeout=LLM_TRACE_WRITE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(f"LLM trace run write failed: {exc.__class__.__name__}")
        return _write_result(
            accepted=False,
            trace_id=trace_id,
            status="failed",
            reason=exc.__class__.__name__,
        )
    return _write_result(
        accepted=True,
        trace_id=trace_id,
        status="recorded",
        reason="",
    )


async def record_llm_trace_step(
    *,
    trace_id: str,
    stage_name: str,
    route_name: str,
    model_name: str,
    messages: Sequence[BaseMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    duration_ms: int,
    output_state_fields: Sequence[str],
    sequence: int = 0,
) -> LLMTraceWriteResult:
    """Record one LLM stage prompt/response trace."""

    if not trace_id:
        return _write_result(
            accepted=False,
            trace_id="",
            status="skipped",
            reason="missing trace id",
        )
    if not _capture_enabled():
        return _write_result(
            accepted=False,
            trace_id=trace_id,
            status="skipped",
            reason="capture mode off",
        )

    created_at = storage_utc_now_iso()
    prompt_text = _combined_prompt_text(messages)
    full_capture = _full_capture_enabled()
    document = {
        "step_id": f"{trace_id}_{uuid4().hex}",
        "trace_id": trace_id,
        "sequence": sequence,
        "stage_name": stage_name,
        "route_name": route_name,
        "model_name": model_name,
        "status": status,
        "parse_status": parse_status,
        "prompt_chars": sum(len(_message_content(message)) for message in messages),
        "output_chars": len(response_text),
        "prompt_sha256": _sha256_text(prompt_text),
        "output_sha256": _sha256_text(response_text),
        "raw_messages": _messages_to_records(messages) if full_capture else [],
        "raw_response_text": response_text if full_capture else "",
        "parsed_output": parsed_output if full_capture else {},
        "output_state_fields": list(output_state_fields),
        "duration_ms": duration_ms,
        "created_at": created_at,
        "expires_at": expiry_from_storage_iso(
            created_at,
            ttl_days=DEBUG_LOG_TTL_DAYS,
        ),
    }
    try:
        await asyncio.wait_for(
            db_llm_tracing.insert_trace_step(document),
            timeout=LLM_TRACE_WRITE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(f"LLM trace step write failed: {exc.__class__.__name__}")
        return _write_result(
            accepted=False,
            trace_id=trace_id,
            status="failed",
            reason=exc.__class__.__name__,
        )
    return _write_result(
        accepted=True,
        trace_id=trace_id,
        status="recorded",
        reason="",
    )


async def finalize_llm_trace_run(
    *,
    trace_id: str,
    status: str,
    final_dialog_count: int,
    delivery_tracking_id: str,
) -> LLMTraceWriteResult:
    """Finalize mutable trace-run status metadata."""

    if not trace_id or not _capture_enabled():
        return _write_result(
            accepted=False,
            trace_id=trace_id,
            status="skipped",
            reason="capture mode off or missing trace id",
        )
    update_doc = {
        "status": status,
        "completed_at": storage_utc_now_iso(),
        "final_dialog_count": final_dialog_count,
        "delivery_tracking_id": delivery_tracking_id,
    }
    try:
        await asyncio.wait_for(
            db_llm_tracing.update_trace_run(
                trace_id=trace_id,
                update_doc=update_doc,
            ),
            timeout=LLM_TRACE_WRITE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning(f"LLM trace run finalize failed: {exc.__class__.__name__}")
        return _write_result(
            accepted=False,
            trace_id=trace_id,
            status="failed",
            reason=exc.__class__.__name__,
        )
    return _write_result(
        accepted=True,
        trace_id=trace_id,
        status="recorded",
        reason="",
    )
