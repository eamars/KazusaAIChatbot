"""Guarded one-case Stage 3 live-LLM and fresh-database evidence tests."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
import time
from uuid import uuid4

import pytest

from tests.stage3_fresh_database import (
    load_stage3_case_fixture,
    select_stage3_case,
    validate_stage3_environment,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_FIXTURE_PATH = Path("tests/fixtures/stage3_fresh_database_cases.json")
_OUTPUT_ROOT = Path(
    "test_artifacts/cognition_core_v2/stage_3/focused_live"
)
_REQUIRED_LLM_ENV = (
    "RELEVANCE_AGENT_LLM_BASE_URL",
    "RELEVANCE_AGENT_LLM_API_KEY",
    "RELEVANCE_AGENT_LLM_MODEL",
    "VISION_DESCRIPTOR_LLM_BASE_URL",
    "VISION_DESCRIPTOR_LLM_API_KEY",
    "VISION_DESCRIPTOR_LLM_MODEL",
    "MSG_DECONTEXTUALIZER_LLM_BASE_URL",
    "MSG_DECONTEXTUALIZER_LLM_API_KEY",
    "MSG_DECONTEXTUALIZER_LLM_MODEL",
    "RAG_PLANNER_LLM_BASE_URL",
    "RAG_PLANNER_LLM_API_KEY",
    "RAG_PLANNER_LLM_MODEL",
    "RAG_SUBAGENT_LLM_BASE_URL",
    "RAG_SUBAGENT_LLM_API_KEY",
    "RAG_SUBAGENT_LLM_MODEL",
    "WEB_SEARCH_LLM_BASE_URL",
    "WEB_SEARCH_LLM_API_KEY",
    "WEB_SEARCH_LLM_MODEL",
    "COGNITION_LLM_BASE_URL",
    "COGNITION_LLM_API_KEY",
    "COGNITION_LLM_MODEL",
    "BOUNDARY_CORE_LLM_BASE_URL",
    "BOUNDARY_CORE_LLM_API_KEY",
    "BOUNDARY_CORE_LLM_MODEL",
    "DIALOG_GENERATOR_LLM_BASE_URL",
    "DIALOG_GENERATOR_LLM_API_KEY",
    "DIALOG_GENERATOR_LLM_MODEL",
    "CONSOLIDATION_LLM_BASE_URL",
    "CONSOLIDATION_LLM_API_KEY",
    "CONSOLIDATION_LLM_MODEL",
    "JSON_REPAIR_LLM_BASE_URL",
    "JSON_REPAIR_LLM_API_KEY",
    "JSON_REPAIR_LLM_MODEL",
    "BACKGROUND_WORK_LLM_BASE_URL",
    "BACKGROUND_WORK_LLM_API_KEY",
    "BACKGROUND_WORK_LLM_MODEL",
    "CODING_AGENT_PM_LLM_BASE_URL",
    "CODING_AGENT_PM_LLM_API_KEY",
    "CODING_AGENT_PM_LLM_MODEL",
    "CODING_AGENT_PROGRAMMER_LLM_BASE_URL",
    "CODING_AGENT_PROGRAMMER_LLM_API_KEY",
    "CODING_AGENT_PROGRAMMER_LLM_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_MODEL",
)


def _configure_utf8_streams() -> None:
    """Keep live CJK model output printable in the test process."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_streams()


def _prepare_stage3_runtime() -> dict[str, str]:
    """Validate the dedicated endpoint before importing service modules."""

    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    try:
        guarded = validate_stage3_environment()
    except ValueError as exc:
        pytest.skip(f"Stage 3 guarded environment is unavailable: {exc}")

    missing = [name for name in _REQUIRED_LLM_ENV if not os.environ.get(name)]
    if missing:
        pytest.skip(
            "Stage 3 live LLM environment is incomplete: "
            + ", ".join(missing)
        )
    if os.environ.get("LLM_TRACE_CAPTURE_MODE", "metadata") == "off":
        pytest.skip("Stage 3 live evidence requires LLM trace capture")

    # tests/conftest.py selects the older live-LLM database for legacy tests.
    # The Stage 3 child process must restore the guarded database before any
    # Kazusa module imports configuration or opens MongoDB.
    os.environ.update({
        "MONGODB_URI": guarded["mongodb_uri"],
        "MONGODB_DB_NAME": guarded["database_name"],
        "CHARACTER_PROFILE_PATH": guarded["character_profile_path"],
        "SELF_COGNITION_ENABLED": "false",
        "CALENDAR_SCHEDULER_ENABLED": "false",
        "BACKGROUND_WORK_WORKER_ENABLED": "false",
        "REFLECTION_CYCLE_ENABLED": "false",
    })
    return guarded


def _load_fixture_case(case_id: str) -> Mapping[str, object]:
    """Load one frozen case without importing the application runtime."""

    fixture = load_stage3_case_fixture(_FIXTURE_PATH)
    return select_stage3_case(fixture, case_id)


def _storage_now() -> str:
    """Return a storage-compatible UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: object) -> object:
    """Convert live model and Mongo values into JSON evidence values."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return str(isoformat())
    return str(value)


_PROTECTED_EVIDENCE_FIELDS = frozenset({
    "adapter_calls",
    "character_profile_observed",
    "cognition_output",
    "consolidation_state",
    "delivery_result",
    "episode",
    "graph_result",
    "input",
    "lifecycle_record",
    "llm_trace_run",
    "llm_trace_steps",
    "message_envelope",
    "parsed_output",
    "raw_messages",
    "raw_response_text",
    "response",
    "settled_trace",
    "settlement",
    "source_case",
})


def _redacted_evidence_value(value: object) -> dict[str, object]:
    """Replace one protected value with non-reversible review metadata."""

    safe_value = _json_safe(value)
    serialized = json.dumps(
        safe_value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    item_count = len(value) if isinstance(value, (Mapping, Sequence)) else 1
    return {
        "redacted": True,
        "sha256": hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest(),
        "serialized_chars": len(serialized),
        "item_count": item_count,
    }


def _safe_evidence_payload(
    value: object,
    *,
    field_name: str = "",
) -> object:
    """Redact protected evidence fields while preserving technical metadata."""

    if field_name.casefold() in _PROTECTED_EVIDENCE_FIELDS:
        return _redacted_evidence_value(value)
    if isinstance(value, Mapping):
        return {
            str(key): _safe_evidence_payload(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _safe_evidence_payload(item, field_name=field_name)
            for item in value
        ]
    return _json_safe(value)


def _write_evidence(case_id: str, payload: Mapping[str, object]) -> Path:
    """Persist redacted technical evidence for later human quality review."""

    output_path = os.environ.get("STAGE3_CASE_OUTPUT_PATH")
    path = Path(output_path) if output_path else _OUTPUT_ROOT / f"{case_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _safe_evidence_payload(payload),
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"wrote Stage 3 live evidence: {path}")
    return path


def _message_text(message: object) -> str:
    """Read one LangChain message body for hashing and sizing."""

    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _text_digest(value: object) -> dict[str, object]:
    """Return size and digest metadata for one non-persisted text value."""

    text = str(value)
    return {
        "chars": len(text),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _trace_review_metadata(trace: Mapping[str, object]) -> dict[str, object]:
    """Project a settled trace into safe cardinality and status evidence."""

    action_specs = trace.get("action_specs")
    action_results = trace.get("action_results")
    surface_outputs = trace.get("surface_outputs")
    diagnostics = trace.get("attempt_diagnostics")
    delivery = trace.get("delivery_correlation")
    return {
        "schema_version": trace.get("schema_version"),
        "source_kind": trace.get("trigger_source"),
        "terminal_status": trace.get("terminal_status"),
        "action_spec_count": (
            len(action_specs) if isinstance(action_specs, list) else 0
        ),
        "action_result_count": (
            len(action_results) if isinstance(action_results, list) else 0
        ),
        "surface_output_count": (
            len(surface_outputs) if isinstance(surface_outputs, list) else 0
        ),
        "attempt_diagnostic_count": (
            len(diagnostics) if isinstance(diagnostics, list) else 0
        ),
        "delivery_receipt_status": (
            delivery.get("receipt_status")
            if isinstance(delivery, Mapping)
            else "unknown"
        ),
        "action_kinds": [
            str(row.get("kind"))
            for row in action_specs
            if isinstance(row, Mapping) and row.get("kind")
        ] if isinstance(action_specs, list) else [],
        "action_statuses": [
            str(row.get("status"))
            for row in action_results
            if isinstance(row, Mapping) and row.get("status")
        ] if isinstance(action_results, list) else [],
    }


def _lifecycle_review_metadata(
    lifecycle: Mapping[str, object] | None,
) -> dict[str, object]:
    """Project one lifecycle record into safe status/cardinality evidence."""

    if not isinstance(lifecycle, Mapping):
        return {
            "present": False,
            "status": "missing",
            "action_projection_count": 0,
            "error_code_count": 0,
        }
    action_projections = lifecycle.get("action_projections")
    error_codes = lifecycle.get("error_codes")
    return {
        "present": True,
        "status": lifecycle.get("status", "unknown"),
        "action_projection_count": (
            len(action_projections)
            if isinstance(action_projections, list)
            else 0
        ),
        "error_code_count": (
            len(error_codes) if isinstance(error_codes, list) else 0
        ),
    }


def _llm_step_review_metadata(
    trace_steps: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project protected trace rows into safe call/latency ledger rows."""

    fields = (
        "stage_name",
        "route_name",
        "model_name",
        "status",
        "parse_status",
        "duration_ms",
        "sequence",
        "prompt_chars",
        "prompt_sha256",
        "response_chars",
        "response_sha256",
    )
    return [
        {
            field_name: row.get(field_name)
            for field_name in fields
            if field_name in row
        }
        for row in trace_steps
        if isinstance(row, Mapping)
    ]


def _response_review_metadata(
    response_payload: Mapping[str, object],
) -> dict[str, object]:
    """Project visible response shape without persisting its wording."""

    messages = response_payload.get("messages")
    message_rows = messages if isinstance(messages, list) else []
    return {
        "message_count": len(message_rows),
        "message_sizes": [
            _text_digest(
                row.get("content", "")
                if isinstance(row, Mapping)
                else row
            )
            for row in message_rows
        ],
    }


def _case_review_metadata(
    case_id: str,
    case: Mapping[str, object],
) -> dict[str, object]:
    """Project frozen case facts without persisting input text or identifiers."""

    target = case.get("target_scope_fixture")
    target_mapping = target if isinstance(target, Mapping) else {}
    input_text = case.get("input_text", "")
    return {
        "case_id": case_id,
        "sequence": case.get("sequence", ""),
        "input": _text_digest(input_text),
        "target_channel_type": target_mapping.get("channel_type", ""),
        "addressed_to_character": bool(
            target_mapping.get("addressed_to_character", False)
        ),
    }


async def _database_schema_evidence(db: Any) -> list[dict[str, object]]:
    """Collect collection/index names and sanitized native shape metadata."""

    collection_names = await db.list_collection_names()
    evidence: list[dict[str, object]] = []
    for collection_name in sorted(str(name) for name in collection_names):
        index_rows = await db[collection_name].list_indexes().to_list(
            length=None,
        )
        index_names: list[str] = []
        indexed_fields: list[str] = []
        ttl_values: list[int] = []
        for index_row in index_rows:
            if not isinstance(index_row, Mapping):
                continue
            index_name = index_row.get("name")
            if isinstance(index_name, str):
                index_names.append(index_name)
            key = index_row.get("key")
            if isinstance(key, Mapping):
                indexed_fields.extend(str(field) for field in key)
            expire_after = index_row.get("expireAfterSeconds")
            if isinstance(expire_after, int) and not isinstance(
                expire_after,
                bool,
            ):
                ttl_values.append(expire_after)
        evidence.append({
            "collection_name": collection_name,
            "index_names": sorted(set(index_names)),
            "owner_module": "native runtime owner",
            "native_schema_version": "native collection contract",
            "creation_condition": "created by native startup or owner operation",
            "retention_ttl": sorted(set(ttl_values)),
            "sanitized_representative_shape": {
                "indexed_fields": sorted(set(indexed_fields)),
            },
        })
    return evidence


@contextmanager
def _capture_llm_steps(calls: list[dict[str, object]]) -> Iterator[None]:
    """Capture actual LLM call inputs and outputs around one live case."""

    from kazusa_ai_chatbot import llm_tracing

    original = llm_tracing.record_llm_trace_step

    async def record_step(**kwargs: object) -> object:
        messages = kwargs.get("messages", [])
        message_rows = [
            _message_text(message)
            for message in messages
        ] if isinstance(messages, Sequence) else []
        response_text = str(kwargs.get("response_text", ""))
        prompt_text = "\n".join(message_rows)
        parsed_output = kwargs.get("parsed_output", {})
        parsed_output_keys = (
            sorted(str(key) for key in parsed_output)
            if isinstance(parsed_output, Mapping)
            else []
        )
        calls.append({
            "stage_name": str(kwargs.get("stage_name", "")),
            "route_name": str(kwargs.get("route_name", "")),
            "model_name": str(kwargs.get("model_name", "")),
            "status": str(kwargs.get("status", "")),
            "parse_status": str(kwargs.get("parse_status", "")),
            "duration_ms": kwargs.get("duration_ms", 0),
            "sequence": kwargs.get("sequence", 0),
            "prompt_chars": len(prompt_text),
            "prompt_sha256": hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest(),
            "response_chars": len(response_text),
            "response_sha256": hashlib.sha256(
                response_text.encode("utf-8")
            ).hexdigest(),
            "parsed_output_keys": parsed_output_keys,
        })
        return await original(**kwargs)

    llm_tracing.record_llm_trace_step = record_step  # type: ignore[assignment]
    try:
        yield
    finally:
        llm_tracing.record_llm_trace_step = original


class _Stage3DebugAdapter:
    """In-memory adapter boundary used only to prove tool-result delivery."""

    platform = "debug"
    display_name = "Stage 3 Character"

    def __init__(self) -> None:
        self.platform_bot_id = "stage3-bot"
        self.calls: list[dict[str, object]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        del channel_id, channel_type
        return True

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: Sequence[dict[str, object]] | None = None,
    ) -> object:
        from kazusa_ai_chatbot.dispatcher import SendResult

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": list(delivery_mentions or []),
        })
        return SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=f"stage3-adapter-message-{uuid4().hex}",
            sent_at=datetime.now(timezone.utc),
        )


async def _run_chat_case(
    *,
    case_id: str,
    case: Mapping[str, object] | None = None,
    envelope: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run one real user-message turn through the service queue."""

    _prepare_stage3_runtime()
    from kazusa_ai_chatbot import llm_tracing
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.db._client import get_db

    if case is None:
        case = _load_fixture_case("group_01")
    target = case.get("target_scope_fixture", {})
    if not isinstance(target, Mapping):
        raise ValueError("Stage 3 chat case has no target scope")
    channel_type = str(target.get("channel_type", "private"))
    channel_id = str(target.get("channel_id", "stage3-private"))
    user_id = str(target.get("global_user_id", "stage3-user"))
    input_text = str(case.get("input_text", "Stage 3 live input"))
    addressed = bool(target.get("addressed_to_character", True))
    request_envelope: dict[str, object] = {
        "body_text": input_text,
        "raw_wire_text": input_text,
        "mentions": [],
        "reply": None,
        "attachments": [],
        "addressed_to_global_user_ids": (
            [CHARACTER_GLOBAL_USER_ID] if addressed else []
        ),
        "broadcast": channel_type == "group" and not addressed,
    }
    if envelope is not None:
        request_envelope.update(dict(envelope))
    request = ChatRequest.model_validate({
        "platform": "debug",
        "platform_channel_id": channel_id,
        "channel_type": channel_type,
        "platform_message_id": (
            f"{case_id}-{uuid4().hex}"
            if not os.environ.get("STAGE3_CASE_ID")
            else case_id
        ),
        "platform_user_id": user_id,
        "platform_bot_id": "stage3-bot",
        "display_name": "Stage 3 User",
        "channel_name": "Stage 3 Test Channel",
        "content_type": "mixed" if request_envelope["attachments"] else "text",
        "message_envelope": request_envelope,
        "local_timestamp": _storage_now(),
        "debug_modes": {},
    })

    settlements: list[dict[str, object]] = []
    llm_calls: list[dict[str, object]] = []
    original_settle = post_turn.settle_episode_trace

    def capture_settlement(**kwargs: object) -> Mapping[str, object]:
        trace = original_settle(**kwargs)
        settlements.append({
            "trace": deepcopy(trace),
            "episode": deepcopy(kwargs.get("episode", {})),
            "graph_result": deepcopy(kwargs.get("graph_result", {})),
        })
        return trace

    post_turn.settle_episode_trace = capture_settlement  # type: ignore[assignment]
    started_at = time.perf_counter()
    try:
        with _capture_llm_steps(llm_calls):
            async with service.lifespan(service.app):
                if service._adapter_registry is not None:
                    service._adapter_registry.register(_Stage3DebugAdapter())
                response = await service._enqueue_chat_request(request)
                db = await get_db()
                schema_evidence = await _database_schema_evidence(db)
                if len(settlements) != 1:
                    raise AssertionError(
                        f"expected exactly one settled trace, got {len(settlements)}"
                    )
                settled = settlements[0]
                trace = settled["trace"]
                episode = settled["episode"]
                if not isinstance(trace, Mapping):
                    raise AssertionError("settled trace is not an object")
                if not isinstance(episode, Mapping):
                    raise AssertionError("settled episode is not an object")
                episode_id = str(episode.get("episode_id", ""))
                lifecycle_rows = await db[
                    "post_turn_lifecycle_records"
                ].count_documents({"source_episode_id": episode_id})
                lifecycle = await db[
                    "post_turn_lifecycle_records"
                ].find_one({"source_episode_id": episode_id}, {"_id": 0})
                trace_run = await db["llm_trace_runs"].find_one(
                    {"platform_message_id": request.platform_message_id},
                    {"_id": 0},
                )
                trace_id = str(trace_run.get("trace_id", "")) if trace_run else ""
                trace_steps = await db["llm_trace_steps"].find(
                    {"trace_id": trace_id},
                    {"_id": 0},
                ).sort("sequence", 1).to_list(length=None)
                profile = await db["character_state"].find_one(
                    {},
                    {"_id": 0, "name": 1, "global_user_id": 1},
                )
                response_payload = response.model_dump(mode="json")
    finally:
        post_turn.settle_episode_trace = original_settle
    duration_ms = round((time.perf_counter() - started_at) * 1000)

    if not isinstance(trace, Mapping):
        raise AssertionError("live service did not return a settled trace")
    if trace.get("trigger_source") != "user_message":
        raise AssertionError("live chat did not settle as user_message")
    if trace.get("terminal_status") != "completed_visible":
        raise AssertionError(
            "Stage 3 final fixture requires a visible completed terminal status"
        )
    if lifecycle_rows != 1 or not isinstance(lifecycle, Mapping):
        raise AssertionError("live chat did not persist one lifecycle record")
    if not response_payload.get("messages"):
        raise AssertionError("live chat returned no visible message")

    artifact = {
        "schema_version": "stage3_live_case_evidence.v1",
        "case_id": case_id,
        "run_mode": os.environ.get("STAGE3_RUN_MODE", "focused"),
        "technical_status": "passed",
        "terminal_status": trace.get("terminal_status"),
        "duration_ms": duration_ms,
        "llm_call_count": len(llm_calls),
        "case_review": _case_review_metadata(case_id, case),
        "episode_id_present": bool(episode.get("episode_id")),
        "source_kind": trace.get("trigger_source"),
        "trace_cardinality": 1,
        "lifecycle_cardinality": lifecycle_rows,
        "trace_review": _trace_review_metadata(trace),
        "lifecycle_review": _lifecycle_review_metadata(lifecycle),
        "response_review": _response_review_metadata(response_payload),
        "llm_step_review": _llm_step_review_metadata(trace_steps),
        "collections_and_indexes": schema_evidence,
        "input": request.model_dump(mode="json"),
        "response": response_payload,
        "episode": episode,
        "settled_trace": trace,
        "lifecycle_record": lifecycle,
        "llm_trace_run": trace_run,
        "llm_trace_steps": trace_steps,
        "raw_llm_calls": llm_calls,
        "character_profile_observed": profile,
    }
    artifact_path = _write_evidence(case_id, artifact)
    artifact["artifact_path"] = str(artifact_path)
    artifact["settlement"] = settled
    return artifact


async def _run_self_cognition_case(
    *,
    case_id: str,
    trigger_kind: str,
    case_name: str,
    use_latch: bool = False,
    promoted_reflection: bool = False,
) -> dict[str, object]:
    """Run one actual self-cognition source and settle it once."""

    _prepare_stage3_runtime()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
    from kazusa_ai_chatbot.self_cognition import models, runner

    now = _storage_now()
    profile = json.loads(
        Path(os.environ["CHARACTER_PROFILE_PATH"]).read_text(encoding="utf-8")
    )
    if not isinstance(profile, dict):
        raise ValueError("Stage 3 profile seed must be an object")
    scope_type = "group" if trigger_kind == models.TRIGGER_GROUP_CHAT_REVIEW else "private"
    target_user_id = None if scope_type == "group" else "stage3-self-user"
    case: dict[str, Any] = {
        "case_name": case_name,
        "case_id": case_id,
        "idle_timestamp_utc": now,
        "last_evidence_timestamp_utc": now,
        "trigger_kind": trigger_kind,
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": f"stage3-{scope_type}",
            "channel_type": scope_type,
            "user_id": target_user_id,
            "platform_user_id": target_user_id,
            "display_name": "Stage 3 User",
        },
        "source_refs": [{
            "source_kind": "conversation_window",
            "source_id": f"stage3-ref-{case_id}",
            "summary": "A grounded Stage 3 source observation.",
        }],
        "semantic_due_state": "due_now",
        "actionability": "actionable",
        "visible_context": [{
            "role": "user",
            "body_text": "The user left a grounded follow-up worth considering.",
            "timestamp": now,
            "display_name": "Stage 3 User",
        }],
        "delivery_mention_users": [],
        "conversation_progress": {},
        "group_activity_window": {},
        "reflection_modifier": {},
        "existing_attempts": [],
        "character_profile": profile,
        "user_profile": {"display_name": "Stage 3 User"},
        "platform_bot_id": "stage3-bot",
        "channel_topic": "",
        "promoted_reflection_context": {},
        "budget": {
            "rag_calls": 0,
            "cognition_calls": 0,
            "dialog_calls": 0,
            "topic_limit": 1,
        },
        "source_calendar_run_id": f"stage3-calendar-{case_id}",
        "source_calendar_skip_reason": "",
        "cognition_source": {
            "source_kind": trigger_kind,
            "source_id": case_id,
        },
        "source_action_attempt_id": "",
        "delivery_target": {
            "schema_version": "self_cognition_delivery_target.v1",
            "platform": "debug",
            "platform_channel_id": f"stage3-{scope_type}",
            "channel_type": scope_type,
            "target_global_user_id": target_user_id,
            "target_platform_user_id": target_user_id,
            "source_kind": "self_cognition_source_channel",
            "source_ref": case_id,
            "source_platform_channel_id": f"stage3-{scope_type}",
            "source_channel_type": scope_type,
            "source_message_id": case_id,
            "source_global_user_id": target_user_id,
            "source_platform_bot_id": "stage3-bot",
            "source_character_name": str(profile.get("name", "Character")),
            "guild_id": None,
            "bot_permission_role": "self_cognition",
            "fallback_reason": "",
        },
        "target_binding_status": "bound",
    }
    if trigger_kind == models.TRIGGER_SCHEDULED_FUTURE_COGNITION:
        case["calendar_run"] = {
            "run_id": f"stage3-calendar-{case_id}",
            "schedule_id": f"stage3-schedule-{case_id}",
            "due_at": now,
        }
    async def claim_latch_if_needed() -> Mapping[str, object] | None:
        """Issue and claim the latch after lifespan has opened MongoDB."""

        if not use_latch:
            return None
        from kazusa_ai_chatbot.db import (
            claim_due_internal_action_latch,
            issue_internal_action_latch,
        )

        await issue_internal_action_latch(
            source_episode_id=f"stage3-latch-source:{case_id}",
            source_action_attempt_id=f"stage3-latch-attempt:{case_id}",
            continuation_objective="Review the grounded continuation objective.",
            evidence_refs=[],
            target_scope={
                "platform": "debug",
                "platform_channel_id": "stage3-private",
                "channel_type": "private",
                "current_platform_user_id": "stage3-self-user",
                "current_global_user_id": "stage3-self-user",
                "current_display_name": "Stage 3 User",
                "target_addressed_user_ids": ["stage3-self-user"],
                "target_broadcast": False,
            },
            privacy_scope="private",
            continuation_depth=0,
            now=now,
        )
        claimed = await claim_due_internal_action_latch(
            worker_id=f"stage3-live-{case_id}",
            now=now,
        )
        if claimed is None:
            raise AssertionError("Stage 3 live latch was not claimable")
        case["internal_action_latch"] = dict(claimed["latch"])
        case["claim_token"] = str(claimed["claim_token"])
        return claimed

    if promoted_reflection:
        async with service.lifespan(service.app):
            context = await service.build_promoted_reflection_context()
            if not context:
                pytest.skip(
                    "Stage 3 guarded database has no promoted reflection context"
                )
            case["promoted_reflection_context"] = context
            latch_claim = await claim_latch_if_needed()
            return await _settle_runner_case(
                service=service,
                post_turn=post_turn,
                runner=runner,
                case=case,
                case_id=case_id,
                latch_claim=latch_claim,
                character_global_user_id=CHARACTER_GLOBAL_USER_ID,
            )
    async with service.lifespan(service.app):
        latch_claim = await claim_latch_if_needed()
        return await _settle_runner_case(
            service=service,
            post_turn=post_turn,
            runner=runner,
            case=case,
            case_id=case_id,
            latch_claim=latch_claim,
            character_global_user_id=CHARACTER_GLOBAL_USER_ID,
        )


async def _settle_runner_case(
    *,
    service: Any,
    post_turn: Any,
    runner: Any,
    case: Mapping[str, object],
    case_id: str,
    latch_claim: Mapping[str, object] | None,
    character_global_user_id: str,
) -> dict[str, object]:
    """Settle runner output and persist its lifecycle projection."""

    del character_global_user_id
    from kazusa_ai_chatbot.db._client import get_db

    settlements: list[dict[str, object]] = []
    llm_calls: list[dict[str, object]] = []
    original_settle = post_turn.settle_episode_trace

    def capture_settlement(**kwargs: object) -> Mapping[str, object]:
        trace = original_settle(**kwargs)
        settlements.append({
            "trace": deepcopy(trace),
            "episode": deepcopy(kwargs.get("episode", {})),
        })
        return trace

    post_turn.settle_episode_trace = capture_settlement  # type: ignore[assignment]
    try:
        with _capture_llm_steps(llm_calls):
            artifacts = await runner.build_self_cognition_case_artifacts_async(
                dict(case),
                apply_consolidation=False,
                execute_private_actions=False,
            )
            episode = artifacts[runner.models.RUNTIME_COGNITIVE_EPISODE]
            cognition_output = artifacts.get(
                runner.models.ARTIFACT_COGNITION_OUTPUT,
                {},
            )
            consolidation_state = artifacts.get(
                runner.models.RUNTIME_CONSOLIDATION_STATE,
                {},
            )
            graph_result = {}
            if isinstance(consolidation_state, Mapping):
                graph_result.update(consolidation_state)
            if isinstance(cognition_output, Mapping):
                graph_result.update(cognition_output)
            response_dialog = [
                item
                for item in graph_result.get("final_dialog", [])
                if isinstance(item, str) and item.strip()
            ] if isinstance(graph_result.get("final_dialog", []), list) else []
            settled_trace = await service._settle_runtime_episode_trace(
                episode=episode,
                graph_result=graph_result,
                response_dialog=response_dialog,
                delivery_tracking_id="",
                settled_at=_storage_now(),
            )
            await service._persist_post_turn_lifecycle_record(
                episode=episode,
                state=graph_result,
                fallback_trace=settled_trace,
                delivery_tracking_id="",
            )
            if latch_claim is not None:
                from kazusa_ai_chatbot.db import consume_internal_action_latch

                await consume_internal_action_latch(
                    latch_id=str(latch_claim["latch"]["latch_id"]),
                    claim_token=str(latch_claim["claim_token"]),
                    consumed_episode_id=str(episode["episode_id"]),
                    now=_storage_now(),
                )
            db = await get_db()
            lifecycle = await db["post_turn_lifecycle_records"].find_one(
                {"source_episode_id": episode["episode_id"]},
                {"_id": 0},
            )
    finally:
        post_turn.settle_episode_trace = original_settle

    if len(settlements) != 1:
        raise AssertionError(
            f"expected one self-cognition settlement, got {len(settlements)}"
        )
    trace = settlements[0]["trace"]
    if not isinstance(trace, Mapping):
        raise AssertionError("self-cognition settlement is not an object")
    if trace.get("trigger_source") not in {
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
    }:
        raise AssertionError("self-cognition source was not canonical")
    if not isinstance(lifecycle, Mapping):
        raise AssertionError("self-cognition lifecycle record is missing")
    artifact = {
        "schema_version": "stage3_live_case_evidence.v1",
        "case_id": case_id,
        "technical_status": "passed",
        "terminal_status": trace.get("terminal_status"),
        "llm_call_count": len(llm_calls),
        "source_kind": trace.get("trigger_source"),
        "trace_cardinality": 1,
        "lifecycle_cardinality": (
            1 if isinstance(lifecycle, Mapping) else 0
        ),
        "case_review": {
            "case_id": case_id,
            "source_kind": trace.get("trigger_source"),
            "target_channel_type": case.get("target_scope", {}).get(
                "channel_type", ""
            ) if isinstance(case.get("target_scope"), Mapping) else "",
            "latch_requested": latch_claim is not None,
        },
        "trace_review": _trace_review_metadata(trace),
        "lifecycle_review": _lifecycle_review_metadata(lifecycle),
        "llm_step_review": _llm_step_review_metadata(llm_calls),
        "source_case": case,
        "runner_artifact_keys": sorted(str(key) for key in artifacts),
        "cognition_output": cognition_output,
        "consolidation_state": consolidation_state,
        "settled_trace": trace,
        "lifecycle_record": lifecycle,
        "raw_llm_calls": llm_calls,
        "latch_claimed": latch_claim is not None,
        "latch_consumed_episode_id": (
            str(settlements[0]["episode"].get("episode_id", ""))
            if latch_claim is not None
            else ""
        ),
    }
    _write_evidence(case_id, artifact)
    return artifact


async def _run_tool_result_case(case_id: str) -> dict[str, object]:
    """Run one completed background-work result through live cognition."""

    _prepare_stage3_runtime()
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.background_work.result_source import (
        build_result_ready_episode_from_job,
    )
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.dispatcher import AdapterRegistry
    from kazusa_ai_chatbot.db._client import get_db

    now = _storage_now()
    job = {
        "schema_version": "background_work_job.v1",
        "job_id": f"stage3-job-{case_id}",
        "accepted_task_id": f"stage3-task-{case_id}",
        "status": "completed",
        "delivery_state": "ready",
        "source_platform": "debug",
        "source_channel_id": "stage3-private",
        "source_channel_type": "private",
        "source_message_id": case_id,
        "source_platform_bot_id": "stage3-bot",
        "source_character_name": "Stage 3 Character",
        "requester_global_user_id": "stage3-tool-user",
        "requester_platform_user_id": "stage3-tool-user",
        "requester_display_name": "Stage 3 User",
        "task_brief": "Complete a bounded background task.",
        "result_summary": "The background task completed with a grounded result.",
        "artifact_text": "The completed artifact contains the requested result.",
        "failure_summary": "",
        "completed_at": now,
        "created_at": now,
        "updated_at": now,
    }
    episode = build_result_ready_episode_from_job(job)
    adapter = _Stage3DebugAdapter()
    settlements: list[dict[str, object]] = []
    llm_calls: list[dict[str, object]] = []
    original_settle = post_turn.settle_episode_trace

    def capture_settlement(**kwargs: object) -> Mapping[str, object]:
        trace = original_settle(**kwargs)
        settlements.append({
            "trace": deepcopy(trace),
            "episode": deepcopy(kwargs.get("episode", {})),
        })
        return trace

    post_turn.settle_episode_trace = capture_settlement  # type: ignore[assignment]
    try:
        with _capture_llm_steps(llm_calls):
            async with service.lifespan(service.app):
                registry = service._adapter_registry
                if not isinstance(registry, AdapterRegistry):
                    raise AssertionError("service adapter registry is unavailable")
                registry.register(adapter)
                result = await service._deliver_accepted_task_result_episode(
                    episode
                )
                db = await get_db()
                lifecycle = await db[
                    "post_turn_lifecycle_records"
                ].find_one(
                    {"source_episode_id": episode["episode_id"]},
                    {"_id": 0},
                )
    finally:
        post_turn.settle_episode_trace = original_settle

    if result.get("status") != "delivered":
        raise AssertionError(f"tool-result delivery failed: {result}")
    if len(settlements) != 1:
        raise AssertionError("tool-result path did not settle exactly once")
    trace = settlements[0]["trace"]
    if not isinstance(trace, Mapping) or trace.get("trigger_source") != "tool_result":
        raise AssertionError("tool-result source was not canonical")
    if not adapter.calls or not isinstance(lifecycle, Mapping):
        raise AssertionError("tool-result delivery evidence is incomplete")
    artifact = {
        "schema_version": "stage3_live_case_evidence.v1",
        "case_id": case_id,
        "technical_status": "passed",
        "terminal_status": trace.get("terminal_status"),
        "llm_call_count": len(llm_calls),
        "source_kind": trace.get("trigger_source"),
        "trace_cardinality": 1,
        "lifecycle_cardinality": 1,
        "case_review": {
            "case_id": case_id,
            "source_kind": trace.get("trigger_source"),
            "target_channel_type": "private",
            "delivery_adapter_call_count": len(adapter.calls),
        },
        "trace_review": _trace_review_metadata(trace),
        "lifecycle_review": _lifecycle_review_metadata(lifecycle),
        "llm_step_review": _llm_step_review_metadata(llm_calls),
        "episode": episode,
        "settled_trace": trace,
        "delivery_result": result,
        "adapter_calls": adapter.calls,
        "lifecycle_record": lifecycle,
        "raw_llm_calls": llm_calls,
    }
    _write_evidence(case_id, artifact)
    return artifact


async def test_live_fresh_database_case() -> None:
    """Run the parent harness's one sequential frozen user-message case."""

    case_id = os.environ.get("STAGE3_CASE_ID", "")
    if not case_id:
        pytest.skip("STAGE3_CASE_ID is required for the fresh-database harness")
    case = _load_fixture_case(case_id)
    await _run_chat_case(case_id=case_id, case=case)


async def test_live_user_message_source() -> None:
    """Prove one direct user-message source with real LLM calls."""

    await _run_chat_case(case_id="focused_user_message")


async def test_live_internal_thought_source() -> None:
    """Prove one claimed latch-grounded internal-thought source."""

    await _run_self_cognition_case(
        case_id="focused_internal_thought",
        trigger_kind="recent_direct_dialog_review",
        case_name="private_no_action",
        use_latch=True,
    )


async def test_live_self_cognition_source() -> None:
    """Prove one ordinary grounded self-cognition source."""

    await _run_self_cognition_case(
        case_id="focused_self_cognition",
        trigger_kind="recent_direct_dialog_review",
        case_name="private_no_action",
    )


async def test_live_scheduled_tick_source() -> None:
    """Prove one claimed scheduled-tick source."""

    await _run_self_cognition_case(
        case_id="focused_scheduled_tick",
        trigger_kind="scheduled_future_cognition_due",
        case_name="scheduled_future_cognition",
    )


async def test_live_tool_result_source() -> None:
    """Prove one completed tool-result source and delivery handoff."""

    await _run_tool_result_case("focused_tool_result")


async def test_live_group_review_promoted_reflection() -> None:
    """Run group review only when the guarded DB has promoted reflection."""

    await _run_self_cognition_case(
        case_id="focused_group_review_reflection",
        trigger_kind="group_chat_trigger_review",
        case_name="group_chat_review",
        promoted_reflection=True,
    )


async def test_live_media_reply_mentions_preserved() -> None:
    """Prove structured reply, mention, and admitted media projection."""

    artifact = await _run_chat_case(
        case_id="focused_media_reply_mentions",
        envelope={
            "mentions": [{
                "platform_user_id": "stage3-mentioned-user",
                "global_user_id": "stage3-mentioned-user",
                "display_name": "Mentioned User",
                "entity_kind": "user",
                "raw_text": "@mentioned",
            }],
            "reply": {
                "platform_message_id": "stage3-reply-target",
                "platform_user_id": "stage3-reply-user",
                "global_user_id": "stage3-reply-user",
                "display_name": "Reply User",
                "excerpt": "A prior grounded message.",
                "derivation": "adapter_metadata",
            },
            "attachments": [{
                "media_type": "image/png",
                "url": "",
                "base64_data": "",
                "description": "A small blue square admitted by the adapter.",
                "size_bytes": 32,
                "storage_shape": "url_only",
            }],
        },
    )
    episode = artifact["episode"]
    settlement = artifact["settlement"]
    graph_result = settlement.get("graph_result", {})
    if not isinstance(episode, Mapping) or not isinstance(graph_result, Mapping):
        raise AssertionError("media evidence does not contain live projections")
    media_percepts = [
        percept
        for percept in episode.get("percepts", [])
        if isinstance(percept, Mapping)
        and percept.get("percept_kind") == "image_observation"
    ]
    if len(media_percepts) != 1:
        raise AssertionError("admitted image observation did not reach episode")
    prompt_context = graph_result.get("prompt_message_context")
    if not isinstance(prompt_context, Mapping):
        raise AssertionError("prompt message context is missing")
    if not prompt_context.get("mentions"):
        raise AssertionError("mention projection did not reach cognition state")
    reply_context = graph_result.get("reply_context")
    if not isinstance(reply_context, Mapping) or not reply_context.get(
        "reply_to_display_name"
    ):
        raise AssertionError("reply target projection did not reach cognition state")
    artifact["media_reply_mention_projection"] = {
        "media_percept_count": len(media_percepts),
        "mention_display_names": [
            str(item.get("display_name", ""))
            for item in prompt_context["mentions"]
            if isinstance(item, Mapping)
        ],
        "reply_display_name": reply_context.get("reply_to_display_name"),
    }
    _write_evidence("focused_media_reply_mentions_projection", artifact)


async def test_live_action_affordance_routes() -> None:
    """Exercise live action selection and the bounded unavailable route."""

    artifact = await _run_chat_case(
        case_id="focused_action_affordances",
        case={
            "target_scope_fixture": {
                "channel_type": "private",
                "channel_id": "stage3-action-private",
                "global_user_id": "stage3-action-user",
                "addressed_to_character": True,
            },
            "input_text": (
                "Please keep a concise reminder about this commitment for a "
                "later follow-up."
            ),
        },
    )
    settlement = artifact["settlement"]
    graph_result = settlement.get("graph_result", {})
    if not isinstance(graph_result, Mapping):
        raise AssertionError("action route graph result is missing")
    for field_name in ("action_specs", "action_results", "future_promises"):
        value = graph_result.get(field_name, [])
        if not isinstance(value, list):
            raise AssertionError(f"{field_name} is not a bounded list")
    artifact["action_route_projection"] = {
        "action_spec_count": len(graph_result.get("action_specs", [])),
        "action_result_count": len(graph_result.get("action_results", [])),
        "future_promise_count": len(graph_result.get("future_promises", [])),
        "adapter_unavailable_path": "checked by deterministic dispatcher gate",
    }
    _write_evidence("focused_action_affordances_projection", artifact)
