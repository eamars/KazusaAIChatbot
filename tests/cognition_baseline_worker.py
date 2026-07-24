"""Run one isolated baseline/V2 case against the public service boundary."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any
from uuid import uuid4


_EXPECTED_PROFILE_SHA256 = (
    "7cd3d773c584fee7656da15eec827cd26b450825ec878716389f1e9a2ae1a484"
)
_EXPECTED_HISTORY_SHA256 = (
    "e42ef1a7a454e1208f5723fd3b87ba70d0e64579a68838ede911b5286e576008"
)
_DATABASE_PREFIX = "_test_"
_WORKSPACE_GUARD_ROOT = (
    Path(__file__).resolve().parents[1] / "test_runs"
).resolve()
_PROFILE_RUNTIME_FIELDS = frozenset({
    "self_image",
    "cognition_state",
    "global_vibe",
    "mood",
    "reflection_summary",
    "updated_at",
})
_KAZUSA_PROJECT_TOKEN = "KazusaAIChatbot"
_KAZUSA_PROJECT_URL = "https://github.com/eamars/KazusaAIChatbot"
_REQUIRED_ENV = (
    "MONGODB_URI",
    "MONGODB_DB_NAME",
    "KAZUSA_TEST_DB_GUARD",
    "STAGE3_DATABASE_GUARD",
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
    """Keep Chinese raw input and model output printable on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_streams()


def _load_json(path: Path) -> dict[str, Any]:
    """Load one worker input or artifact object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one UTF-8 raw evidence artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one fixed file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_digest(value: object) -> str:
    """Hash one canonical JSON-shaped worker value."""

    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _case_digest(case: Mapping[str, Any]) -> str:
    """Hash a scored case without execution-specific scheduling fields."""

    stable_case = {
        key: value
        for key, value in case.items()
        if key not in {"execution_id", "repetition_ordinal"}
    }
    return _json_digest(stable_case)


def _json_safe(value: object) -> object:
    """Convert Mongo and model values into JSON evidence values."""

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


def _walk_strings(value: object, path: str = "$") -> list[tuple[str, str]]:
    """Return every string value with its JSON-shaped path."""

    rows: list[tuple[str, str]] = []
    if isinstance(value, str):
        rows.append((path, value))
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            rows.extend(_walk_strings(nested, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            rows.extend(_walk_strings(nested, f"{path}[{index}]"))
    return rows


def _target_git_state(target_root: Path) -> dict[str, str]:
    """Capture target worktree SHA and status before/after a case."""

    sha_result = subprocess.run(
        ["git", "-C", str(target_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    status_result = subprocess.run(
        ["git", "-C", str(target_root), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "sha": sha_result.stdout.strip(),
        "status": status_result.stdout,
    }


def _filesystem_digest(root: Path) -> dict[str, Any]:
    """Digest the disposable workspace without reading outside its root."""

    if not root.exists():
        return {"exists": False, "file_count": 0, "sha256": ""}
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
        file_count += 1
    return {
        "exists": True,
        "file_count": file_count,
        "sha256": digest.hexdigest(),
    }


def _reply_target(reply_context: object) -> dict[str, str] | None:
    """Project a fixture reply context into the ChatRequest contract."""

    if not isinstance(reply_context, Mapping):
        return None
    reply = {
        "platform_message_id": str(
            reply_context.get("reply_to_message_id") or ""
        ),
        "platform_user_id": str(
            reply_context.get("reply_to_platform_user_id") or ""
        ),
        "global_user_id": str(
            reply_context.get("reply_to_global_user_id") or ""
        ),
        "display_name": str(
            reply_context.get("reply_to_display_name") or ""
        ),
        "excerpt": str(reply_context.get("reply_excerpt") or ""),
        "derivation": "platform_native",
    }
    if not any(reply.values()):
        return None
    return reply


def _build_request(
    case: Mapping[str, Any],
    *,
    fixed_local_timestamp: str,
    character_global_id: str,
) -> Any:
    """Build a ChatRequest from the neutral effective input payload."""

    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    effective_input = case.get("effective_input")
    if not isinstance(effective_input, Mapping):
        raise ValueError("worker case has no effective input")
    raw_mentions = effective_input.get("mentions")
    mentions = [
        dict(mention)
        for mention in raw_mentions
        if isinstance(mention, Mapping)
    ] if isinstance(raw_mentions, list) else []
    raw_attachments = effective_input.get("attachments")
    attachments = [
        dict(attachment)
        for attachment in raw_attachments
        if isinstance(attachment, Mapping)
    ] if isinstance(raw_attachments, list) else []
    body_text = str(effective_input.get("body_text") or "")
    envelope = {
        "body_text": body_text,
        "raw_wire_text": str(
            effective_input.get("raw_wire_text") or body_text
        ),
        "mentions": mentions,
        "reply": _reply_target(effective_input.get("reply_context")),
        "attachments": attachments,
        "addressed_to_global_user_ids": [
            str(value)
            for value in effective_input.get(
                "addressed_to_global_user_ids",
                [],
            )
            if str(value).strip()
        ],
        "broadcast": bool(effective_input.get("broadcast", False)),
    }
    return ChatRequest.model_validate({
        "platform": str(effective_input.get("platform") or "debug"),
        "platform_channel_id": str(
            effective_input.get("platform_channel_id") or "baseline"
        ),
        "channel_type": str(
            effective_input.get("channel_type") or "group"
        ),
        "platform_message_id": str(
            effective_input.get("platform_message_id")
            or f"worker-{uuid4().hex}"
        ),
        "platform_user_id": str(
            effective_input.get("platform_user_id") or "baseline-user"
        ),
        "platform_bot_id": str(
            effective_input.get("platform_bot_id") or character_global_id
        ),
        "display_name": str(
            effective_input.get("display_name") or "基线测试用户"
        ),
        "channel_name": str(
            effective_input.get("channel_name") or "baseline-replay"
        ),
        "content_type": str(
            effective_input.get("content_type") or "text"
        ),
        "message_envelope": envelope,
        "local_timestamp": build_turn_clock(
            fixed_local_timestamp
        )["local_timestamp"],
        "debug_modes": {},
    })


async def _wait_for_trace_run(
    db: Any,
    *,
    platform_message_id: str,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Wait until the live queue writes a terminal trace run."""

    deadline = time.perf_counter() + timeout_seconds
    latest: Mapping[str, Any] | None = None
    while time.perf_counter() < deadline:
        row = await db.llm_trace_runs.find_one(
            {"platform_message_id": platform_message_id},
            {"_id": 0},
        )
        if isinstance(row, Mapping):
            latest = row
            if str(row.get("status") or "") != "running":
                return dict(row)
        await asyncio.sleep(0.1)
    if latest is None:
        raise TimeoutError("trace run was not persisted")
    raise TimeoutError(
        f"trace run remained running: {latest.get('status', '')}"
    )


def _extract_canonical_monologue(
    response_payload: Mapping[str, Any],
) -> tuple[str, str]:
    """Extract exactly one canonical L2 reasoning monologue."""

    graph = response_payload.get("cognition_graph")
    if not isinstance(graph, Mapping):
        return "", ""
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return "", ""
    reasoning_nodes = [
        node
        for node in nodes
        if isinstance(node, Mapping) and node.get("id") == "l2.reasoning"
    ]
    if len(reasoning_nodes) != 1:
        return "", ""
    detail = reasoning_nodes[0].get("detail")
    if not isinstance(detail, Mapping):
        return "", ""
    value = detail.get("internal_monologue")
    if not isinstance(value, str) or not value.strip():
        return "", ""
    return (
        value,
        'response.cognition_graph.nodes[id="l2.reasoning"]'
        ".detail.internal_monologue",
    )


def _extract_final_cognition_monologue(
    graph_result: Mapping[str, Any],
) -> str:
    """Read the final cognition-core private monologue from graph state."""

    direct_value = graph_result.get("private_monologue")
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value
    for key in ("cognition_core_output", "cognition_output"):
        output = graph_result.get(key)
        if isinstance(output, Mapping):
            value = output.get("private_monologue")
            if isinstance(value, str) and value.strip():
                return value
    settlement = graph_result.get("settlement")
    if isinstance(settlement, Mapping):
        value = _extract_final_cognition_monologue(settlement)
        if value:
            return value
    return ""


def _trace_status_is_valid(
    *,
    output_mode: str,
    trace_run: Mapping[str, Any],
    source_kind: str = "",
) -> bool:
    """Accept the terminal trace status owned by each output surface."""

    status = str(trace_run.get("status") or "")
    if source_kind in {
        "internal_thought",
        "self_cognition",
        "scheduled_tick",
        "proactive_proposal",
    }:
        if output_mode == "private":
            return status in {"completed_private", "completed_action"}
        return status == "completed_visible"
    if output_mode == "private":
        return status in {"succeeded", "completed_private"}
    return status == "succeeded"


def _v2_requires_final_cognition_monologue(source_kind: str) -> bool:
    """Require live cognition wording only for sources that own it."""

    return source_kind != "reflection"


def _requires_live_monologue(*, source_kind: str, output_mode: str) -> bool:
    """Require a monologue only when the source reaches a wording surface."""

    if output_mode == "silent":
        return False
    return source_kind != "reflection"


def _extract_background_monologue(
    graph_result: Mapping[str, Any],
) -> tuple[str, str]:
    """Extract the background result's legacy or V2 cognition monologue."""

    direct = _extract_final_cognition_monologue(graph_result)
    if direct:
        return direct, "settlement.graph_result.private_monologue"
    candidates = [
        value
        for value in _mapping_values(graph_result, "internal_monologue")
        if isinstance(value, str) and value.strip()
    ]
    if candidates:
        return candidates[-1], "settlement.graph_result.internal_monologue"
    return "", ""


def _background_result_summary(fixture_case: Mapping[str, Any]) -> str:
    """Return the exact completed-result text declared by the fixture."""

    explicit_summary = fixture_case.get("result_summary")
    if isinstance(explicit_summary, str) and explicit_summary.strip():
        return explicit_summary
    return str(fixture_case.get("input_text") or "")


def _response_surface_status(response_payload: Mapping[str, Any]) -> str:
    """Classify the observable response surface without rewriting it."""

    if isinstance(response_payload.get("operational_error"), Mapping):
        return "operational_error"
    messages = response_payload.get("messages")
    if isinstance(messages, list) and messages:
        return "visible_dialog"
    return "no_visible_dialog"


def _source_leaks(
    case: Mapping[str, Any],
    model_visible: object,
) -> list[dict[str, str]]:
    """Find source character identity tokens in Asuna model-visible evidence."""

    source_character = case.get("source_character")
    if not isinstance(source_character, Mapping) or not source_character:
        return []
    tokens = {
        str(source_character.get("global_user_id") or ""),
        str(source_character.get("display_name") or ""),
        *[
            str(value)
            for value in source_character.get("platform_user_ids", [])
        ],
        *[
            str(value)
            for value in source_character.get("aliases", [])
        ],
    }
    tokens = {token for token in tokens if token}
    leaks: list[dict[str, str]] = []
    for path, value in _walk_strings(model_visible):
        for token in sorted(tokens, key=len, reverse=True):
            if token in value:
                leaks.append({
                    "path": path,
                    "token": token,
                    "value": value,
                })
    return leaks


def _immutable_external_failures(case: Mapping[str, Any]) -> list[str]:
    """Verify declared external text survives at its source location."""

    failures: list[str] = []
    spans = case.get("immutable_external_spans")
    declared_values = {
        str(span.get("value") or "")
        for span in spans
        if isinstance(span, Mapping) and str(span.get("value") or "")
    } if isinstance(spans, list) else set()
    source_input = case.get("source_input")
    effective_input = case.get("effective_input")
    source_body = str(case.get("input_text") or "")
    if isinstance(source_input, Mapping):
        source_body = str(source_input.get("body_text") or "") or source_body
    effective_body = (
        str(effective_input.get("body_text") or "")
        if isinstance(effective_input, Mapping)
        else ""
    )
    for span in spans if isinstance(spans, list) else []:
        if not isinstance(span, Mapping):
            failures.append("immutable external span is not an object")
            continue
        token = str(span.get("value") or "")
        if not token:
            failures.append("immutable external span has no value")
            continue
        source_count = source_body.count(token)
        effective_count = effective_body.count(token)
        if source_count != effective_count:
            failures.append(
                f"immutable external span changed at "
                f"{span.get('json_path', '')}: {token} "
                f"source={source_count} effective={effective_count}"
            )
    for token in (_KAZUSA_PROJECT_TOKEN, _KAZUSA_PROJECT_URL):
        if token in source_body and token not in declared_values:
            failures.append(
                f"immutable external token has no declared span: {token}"
            )
    return failures


def _profile_matches(
    observed: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
) -> list[str]:
    """Compare static profile fields observed in the target DB."""

    if observed is None:
        return ["character profile singleton is missing"]
    failures: list[str] = []
    observed_keys = set(observed)
    expected_keys = set(profile)
    unexpected_keys = sorted(
        observed_keys - expected_keys - _PROFILE_RUNTIME_FIELDS
    )
    missing_keys = sorted(expected_keys - observed_keys)
    failures.extend(
        f"character profile has unexpected field: {key}"
        for key in unexpected_keys
    )
    failures.extend(
        f"character profile is missing field: {key}"
        for key in missing_keys
    )
    for key, expected in profile.items():
        if observed.get(key) != expected:
            failures.append(f"character profile field differs: {key}")
    return failures


def _validate_worker_structure(
    input_payload: Mapping[str, Any],
    *,
    target_root: Path,
    workspace_root: Path,
    profile_path: Path,
    history_path: Path,
) -> dict[str, str]:
    """Validate immutable worker inputs before environment gating."""

    database_name = str(input_payload.get("database_name") or "")
    _validate_database_guard(database_name)
    if not target_root.is_dir():
        raise ValueError(f"target root is missing: {target_root}")
    if _WORKSPACE_GUARD_ROOT not in workspace_root.parents:
        raise ValueError(
            f"workspace escaped test_runs guard: {workspace_root}"
        )
    if _sha256(profile_path) != _EXPECTED_PROFILE_SHA256:
        raise ValueError("profile hash differs from frozen evidence copy")
    if _sha256(history_path) != _EXPECTED_HISTORY_SHA256:
        raise ValueError("history hash differs from frozen evidence copy")
    case = input_payload.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("worker input case is missing")
    expected_case_sha = str(input_payload.get("case_sha256") or "")
    if _case_digest(case) != expected_case_sha:
        raise ValueError("worker case differs from manifest digest")
    target_state = _target_git_state(target_root)
    expected_revision = str(input_payload.get("revision_sha") or "")
    if target_state["sha"] != expected_revision:
        raise ValueError(
            "target revision differs from worker manifest: "
            f"{target_state['sha']} != {expected_revision}"
        )
    return target_state


def _validate_database_guard(database_name: str) -> None:
    """Require the worker to use the exact configured guarded test database."""

    if not database_name.startswith(_DATABASE_PREFIX):
        raise ValueError(
            f"database name is outside reserved prefix: {database_name}"
        )
    configured_name = os.environ.get("MONGODB_DB_NAME", "").strip()
    if configured_name != database_name:
        raise ValueError(
            "worker database differs from configured MONGODB_DB_NAME: "
            f"{database_name} != {configured_name}"
        )
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        raise ValueError("KAZUSA_TEST_DB_GUARD must be enabled")
    if os.environ.get("STAGE3_DATABASE_GUARD") != "1":
        raise ValueError("STAGE3_DATABASE_GUARD must be enabled")


async def _require_empty_database(
    input_payload: Mapping[str, Any],
) -> Any:
    """Require the reserved database to be empty before case setup."""

    from kazusa_ai_chatbot.db._client import get_db

    database_name = str(input_payload["database_name"])
    _validate_database_guard(database_name)
    db = await get_db()
    counts = await _collection_counts(db)
    if counts:
        raise ValueError(
            "reserved test database is not empty: "
            f"{json.dumps(counts, ensure_ascii=False, sort_keys=True)}"
        )
    return db


async def _cleanup_database(
    input_payload: Mapping[str, Any],
    db: Any,
) -> None:
    """Drop one reserved database and verify that it disappeared."""

    from kazusa_ai_chatbot.db._client import close_db, get_db

    database_name = str(input_payload["database_name"])
    _validate_database_guard(database_name)
    del db
    cleanup_error: Exception | None = None
    try:
        cleanup_db = await get_db()
        await cleanup_db.client.drop_database(database_name)
        if database_name in await cleanup_db.client.list_database_names():
            raise RuntimeError(
                f"reserved test database still exists: {database_name}"
            )
    except Exception as exc:
        cleanup_error = exc
    finally:
        await close_db()
    if cleanup_error is not None:
        raise RuntimeError(
            f"test database cleanup failed: {cleanup_error}"
        ) from cleanup_error


def _mapping_values(value: object, key: str) -> list[object]:
    """Collect values for one key from nested JSON-shaped evidence."""

    values: list[object] = []
    if isinstance(value, Mapping):
        if key in value:
            values.append(value[key])
        for nested in value.values():
            values.extend(_mapping_values(nested, key))
    elif isinstance(value, list):
        for nested in value:
            values.extend(_mapping_values(nested, key))
    return values


def _has_nonempty_key(value: object, keys: Sequence[str]) -> bool:
    """Return whether one declared evidence key has a meaningful value."""

    for key in keys:
        for nested in _mapping_values(value, key):
            if nested not in (None, "", [], {}):
                return True
    return False


def _response_overlaps_source_evidence(
    case: Mapping[str, Any],
    response_payload: Mapping[str, Any],
    monologue: str,
) -> bool:
    """Require a visible answer to reuse a substantive source fact."""

    effective_input = case.get("effective_input")
    if not isinstance(effective_input, Mapping):
        return False
    source_text = str(effective_input.get("body_text") or "")
    if not source_text.strip():
        return False
    messages = response_payload.get("messages")
    visible_text = " ".join(
        str(message)
        for message in messages
        if isinstance(messages, list)
        and isinstance(message, str)
    ) if isinstance(messages, list) else ""
    comparison_text = f"{visible_text} {monologue}"
    identity_tokens = {
        str(effective_input.get("display_name") or ""),
    }
    mentions = effective_input.get("mentions")
    if isinstance(mentions, list):
        identity_tokens.update(
            str(mention.get("display_name") or "")
            for mention in mentions
            if isinstance(mention, Mapping)
        )
    source_runs = re.findall(
        r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+",
        source_text,
    )
    source_spans: list[str] = []
    for run in source_runs:
        minimum_length = 2 if re.fullmatch(r"[\u4e00-\u9fff]+", run) else 3
        for span_length in range(minimum_length, min(len(run), 4) + 1):
            source_spans.extend(
                run[index:index + span_length]
                for index in range(len(run) - span_length + 1)
            )
    substantive_spans = [
        span
        for span in source_spans
        if not any(
            span in identity_token
            for identity_token in identity_tokens
            if identity_token
        )
    ]
    return any(span in comparison_text for span in substantive_spans)


def _delivery_text_is_grounded(
    expected_delivery_text: str,
    delivery_calls: Sequence[Mapping[str, Any]],
) -> bool:
    """Accept a paraphrased result only when it retains a substantive fact."""

    if not expected_delivery_text.strip():
        return False
    delivered_text = " ".join(
        str(call.get("text") or "")
        for call in delivery_calls
    )
    for run in re.findall(
        r"[\u4e00-\u9fff]+|[A-Za-z0-9_]+",
        expected_delivery_text,
    ):
        minimum_length = (
            2
            if re.fullmatch(r"[\u4e00-\u9fff]+", run)
            else 3
        )
        if len(run) < minimum_length:
            continue
        spans = {
            run[index:index + span_length]
            for span_length in range(
                minimum_length,
                min(len(run), 4) + 1,
            )
            for index in range(len(run) - span_length + 1)
        }
        if any(span in delivered_text for span in spans):
            return True
    return False


def _response_contains_clarification_question(
    response_payload: Mapping[str, Any],
) -> bool:
    """Recognize a visible clarification question without rewriting it."""

    messages = response_payload.get("messages")
    if not isinstance(messages, list):
        return False
    return any(
        isinstance(message, str)
        and ("?" in message or "？" in message)
        for message in messages
    )


def _response_contains_source_attribution(
    response_payload: Mapping[str, Any],
) -> bool:
    """Recognize a concrete HTTP(S) citation in the visible answer."""

    messages = response_payload.get("messages")
    if not isinstance(messages, list):
        return False
    visible_text = " ".join(
        message for message in messages if isinstance(message, str)
    )
    return bool(re.search(r"https?://[^\s]+", visible_text))


def _response_repeats_source_time_expression(
    case: Mapping[str, Any],
    response_payload: Mapping[str, Any],
) -> bool:
    """Match an explicit Chinese relative-time phrase from input to output."""

    source_text = str(
        case.get("input_text")
        or (
            case.get("effective_input", {}).get("body_text", "")
            if isinstance(case.get("effective_input"), Mapping)
            else ""
        )
    )
    messages = response_payload.get("messages")
    if not source_text.strip() or not isinstance(messages, list):
        return False
    visible_text = " ".join(
        message for message in messages if isinstance(message, str)
    )
    expressions = re.findall(
        r"(?:今天|明天|后天)(?:上午|下午|晚上)?"
        r"[零一二三四五六七八九十\d]+点"
        r"(?:[零一二三四五六七八九十\d]+分)?",
        source_text,
    )
    return any(expression in visible_text for expression in expressions)


def _status_values(value: object) -> list[str]:
    """Collect status-like strings from nested lifecycle evidence."""

    statuses: list[str] = []
    for key in (
        "status",
        "state",
        "terminal_status",
        "result_status",
        "delivery_state",
    ):
        statuses.extend(
            str(item).strip().lower()
            for item in _mapping_values(value, key)
            if isinstance(item, str) and item.strip()
        )
    return statuses


def _has_status(value: object, statuses: set[str]) -> bool:
    """Return whether lifecycle evidence contains one expected status."""

    return any(item in statuses for item in _status_values(value))


def _has_unavailable_evidence(value: object) -> bool:
    """Require an explicit unavailable/limitation state for truthful limits."""

    for key in ("availability", "scheduler", "limitation", "unavailable"):
        for nested in _mapping_values(value, key):
            if isinstance(nested, Mapping):
                if nested.get("available") is False:
                    return True
                if str(nested.get("status") or "").lower() in {
                    "unavailable",
                    "disabled",
                    "blocked",
                }:
                    return True
            elif key == "unavailable" and nested:
                return True
    for runtime in _mapping_values(value, "action_availability_runtime"):
        if not isinstance(runtime, Mapping):
            continue
        if _unavailable_status(runtime.get("scheduler_status")):
            return True
        worker_status = runtime.get("worker_status")
        if isinstance(worker_status, Mapping) and any(
            _unavailable_status(status)
            for status in worker_status.values()
        ):
            return True
        adapter_status = runtime.get("adapter_target_status")
        if isinstance(adapter_status, Mapping) and any(
            _unavailable_status(status)
            for status in adapter_status.values()
        ):
            return True
    return False


def _has_unavailable_scheduler_evidence(value: object) -> bool:
    """Require a trusted scheduler-owner outage in nested runtime evidence."""

    for runtime in _mapping_values(value, "action_availability_runtime"):
        if not isinstance(runtime, Mapping):
            continue
        if _unavailable_status(runtime.get("scheduler_status")):
            return True
        worker_status = runtime.get("worker_status")
        if not isinstance(worker_status, Mapping):
            continue
        if _unavailable_status(worker_status.get("orchestrator")):
            return True
    return False


def _has_runtime_capability_limit_contract(value: object) -> bool:
    """Require the surface contract to carry the trusted outage boundary."""

    for limits in _mapping_values(value, "runtime_capability_limits"):
        if not isinstance(limits, list):
            continue
        if any(
            isinstance(limit, str)
            and "不可用" in limit
            for limit in limits
        ):
            return True
    return False


def _truthful_unavailable_scheduler_outcome(value: object) -> bool:
    """Require both owner evidence and its propagated surface contract."""

    return (
        _has_unavailable_scheduler_evidence(value)
        and _has_runtime_capability_limit_contract(value)
    )


def _truthful_unavailable_repository_outcome(value: object) -> bool:
    """Accept a typed unavailable repository owner without reader evidence."""

    if not isinstance(value, Mapping):
        return False
    graph_result = value.get("graph_result")
    if not isinstance(graph_result, Mapping):
        return False
    runtime = graph_result.get("action_availability_runtime")
    if not isinstance(runtime, Mapping):
        return False
    worker_status = runtime.get("worker_status")
    if not isinstance(worker_status, Mapping):
        return False
    if not all(
        _unavailable_status(worker_status.get(owner))
        for owner in ("accepted_task", "background_work")
    ):
        return False
    cognition_output = graph_result.get("cognition_core_output")
    if not isinstance(cognition_output, Mapping):
        return False
    if cognition_output.get("goal_resolution") not in {
        "blocked",
        "requires_user_input",
    }:
        return False
    if cognition_output.get("action_requests") != []:
        return False
    resolver_requests = cognition_output.get("resolver_requests")
    if not isinstance(resolver_requests, list):
        return False
    if any(
        not isinstance(row, Mapping)
        or row.get("capability") != "human_clarification"
        for row in resolver_requests
    ):
        return False
    return "仓库代码读取 owner 不可用" in json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _truthful_unavailable_coding_response(value: object) -> bool:
    """Accept a truthful current limitation without an attempted coding effect."""

    if not isinstance(value, Mapping):
        return False
    graph_result = value.get("graph_result")
    if not isinstance(graph_result, Mapping):
        return False
    runtime = graph_result.get("action_availability_runtime")
    if not isinstance(runtime, Mapping):
        return False
    worker_status = runtime.get("worker_status")
    if not isinstance(worker_status, Mapping):
        return False
    if not all(
        _unavailable_status(worker_status.get(owner))
        for owner in ("accepted_task", "background_work", "orchestrator")
    ):
        return False
    cognition_output = graph_result.get("cognition_core_output")
    if not isinstance(cognition_output, Mapping):
        return False
    if cognition_output.get("goal_resolution") != "answerable_now":
        return False
    if cognition_output.get("action_requests") != []:
        return False
    if cognition_output.get("resolver_requests") != []:
        return False
    return "仓库代码读取 owner 不可用" in json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _scheduler_disabled_in_test_runtime() -> bool:
    """Return whether the isolated corpus process disables the scheduler."""

    return os.environ.get("CALENDAR_SCHEDULER_ENABLED", "").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


def _unavailable_status(value: object) -> bool:
    """Return whether one trusted runtime status states an outage."""

    return str(value or "").strip().lower() in {
        "down",
        "unavailable",
        "disabled",
        "blocked",
    }


def _contains_text(value: object, text: str) -> bool:
    """Search evidence text without changing or normalizing its content."""

    return text in json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _count_named_delta(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    token: str,
) -> int:
    """Count newly persisted rows in collections containing one token."""

    before_count = sum(
        int(value)
        for key, value in before.items()
        if token in str(key).lower() and isinstance(value, int)
    )
    after_count = sum(
        int(value)
        for key, value in after.items()
        if token in str(key).lower() and isinstance(value, int)
    )
    return after_count - before_count


def _evaluate_hard_gates(
    input_payload: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    response_payload: Mapping[str, Any],
    monologue: str,
    monologue_path: str,
    graph_result: Mapping[str, Any],
    persisted_profile: Mapping[str, Any] | None,
    adapter_calls: Sequence[Mapping[str, Any]],
    counts_before: Mapping[str, Any],
    counts_after: Mapping[str, Any],
    workspace_before: Mapping[str, Any],
    workspace_after: Mapping[str, Any],
    expected_delivery_text: str,
) -> tuple[list[str], dict[str, bool]]:
    """Evaluate every fixture-declared gate and fail closed when unknown."""

    messages = response_payload.get("messages")
    visible = isinstance(messages, list) and any(
        isinstance(message, str) and message.strip()
        for message in messages
    )
    output_mode = str(case.get("output_mode") or "visible")
    private_surface = output_mode == "private" and bool(monologue) and not visible
    delivery_calls = [
        call
        for call in adapter_calls
        if str(call.get("text") or "").strip()
    ]
    combined_evidence = {
        "graph_result": graph_result,
        "response": response_payload,
        "adapter_calls": adapter_calls,
        "counts_after": counts_after,
        "persisted_profile": persisted_profile,
    }
    workspace_effect = workspace_before != workspace_after
    schedule_delta = _count_named_delta(
        counts_before,
        counts_after,
        "schedule",
    )
    truthful_unavailable_scheduler = (
        _scheduler_disabled_in_test_runtime()
        and _truthful_unavailable_scheduler_outcome(graph_result)
    )
    truthful_unavailable_repository = (
        _truthful_unavailable_repository_outcome(combined_evidence)
    )
    truthful_unavailable_coding = (
        str(case.get("case_id") or "") == "C11"
        and _truthful_unavailable_coding_response(combined_evidence)
    )
    coding_run_present = _has_nonempty_key(
        combined_evidence,
        ("coding_run", "coding_run_state", "coding_run_context"),
    )
    accepted_task_delta = _count_named_delta(
        counts_before,
        counts_after,
        "accepted_task",
    )
    background_work_delta = _count_named_delta(
        counts_before,
        counts_after,
        "background_work",
    )
    accepted_task_count_after = _count_named_delta(
        {},
        counts_after,
        "accepted_task",
    )
    accepted_task_state_present = _has_nonempty_key(
        combined_evidence,
        (
            "accepted_task",
            "accepted_task_state",
        ),
    )
    coding_owner_unavailable = (
        (truthful_unavailable_repository or truthful_unavailable_coding)
        and accepted_task_delta == 0
        and background_work_delta == 0
        and not coding_run_present
        and not workspace_effect
    )
    cancelled = _has_status(combined_evidence, {"cancelled", "canceled"})
    status_terminal = _has_status(
        combined_evidence,
        {"completed", "succeeded", "terminal", "finished"},
    )
    approval_pending = _has_status(
        combined_evidence,
        {"pending", "proposal_ready", "awaiting_approval", "approval_pending"},
    )
    gate_results: dict[str, bool] = {}
    for raw_gate in case.get("hard_gates", []):
        gate = str(raw_gate)
        passed = False
        if gate == "visible_dialog":
            passed = visible
        elif gate in {"no_visible_dialog", "relevance_silence"}:
            passed = not visible
        elif gate == "canonical_monologue":
            passed = bool(monologue) and "internal_monologue" in monologue_path
        elif gate == "no_source_identity_leak":
            passed = not _source_leaks(
                case,
                {
                    "effective_input": case.get("effective_input"),
                    "effective_context": case.get("effective_context"),
                    "response": response_payload,
                    "monologue": monologue,
                },
            )
        elif gate == "immutable_external_text":
            passed = not _immutable_external_failures(case)
        elif gate in {"private_cognition", "permission_bound_private_result"}:
            passed = private_surface
        elif gate in {
            "no_delivery",
            "no_unchecked_delivery",
            "no_delivery_without_contact_permission",
            "permission_negative",
        }:
            passed = not delivery_calls
        elif gate == "one_authorized_delivery":
            passed = len(delivery_calls) == 1
        elif gate == "due_trigger":
            source_case = graph_result.get("calendar_run")
            passed = (
                str(case.get("source_kind") or "") == "scheduled_tick"
                and (
                    (
                        isinstance(source_case, Mapping)
                        and bool(source_case.get("due_at"))
                    )
                    or _has_nonempty_key(
                        graph_result,
                        ("cognitive_episode", "semantic_due_state"),
                    )
                )
            )
        elif gate == "reflection_outside_live_wording":
            passed = (
                str(case.get("source_kind") or "") == "reflection"
                and not visible
                and _has_nonempty_key(graph_result, ("reflection_result",))
            )
        elif gate == "affect_settled":
            persisted_affect = (
                isinstance(persisted_profile, Mapping)
                and all(
                    str(persisted_profile.get(field) or "").strip()
                    for field in ("mood", "global_vibe", "reflection_summary")
                )
            )
            passed = (
                _has_nonempty_key(graph_result, ("reflection_result",))
                and _has_nonempty_key(
                    graph_result,
                    ("character_state", "persisted_profile"),
                )
            ) or (
                _has_nonempty_key(graph_result, ("reflection_result",))
                and persisted_affect
            )
        elif gate == "reply_response":
            effective_input = case.get("effective_input")
            passed = (
                visible
                and isinstance(effective_input, Mapping)
                and isinstance(effective_input.get("reply_context"), Mapping)
                and bool(effective_input["reply_context"].get(
                    "reply_to_message_id"
                ))
            )
        elif gate == "evidence_grounded":
            passed = _has_nonempty_key(
                graph_result,
                ("evidence", "evidence_handles", "selected_evidence_handles"),
            ) or _response_overlaps_source_evidence(
                case,
                response_payload,
                monologue,
            )
        elif gate == "source_attribution":
            passed = _has_nonempty_key(
                graph_result,
                ("evidence_refs", "source_attribution", "evidence"),
            ) or _response_contains_source_attribution(response_payload)
        elif gate == "media_evidence_before_answer":
            passed = _has_nonempty_key(
                graph_result,
                ("media_evidence", "media_observations"),
            )
        elif gate == "local_recall":
            passed = _has_nonempty_key(
                graph_result,
                (
                    "conversation_evidence",
                    "conversation_progress",
                    "memory_evidence",
                    "recall_evidence",
                ),
            )
        elif gate == "minimal_clarification":
            passed = _has_nonempty_key(
                graph_result,
                ("clarification", "clarification_question", "human_clarification"),
            ) or _response_contains_clarification_question(response_payload)
        elif gate == "repository_map_evidence":
            passed = _has_nonempty_key(
                combined_evidence,
                (
                    "repository_map",
                    "repository_map_evidence",
                    "workspace_read_evidence",
                    "repository_evidence",
                ),
            ) or coding_owner_unavailable
        elif gate == "coding_reader_route":
            passed = (
                _contains_text(combined_evidence, "coding_reader")
                or _contains_text(combined_evidence, "code_reading")
                or coding_owner_unavailable
            )
        elif gate == "terminal_result":
            passed = (
                _has_nonempty_key(
                    combined_evidence,
                    (
                        "terminal_result",
                        "terminal_status",
                        "execution_result",
                        "result_summary",
                    ),
                )
                or status_terminal
            )
        elif gate == "accepted_task_persisted":
            passed = accepted_task_delta > 0
        elif gate == "accepted_task_status":
            passed = accepted_task_count_after > 0 or accepted_task_state_present
        elif gate == "accepted_coding_task_persisted":
            passed = (
                accepted_task_delta > 0 and coding_run_present
            ) or coding_owner_unavailable
        elif gate in {"coding_run_bound", "no_unbound_run"}:
            passed = coding_run_present or coding_owner_unavailable
        elif gate == "coding_run_status":
            passed = coding_run_present and bool(_status_values(graph_result))
        elif gate == "coding_run_unblocked":
            passed = coding_run_present and not _has_status(
                graph_result,
                {"blocked"},
            )
        elif gate == "bounded_progress":
            passed = _has_nonempty_key(
                graph_result,
                ("progress", "conversation_progress", "bounded_progress"),
            )
        elif gate in {"file_effect", "guarded_workspace_effect"}:
            passed = workspace_effect or coding_owner_unavailable
        elif gate == "approval_bound":
            passed = approval_pending or _has_nonempty_key(
                graph_result,
                ("approval", "approval_state", "approval_result"),
            )
        elif gate == "approval_pending":
            passed = approval_pending
        elif gate == "no_effect_before_approval":
            passed = not workspace_effect and not delivery_calls
        elif gate == "cancel_once":
            passed = cancelled and len(delivery_calls) <= 1
        elif gate == "no_post_cancel_effect":
            passed = cancelled and not workspace_effect
        elif gate == "schedule_once":
            if _scheduler_disabled_in_test_runtime():
                passed = truthful_unavailable_scheduler and (
                    _response_repeats_source_time_expression(
                        case,
                        response_payload,
                    )
                )
            else:
                passed = schedule_delta == 1 or _has_nonempty_key(
                    graph_result,
                    ("future_promises", "scheduled_tasks", "schedule"),
                )
        elif gate == "schedule_time_exact":
            if _scheduler_disabled_in_test_runtime():
                passed = truthful_unavailable_scheduler and (
                    _response_repeats_source_time_expression(
                        case,
                        response_payload,
                    )
                )
            else:
                scheduled_timestamp = str(
                    input_payload.get("fixed_scheduled_local_timestamp") or ""
                )
                scheduled_date, _, scheduled_time = (
                    scheduled_timestamp.partition(" ")
                )
                passed = _contains_text(
                    combined_evidence,
                    scheduled_timestamp,
                ) or (
                    bool(scheduled_date)
                    and bool(scheduled_time)
                    and _contains_text(combined_evidence, scheduled_date)
                    and _contains_text(combined_evidence, scheduled_time)
                ) or _response_repeats_source_time_expression(
                    case,
                    response_payload,
                )
        elif gate == "no_unowned_delayed_side_effect":
            if _scheduler_disabled_in_test_runtime():
                passed = (
                    _count_named_delta(
                        counts_before,
                        counts_after,
                        "accepted_task",
                    ) == 0
                    and _count_named_delta(
                        counts_before,
                        counts_after,
                        "background_work_job",
                    ) == 0
                )
            else:
                passed = True
        elif gate == "memory_lifecycle":
            passed = _has_nonempty_key(
                graph_result,
                (
                    "memory_lifecycle",
                    "memory_updates",
                    "consolidation_state",
                    "consolidation_result",
                ),
            )
        elif gate == "three_private_outcomes":
            private_outcomes = []
            for key in ("private_outcomes", "internal_results", "semantic_outcomes"):
                private_outcomes.extend(_mapping_values(graph_result, key))
            passed = any(
                isinstance(value, list) and len(value) >= 3
                for value in private_outcomes
            )
        elif gate in {"tool_result_delivery", "immediate_result"}:
            passed = _delivery_text_is_grounded(
                expected_delivery_text,
                delivery_calls,
            )
        elif gate == "no_duplicate_run":
            passed = len(delivery_calls) <= 1
        elif gate in {"truthful_limitation", "no_false_promise"}:
            passed = _truthful_unavailable_scheduler_outcome(graph_result)
        else:
            passed = False
        gate_results[gate] = passed

    failures = [
        f"hard gate failed: {gate}"
        for gate, passed in gate_results.items()
        if not passed
    ]
    return failures, gate_results


async def _bootstrap_main_profile(
    input_payload: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> None:
    """Seed the legacy singleton profile before the main runtime starts."""

    if str(input_payload.get("revision") or "") != "main":
        return
    from kazusa_ai_chatbot.db import db_bootstrap, save_character_profile

    await db_bootstrap()
    await save_character_profile(dict(profile))


def _default_seed_timestamp_before_turn(fixed_local_timestamp: str) -> str:
    """Place an untimestamped fixture row before the active fixed turn."""

    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    active_clock = build_turn_clock(fixed_local_timestamp)
    active_timestamp = datetime.fromisoformat(
        str(active_clock["storage_timestamp_utc"]).replace("Z", "+00:00")
    )
    seed_timestamp = active_timestamp - timedelta(minutes=1)
    return seed_timestamp.isoformat()


async def _seed_conversation_rows(
    *,
    case: Mapping[str, Any],
    channel_id: str,
    fixed_local_timestamp: str,
    current_author_global_user_id: str = "",
    current_author_platform_user_id: str = "",
) -> list[dict[str, Any]]:
    """Seed declared context rows with chronology before the active turn."""

    context_rows = case.get("effective_context")
    state_seed = case.get("state_seed")
    default_seed_timestamp = _default_seed_timestamp_before_turn(
        fixed_local_timestamp
    )
    extra_rows = []
    if isinstance(state_seed, Mapping):
        seed_value = state_seed.get("conversation_rows")
        if isinstance(seed_value, list):
            extra_rows.extend(
                row for row in seed_value if isinstance(row, Mapping)
            )
        prior_user_row = state_seed.get("prior_user_row")
        if isinstance(prior_user_row, str):
            prior_user_document: dict[str, Any] = {
                "role": "user",
                "body_text": prior_user_row,
            }
            if current_author_global_user_id:
                prior_user_document["global_user_id"] = (
                    current_author_global_user_id
                )
            if current_author_platform_user_id:
                prior_user_document["platform_user_id"] = (
                    current_author_platform_user_id
                )
            extra_rows.append(prior_user_document)
        prior_message = state_seed.get("prior_message")
        if isinstance(prior_message, str):
            extra_rows.append({
                "role": "assistant",
                "body_text": prior_message,
            })
    source_rows = [
        row for row in context_rows
        if isinstance(row, Mapping)
    ] if isinstance(context_rows, list) else []
    source_rows.extend(extra_rows)
    if not source_rows:
        return []
    from kazusa_ai_chatbot.db import (
        get_document_text_embeddings_batch,
        save_conversation,
    )

    documents: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        body_text = str(row.get("body_text") or "").strip()
        if not body_text:
            continue
        document = dict(row)
        document.update({
            "platform": "debug",
            "platform_channel_id": channel_id,
            "platform_message_id": (
                str(row.get("platform_message_id") or "")
                or f"seed-{index:03d}"
            ),
            "channel_type": str(row.get("channel_type") or "group"),
            "channel_name": "baseline-replay",
            "body_text": body_text,
            "raw_wire_text": str(row.get("raw_wire_text") or body_text),
            "content_type": str(row.get("content_type") or "text"),
            "attachments": (
                row.get("attachments", [])
                if isinstance(row.get("attachments", []), list)
                else []
            ),
            "mentions": (
                row.get("mentions", [])
                if isinstance(row.get("mentions", []), list)
                else []
            ),
            "addressed_to_global_user_ids": (
                row.get("addressed_to_global_user_ids", [])
                if isinstance(row.get("addressed_to_global_user_ids", []), list)
                else []
            ),
            "broadcast": bool(row.get("broadcast", False)),
            "reply_context": (
                row.get("reply_context", {})
                if isinstance(row.get("reply_context", {}), Mapping)
                else {}
            ),
            "timestamp": str(
                row.get("timestamp")
                or default_seed_timestamp
            ),
        })
        documents.append(document)
    embeddings = await get_document_text_embeddings_batch([
        document["body_text"] for document in documents
    ])
    if len(embeddings) != len(documents):
        raise ValueError("seed embedding count differs from seed row count")
    for document, embedding in zip(documents, embeddings, strict=True):
        document["embedding"] = embedding
        await save_conversation(document)
    return documents


def _build_seeded_coding_task_document(
    *,
    case: Mapping[str, Any],
    fixed_local_timestamp: str,
    source_platform: str,
    source_channel_id: str,
    source_channel_type: str,
    source_message_id: str,
    source_platform_bot_id: str,
    source_character_name: str,
    requester_global_user_id: str,
    requester_platform_user_id: str,
    requester_display_name: str,
) -> dict[str, Any] | None:
    """Build one active accepted-task row from a declared coding seed."""

    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    state_seed = case.get("state_seed")
    if not isinstance(state_seed, Mapping):
        return None
    coding_seed = state_seed.get("coding_run")
    if not isinstance(coding_seed, Mapping):
        return None
    case_id = str(case.get("case_id") or "").strip()
    run_id = str(coding_seed.get("run_id") or "").strip()
    status = str(coding_seed.get("status") or "").strip()
    raw_actions = coding_seed.get("action_set")
    allowed_statuses = {
        "created",
        "source_resolved",
        "evidence_collected",
        "proposal_ready",
        "awaiting_approval",
        "applying",
        "verifying",
        "repairing",
        "completed",
        "blocked",
        "rejected",
        "failed",
        "cancelled",
    }
    allowed_actions = {
        "revise_proposal",
        "summarize",
        "status",
        "approve_and_verify",
        "respond_to_blocker",
        "cancel",
    }
    if not case_id or not run_id:
        raise ValueError("coding seed requires case_id and run_id")
    if status not in allowed_statuses:
        raise ValueError(f"coding seed has unsupported status: {status}")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("coding seed requires a non-empty action_set")
    action_set = [
        str(action)
        for action in raw_actions
        if isinstance(action, str) and action in allowed_actions
    ]
    if len(action_set) != len(raw_actions):
        raise ValueError("coding seed action_set contains an unsupported action")
    storage_timestamp = build_turn_clock(
        fixed_local_timestamp,
    )["storage_timestamp_utc"]
    objective_summary = str(
        coding_seed.get("objective_summary")
        or coding_seed.get("task_summary")
        or case.get("input_text")
        or ""
    ).strip()[:500]
    open_blocker = coding_seed.get("open_blocker")
    active_blocker = None
    if isinstance(open_blocker, str) and open_blocker.strip():
        active_blocker = {
            "blocker_kind": "needs_user_input",
            "question": open_blocker[:500],
            "options": [
                str(option)[:120]
                for option in coding_seed.get("blocker_options", [])
                if isinstance(option, str) and option.strip()
            ][:5],
        }
    context = {
        "schema_version": "coding_run_context.v1",
        "coding_run_ref": f"coding_run:{run_id}",
        "status": status,
        "objective_summary": objective_summary,
        "allowed_next_actions": list(dict.fromkeys(action_set)),
        "active_blocker": active_blocker,
        "followup_open": status not in {
            "completed",
            "rejected",
            "failed",
            "cancelled",
        },
        "updated_at": storage_timestamp,
    }
    task_id = f"baseline-seeded-task-{case_id}"
    return {
        "_id": task_id,
        "schema_version": "accepted_task.v1",
        "accepted_task_id": task_id,
        "task_identity_key": f"baseline-seeded-identity-{case_id}",
        "active_identity_key": f"baseline-seeded-active-{case_id}",
        "action_kind": "accepted_coding_task_request",
        "first_source_message_id": source_message_id,
        "related_source_message_ids": [],
        "source_trigger_source": "user_message",
        "state": "pending",
        "result_kind": "none",
        "executor_kind": "background_work",
        "executor_ref": f"baseline-seeded-executor-{case_id}",
        "accepted_task_summary": objective_summary,
        "source_context": objective_summary,
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 4000,
        "source_platform": source_platform,
        "source_channel_id": source_channel_id,
        "source_channel_type": source_channel_type,
        "source_platform_bot_id": source_platform_bot_id,
        "source_character_name": source_character_name,
        "requester_global_user_id": requester_global_user_id,
        "requester_platform_user_id": requester_platform_user_id,
        "requester_display_name": requester_display_name,
        "created_at": storage_timestamp,
        "updated_at": storage_timestamp,
        "started_at": "",
        "completed_at": "",
        "delivered_at": "",
        "result_summary": "",
        "artifact_text": "",
        "failure_summary": "",
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "last_progress_reported_at": "",
        "coding_run_context": context,
    }


def _build_self_cognition_case(
    input_payload: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a public self-cognition source case from one fixture row."""

    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        build_acquaintance_user_state,
    )
    from kazusa_ai_chatbot.self_cognition import models
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    fixture_case = input_payload["case"]
    case_id = str(fixture_case["case_id"])
    source_kind = str(fixture_case.get("source_kind") or "self_cognition")
    if source_kind == "scheduled_tick":
        trigger_kind = models.TRIGGER_SCHEDULED_FUTURE_COGNITION
        case_name = models.CASE_SCHEDULED_FUTURE_COGNITION
        due_state = models.DUE_STATE_DUE_NOW
    elif source_kind == "internal_thought":
        trigger_kind = models.TRIGGER_RECENT_DIRECT_DIALOG_REVIEW
        case_name = models.CASE_PRIVATE_NO_ACTION
        due_state = None
    else:
        trigger_kind = models.TRIGGER_CONVERSATION_PROGRESS_REVIEW
        case_name = models.CASE_PRIVATE_NO_ACTION
        due_state = None
    turn_clock = build_turn_clock(
        str(input_payload["fixed_local_timestamp"])
    )
    storage_timestamp = turn_clock["storage_timestamp_utc"]
    channel_id = f"baseline-self-{case_id}"
    target_user_id = "baseline-current-user"
    character_profile = dict(profile)
    character_profile["global_user_id"] = "character-global"
    delivery_target = {
        "schema_version": "self_cognition_delivery_target.v1",
        "platform": "debug",
        "platform_channel_id": channel_id,
        "channel_type": "private",
        "target_global_user_id": target_user_id,
        "target_platform_user_id": "baseline-current-user-platform",
        "source_kind": "target_private_channel",
        "source_ref": case_id,
        "source_platform_channel_id": channel_id,
        "source_channel_type": "private",
        "source_message_id": case_id,
        "source_global_user_id": target_user_id,
        "source_platform_bot_id": "baseline-character-platform",
        "source_character_name": str(profile.get("name") or ""),
        "guild_id": None,
        "bot_permission_role": "self_cognition",
        "fallback_reason": "",
    }
    source_text = str(fixture_case.get("input_text") or "")
    case: dict[str, Any] = {
        "case_name": case_name,
        "case_id": f"{case_id}:{uuid4().hex[:8]}",
        "idle_timestamp_utc": storage_timestamp,
        "last_evidence_timestamp_utc": storage_timestamp,
        "trigger_kind": trigger_kind,
        "target_scope": {
            "platform": "debug",
            "platform_channel_id": channel_id,
            "channel_type": "private",
            "user_id": target_user_id,
            "platform_user_id": "baseline-current-user-platform",
            "display_name": "基线测试用户",
        },
        "source_refs": [{
            "source_kind": source_kind,
            "source_id": case_id,
            "summary": source_text,
            "due_at": storage_timestamp if due_state else None,
        }],
        "semantic_due_state": due_state,
        "actionability": "actionable",
        "visible_context": [{
            "role": "user" if source_kind != "internal_thought" else "internal_thought",
            "body_text": source_text,
            "timestamp": storage_timestamp,
            "display_name": "基线测试用户",
        }],
        "delivery_mention_users": [],
        "conversation_progress": {},
        "group_activity_window": {},
        "reflection_modifier": {},
        "existing_attempts": [],
        "character_profile": character_profile,
        "user_profile": {
            "global_user_id": "baseline-current-user",
            "display_name": "基线测试用户",
            "affinity": 500,
            "affinity_level": "普通",
            "last_relationship_insight": "",
            "cognition_state": build_acquaintance_user_state(
                global_user_id="baseline-current-user",
                updated_at=storage_timestamp.replace("+00:00", "Z"),
            ),
        },
        "platform_bot_id": "baseline-character-platform",
        "channel_topic": "",
        "promoted_reflection_context": {},
        "budget": {
            "rag_calls": 0,
            "cognition_calls": 0,
            "dialog_calls": 0,
            "topic_limit": 1,
        },
        "source_calendar_run_id": (
            f"baseline-calendar-{case_id}"
            if source_kind == "scheduled_tick" else ""
        ),
        "source_calendar_skip_reason": "",
        "cognition_source": {
            "source_kind": source_kind,
            "source_id": case_id,
        },
        "source_action_attempt_id": "",
        "delivery_target": delivery_target,
        "target_binding_status": "bound",
    }
    if source_kind == "scheduled_tick":
        case["calendar_run"] = {
            "run_id": f"baseline-calendar-{case_id}",
            "schedule_id": f"baseline-schedule-{case_id}",
            "due_at": storage_timestamp,
        }
    return case


async def _run_self_cognition_case(
    input_payload: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Run an internal/self-cognition case through its public runner."""

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.db._client import get_db
    from kazusa_ai_chatbot.self_cognition import runner, worker
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    case = _build_self_cognition_case(input_payload, profile=profile)
    adapter = _WorkerDebugAdapter()
    fixed_clock = build_turn_clock(
        str(input_payload["fixed_local_timestamp"])
    )
    source_kind = str(input_payload["case"].get("source_kind") or "")
    settlements: list[dict[str, Any]] = []
    captured_artifacts: list[dict[str, Any]] = []
    original_settle = getattr(post_turn, "settle_episode_trace", None)

    def capture_settlement(**kwargs: object) -> Mapping[str, Any]:
        if original_settle is None:
            return {}
        trace = original_settle(**kwargs)
        graph_result = kwargs.get("graph_result")
        settlements.append({
            "trace": deepcopy(trace),
            "graph_result": deepcopy(graph_result)
            if isinstance(graph_result, Mapping) else {},
        })
        return trace

    if original_settle is not None:
        post_turn.settle_episode_trace = capture_settlement  # type: ignore[assignment]
    db = None
    try:
        db = await _require_empty_database(input_payload)
        await _bootstrap_main_profile(input_payload, profile)
        async with service.lifespan(service.app):
            service.register_runtime_adapter(adapter)
            db = await get_db()
            from kazusa_ai_chatbot.db import create_user_profile

            if await db.user_profiles.find_one(
                {"global_user_id": "baseline-current-user"},
                {"_id": 1},
            ) is None:
                await create_user_profile({
                    "global_user_id": "baseline-current-user",
                })
            counts_before = await _collection_counts(db)
            if source_kind == "scheduled_tick":
                async def collect_cases(**kwargs: object) -> list[dict[str, Any]]:
                    del kwargs
                    return [case]

                async def claim_calendar_run(
                    *args: object,
                    **kwargs: object,
                ) -> bool:
                    del args, kwargs
                    return True

                async def no_op_calendar_run(
                    *args: object,
                    **kwargs: object,
                ) -> None:
                    del args, kwargs

                async def empty_attempts(
                    *args: object,
                    **kwargs: object,
                ) -> list[dict[str, Any]]:
                    del args, kwargs
                    return []

                async def run_self_case(
                    *,
                    case: Mapping[str, Any],
                    pipeline_run_handle: object = None,
                ) -> dict[str, Any]:
                    artifacts = (
                        await runner.build_self_cognition_case_artifacts_async(
                            case,
                            apply_consolidation=(
                                str(input_payload.get("revision")) != "v2"
                            ),
                            execute_private_actions=True,
                            pipeline_run_handle=pipeline_run_handle,
                        )
                    )
                    captured_artifacts.append(artifacts)
                    return artifacts

                worker_result = await worker.run_self_cognition_worker_tick(
                    now=datetime.fromisoformat(
                        fixed_clock["storage_timestamp_utc"]
                    ),
                    is_primary_interaction_busy=lambda: False,
                    character_profile={
                        **dict(profile),
                        "global_user_id": "character-global",
                    },
                    collect_cases_func=collect_cases,
                    read_attempts_func=empty_attempts,
                    record_attempt_func=no_op_calendar_run,
                    claim_calendar_run_func=claim_calendar_run,
                    complete_calendar_run_func=no_op_calendar_run,
                    fail_calendar_run_func=no_op_calendar_run,
                    skip_calendar_run_func=no_op_calendar_run,
                    defer_calendar_run_func=no_op_calendar_run,
                    adapter_registry_provider=(
                        lambda: service._adapter_registry
                    ),
                    run_case_func=run_self_case,
                    max_cases=1,
                )
                if settlements and not captured_artifacts:
                    settlement = settlements[-1]
                    graph_result = dict(
                        settlement.get("graph_result", {})
                    )
                    settled_trace = settlement["trace"]
                elif captured_artifacts:
                    artifact_payloads = captured_artifacts[-1]
                    cognition_output = artifact_payloads.get(
                        runner.models.ARTIFACT_COGNITION_OUTPUT,
                        {},
                    )
                    consolidation_key = getattr(
                        runner.models,
                        "RUNTIME_CONSOLIDATION_STATE",
                        None,
                    )
                    consolidation_state = (
                        artifact_payloads.get(consolidation_key, {})
                        if consolidation_key is not None
                        else {}
                    )
                    graph_result = {}
                    if isinstance(consolidation_state, Mapping):
                        graph_result.update(consolidation_state)
                    if isinstance(cognition_output, Mapping):
                        graph_result.update(cognition_output)
                    trace_key = getattr(
                        runner.models,
                        "RUNTIME_EPISODE_TRACE",
                        None,
                    )
                    settled_trace = (
                        artifact_payloads.get(trace_key)
                        if trace_key is not None
                        else None
                    )
                    if not isinstance(settled_trace, Mapping):
                        settled_trace = graph_result.get("episode_trace")
                    if not isinstance(settled_trace, Mapping):
                        settled_trace = {
                            "terminal_status": "succeeded",
                            "trace_id": "",
                        }
                    candidate_key = getattr(
                        runner.models,
                        "ARTIFACT_ACTION_CANDIDATE",
                        None,
                    )
                    candidate = (
                        artifact_payloads.get(candidate_key, {})
                        if candidate_key is not None
                        else {}
                    )
                    if isinstance(candidate, Mapping):
                        candidate_text = str(candidate.get("text") or "")
                        if candidate_text:
                            graph_result["final_dialog"] = [candidate_text]
                else:
                    raise ValueError(
                        f"scheduled worker produced no settlement: {worker_result}"
                    )
            else:
                artifacts = (
                    await runner.build_self_cognition_case_artifacts_async(
                        case,
                        apply_consolidation=False,
                        execute_private_actions=False,
                    )
                )
                episode_key = getattr(
                    runner.models,
                    "RUNTIME_COGNITIVE_EPISODE",
                    None,
                )
                episode = (
                    artifacts.get(episode_key)
                    if episode_key is not None
                    else None
                )
                cognition_output = artifacts.get(
                    runner.models.ARTIFACT_COGNITION_OUTPUT,
                    {},
                )
                consolidation_key = getattr(
                    runner.models,
                    "RUNTIME_CONSOLIDATION_STATE",
                    None,
                )
                consolidation_state = (
                    artifacts.get(consolidation_key, {})
                    if consolidation_key is not None
                    else {}
                )
                graph_result = {}
                if isinstance(consolidation_state, Mapping):
                    graph_result.update(consolidation_state)
                if isinstance(cognition_output, Mapping):
                    graph_result.update(cognition_output)
                settled_at = fixed_clock["storage_timestamp_utc"]
                settle_runtime = getattr(
                    service,
                    "_settle_runtime_episode_trace",
                    None,
                )
                if isinstance(episode, Mapping) and callable(settle_runtime):
                    settled_trace = await settle_runtime(
                        episode=episode,
                        graph_result=graph_result,
                        response_dialog=[],
                        delivery_tracking_id="",
                        settled_at=settled_at,
                    )
                    persist_lifecycle = getattr(
                        service,
                        "_persist_post_turn_lifecycle_record",
                        None,
                    )
                    if callable(persist_lifecycle):
                        await persist_lifecycle(
                            episode=episode,
                            state=graph_result,
                            fallback_trace=settled_trace,
                            delivery_tracking_id="",
                        )
                else:
                    settled_trace = graph_result.get("episode_trace")
                    if not isinstance(settled_trace, Mapping):
                        settled_trace = {
                            "terminal_status": "succeeded",
                            "trace_id": "",
                        }
                candidate_key = getattr(
                    runner.models,
                    "ARTIFACT_ACTION_CANDIDATE",
                    None,
                )
                candidate = (
                    artifacts.get(candidate_key, {})
                    if candidate_key is not None
                    else {}
                )
                if isinstance(candidate, Mapping):
                    candidate_text = str(candidate.get("text") or "")
                    if candidate_text:
                        graph_result["final_dialog"] = [candidate_text]
            final_dialog = [
                item
                for item in graph_result.get("final_dialog", [])
                if isinstance(item, str) and item.strip()
            ] if isinstance(graph_result.get("final_dialog"), list) else []
            if not final_dialog:
                final_dialog = [
                    str(call.get("text") or "")
                    for call in adapter.calls
                    if str(call.get("text") or "").strip()
                ]
            persisted_profile = await db.character_state.find_one(
                {"_id": "global"},
                {"_id": 0},
            )
            counts_after = await _collection_counts(db)
            private_monologue = str(
                graph_result.get("private_monologue")
                or graph_result.get("internal_monologue")
                or ""
            )
            return {
                "response": {
                    "messages": final_dialog,
                    "cognition_graph": None,
                },
                "trace_run": {
                    "status": settled_trace.get("terminal_status", ""),
                    "trace_id": settled_trace.get("trace_id", ""),
                },
                "trace_steps": [],
                "persisted_profile": _json_safe(persisted_profile),
                "graph_result": _json_safe(graph_result),
                "seeded_context": [],
                "adapter_calls": _json_safe(adapter.calls),
                "counts_before_turn": counts_before,
                "counts_after_turn": counts_after,
                "request": {
                    "source_case": _json_safe(case),
                },
                "private_monologue": private_monologue,
                "private_monologue_path": (
                    "self_cognition_worker.settlement.graph_result"
                    ".private_monologue"
                    if source_kind == "scheduled_tick"
                    else "self_cognition_artifacts"
                    ".cognition_output.private_monologue"
                ),
            }
    finally:
        if original_settle is not None:
            post_turn.settle_episode_trace = original_settle  # type: ignore[assignment]
        await _cleanup_database(input_payload, db)


async def _run_reflection_case(
    input_payload: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Run reflection affect settling outside the live wording path."""

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.db._client import get_db
    from kazusa_ai_chatbot.reflection_cycle.affect_settling import (
        run_daily_affect_settling,
    )

    db = None
    try:
        db = await _require_empty_database(input_payload)
        await _bootstrap_main_profile(input_payload, profile)
        async with service.lifespan(service.app):
            db = await get_db()
            state_seed = input_payload["case"].get("state_seed")
            seeded_character_state = (
                state_seed.get("character_state")
                if isinstance(state_seed, Mapping)
                else None
            )
            if isinstance(seeded_character_state, Mapping):
                await db.character_state.update_one(
                    {"_id": "global"},
                    {"$set": dict(seeded_character_state)},
                )
            counts_before = await _collection_counts(db)
            result = await run_daily_affect_settling(
                settling_local_date="2026-07-24",
                dry_run=False,
                enable_character_state_write=True,
                character_state_refresh_callback=(
                    service._refresh_runtime_character_state
                ),
            )
            counts_after = await _collection_counts(db)
            return {
                "response": {"messages": [], "cognition_graph": None},
                "trace_run": {
                    "status": "succeeded",
                    "trace_id": "",
                },
                "trace_steps": [],
                "persisted_profile": _json_safe(
                    await db.character_state.find_one(
                        {"_id": "global"},
                        {"_id": 0},
                    )
                ),
                "graph_result": {"reflection_result": _json_safe(result)},
                "seeded_context": [],
                "adapter_calls": [],
                "counts_before_turn": counts_before,
                "counts_after_turn": counts_after,
                "request": {"source_case": _json_safe(input_payload["case"])},
                "private_monologue": "",
                "private_monologue_path": "",
            }
    finally:
        await _cleanup_database(input_payload, db)


async def _run_background_result_case(
    input_payload: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Deliver one completed tool result through the public runtime tick."""

    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.background_work.runtime import (
        run_background_work_runtime_tick,
    )
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.db import (
        create_user_profile,
        insert_background_work_job,
    )
    from kazusa_ai_chatbot.db._client import get_db
    from kazusa_ai_chatbot.time_boundary import build_turn_clock

    fixture_case = input_payload["case"]
    case_id = str(fixture_case["case_id"])
    now = build_turn_clock(
        str(input_payload["fixed_local_timestamp"])
    )["storage_timestamp_utc"]
    requester_global_id = "baseline-current-user"
    requester_platform_id = "baseline-current-user-platform"
    job = {
        "schema_version": "background_work_job.v1",
        "job_id": f"baseline-job-{case_id}",
        "idempotency_key": f"baseline-idempotency-{case_id}",
        "source_action_attempt_id": f"baseline-attempt-{case_id}",
        "accepted_task_id": "baseline-run-011",
        "task_identity_key": f"baseline-task-identity-{case_id}",
        "status": "completed",
        "delivery_state": "ready",
        "task_brief": str(fixture_case.get("input_text") or ""),
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 4000,
        "source_platform": "debug",
        "source_channel_id": f"baseline-tool-{case_id}",
        "source_channel_type": "private",
        "source_message_id": case_id,
        "source_platform_bot_id": "baseline-character-platform",
        "source_character_name": str(profile.get("name") or ""),
        "requester_global_user_id": requester_global_id,
        "requester_platform_user_id": requester_platform_id,
        "requester_display_name": "基线测试用户",
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
        "lease_owner": None,
        "lease_expires_at": None,
        "attempt_count": 1,
        "max_attempts": 3,
        "router_action": "execute",
        "worker": "coding_agent",
        "routed_task": "tool_result",
        "router_reason": "frozen completed result fixture",
        "source_context": "",
        "requested_worker": "coding_agent",
        "worker_payload": {},
        "artifact_text": str(fixture_case.get("input_text") or ""),
        "artifact_char_count": len(str(fixture_case.get("input_text") or "")),
        "failure_summary": "",
        "result_summary": _background_result_summary(fixture_case),
        "worker_metadata": {},
        "delivery_attempt_count": 0,
        "delivery_failure_summary": "",
        "delivery_tracking_id": "",
        "delivered_conversation_message_id": "",
        "delivered_at": "",
    }
    adapter = _WorkerDebugAdapter()
    settlements: list[dict[str, Any]] = []
    cognition_results: list[dict[str, Any]] = []
    original_settle = getattr(post_turn, "settle_episode_trace", None)
    original_persona_supervisor2 = getattr(
        service,
        "persona_supervisor2",
        None,
    )

    def capture_settlement(**kwargs: object) -> Mapping[str, Any]:
        if original_settle is None:
            return {}
        trace = original_settle(**kwargs)
        graph_result = kwargs.get("graph_result")
        settlements.append({
            "trace": deepcopy(trace),
            "graph_result": deepcopy(graph_result)
            if isinstance(graph_result, Mapping) else {},
        })
        return trace

    async def capture_persona_supervisor2(
        *args: object,
        **kwargs: object,
    ) -> Mapping[str, Any]:
        if not callable(original_persona_supervisor2):
            return {}
        result = await original_persona_supervisor2(*args, **kwargs)
        if isinstance(result, Mapping):
            cognition_results.append(deepcopy(dict(result)))
        return result

    if original_settle is not None:
        post_turn.settle_episode_trace = capture_settlement  # type: ignore[assignment]
    if callable(original_persona_supervisor2):
        service.persona_supervisor2 = capture_persona_supervisor2  # type: ignore[assignment]
    db = None
    try:
        db = await _require_empty_database(input_payload)
        await _bootstrap_main_profile(input_payload, profile)
        async with service.lifespan(service.app):
            service.register_runtime_adapter(adapter)
            db = await get_db()
            if await db.user_profiles.find_one(
                {"global_user_id": requester_global_id},
                {"_id": 1},
            ) is None:
                await create_user_profile({
                    "global_user_id": requester_global_id,
                })
            await insert_background_work_job(job)
            counts_before = await _collection_counts(db)
            runtime_result = await run_background_work_runtime_tick(
                is_primary_interaction_busy=lambda: False,
                deliver_result_episode_func=(
                    service._deliver_accepted_task_result_episode
                ),
            )
            counts_after = await _collection_counts(db)
            trace_rows = await db.llm_trace_runs.find(
                {"platform_message_id": case_id},
                {"_id": 0},
            ).sort("created_at", -1).to_list(length=1)
            latest_trace = trace_rows[0] if trace_rows else {
                "status": "succeeded",
                "trace_id": "",
            }
            graph_result = (
                settlements[-1]["graph_result"]
                if settlements else {}
            )
            if not graph_result and cognition_results:
                graph_result = cognition_results[-1]
            private_monologue, private_monologue_path = (
                _extract_background_monologue(
                    graph_result
                    if isinstance(graph_result, Mapping) else {}
                )
            )
            messages = [
                str(call.get("text") or "")
                for call in adapter.calls
                if str(call.get("text") or "").strip()
            ]
            return {
                "response": {
                    "messages": messages,
                    "cognition_graph": None,
                },
                "trace_run": _json_safe(latest_trace),
                "trace_steps": [],
                "persisted_profile": _json_safe(
                    await db.character_state.find_one(
                        {"_id": "global"},
                        {"_id": 0},
                    )
                ),
                "graph_result": _json_safe({
                    "background_runtime": runtime_result,
                    "settlement": graph_result,
                }),
                "seeded_context": [],
                "adapter_calls": _json_safe(adapter.calls),
                "counts_before_turn": counts_before,
                "counts_after_turn": counts_after,
                "request": {"source_case": _json_safe(fixture_case)},
                "expected_delivery_text": str(
                    fixture_case.get("input_text")
                    or job["result_summary"]
                ),
                "private_monologue": private_monologue,
                "private_monologue_path": private_monologue_path,
            }
    finally:
        if original_settle is not None:
            post_turn.settle_episode_trace = original_settle  # type: ignore[assignment]
        if callable(original_persona_supervisor2):
            service.persona_supervisor2 = original_persona_supervisor2  # type: ignore[assignment]
        await _cleanup_database(input_payload, db)


async def _collection_counts(db: Any) -> dict[str, int]:
    """Count rows in each collection for before/after evidence."""

    names = await db.list_collection_names()
    return {
        str(name): int(await db[str(name)].count_documents({}))
        for name in sorted(names)
    }


class _WorkerDebugAdapter:
    """In-memory adapter used to capture delivery without external effects."""

    platform = "debug"
    display_name = "Baseline Replay Character"

    def __init__(self) -> None:
        self.platform_bot_id = "baseline-character-platform"
        self.calls: list[dict[str, Any]] = []

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
            message_id=f"baseline-delivery-{uuid4().hex}",
            sent_at=datetime.now(timezone.utc),
        )


async def _run_chat_case(
    input_payload: Mapping[str, Any],
    *,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    """Run a chat case through the public FastAPI chat boundary."""

    from fastapi import BackgroundTasks
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.brain_service import post_turn
    from kazusa_ai_chatbot.db._client import get_db

    case = input_payload["case"]
    request = _build_request(
        case,
        fixed_local_timestamp=str(input_payload["fixed_local_timestamp"]),
        character_global_id="character-global",
    )
    graph_results: list[dict[str, Any]] = []
    settlements: list[dict[str, Any]] = []
    original_settle = getattr(post_turn, "settle_episode_trace", None)
    original_runtime_settle = getattr(
        service,
        "_settle_runtime_episode_trace",
        None,
    )

    def capture_settle(**kwargs: object) -> Mapping[str, Any]:
        if original_settle is None:
            return_value: Mapping[str, Any] = {}
            return return_value
        result = original_settle(**kwargs)
        graph_result = kwargs.get("graph_result")
        settlements.append({
            "trace": deepcopy(result),
            "graph_result": deepcopy(graph_result)
            if isinstance(graph_result, Mapping) else {},
        })
        return result

    async def capture_runtime_settle(**kwargs: object) -> Mapping[str, Any]:
        graph_result = kwargs.get("graph_result")
        if isinstance(graph_result, Mapping):
            graph_results.append(deepcopy(dict(graph_result)))
        if original_runtime_settle is None:
            return {}
        return await original_runtime_settle(**kwargs)

    if original_settle is not None:
        post_turn.settle_episode_trace = capture_settle  # type: ignore[assignment]
    if original_runtime_settle is not None:
        service._settle_runtime_episode_trace = (  # type: ignore[assignment]
            capture_runtime_settle
        )
    db = None
    try:
        db = await _require_empty_database(input_payload)
        await _bootstrap_main_profile(input_payload, profile)
        adapter = _WorkerDebugAdapter()
        async with service.lifespan(service.app):
            service.register_runtime_adapter(adapter)
            db = await get_db()
            from kazusa_ai_chatbot.db.users import resolve_global_user_id

            current_author_global_user_id = await resolve_global_user_id(
                request.platform,
                request.platform_user_id,
                request.display_name,
            )
            seeded = await _seed_conversation_rows(
                case=case,
                channel_id=request.platform_channel_id,
                fixed_local_timestamp=str(
                    input_payload["fixed_local_timestamp"]
                ),
                current_author_global_user_id=current_author_global_user_id,
                current_author_platform_user_id=request.platform_user_id,
            )
            seeded_coding_run = _build_seeded_coding_task_document(
                case=case,
                fixed_local_timestamp=str(
                    input_payload["fixed_local_timestamp"]
                ),
                source_platform=request.platform,
                source_channel_id=request.platform_channel_id,
                source_channel_type=request.channel_type,
                source_message_id=request.platform_message_id,
                source_platform_bot_id=request.platform_bot_id,
                source_character_name=str(profile.get("name") or ""),
                requester_global_user_id=current_author_global_user_id,
                requester_platform_user_id=request.platform_user_id,
                requester_display_name=request.display_name,
            )
            if seeded_coding_run is not None:
                from kazusa_ai_chatbot.accepted_task.models import (
                    ACCEPTED_TASKS_COLLECTION,
                )

                await db[ACCEPTED_TASKS_COLLECTION].insert_one(
                    seeded_coding_run,
                )
            counts_before_turn = await _collection_counts(db)
            response = await service.chat(request, BackgroundTasks())
            trace_run = await _wait_for_trace_run(
                db,
                platform_message_id=request.platform_message_id,
            )
            response_payload = response.model_dump(mode="json")
            trace_id = str(trace_run.get("trace_id") or "")
            trace_steps = await db.llm_trace_steps.find(
                {"trace_id": trace_id},
                {"_id": 0},
            ).sort("sequence", 1).to_list(length=None)
            persisted_profile = await db.character_state.find_one(
                {"_id": "global"},
                {"_id": 0},
            )
            counts_after_turn = await _collection_counts(db)
            graph_result = (
                graph_results[-1]
                if graph_results
                else settlements[-1]["graph_result"]
                if settlements
                else response_payload.get("cognition_graph")
                if isinstance(response_payload.get("cognition_graph"), Mapping)
                else {}
            )
            return {
                "response": _json_safe(response_payload),
                "trace_run": _json_safe(trace_run),
                "trace_steps": _json_safe(trace_steps),
                "persisted_profile": _json_safe(persisted_profile),
                "graph_result": _json_safe(graph_result),
                "seeded_context": _json_safe(seeded),
                "seeded_coding_run": _json_safe(seeded_coding_run),
                "adapter_calls": _json_safe(adapter.calls),
                "counts_before_turn": counts_before_turn,
                "counts_after_turn": counts_after_turn,
                "request": _json_safe(request.model_dump(mode="json")),
            }
    finally:
        if original_settle is not None:
            post_turn.settle_episode_trace = original_settle  # type: ignore[assignment]
        if original_runtime_settle is not None:
            service._settle_runtime_episode_trace = (  # type: ignore[assignment]
                original_runtime_settle
            )
        await _cleanup_database(input_payload, db)


async def _execute(input_payload: Mapping[str, Any]) -> int:
    """Validate worker boundaries and execute one case."""

    output_path = Path(str(input_payload["output_path"])).resolve()
    target_root = Path(str(input_payload["target_root"])).resolve()
    workspace_root = Path(str(input_payload["workspace_root"])).resolve()
    profile_path = Path(str(input_payload["profile_path"])).resolve()
    history_path = Path(str(input_payload["history_path"])).resolve()
    database_name = str(input_payload["database_name"])
    base_artifact = {
        "schema_version": "cognition_baseline_worker_evidence.v1",
        "execution_id": str(input_payload.get("execution_id") or ""),
        "corpus": str(input_payload.get("corpus") or ""),
        "revision": str(input_payload.get("revision") or ""),
        "revision_sha": str(input_payload.get("revision_sha") or ""),
        "profile_sha256": str(input_payload.get("profile_sha256") or ""),
        "history_sha256": str(input_payload.get("history_sha256") or ""),
        "case_sha256": str(input_payload.get("case_sha256") or ""),
        "target_root": str(target_root),
        "database_name": database_name,
        "case_id": str(
            input_payload.get("case", {}).get("case_id") or ""
        ),
        "output_mode": str(
            input_payload.get("case", {}).get("output_mode") or ""
        ),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    target_before = _validate_worker_structure(
        input_payload,
        target_root=target_root,
        workspace_root=workspace_root,
        profile_path=profile_path,
        history_path=history_path,
    )
    missing_env = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing_env:
        _write_json(output_path, {
            **base_artifact,
            "technical_status": "blocked_environment",
            "target_git_before": target_before,
            "missing_environment": missing_env,
            "blocking_reason": (
                "explicit live-LLM/Mongo environment is incomplete; "
                "the controller did not provide all configured .env values"
            ),
        })
        print(json.dumps({
            "technical_status": "blocked_environment",
            "missing_environment": missing_env,
        }, ensure_ascii=False, indent=2))
        return 2
    profile = _load_json(profile_path)
    target_source = target_root / "src"
    sys.path.insert(0, str(target_source))
    case = input_payload.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("worker input case is missing")
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace_before = _filesystem_digest(workspace_root)
    started_at = time.perf_counter()
    try:
        event_kind = str(case.get("event_kind") or "chat")
        source_kind = str(case.get("source_kind") or "user_message")
        if event_kind == "chat":
            live_result = await _run_chat_case(
                input_payload,
                profile=profile,
            )
        elif source_kind in {
            "internal_thought",
            "self_cognition",
            "scheduled_tick",
            "proactive_proposal",
        }:
            live_result = await _run_self_cognition_case(
                input_payload,
                profile=profile,
            )
        elif source_kind == "reflection":
            live_result = await _run_reflection_case(
                input_payload,
                profile=profile,
            )
        elif source_kind == "tool_result":
            live_result = await _run_background_result_case(
                input_payload,
                profile=profile,
            )
        else:
            raise ValueError(
                f"unsupported public source kind: {source_kind}"
            )
        response_payload = live_result["response"]
        if not isinstance(response_payload, Mapping):
            raise ValueError("worker response is not an object")
        graph_result = live_result.get("graph_result")
        graph_result_mapping = (
            graph_result if isinstance(graph_result, Mapping) else {}
        )
        raw_adapter_calls = live_result.get("adapter_calls", [])
        adapter_calls = [
            call
            for call in raw_adapter_calls
            if isinstance(call, Mapping)
        ] if isinstance(raw_adapter_calls, list) else []
        delivered_messages = [
            str(call.get("text") or "")
            for call in adapter_calls
            if str(call.get("text") or "").strip()
        ]
        messages = response_payload.get("messages")
        if (
            not isinstance(messages, list) or not messages
        ) and delivered_messages:
            response_payload = dict(response_payload)
            response_payload["messages"] = delivered_messages
            messages = delivered_messages
        monologue = str(live_result.get("private_monologue") or "")
        monologue_path = str(live_result.get("private_monologue_path") or "")
        if not monologue:
            monologue, monologue_path = _extract_canonical_monologue(
                response_payload
            )
        if not monologue and event_kind == "self_cognition":
            monologue, monologue_path = _extract_background_monologue(
                graph_result_mapping
            )
        surface_status = _response_surface_status(response_payload)
        visible_messages = messages if isinstance(messages, list) else []
        final_cognition_monologue = _extract_final_cognition_monologue(
            graph_result_mapping
        )
        output_mode = str(case.get("output_mode") or "visible")
        hard_gate_failures: list[str] = []
        hard_gate_failures.extend(_immutable_external_failures(case))
        source_leaks = _source_leaks(
            case,
            {
                "effective_input": case.get("effective_input"),
                "effective_context": case.get("effective_context"),
                "response": response_payload,
                "monologue": monologue,
            },
        )
        if source_leaks:
            hard_gate_failures.append("source identity leaked")
        if _requires_live_monologue(
            source_kind=source_kind,
            output_mode=output_mode,
        ) and not monologue:
            hard_gate_failures.append("private monologue is missing")
        if output_mode != "silent":
            if not isinstance(messages, list):
                hard_gate_failures.append("response messages is not a list")
        if output_mode == "private" and (
            isinstance(messages, list) and messages
        ):
            hard_gate_failures.append(
                "private source produced visible dialog"
            )
        if (
            str(input_payload.get("revision")) == "v2"
            and _requires_live_monologue(
                source_kind=source_kind,
                output_mode=output_mode,
            )
        ):
            if not final_cognition_monologue:
                hard_gate_failures.append(
                    "V2 final cognition_core_output.private_monologue missing"
                )
            elif final_cognition_monologue != monologue:
                hard_gate_failures.append(
                    "V2 final cognition monologue differs from canonical node"
                )
        trace_run = live_result.get("trace_run")
        if not isinstance(trace_run, Mapping):
            hard_gate_failures.append("trace run evidence missing")
        elif not _trace_status_is_valid(
            output_mode=output_mode,
            trace_run=trace_run,
            source_kind=str(case.get("source_kind") or ""),
        ):
            hard_gate_failures.append("trace run did not succeed")
        target_after = _target_git_state(target_root)
        workspace_after = _filesystem_digest(workspace_root)
        if target_after != target_before:
            hard_gate_failures.append("target worktree changed")
        counts_before = live_result.get("counts_before_turn", {})
        counts_after = live_result.get("counts_after_turn", {})
        counts_before_mapping = (
            counts_before if isinstance(counts_before, Mapping) else {}
        )
        counts_after_mapping = (
            counts_after if isinstance(counts_after, Mapping) else {}
        )
        gate_failures, gate_results = _evaluate_hard_gates(
            input_payload,
            case,
            response_payload=response_payload,
            monologue=monologue,
            monologue_path=monologue_path,
            graph_result=graph_result_mapping,
            persisted_profile=(
                live_result.get("persisted_profile")
                if isinstance(live_result.get("persisted_profile"), Mapping)
                else None
            ),
            adapter_calls=adapter_calls,
            counts_before=counts_before_mapping,
            counts_after=counts_after_mapping,
            workspace_before=workspace_before,
            workspace_after=workspace_after,
            expected_delivery_text=str(
                live_result.get("expected_delivery_text") or ""
            ),
        )
        hard_gate_failures.extend(gate_failures)
        profile_failures = _profile_matches(
            live_result.get("persisted_profile"),
            profile,
        )
        hard_gate_failures.extend(profile_failures)
        artifact = {
            **base_artifact,
            "technical_status": "passed" if not hard_gate_failures else "failed",
            "case": _json_safe(case),
            "request": live_result.get("request"),
            "response": response_payload,
            "response_surface_status": surface_status,
            "visible_messages": visible_messages,
            "visible_message_count": len(visible_messages),
            "private_monologue": monologue,
            "private_monologue_path": monologue_path,
            "final_cognition_core_private_monologue": final_cognition_monologue,
            "trace_run": live_result.get("trace_run"),
            "trace_steps": live_result.get("trace_steps", []),
            "graph_result": live_result.get("graph_result", {}),
            "persisted_profile": live_result.get("persisted_profile"),
            "seeded_context": live_result.get("seeded_context", []),
            "seeded_coding_run": live_result.get("seeded_coding_run"),
            "adapter_calls": live_result.get("adapter_calls", []),
            "expected_delivery_text": live_result.get(
                "expected_delivery_text",
                "",
            ),
            "counts_before_turn": live_result.get("counts_before_turn", {}),
            "counts_after_turn": live_result.get("counts_after_turn", {}),
            "target_git_before": target_before,
            "target_git_after": target_after,
            "workspace_before": workspace_before,
            "workspace_after": workspace_after,
            "source_leaks": source_leaks,
            "hard_gate_results": gate_results,
            "hard_gate_failures": hard_gate_failures,
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        }
    except Exception as exc:
        artifact = {
            **base_artifact,
            "technical_status": "failed",
            "failure_type": exc.__class__.__name__,
            "failure_message": str(exc),
            "failure_traceback": traceback.format_exc(),
            "target_git_before": target_before,
            "workspace_before": workspace_before,
            "hard_gate_failures": ["worker exception"],
            "duration_ms": round((time.perf_counter() - started_at) * 1000),
        }
    _write_json(output_path, artifact)
    print(json.dumps({
        "execution_id": artifact["execution_id"],
        "technical_status": artifact["technical_status"],
        "case_id": artifact["case_id"],
        "surface": artifact.get("response_surface_status", ""),
        "visible_message_count": artifact.get("visible_message_count", 0),
        "hard_gate_failures": artifact.get("hard_gate_failures", []),
        "private_monologue": artifact.get("private_monologue", ""),
        "visible_messages": artifact.get("visible_messages", []),
        "artifact": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0 if artifact["technical_status"] == "passed" else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the worker command-line parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one worker case."""

    args = _build_parser().parse_args(argv)
    input_payload = _load_json(args.input.resolve())
    return asyncio.run(_execute(input_payload))


if __name__ == "__main__":
    sys.exit(main())
