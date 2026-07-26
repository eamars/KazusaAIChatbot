"""Full 20-turn Asuna private R18 conversation E2E test."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any
from uuid import uuid4

import httpx
import pytest


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm, pytest.mark.live_db]

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2"
    / "private_r18_replay"
    / "replay_manifest.json"
)
_ARTIFACT_ROOT = (
    _ROOT
    / "test_artifacts"
    / "cognition_core_v2"
    / "asuna_private_r18_affinity_replay"
)
_TEST_DATABASE_NAME = "_test_kazusa_live_llm"
_CHARACTER_TIME_ZONE = "Pacific/Auckland"
_REPLAY_BOT_PLATFORM_ID = "3768713357"
_TRACE_WAIT_SECONDS = 600.0
_CONDITIONS = frozenset({"high_affinity", "default_affinity"})
_RELATIONSHIP_FIELDS = (
    "familiarity",
    "positive_regard",
    "trust",
    "attachment",
    "desired_closeness",
    "perceived_closeness",
    "care",
    "boundary_safety",
    "exclusivity",
    "unresolved_injury",
    "salience",
)
_DEFAULT_RELATIONSHIP_VALUES = {
    "familiarity": 10,
    "positive_regard": 0,
    "trust": 0,
    "attachment": 0,
    "desired_closeness": 10,
    "perceived_closeness": 10,
    "care": 0,
    "boundary_safety": 0,
    "exclusivity": 0,
    "unresolved_injury": 0,
    "salience": 0,
}


class FatalSequenceError(RuntimeError):
    """Raised when the E2E sequence cannot safely advance to the next turn."""


def _configure_utf8_streams() -> None:
    """Keep exact CJK inputs and live model output printable on Windows."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


_configure_utf8_streams()


def _json_safe(value: object) -> object:
    """Convert Mongo, Pydantic, and model values into JSON evidence."""

    if isinstance(value, Mapping):
        json_value = {
            str(key): _json_safe(item)
            for key, item in value.items()
        }
    elif isinstance(value, (list, tuple)):
        json_value = [_json_safe(item) for item in value]
    elif isinstance(value, (str, int, float, bool)) or value is None:
        json_value = value
    else:
        isoformat = getattr(value, "isoformat", None)
        if callable(isoformat):
            json_value = str(isoformat())
        else:
            json_value = str(value)
    return json_value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write one UTF-8 test evidence object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
    """Require one mapping-shaped fixture field."""

    if not isinstance(value, Mapping):
        raise ValueError(f"fixture field is not an object: {field_name}")
    return value


def _manifest_cases(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Validate and return the exact twenty source-message cases."""

    if manifest.get("schema_version") != "real_conversation_replay.v2":
        raise ValueError("private R18 manifest schema is invalid")
    if manifest.get("scenario") != "private_r18":
        raise ValueError("private R18 manifest scenario is invalid")
    source = _mapping(manifest.get("source"), "source")
    for field_name in (
        "platform",
        "platform_channel_id",
        "source_user_global_id",
        "source_user_platform_id",
        "character_global_id",
    ):
        value = source.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"private R18 source field is invalid: {field_name}")
    cases_value = manifest.get("cases")
    if not isinstance(cases_value, list) or len(cases_value) != 20:
        raise ValueError("private R18 manifest must contain exactly 20 cases")
    cases = [
        _mapping(case, "cases[]")
        for case in cases_value
    ]
    indexes = [case.get("case_index") for case in cases]
    if indexes != list(range(1, 21)):
        raise ValueError("private R18 cases are not chronological")
    for case in cases:
        message = _mapping(case.get("source_message"), "source_message")
        body_text = message.get("body_text")
        timestamp = message.get("timestamp")
        if not isinstance(body_text, str) or not body_text.strip():
            raise ValueError("private R18 source message body is invalid")
        if not isinstance(timestamp, str) or not timestamp.strip():
            raise ValueError("private R18 source message timestamp is invalid")
        if message.get("role") != "user":
            raise ValueError("private R18 sequence contains a non-user input")
    return cases


def _load_manifest() -> dict[str, Any]:
    """Load the canonical private R18 input manifest."""

    manifest = _load_json_object(_MANIFEST_PATH)
    return manifest


def _condition_name() -> str:
    """Return the affinity condition selected for this child process."""

    condition = os.environ.get("ASUNA_R18_CONDITION", "").strip()
    if condition not in _CONDITIONS:
        raise ValueError(
            "ASUNA_R18_CONDITION must be high_affinity or default_affinity"
        )
    return condition


def _run_id() -> str:
    """Return the parent-assigned condition run identifier."""

    value = os.environ.get("ASUNA_R18_RUN_ID", "").strip()
    if not value:
        value = f"manual_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    return value


def _guarded_path(path: Path, field_name: str = "path") -> Path:
    """Require an evidence path to stay below the replay artifact root."""

    candidate = path.resolve()
    guard_root = _ARTIFACT_ROOT.resolve()
    if candidate == guard_root or guard_root not in candidate.parents:
        raise ValueError(f"{field_name} escaped the replay artifact root")
    return candidate


def _output_root(*, run_id: str, condition: str) -> Path:
    """Return the guarded output directory for one complete sequence."""

    configured = os.environ.get("ASUNA_R18_OUTPUT_ROOT", "").strip()
    if configured:
        output_root = _guarded_path(
            Path(configured),
            "ASUNA_R18_OUTPUT_ROOT",
        )
    else:
        output_root = _guarded_path(_ARTIFACT_ROOT / run_id / condition)
    return output_root


def _storage_timestamp() -> str:
    """Return a cognition-contract-compatible UTC timestamp."""

    timestamp = datetime.now(timezone.utc).isoformat(
        timespec="milliseconds",
    ).replace(
        "+00:00",
        "Z",
    )
    return timestamp


def _relationship_seed_values(
    condition: str,
    manifest: Mapping[str, Any],
) -> dict[str, int]:
    """Return only the native relationship axes for one condition."""

    if condition == "default_affinity":
        values = dict(_DEFAULT_RELATIONSHIP_VALUES)
    else:
        if condition != "high_affinity":
            raise ValueError(f"unsupported affinity condition: {condition}")
        fixture = _mapping(manifest.get("frozen_fixture"), "frozen_fixture")
        axes = _mapping(
            fixture.get("relationship_axes"),
            "relationship_axes",
        )
        values = {}
        for field_name in _RELATIONSHIP_FIELDS:
            value = axes.get(field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"relationship axis is invalid: {field_name}"
                )
            if not 0 <= value <= 100:
                raise ValueError(
                    f"relationship axis is out of range: {field_name}"
                )
            values[field_name] = value
    return values


def _prepare_runtime_environment(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Validate the adult profile and guarded live-test environment."""

    if os.environ.get("MONGODB_DB_NAME", "").strip() != _TEST_DATABASE_NAME:
        raise AssertionError(
            "Asuna R18 E2E requires the exact guarded database "
            f"{_TEST_DATABASE_NAME!r}"
        )
    if os.environ.get("KAZUSA_TEST_DB_GUARD") != "1":
        raise AssertionError("Asuna R18 E2E requires KAZUSA_TEST_DB_GUARD=1")
    from kazusa_ai_chatbot.character_profile import (
        load_packaged_character_profile_seed,
    )

    profile = load_packaged_character_profile_seed()
    age = profile.get("age")
    if isinstance(age, bool) or not isinstance(age, int) or age < 18:
        raise AssertionError("Asuna R18 E2E requires an adult profile")
    source = _mapping(manifest.get("source"), "source")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["CHARACTER_GLOBAL_USER_ID"] = str(
        source["character_global_id"]
    )
    os.environ["CHARACTER_TIME_ZONE"] = _CHARACTER_TIME_ZONE
    for variable_name in (
        "SELF_COGNITION_ENABLED",
        "CALENDAR_SCHEDULER_ENABLED",
        "BACKGROUND_WORK_WORKER_ENABLED",
        "REFLECTION_CYCLE_ENABLED",
    ):
        os.environ[variable_name] = "false"
    runtime = {
        "profile_name": str(profile.get("name", "")),
        "character_global_id": str(source["character_global_id"]),
        "user_global_id": str(source["source_user_global_id"]),
        "user_platform_id": str(source["source_user_platform_id"]),
        "channel_id": str(source["platform_channel_id"]),
        "platform": str(source["platform"]),
    }
    return runtime


def _reply_target(value: object) -> dict[str, str] | None:
    """Project an input reply envelope without changing its text."""

    if not isinstance(value, Mapping) or not value:
        return None
    field_map = {
        "reply_to_message_id": "platform_message_id",
        "reply_to_platform_user_id": "platform_user_id",
        "reply_to_global_user_id": "global_user_id",
        "reply_to_display_name": "display_name",
        "reply_excerpt": "excerpt",
    }
    reply: dict[str, str] = {}
    for source_name, target_name in field_map.items():
        field_value = value.get(source_name)
        if isinstance(field_value, str) and field_value.strip():
            reply[target_name] = field_value
    if not reply:
        return None
    reply["derivation"] = "platform_native"
    return reply


def _build_request(
    *,
    case: Mapping[str, Any],
    runtime: Mapping[str, str],
) -> Any:
    """Build the public ChatRequest from one exact user input message."""

    from kazusa_ai_chatbot.brain_service.contracts import ChatRequest

    source = _mapping(case.get("source_message"), "source_message")
    mentions_value = source.get("mentions")
    attachments_value = source.get("attachments")
    addressed_value = source.get("addressed_to_global_user_ids")
    mentions = [
        dict(item)
        for item in mentions_value
        if isinstance(item, Mapping)
    ] if isinstance(mentions_value, list) else []
    attachments = [
        dict(item)
        for item in attachments_value
        if isinstance(item, Mapping)
    ] if isinstance(attachments_value, list) else []
    addressed = [
        str(item)
        for item in addressed_value
        if str(item).strip()
    ] if isinstance(addressed_value, list) else []
    body_text = str(source["body_text"])
    envelope = {
        "body_text": body_text,
        "raw_wire_text": str(source.get("raw_wire_text") or body_text),
        "mentions": mentions,
        "reply": _reply_target(source.get("reply_context")),
        "attachments": attachments,
        "addressed_to_global_user_ids": addressed,
        "broadcast": bool(source.get("broadcast", False)),
    }
    request = ChatRequest.model_validate({
        "platform": str(source.get("platform") or runtime["platform"]),
        "platform_channel_id": str(
            source.get("platform_channel_id") or runtime["channel_id"]
        ),
        "channel_type": str(source.get("channel_type") or "private"),
        "platform_message_id": str(source["platform_message_id"]),
        "platform_user_id": str(source["platform_user_id"]),
        "platform_bot_id": _REPLAY_BOT_PLATFORM_ID,
        "display_name": str(source.get("display_name") or "replay-user"),
        "channel_name": f"QQ private {runtime['channel_id']}",
        "content_type": str(source.get("content_type") or "text"),
        "message_envelope": envelope,
        "local_timestamp": "",
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
    })
    return request


async def _reset_database() -> Any:
    """Drop and recreate only the reserved live-test database."""

    from kazusa_ai_chatbot.db import db_bootstrap
    from kazusa_ai_chatbot.db._client import get_db

    database = await get_db()
    if database.name != _TEST_DATABASE_NAME:
        raise AssertionError("database guard returned an unexpected database")
    for collection_name in await database.list_collection_names():
        await database[collection_name].drop()
    await db_bootstrap()
    return database


async def _collection_counts(database: Any) -> dict[str, int]:
    """Return document counts for all currently materialized collections."""

    names = sorted(await database.list_collection_names())
    counts = {
        name: int(await database[name].count_documents({}))
        for name in names
    }
    return counts


async def _seed_empty_condition_baseline(
    *,
    manifest: Mapping[str, Any],
    condition: str,
    runtime: Mapping[str, str],
) -> dict[str, Any]:
    """Seed only identity and affinity; the twenty inputs create all history."""

    from kazusa_ai_chatbot.cognition_core_v2.state_models import (
        build_acquaintance_user_state,
    )
    from kazusa_ai_chatbot.db import create_user_profile

    database = await _reset_database()
    relationship_values = _relationship_seed_values(condition, manifest)
    state_timestamp = _storage_timestamp()
    cognition_state = build_acquaintance_user_state(
        global_user_id=runtime["user_global_id"],
        updated_at=state_timestamp,
    )
    relationship = cognition_state["relationship"]
    for field_name, field_value in relationship_values.items():
        relationship[field_name] = field_value
    await create_user_profile({
        "global_user_id": runtime["user_global_id"],
        "platform_accounts": [{
            "platform": runtime["platform"],
            "platform_user_id": runtime["user_platform_id"],
            "display_name": "R18 E2E User",
            "linked_at": state_timestamp,
        }],
        "suspected_aliases": [],
        "cognition_state": cognition_state,
    })
    counts = await _collection_counts(database)
    for collection_name in (
        "conversation_history",
        "user_memory_units",
        "internal_monologue_residue",
    ):
        if counts.get(collection_name, 0) != 0:
            raise AssertionError(
                "empty E2E baseline unexpectedly contains rows: "
                f"{collection_name}={counts.get(collection_name)}"
            )
    source_cases = _manifest_cases(manifest)
    input_digest = hashlib.sha256(
        json.dumps(
            [
                _mapping(case["source_message"], "source_message")
                for case in source_cases
            ],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    baseline = {
        "schema_version": "asuna_private_r18_empty_e2e_baseline.v1",
        "database_name": database.name,
        "source_inputs_only": True,
        "history_seeded": 0,
        "memory_seeded": 0,
        "residue_seeded": 0,
        "profile_seeded": 1,
        "relationship": deepcopy(relationship),
        "input_digest": input_digest,
        "counts_after_seed": counts,
    }
    return baseline


def _relationship_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the native relationship state from one user profile."""

    cognition_state = _mapping(profile.get("cognition_state"), "cognition_state")
    relationship = _mapping(cognition_state.get("relationship"), "relationship")
    relationship_state = dict(relationship)
    return relationship_state


def _state_from_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the complete native user cognition state."""

    state = _mapping(profile.get("cognition_state"), "cognition_state")
    cognition_state = dict(state)
    return cognition_state


def _response_surface(response: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the returned surface without classifying model wording."""

    operational_error = response.get("operational_error")
    messages = response.get("messages")
    if isinstance(operational_error, Mapping):
        response_surface = {
            "status": "operational_error",
            "error_code": str(operational_error.get("error_code", "")),
            "message_count": len(messages) if isinstance(messages, list) else 0,
        }
    elif isinstance(messages, list) and messages:
        response_surface = {
            "status": "visible_dialog",
            "error_code": "",
            "message_count": len(messages),
        }
    else:
        response_surface = {
            "status": "no_visible_dialog",
            "error_code": "",
            "message_count": 0,
        }
    return response_surface


def _trace_dispositions(trace_steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """Project observed relevance/boundary dispositions for review."""

    dispositions: list[str] = []
    for step in trace_steps:
        parsed = step.get("parsed_output")
        if not isinstance(parsed, Mapping):
            continue
        for field_name in (
            "semantic_disposition",
            "response_action",
            "intake_action",
        ):
            value = parsed.get(field_name)
            if isinstance(value, str) and value.strip():
                dispositions.append(value.strip())
                break
    return dispositions


def _assert_conversation_chronology(
    conversation_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require assistant blocks to follow their trace-owning user rows.

    Args:
        conversation_rows: Persisted conversation rows in model-facing order.

    Raises:
        FatalSequenceError: If any trace pair is absent, reversed, or split.
    """

    user_index_by_trace: dict[str, int] = {}
    for index, row in enumerate(conversation_rows):
        if row.get("role") != "user":
            continue
        trace_id = str(row.get("llm_trace_id") or "")
        if trace_id:
            user_index_by_trace[trace_id] = index

    for assistant_index, row in enumerate(conversation_rows):
        if row.get("role") != "assistant":
            continue
        trace_id = str(row.get("llm_trace_id") or "")
        user_index = user_index_by_trace.get(trace_id)
        if user_index is None or user_index >= assistant_index:
            raise FatalSequenceError(
                "conversation chronology lacks an earlier trace-owning user row"
            )
        intervening_rows = conversation_rows[
            user_index + 1:assistant_index + 1
        ]
        for intervening in intervening_rows:
            role = intervening.get("role")
            intervening_trace_id = str(
                intervening.get("llm_trace_id") or ""
            )
            if role == "user" or (
                role == "assistant"
                and intervening_trace_id != trace_id
            ):
                raise FatalSequenceError(
                    "conversation chronology split a user/assistant turn pair"
                )


def _assert_full_trace_capture(
    trace_steps: Sequence[Mapping[str, Any]],
) -> None:
    """Require protected raw model evidence before database restoration.

    Args:
        trace_steps: Protected trace rows captured for one replay turn.

    Raises:
        FatalSequenceError: If any required raw or parsed evidence is absent.
    """

    if not trace_steps:
        raise FatalSequenceError("full trace capture contains no LLM steps")
    for step in trace_steps:
        raw_messages = step.get("raw_messages")
        raw_response_text = step.get("raw_response_text")
        if (
            not isinstance(raw_messages, list)
            or not raw_messages
            or not isinstance(raw_response_text, str)
            or not raw_response_text
            or "parsed_output" not in step
        ):
            stage_name = str(step.get("stage_name") or "unknown")
            raise FatalSequenceError(
                f"full trace capture is incomplete for stage {stage_name}"
            )


async def _wait_for_trace_run(
    database: Any,
    *,
    platform_message_id: str,
) -> dict[str, Any]:
    """Wait for one trace row to reach a terminal status."""

    deadline = time.perf_counter() + _TRACE_WAIT_SECONDS
    latest: Mapping[str, Any] | None = None
    while time.perf_counter() < deadline:
        row = await database.llm_trace_runs.find_one(
            {"platform_message_id": platform_message_id},
            {"_id": 0},
        )
        if isinstance(row, Mapping):
            latest = row
            if str(row.get("status") or "") != "running":
                trace_run = dict(row)
                return trace_run
        await asyncio.sleep(0.2)
    if latest is None:
        raise FatalSequenceError(
            "live Asuna E2E did not persist an LLM trace run"
        )
    raise FatalSequenceError(
        "live Asuna E2E trace remained running: "
        f"{latest.get('status', '')}"
    )


class _ReplayAdapter:
    """In-memory QQ adapter at the production delivery boundary."""

    platform = "qq"
    display_name = "Asuna R18 E2E"

    def __init__(self) -> None:
        """Initialize stable bot identity and captured delivery calls."""

        self.platform_bot_id = _REPLAY_BOT_PLATFORM_ID
        self.calls: list[dict[str, Any]] = []

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Confirm that the guarded in-memory delivery target is available.

        Args:
            channel_id: Platform channel selected by production delivery.
            channel_type: Platform channel classification.

        Returns:
            True because the in-memory replay adapter is always available.
        """

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
    ) -> Any:
        """Capture one production delivery call and return its receipt.

        Args:
            channel_id: Platform channel selected for delivery.
            text: Visible character dialog sent by the service.
            channel_type: Platform channel classification.
            reply_to_msg_id: Optional platform-native reply target.
            delivery_mentions: Optional normalized delivery mentions.

        Returns:
            A successful synthetic platform delivery receipt.
        """

        from kazusa_ai_chatbot.dispatcher import SendResult

        self.calls.append({
            "channel_id": channel_id,
            "text": text,
            "channel_type": channel_type,
            "reply_to_msg_id": reply_to_msg_id,
            "delivery_mentions": list(delivery_mentions or []),
        })
        send_result = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=f"asuna-r18-e2e-delivery-{uuid4().hex}",
            sent_at=datetime.now(timezone.utc),
        )
        return send_result


def _episode_id(request: Any) -> str:
    """Build the service's deterministic user-message episode identifier."""

    episode_id = (
        f"user_message:{request.platform}:"
        f"{request.platform_channel_id}:{request.platform_message_id}"
    )
    return episode_id


async def _run_one_turn(
    *,
    client: httpx.AsyncClient,
    database: Any,
    adapter: _ReplayAdapter,
    request: Any,
    source_message: Mapping[str, Any],
    prior_turn: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run one input through the public FastAPI /chat route."""

    from kazusa_ai_chatbot.brain_service.contracts import ChatResponse

    user_id = str(request.platform_user_id)
    profile_before = await database.user_profiles.find_one(
        {"platform_accounts": {
            "$elemMatch": {
                "platform": request.platform,
                "platform_user_id": user_id,
            },
        }},
        {"_id": 0},
    )
    if not isinstance(profile_before, Mapping):
        raise FatalSequenceError("E2E user profile is missing before a turn")
    counts_before = await _collection_counts(database)
    adapter_call_start = len(adapter.calls)
    started_at = time.perf_counter()
    request_payload = request.model_dump(mode="json")
    try:
        http_response = await client.post("/chat", json=request_payload)
    except Exception as exc:
        raise FatalSequenceError(
            f"public /chat call crashed: {exc}"
        ) from exc
    if http_response.status_code != 200:
        raise FatalSequenceError(
            "public /chat returned HTTP "
            f"{http_response.status_code}: {http_response.text[:1000]}"
        )
    try:
        response_payload = http_response.json()
        response = ChatResponse.model_validate(response_payload)
    except Exception as exc:
        raise FatalSequenceError(
            f"public /chat returned an invalid ChatResponse: {exc}"
        ) from exc

    trace_run = await _wait_for_trace_run(
        database,
        platform_message_id=str(request.platform_message_id),
    )
    user_row = await database.conversation_history.find_one(
        {
            "platform": request.platform,
            "platform_channel_id": request.platform_channel_id,
            "platform_message_id": request.platform_message_id,
        },
        {"_id": 0},
    )
    if not isinstance(user_row, Mapping):
        raise FatalSequenceError("current E2E user row was not persisted")
    trace_id = str(
        trace_run.get("trace_id")
        or user_row.get("llm_trace_id")
        or ""
    )
    if not trace_id:
        raise FatalSequenceError("current E2E turn has no trace identifier")
    trace_steps = await database.llm_trace_steps.find(
        {"trace_id": trace_id},
        {"_id": 0},
    ).sort("sequence", 1).to_list(length=None)
    trace_steps = [
        dict(step)
        for step in trace_steps
        if isinstance(step, Mapping)
    ]
    assistant_rows = await database.conversation_history.find(
        {"llm_trace_id": trace_id, "role": "assistant"},
        {"_id": 0},
    ).sort("timestamp", 1).to_list(length=None)
    lifecycle = await database.post_turn_lifecycle_records.find_one(
        {"source_episode_id": _episode_id(request)},
        {"_id": 0},
    )
    profile_after = await database.user_profiles.find_one(
        {"global_user_id": profile_before["global_user_id"]},
        {"_id": 0},
    )
    if not isinstance(profile_after, Mapping):
        raise FatalSequenceError("E2E user profile is missing after a turn")
    character_after = await database.character_state.find_one(
        {"_id": "global"},
        {"_id": 0},
    )
    conversation_rows = await database.conversation_history.find(
        {"platform_channel_id": request.platform_channel_id},
        {"_id": 0},
    ).sort("timestamp", 1).to_list(length=None)
    _assert_conversation_chronology(conversation_rows)
    _assert_full_trace_capture(trace_steps)
    counts_after = await _collection_counts(database)
    if counts_after.get("conversation_history", 0) <= counts_before.get(
        "conversation_history",
        0,
    ):
        raise FatalSequenceError(
            "conversation history did not advance after the E2E turn"
        )
    response_payload = response.model_dump(mode="json")
    trace_status = str(trace_run.get("status") or "")
    technical_status = "passed" if trace_status == "succeeded" else "service_failure"
    adapter_calls = adapter.calls[adapter_call_start:]
    relationship_before = _relationship_from_profile(profile_before)
    relationship_after = _relationship_from_profile(profile_after)
    artifact = {
        "schema_version": "asuna_private_r18_affinity_e2e_turn.v1",
        "technical_status": technical_status,
        "case_index": int(source_message.get("case_index", 0)),
        "input": {
            "source_message": dict(source_message),
            "request": request_payload,
        },
        "response": response_payload,
        "response_surface": _response_surface(response_payload),
        "trace_run": _json_safe(trace_run),
        "trace_id": trace_id,
        "trace_steps": _json_safe(trace_steps),
        "trace_dispositions": _trace_dispositions(trace_steps),
        "lifecycle_record": _json_safe(lifecycle),
        "persisted_user_row": _json_safe(user_row),
        "persisted_assistant_rows": _json_safe(assistant_rows),
        "conversation_rows_after": _json_safe(conversation_rows),
        "user_state_before": _json_safe(_state_from_profile(profile_before)),
        "user_state_after": _json_safe(_state_from_profile(profile_after)),
        "relationship_before": _json_safe(relationship_before),
        "relationship_after": _json_safe(relationship_after),
        "character_state_after": _json_safe(character_after),
        "adapter_calls": _json_safe(adapter_calls),
        "adapter_call_count_total": len(adapter.calls),
        "collection_counts_before": counts_before,
        "collection_counts_after": counts_after,
        "continuity": {
            "prior_case_index": (
                prior_turn.get("case_index") if prior_turn else None
            ),
            "prior_history_count": (
                prior_turn.get("history_count_after") if prior_turn else 0
            ),
            "history_count_before": counts_before.get(
                "conversation_history",
                0,
            ),
            "history_count_after": counts_after.get(
                "conversation_history",
                0,
            ),
            "relationship_before_matches_prior_after": (
                prior_turn is None
                or relationship_before == prior_turn.get("relationship_after")
            ),
        },
        "duration_ms": round((time.perf_counter() - started_at) * 1000),
    }
    return artifact


def _input_sequence(manifest: Mapping[str, Any]) -> list[dict[str, object]]:
    """Project the twenty inputs without importing old dialog or residue."""

    input_sequence = [
        {
            "case_index": int(case["case_index"]),
            "platform_message_id": str(
                _mapping(case["source_message"], "source_message")[
                    "platform_message_id"
                ]
            ),
            "body_text": str(
                _mapping(case["source_message"], "source_message")["body_text"]
            ),
            "timestamp": str(
                _mapping(case["source_message"], "source_message")["timestamp"]
            ),
        }
        for case in _manifest_cases(manifest)
    ]
    return input_sequence


async def _run_sequence() -> dict[str, Any]:
    """Run all twenty inputs in one persistent service session."""

    manifest = _load_manifest()
    cases = _manifest_cases(manifest)
    condition = _condition_name()
    run_id = _run_id()
    output_root = _output_root(run_id=run_id, condition=condition)
    run_manifest_path = _guarded_path(output_root / "run_manifest.json")
    if run_manifest_path.exists():
        raise ValueError(f"condition run already exists: {run_manifest_path}")
    run_manifest: dict[str, Any] = {
        "schema_version": "asuna_private_r18_affinity_e2e_run.v2",
        "technical_status": "running",
        "run_id": run_id,
        "condition": condition,
        "database_name": _TEST_DATABASE_NAME,
        "database_guard": os.environ.get("KAZUSA_TEST_DB_GUARD", ""),
        "manifest_path": str(_MANIFEST_PATH),
        "input_sequence": _input_sequence(manifest),
        "sequence_execution": {
            "one_persistent_service_lifespan": True,
            "one_public_chat_route_call_per_input": True,
            "child_processes_per_condition": 1,
            "preloaded_history": 0,
            "preloaded_memory": 0,
            "preloaded_residue": 0,
        },
        "turns": [],
        "completed_case_indexes": [],
        "created_at": _storage_timestamp(),
    }
    _write_json(run_manifest_path, run_manifest)
    try:
        runtime = _prepare_runtime_environment(manifest)
        run_manifest["runtime"] = runtime
        seed = await _seed_empty_condition_baseline(
            manifest=manifest,
            condition=condition,
            runtime=runtime,
        )
        run_manifest["baseline"] = _json_safe(seed)
        _write_json(run_manifest_path, run_manifest)

        from kazusa_ai_chatbot import service
        from kazusa_ai_chatbot.db._client import get_db

        adapter = _ReplayAdapter()
        prior_turn: dict[str, Any] | None = None
        service_failures = 0
        async with service.lifespan(service.app):
            service.register_runtime_adapter(adapter)
            transport = httpx.ASGITransport(app=service.app)
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://asuna-r18-e2e",
                timeout=None,
            ) as client:
                health_response = await client.get("/health")
                if health_response.status_code != 200:
                    raise FatalSequenceError(
                        "service health endpoint returned HTTP "
                        f"{health_response.status_code}"
                    )
                run_manifest["health"] = health_response.json()
                database = await get_db()
                for case in cases:
                    case_index = int(case["case_index"])
                    source_message = _mapping(
                        case["source_message"],
                        "source_message",
                    )
                    source_with_index = {
                        "case_index": case_index,
                        **dict(source_message),
                    }
                    request = _build_request(case=case, runtime=runtime)
                    try:
                        artifact = await _run_one_turn(
                            client=client,
                            database=database,
                            adapter=adapter,
                            request=request,
                            source_message=source_with_index,
                            prior_turn=prior_turn,
                        )
                    except Exception as exc:
                        fatal_artifact = {
                            "schema_version": (
                                "asuna_private_r18_affinity_e2e_fatal.v1"
                            ),
                            "technical_status": "fatal_crash",
                            "condition": condition,
                            "run_id": run_id,
                            "case_index": case_index,
                            "input": {
                                "source_message": source_with_index,
                                "request": _json_safe(
                                    request.model_dump(mode="json")
                                ),
                            },
                            "failure_type": exc.__class__.__name__,
                            "failure_message": str(exc),
                            "failure_traceback": traceback.format_exc(),
                        }
                        _write_json(
                            _guarded_path(
                                output_root
                                / f"turn_{case_index:02d}_fatal.json",
                            ),
                            fatal_artifact,
                        )
                        run_manifest.update({
                            "technical_status": "fatal_crash",
                            "fatal_case_index": case_index,
                            "failure_type": exc.__class__.__name__,
                            "failure_message": str(exc),
                            "failure_traceback": traceback.format_exc(),
                            "updated_at": _storage_timestamp(),
                        })
                        _write_json(run_manifest_path, run_manifest)
                        raise
                    artifact_path = _guarded_path(
                        output_root / f"turn_{case_index:02d}.json"
                    )
                    _write_json(artifact_path, artifact)
                    if artifact["technical_status"] == "service_failure":
                        service_failures += 1
                    prior_turn = {
                        "case_index": case_index,
                        "history_count_after": artifact[
                            "continuity"
                        ]["history_count_after"],
                        "relationship_after": artifact[
                            "relationship_after"
                        ],
                    }
                    run_manifest["turns"].append({
                        "case_index": case_index,
                        "artifact": str(artifact_path),
                        "technical_status": artifact["technical_status"],
                        "input_text": source_message["body_text"],
                        "response_surface": artifact["response_surface"],
                        "trace_status": artifact["trace_run"].get(
                            "status",
                            "",
                        ),
                        "history_count_after": artifact[
                            "continuity"
                        ]["history_count_after"],
                        "relationship_before": artifact[
                            "relationship_before"
                        ],
                        "relationship_after": artifact[
                            "relationship_after"
                        ],
                    })
                    run_manifest["completed_case_indexes"].append(case_index)
                    run_manifest["updated_at"] = _storage_timestamp()
                    _write_json(run_manifest_path, run_manifest)
                    print(json.dumps({
                        "condition": condition,
                        "case_index": case_index,
                        "technical_status": artifact["technical_status"],
                        "response_surface": artifact["response_surface"],
                        "artifact": str(artifact_path),
                    }, ensure_ascii=False))
                final_database = await get_db()
                final_profile = await final_database.user_profiles.find_one(
                    {"global_user_id": runtime["user_global_id"]},
                    {"_id": 0},
                )
                run_manifest["final_counts"] = await _collection_counts(
                    final_database
                )
                run_manifest["final_user_state"] = _json_safe(
                    _state_from_profile(final_profile)
                    if isinstance(final_profile, Mapping)
                    else None
                )
        run_manifest.update({
            "technical_status": (
                "completed_with_service_failures"
                if service_failures
                else "passed"
            ),
            "service_failure_count": service_failures,
            "adapter_call_count": len(adapter.calls),
            "completed_at": _storage_timestamp(),
            "updated_at": _storage_timestamp(),
        })
        _write_json(run_manifest_path, run_manifest)
        return run_manifest
    except Exception:
        if run_manifest.get("technical_status") == "running":
            run_manifest.update({
                "technical_status": "fatal_crash",
                "failure_traceback": traceback.format_exc(),
                "updated_at": _storage_timestamp(),
            })
            _write_json(run_manifest_path, run_manifest)
        raise


async def _restore_default_baseline() -> dict[str, Any]:
    """Reset the guarded DB to the empty default-affinity baseline."""

    manifest = _load_manifest()
    runtime = _prepare_runtime_environment(manifest)
    seed = await _seed_empty_condition_baseline(
        manifest=manifest,
        condition="default_affinity",
        runtime=runtime,
    )
    from kazusa_ai_chatbot import service
    from kazusa_ai_chatbot.db._client import get_db

    async with service.lifespan(service.app):
        database = await get_db()
        counts = await _collection_counts(database)
        character = await database.character_state.find_one(
            {"_id": "global"},
            {"_id": 0},
        )
    restore_result = {
        "schema_version": "asuna_private_r18_affinity_restore.v2",
        "technical_status": "passed",
        "condition": "default_affinity",
        "database_name": _TEST_DATABASE_NAME,
        "baseline": _json_safe(seed),
        "counts_after_restore": counts,
        "character_profile_present": isinstance(character, Mapping),
        "restored_at": _storage_timestamp(),
    }
    return restore_result


@pytest.mark.live_llm
async def test_live_asuna_private_r18_affinity_sequence() -> None:
    """Run all twenty exact inputs through one persistent public E2E session."""

    result = await _run_sequence()
    assert result["completed_case_indexes"] == list(range(1, 21))
    assert result["technical_status"] in {
        "passed",
        "completed_with_service_failures",
    }


@pytest.mark.live_db
async def test_restore_asuna_private_r18_baseline() -> None:
    """Restore the guarded database after the two condition sequences."""

    raw_path = os.environ.get("ASUNA_R18_RESTORE_OUTPUT_PATH", "").strip()
    if not raw_path:
        pytest.skip("ASUNA_R18_RESTORE_OUTPUT_PATH is not configured")
    output_path = _guarded_path(Path(raw_path), "ASUNA_R18_RESTORE_OUTPUT_PATH")
    try:
        artifact = await _restore_default_baseline()
    except Exception as exc:
        artifact = {
            "schema_version": "asuna_private_r18_affinity_restore.v2",
            "technical_status": "failed",
            "failure_type": exc.__class__.__name__,
            "failure_message": str(exc),
            "failure_traceback": traceback.format_exc(),
        }
        _write_json(output_path, artifact)
        raise
    _write_json(output_path, artifact)
    assert artifact["technical_status"] == "passed"
