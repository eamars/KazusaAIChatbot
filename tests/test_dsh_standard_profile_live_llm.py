"""Individual real-local-model coverage for Standard and semantic tools."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "dsh_standard_coding"
DEBUG_ROOT = PROJECT_ROOT / ".dsh-debug"

pytestmark = pytest.mark.live_llm


def _require_live_backend() -> None:
    """Require explicit local-model execution for this coverage module."""

    if os.environ.get("KAZUSA_RUN_LIVE_LLM") != "1":
        pytest.skip("set KAZUSA_RUN_LIVE_LLM=1 for real-local-model coverage")
    required = (
        "AGENTIC_RESOLVER_LLM_API_KEY",
        "AGENTIC_RESOLVER_LLM_BASE_URL",
        "AGENTIC_RESOLVER_LLM_MODEL",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        pytest.fail(f"live model configuration is missing: {', '.join(missing)}")


async def _resolve_live(
    tmp_path: Path,
    objective: str,
    operation_id: str,
    *,
    facts_factory: Callable[[Mapping[str, object]], list[str]] | None = None,
    global_user_id: str | None = None,
) -> dict[str, Any]:
    """Run one real V2 resolution and normalize its terminal projection."""

    from agentic_resolver import AgenticResolverRuntime

    runtime = AgenticResolverRuntime.from_environment(data_root=tmp_path)
    scoped_user_id = global_user_id or f"live-{operation_id}-{uuid4().hex}"
    authority = runtime.new_runtime_authority(
        objective_ref=operation_id,
        brain_conversation_ref=f"chat:live:{operation_id}:{uuid4().hex}",
        service_scope={
            "platform": "debug",
            "platform_channel_id": "live",
            "global_user_id": scoped_user_id,
        },
        audience={"kind": "operator", "operation": operation_id},
        interaction_issuer="kazusa-brain",
    )
    intake = AgenticResolverRuntime.build_intake(
        authority,
        objective=objective,
        facts=facts_factory(authority) if facts_factory is not None else [],
    )
    result = await runtime.resolve(intake.to_dict())
    if hasattr(result, "to_dict"):
        result = result.to_dict()
    assert isinstance(result, dict)
    assert result.get("kind") == "terminal", json.dumps(result, sort_keys=True)
    return result


def _assert_semantic_tool_used(result: dict[str, Any], name: str) -> None:
    """Require one named semantic tool in the inspected live terminal record."""

    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert name in rendered


def _read_root_approval_events(
    result: Mapping[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Read approval audit events from the exact root session for one result."""

    from agentic_resolver.contracts import DSH_RELEASE

    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        pytest.fail("live result has no identity for session attribution")
    resolution_thread_id = identity.get("resolution_thread_id")
    segment_id = identity.get("segment_id")
    if (
        not isinstance(resolution_thread_id, str)
        or not resolution_thread_id.strip()
        or not isinstance(segment_id, str)
        or not segment_id.strip()
    ):
        pytest.fail("live result identity cannot identify a DSH root session")

    session_identity = f"{resolution_thread_id}\0{segment_id}".encode()
    session_id = (
        "kazusa-resolution-"
        f"{sha256(session_identity).hexdigest()[:32]}"
    )
    data_root = os.environ.get("KAZUSA_DSH_DATA_ROOT")
    if data_root is None or not data_root.strip():
        pytest.fail("KAZUSA_DSH_DATA_ROOT is unavailable for audit inspection")
    store_path = Path(data_root) / "dsh" / DSH_RELEASE / "sessions.sqlite"
    if not store_path.is_file():
        pytest.fail(f"DSH session store is unavailable: {store_path}")

    uri = f"file:{store_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        session_rows = connection.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchall()
        if session_rows != [(session_id,)]:
            pytest.fail(
                "exact DSH root session was not uniquely attributable: "
                f"{session_id}"
            )
        event_rows = connection.execute(
            "SELECT type, data FROM events "
            "WHERE session_id = ? AND type IN (?, ?) ORDER BY seq",
            (session_id, "approval/asked", "approval/decided"),
        ).fetchall()

    events: list[tuple[str, dict[str, Any]]] = []
    for event_type, raw_data in event_rows:
        if not isinstance(raw_data, str):
            pytest.fail(f"approval audit event is not JSON text: {event_type}")
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as error:
            pytest.fail(f"approval audit event is malformed: {error}")
        if not isinstance(data, dict):
            pytest.fail(f"approval audit event is not an object: {event_type}")
        events.append((event_type, data))
    return events


@pytest.mark.asyncio
async def test_qwen27b_conversation_people_and_memory_read_tools(tmp_path: Path) -> None:
    """Exercise conversation, people, and memory semantic reads."""

    _require_live_backend()
    from kazusa_ai_chatbot.db._client import (
        get_db,
        get_document_text_embedding,
    )

    suffix = uuid4().hex
    global_user_id = f"dsh-live-user-{suffix}"
    display_name = f"DshLiveUser{suffix[:8]}"
    memory_id = f"dsh-live-memory-{suffix}"
    database = await get_db()
    conversation_text = f"The bounded live topic is cobalt {suffix}."
    memory_text = f"The bounded live memory is cobalt {suffix}."
    conversation_embedding = await get_document_text_embedding(
        conversation_text
    )
    memory_embedding = await get_document_text_embedding(memory_text)
    conversation = await database.conversation_history.insert_one({
        "platform": "debug",
        "platform_channel_id": "live",
        "role": "user",
        "platform_user_id": global_user_id,
        "global_user_id": global_user_id,
        "display_name": display_name,
        "body_text": conversation_text,
        "raw_wire_text": conversation_text,
        "addressed_to_global_user_ids": [],
        "mentions": [],
        "broadcast": False,
        "attachments": [],
        "timestamp": "2026-08-29T00:00:00Z",
        "embedding": conversation_embedding,
    })
    await database.user_profiles.insert_one({
        "global_user_id": global_user_id,
        "platform_accounts": [{
            "platform": "debug",
            "platform_user_id": global_user_id,
            "display_name": display_name,
        }],
        "facts": [f"Knows the cobalt topic {suffix}."],
        "suspected_aliases": [],
    })
    await database.memory.insert_one({
        "memory_unit_id": memory_id,
        "lineage_id": f"lineage-{suffix}",
        "version": 1,
        "memory_name": f"Cobalt topic {suffix}",
        "content": memory_text,
        "source_global_user_id": global_user_id,
        "memory_type": "experience",
        "source_kind": "conversation_extracted",
        "authority": "conversation_accepted",
        "status": "active",
        "supersedes_memory_unit_ids": [],
        "merged_from_memory_unit_ids": [],
        "evidence_refs": [],
        "privacy_review": {"global_applicability": "scoped"},
        "confidence_note": "live fixture",
        "timestamp": "2026-08-29T00:00:00Z",
        "updated_at": "2026-08-29T00:00:00Z",
        "expiry_timestamp": None,
        "embedding": memory_embedding,
    })

    try:
        result = await _resolve_live(
            tmp_path,
            (
                f"Search conversation history for cobalt {suffix}, then read the "
                "returned conversation_entry_ref. Summarize conversation "
                f"participants. Find {display_name} by exact display name, then "
                "read the returned person_ref. Search current-user experience "
                f"memories for cobalt {suffix}, then read the returned memory_ref. "
                "After all seven semantic read capabilities have completed, "
                "submit a grounded terminal result."
            ),
            "reads",
            global_user_id=global_user_id,
        )
    finally:
        await database.conversation_history.delete_one(
            {"_id": conversation.inserted_id}
        )
        await database.user_profiles.delete_one(
            {"global_user_id": global_user_id}
        )
        await database.memory.delete_one({"memory_unit_id": memory_id})
    for name in (
        "kazusa_search_conversation_history",
        "kazusa_read_conversation_entries",
        "kazusa_summarize_conversation_participants",
        "kazusa_find_people_by_name",
        "kazusa_read_person_profiles",
        "kazusa_search_memories",
        "kazusa_read_memories",
    ):
        _assert_semantic_tool_used(result, name)


@pytest.mark.asyncio
async def test_qwen27b_memory_write_revision_lifecycle_and_readback(tmp_path: Path) -> None:
    """Exercise idempotent memory creation, revision, lifecycle, and readback."""

    _require_live_backend()
    from kazusa_ai_chatbot.db._client import get_db

    suffix = uuid4().hex
    global_user_id = f"dsh-memory-user-{suffix}"
    information = f"The Plan 2 live marker is cobalt-{suffix}."
    revised = f"The Plan 2 live marker is indigo-{suffix}."
    database = await get_db()
    try:
        result = await _resolve_live(
            tmp_path,
            (
                f"Remember this exact current-user experience: {information} "
                "Use current_task provenance and explain that this is a bounded "
                f"live test. Then revise the returned memory to: {revised} "
                "Complete that revised memory's lifecycle and read the final "
                "memory back through its opaque reference before submitting."
            ),
            "memory-lifecycle",
            global_user_id=global_user_id,
        )
    finally:
        await database.memory.delete_many(
            {"source_global_user_id": global_user_id}
        )
    for name in (
        "kazusa_remember_information",
        "kazusa_revise_memory",
        "kazusa_change_memory_lifecycle",
        "kazusa_read_memories",
    ):
        _assert_semantic_tool_used(result, name)


@pytest.mark.asyncio
async def test_qwen27b_active_recall_calendar_and_attached_media_tools(tmp_path: Path) -> None:
    """Exercise active context, calendar, and attached-media semantic reads."""

    _require_live_backend()
    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        verify_activation_token,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.media import (
        issue_attached_media_reference,
        persist_attached_media,
    )
    from kazusa_ai_chatbot.media_inspection.session_cache import (
        clear_session_media,
        put_session_media,
    )

    global_user_id = f"dsh-media-user-{uuid4().hex}"
    scope = ("debug", "live", global_user_id)
    references = put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
            "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
        ),
        "source_summary": "one-pixel live fixture",
    }])
    persist_attached_media(scope, references)

    def facts(authority: Mapping[str, object]) -> list[str]:
        secret = os.environ["KAZUSA_DSH_TOOL_GATEWAY_SECRET"].encode("utf-8")
        token = authority["semantic_tool_authority"]["token"]  # type: ignore[index]
        verified = verify_activation_token(str(token), secret=secret)
        codec = OpaqueReferenceCodec(secret).with_authority(verified)
        reference = issue_attached_media_reference(
            codec=codec,
            scope=scope,
            cache_ref=str(references[0]["cache_ref"]),
        )
        return [f"Exact attached_media_ref: {reference}"]

    try:
        result = await _resolve_live(
            tmp_path,
            (
                "Recall active commitments and history, read schedules calendar "
                "context, inspect the supplied attached_media_ref by asking what "
                "is visibly present, then submit the evidence-bound result."
            ),
            "context-media",
            facts_factory=facts,
            global_user_id=global_user_id,
        )
    finally:
        clear_session_media(scope)
    for name in (
        "kazusa_recall_active_context",
        "kazusa_read_calendar_context",
        "kazusa_inspect_attached_media",
    ):
        _assert_semantic_tool_used(result, name)


@pytest.mark.asyncio
async def test_qwen27b_selects_description_stripped_semantic_and_native_tools_then_submits_grounded_terminal(tmp_path: Path) -> None:
    """Exercise mixed native and semantic selection with one terminal call."""

    _require_live_backend()
    result = await _resolve_live(
        tmp_path,
        "Use the native workspace inspection and one relevant Kazusa semantic capability, then submit a grounded result.",
        "mixed-tools",
    )
    assert result["terminal"]["status"] in {"resolved", "partial"}
    assert "native_precedence" not in json.dumps(result, ensure_ascii=False)


@pytest.mark.asyncio
async def test_qwen27b_standard_mode_repairs_fixture_with_native_workspace_tools(
    tmp_path: Path,
) -> None:
    """Exercise Standard native editing and test execution on the fixture."""

    _require_live_backend()
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = DEBUG_ROOT / f"dsh-standard-coding-{uuid4().hex}"
    shutil.copytree(FIXTURE_ROOT, workspace)
    relative_workspace = workspace.relative_to(PROJECT_ROOT).as_posix()
    try:
        result = await _resolve_live(
            tmp_path,
            (
                "Use only native workspace tools to inspect, repair, and run "
                f"the deterministic test in {relative_workspace}."
            ),
            "native-coding",
        )
        environment = os.environ.copy()
        environment["KAZUSA_RUN_DSH_FIXTURE"] = "1"
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "test_calculator.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert result["terminal"]["status"] == "resolved"
    finally:
        shutil.rmtree(workspace)


@pytest.mark.asyncio
async def test_qwen27b_outside_workspace_request_round_trips_through_brain_once(
    tmp_path: Path,
) -> None:
    """Exercise Brain approval and one same-thread retry."""

    _require_live_backend()
    result = await _resolve_live(
        tmp_path,
        "Request one operation outside the admitted workspace, let Brain judge it, and retry only after its one-shot grant.",
        "brain-approval",
    )
    events = _read_root_approval_events(result)
    asked = [data for event_type, data in events if event_type == "approval/asked"]
    decided = [
        data for event_type, data in events if event_type == "approval/decided"
    ]
    assert len(asked) == 1
    assert len(decided) == 1

    asked_event = asked[0]
    decided_event = decided[0]
    approval_id = asked_event.get("id")
    assert isinstance(approval_id, str) and approval_id.strip()
    assert decided_event.get("id") == approval_id
    call_id = asked_event.get("callId")
    assert isinstance(call_id, str) and call_id.strip()

    outcome = decided_event.get("outcome")
    assert outcome in {"allowed-once", "rejected"}
    terminal = result.get("terminal")
    assert isinstance(terminal, Mapping)
    terminal_status = terminal.get("status")
    if outcome == "allowed-once":
        assert terminal_status in {"resolved", "partial"}
    else:
        assert terminal_status in {"resolved", "partial", "unavailable"}
