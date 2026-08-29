"""Named real-local-model end-to-end sign-off cases for DSH V2."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any
from uuid import uuid4

import httpx
import pytest
from dotenv import dotenv_values

from tests.test_dsh_standard_profile_live_llm import (
    DEBUG_ROOT,
    FIXTURE_ROOT,
    _require_live_backend,
    _resolve_live,
)
from kazusa_ai_chatbot.time_boundary import build_turn_clock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.live_llm


class _LoopbackAdapterServer:
    """Capture one registered Brain adapter's capability and send calls."""

    def __init__(self, *, platform: str, shared_secret: str) -> None:
        if not platform.strip():
            raise ValueError("loopback adapter platform is required")
        if not shared_secret.strip():
            raise ValueError("loopback adapter shared secret is required")
        self._platform = platform
        self._shared_secret = shared_secret
        self._lock = Lock()
        self._capability_payloads: list[dict[str, Any]] = []
        self._delivery_payloads: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            """Serve the two typed callback endpoints for this test."""

            def do_POST(self) -> None:
                owner._handle_post(self)

            def log_message(self, format: str, *args: object) -> None:
                del format, args

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = Thread(
            target=self._server.serve_forever,
            name="dsh-e2e-loopback-adapter",
            daemon=True,
        )

    @property
    def callback_url(self) -> str:
        """Return the bound callback base URL."""

        address = self._server.server_address
        return f"http://127.0.0.1:{address[1]}"

    def start(self) -> None:
        """Start serving callback requests."""

        self._thread.start()

    def stop(self) -> None:
        """Stop the callback server and join its bounded worker thread."""

        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("loopback adapter thread did not stop")

    def capability_payloads(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of capability probes."""

        with self._lock:
            return [dict(item) for item in self._capability_payloads]

    def delivery_payloads(self) -> list[dict[str, Any]]:
        """Return a thread-safe snapshot of durable send responses."""

        with self._lock:
            return [dict(item) for item in self._delivery_payloads]

    def _handle_post(self, request: BaseHTTPRequestHandler) -> None:
        """Handle one authenticated adapter callback request."""

        expected_authorization = f"Bearer {self._shared_secret}"
        if request.headers.get("Authorization") != expected_authorization:
            self._respond(request, 401, {"available": False})
            return
        raw_length = request.headers.get("Content-Length")
        try:
            length = int(raw_length or "")
        except ValueError:
            self._respond(request, 400, {"error": "invalid content length"})
            return
        try:
            payload_value = json.loads(
                request.rfile.read(length).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._respond(request, 400, {"error": "invalid JSON"})
            return
        if not isinstance(payload_value, dict):
            self._respond(request, 400, {"error": "JSON object required"})
            return

        if request.path == "/send_message/capability":
            with self._lock:
                self._capability_payloads.append(dict(payload_value))
            self._respond(request, 200, {"available": True})
            return
        if request.path != "/send_message":
            self._respond(request, 404, {"error": "unknown callback"})
            return

        message_id = f"dsh-e2e-adapter-message-{uuid4().hex}"
        sent_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        delivery = {
            **payload_value,
            "message_id": message_id,
            "sent_at": sent_at,
        }
        with self._lock:
            self._delivery_payloads.append(delivery)
        self._respond(
            request,
            200,
            {
                "platform": self._platform,
                "channel_id": payload_value.get("channel_id", ""),
                "message_id": message_id,
                "sent_at": sent_at,
            },
        )

    @staticmethod
    def _respond(
        request: BaseHTTPRequestHandler,
        status_code: int,
        payload: Mapping[str, object],
    ) -> None:
        """Send one compact JSON callback response."""

        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request.send_response(status_code)
        request.send_header("Content-Type", "application/json")
        request.send_header("Content-Length", str(len(encoded)))
        request.send_header("Connection", "close")
        request.end_headers()
        request.wfile.write(encoded)


def _read_exact_dsh_events(
    *,
    resolution_thread_id: str,
    segment_id: str,
    event_types: tuple[str, ...] | None = None,
) -> list[tuple[int, str, dict[str, Any]]]:
    """Read exact-session events, optionally restricted to event types."""

    import zstandard
    from agentic_resolver.contracts import DSH_RELEASE

    if not resolution_thread_id.strip() or not segment_id.strip():
        pytest.fail("DSH session identity is incomplete for audit")
    session_identity = f"{resolution_thread_id}\0{segment_id}".encode()
    session_id = (
        "kazusa-resolution-"
        f"{sha256(session_identity).hexdigest()[:32]}"
    )
    data_root = os.environ.get("KAZUSA_DSH_DATA_ROOT")
    if data_root is None or not data_root.strip():
        pytest.fail("KAZUSA_DSH_DATA_ROOT is unavailable for DSH audit")
    store_path = Path(data_root) / "dsh" / DSH_RELEASE / "sessions.sqlite"
    if not store_path.is_file():
        pytest.fail("DSH session store is unavailable for the live audit")

    uri = f"file:{store_path.as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
        session_rows = connection.execute(
            "SELECT id FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchall()
        if session_rows != [(session_id,)]:
            pytest.fail("the DSH identity did not resolve one root session")
        if event_types is None:
            event_rows = connection.execute(
                "SELECT seq, type, data FROM events "
                "WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        else:
            placeholders = ", ".join("?" for _ in event_types)
            event_rows = connection.execute(
                "SELECT seq, type, data FROM events "
                f"WHERE session_id = ? AND type IN ({placeholders}) "
                "ORDER BY seq",
                (session_id, *event_types),
            ).fetchall()

    decoder = zstandard.ZstdDecompressor()
    events: list[tuple[int, str, dict[str, Any]]] = []
    for sequence, event_type, raw_data in event_rows:
        event_data: object = raw_data
        if isinstance(event_data, bytes):
            try:
                event_data = decoder.decompress(event_data).decode("utf-8")
            except (UnicodeDecodeError, zstandard.ZstdError) as exc:
                pytest.fail(
                    f"DSH {event_type} at sequence {sequence} cannot be decoded: "
                    f"{exc}"
                )
        if not isinstance(event_data, str):
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is not JSON text"
            )
        try:
            parsed = json.loads(event_data)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is malformed: {exc}"
            )
        if not isinstance(parsed, dict):
            pytest.fail(
                f"DSH {event_type} at sequence {sequence} is not an object"
            )
        events.append((sequence, event_type, parsed))
    return events


def _read_exact_dsh_tool_events(
    result: Mapping[str, object],
) -> list[tuple[int, str, dict[str, Any]]]:
    """Read tool events from the root session named by one live result."""

    identity = result.get("identity")
    if not isinstance(identity, Mapping):
        pytest.fail("live result identity is unavailable for DSH audit")
    resolution_thread_id = identity.get("resolution_thread_id")
    segment_id = identity.get("segment_id")
    if (
        not isinstance(resolution_thread_id, str)
        or not resolution_thread_id.strip()
        or not isinstance(segment_id, str)
        or not segment_id.strip()
    ):
        pytest.fail("live result identity cannot identify a DSH root session")
    return _read_exact_dsh_events(
        resolution_thread_id=resolution_thread_id,
        segment_id=segment_id,
        event_types=("tool/call", "tool/result"),
    )


def _matching_dsh_tool_result(
    events: list[tuple[int, str, dict[str, Any]]],
    *,
    tool_name: str,
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, object]]:
    """Return one exact tool call and its correlated stored result."""

    calls = [
        (sequence, event)
        for sequence, event_type, event in events
        if event_type == "tool/call" and event.get("name") == tool_name
    ]
    if not calls:
        pytest.fail(f"exact DSH session has no {tool_name} tool call")
    call_sequence, call = calls[0]
    call_id = call.get("callId")
    if not isinstance(call_id, str) or not call_id.strip():
        pytest.fail(f"{tool_name} call at sequence {call_sequence} has no call id")

    matches: list[tuple[int, dict[str, Any], Mapping[str, object]]] = []
    for sequence, event_type, event in events:
        if event_type != "tool/result":
            continue
        message = event.get("message")
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            if item.get("toolCallId") == call_id:
                matches.append((sequence, event, item))
    if len(matches) != 1:
        pytest.fail(
            f"{tool_name} call at sequence {call_sequence} has "
            f"{len(matches)} matching tool results"
        )
    return call, matches[0][1], matches[0][2]


@pytest.mark.asyncio
async def test_e2e_context_people_memory_recall_and_calendar(tmp_path: Path) -> None:
    """Sign off context, people, memory, recall, and calendar traversal."""

    _require_live_backend()
    result = await _resolve_live(
        tmp_path,
        "Inspect relevant people, memories, active recall, and calendar context; summarize what is available, explicitly state which requested categories are empty, and submit the grounded result.",
        "e2e-context",
    )
    rendered = json.dumps(result, ensure_ascii=False)
    terminal = result.get("terminal")
    assert isinstance(terminal, Mapping)
    assert terminal.get("status") in {"resolved", "partial"}
    assert "kazusa_" in rendered


@pytest.mark.asyncio
async def test_e2e_memory_create_revise_lifecycle_and_readback(tmp_path: Path) -> None:
    """Sign off durable semantic memory lifecycle and idempotent readback."""

    _require_live_backend()
    from kazusa_ai_chatbot.db._client import get_db

    suffix = uuid4().hex
    global_user_id = f"dsh-e2e-memory-user-{suffix}"
    database = await get_db()
    try:
        result = await _resolve_live(
            tmp_path,
            (
                f"Remember the exact current-user experience marker teal-{suffix} "
                "using current_task provenance, revise the returned memory to "
                f"violet-{suffix}, complete its lifecycle, read that final memory "
                "through the returned opaque reference, and submit the result."
            ),
            "e2e-memory",
            global_user_id=global_user_id,
        )
    finally:
        await database.memory.delete_many(
            {"source_global_user_id": global_user_id}
        )
    rendered = json.dumps(result, ensure_ascii=False)
    for name in (
        "kazusa_remember_information",
        "kazusa_revise_memory",
        "kazusa_change_memory_lifecycle",
        "kazusa_read_memories",
    ):
        assert name in rendered


@pytest.mark.asyncio
async def test_e2e_attached_media_native_web_and_semantic_evidence(tmp_path: Path) -> None:
    """Sign off native web, attached media, and evidence-bound resolution."""

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

    global_user_id = f"dsh-e2e-media-user-{uuid4().hex}"
    scope = ("debug", "live", global_user_id)
    references = put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNg"
            "YAAAAAMAASsJTYQAAAAASUVORK5CYII="
        ),
        "source_summary": "one-pixel E2E fixture",
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
                "Use native web_search to find the official Python homepage and "
                "report one returned source URL, then inspect the supplied "
                "attached_media_ref. If media inspection reports a typed "
                "unavailable or unsupported limitation, preserve it as a "
                "remaining need and submit a partial grounded result after "
                "completing the web task; otherwise submit a resolved or "
                "partial grounded result."
            ),
            "e2e-media-web",
            facts_factory=facts,
            global_user_id=global_user_id,
        )
    finally:
        clear_session_media(scope)

    events = _read_exact_dsh_tool_events(result)
    _, web_result, web_result_item = _matching_dsh_tool_result(
        events,
        tool_name="web_search",
    )
    if web_result_item.get("isError") is not False:
        pytest.fail("the correlated web_search result reported an error")
    web_meta = web_result.get("meta")
    if not isinstance(web_meta, Mapping):
        pytest.fail("the correlated web_search result has no metadata")
    web_sources = web_meta.get("sources")
    if not isinstance(web_sources, list) or not any(
        isinstance(source, Mapping)
        and isinstance(source.get("url"), str)
        and bool(source.get("url", "").strip())
        for source in web_sources
    ):
        pytest.fail("the successful web_search result has no source URL")

    _, media_result, media_result_item = _matching_dsh_tool_result(
        events,
        tool_name="kazusa_inspect_attached_media",
    )
    if media_result_item.get("isError") is not False:
        pytest.fail("the correlated attached-media result reported an error")
    media_meta = media_result.get("meta")
    if not isinstance(media_meta, Mapping) or media_meta.get("status") != "ok":
        pytest.fail("the attached-media semantic envelope was not successful")
    media_entities = media_meta.get("entities")
    if not isinstance(media_entities, list):
        pytest.fail("the attached-media result has no entity list")
    media_statuses: list[str] = []
    for entity in media_entities:
        if not isinstance(entity, Mapping):
            pytest.fail("the attached-media entity is not an object")
        status = entity.get("status")
        if isinstance(status, str):
            media_statuses.append(status)
    if not media_statuses:
        pytest.fail("the attached-media result has no entity status")

    evidence = result.get("evidence")
    if not isinstance(evidence, list):
        pytest.fail("the live result has no public evidence list")
    evidence_ids: list[str] = []
    attached_media_ids: list[str] = []
    for receipt in evidence:
        if not isinstance(receipt, Mapping):
            pytest.fail("the public evidence list contains a non-object")
        evidence_id = receipt.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            pytest.fail("the public evidence list contains an invalid receipt")
        evidence_ids.append(evidence_id)
        if receipt.get("source_kind") == "attached_media":
            attached_media_ids.append(evidence_id)
    if not attached_media_ids:
        pytest.fail("the public evidence list has no attached-media receipt")
    if len(evidence_ids) != len(set(evidence_ids)):
        pytest.fail("the public evidence list contains duplicate receipt ids")

    terminal = result.get("terminal")
    if not isinstance(terminal, Mapping):
        pytest.fail("the live result has no terminal projection")
    terminal_status = terminal.get("status")
    if terminal_status not in {"resolved", "partial"}:
        pytest.fail("the live terminal projection has an invalid status")
    if any(status in {"failed", "unsupported"} for status in media_statuses):
        remaining_needs = terminal.get("remaining_needs")
        warnings = terminal.get("warnings")
        has_remaining_needs = (
            isinstance(remaining_needs, list) and bool(remaining_needs)
        ) or (
            isinstance(remaining_needs, str) and bool(remaining_needs.strip())
        )
        has_warnings = (
            isinstance(warnings, list) and bool(warnings)
        ) or (isinstance(warnings, str) and bool(warnings.strip()))
        if terminal_status != "partial" or not (
            has_remaining_needs or has_warnings
        ):
            pytest.fail(
                "a typed attached-media limitation requires a partial "
                "terminal with a remaining need or warning"
            )


@pytest.mark.asyncio
async def test_e2e_native_coding_repairs_and_verifies_workspace_fixture(
    tmp_path: Path,
) -> None:
    """Sign off native Standard coding and deterministic fixture verification."""

    _require_live_backend()
    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    workspace = DEBUG_ROOT / f"dsh-e2e-coding-{uuid4().hex}"
    shutil.copytree(FIXTURE_ROOT, workspace)
    relative_workspace = workspace.relative_to(Path.cwd()).as_posix()
    try:
        result = await _resolve_live(
            tmp_path,
            (
                "Use only native Standard workspace tools to repair the "
                f"calculator in {relative_workspace} and run its deterministic test."
            ),
            "e2e-coding",
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
        terminal = result.get("terminal")
        assert isinstance(terminal, Mapping)
        assert terminal.get("status") == "resolved"
    finally:
        shutil.rmtree(workspace)


@pytest.mark.asyncio
async def test_e2e_brain_judgment_checkpoint_relay_resume_and_terminal(
    tmp_path: Path,
) -> None:
    """Sign off one full Brain relay, typed reply, and native continuation."""

    _require_live_backend()
    from kazusa_ai_chatbot.db import dsh_interactions, resolution_threads
    from kazusa_ai_chatbot.db import resolve_global_user_id
    from kazusa_ai_chatbot.db import _client as db_client

    suffix = uuid4().hex
    platform = f"dsh-e2e-brain-{suffix}"
    platform_user_id = f"dsh-e2e-user-{suffix}"
    platform_channel_id = f"dsh-e2e-channel-{suffix}"
    display_name = f"Dsh E2E User {suffix[:8]}"
    character_global_user_id = os.environ.get(
        "CHARACTER_GLOBAL_USER_ID",
        "character-global",
    ).strip()
    if not character_global_user_id:
        pytest.fail("the active character identity is unavailable")
    platform_bot_id = character_global_user_id
    callback_secret = f"dsh-e2e-callback-{suffix}"
    adapter = _LoopbackAdapterServer(
        platform=platform,
        shared_secret=callback_secret,
    )

    settings = dotenv_values(PROJECT_ROOT / ".env")
    brain_database_name = settings.get("MONGODB_DB_NAME")
    if not isinstance(brain_database_name, str) or not brain_database_name.strip():
        pytest.fail(".env does not provide the Brain MongoDB database name")
    adapter.start()
    original_database_name = db_client.MONGODB_DB_NAME
    original_guard = os.environ.get("KAZUSA_TEST_DB_GUARD")
    database = None
    global_user_id: str | None = None
    profile_created = False
    baseline_conversation_ids: set[object] = set()
    interaction_id: str | None = None
    resolution_thread_id: str | None = None
    brain_conversation_ref = f"chat:live:e2e-brain:{suffix}"
    seed_platform_message_id = f"dsh-e2e-seed-{suffix}"
    reply_platform_message_id = f"dsh-e2e-reply-{suffix}"
    outside_file = tmp_path / f"dsh-brain-outside-{suffix}.txt"
    marker = f"DSH_BRAIN_GATE5_{suffix}"

    try:
        outside_file.write_text(
            f"nonce-marker={marker}\n",
            encoding="utf-8",
        )
        workspace_root = Path(
            os.environ["AGENTIC_RESOLVER_WORKSPACE_ROOT"]
        ).resolve()
        if outside_file.resolve().is_relative_to(workspace_root):
            pytest.fail("the Brain approval fixture must be outside the workspace")

        await db_client.close_db()
        db_client.MONGODB_DB_NAME = brain_database_name
        os.environ.pop("KAZUSA_TEST_DB_GUARD", None)
        database = await db_client.get_db()
        account_filter = {
            "platform_accounts": {
                "$elemMatch": {
                    "platform": platform,
                    "platform_user_id": platform_user_id,
                },
            },
        }
        existing_profile = await database.user_profiles.find_one(
            account_filter,
            {"_id": 1},
        )
        if existing_profile is not None:
            pytest.fail("the generated Brain identity already exists")
        global_user_id = await resolve_global_user_id(
            platform=platform,
            platform_user_id=platform_user_id,
            display_name=display_name,
        )
        profile_created = True

        baseline_rows = await database.conversation_history.find(
            {
                "platform": platform,
                "platform_channel_id": platform_channel_id,
            },
            {"_id": 1},
        ).to_list(length=None)
        baseline_conversation_ids = {
            row["_id"] for row in baseline_rows if "_id" in row
        }

        brain_url = os.environ["KAZUSA_DSH_BRAIN_URL"].rstrip("/")
        timeout = httpx.Timeout(
            600.0,
            connect=15.0,
            write=15.0,
            pool=15.0,
        )
        async with httpx.AsyncClient(timeout=timeout) as client:
            registration_response = await client.post(
                f"{brain_url}/runtime/adapters/register",
                json={
                    "platform": platform,
                    "callback_url": adapter.callback_url,
                    "platform_bot_id": platform_bot_id,
                    "shared_secret": callback_secret,
                    "timeout_seconds": 30.0,
                },
            )
            assert registration_response.status_code == 200, (
                "runtime adapter registration failed: "
                f"HTTP {registration_response.status_code}"
            )
            registration = registration_response.json()
            assert isinstance(registration, Mapping)
            assert registration.get("status") == "registered"
            assert registration.get("platform") == platform

            seed_text = (
                "For the next native PowerShell command, ask me for explicit "
                "one-time permission before proceeding. This message does "
                "not grant that permission."
            )
            seed_response = await client.post(
                f"{brain_url}/chat",
                json={
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "channel_type": "private",
                    "platform_message_id": seed_platform_message_id,
                    "platform_user_id": platform_user_id,
                    "platform_bot_id": platform_bot_id,
                    "display_name": display_name,
                    "channel_name": f"DSH E2E {suffix[:8]}",
                    "content_type": "text",
                    "message_envelope": {
                        "body_text": seed_text,
                        "raw_wire_text": seed_text,
                        "mentions": [],
                        "reply": None,
                        "attachments": [],
                        "addressed_to_global_user_ids": [
                            character_global_user_id,
                        ],
                        "broadcast": False,
                    },
                    "local_timestamp": build_turn_clock()["local_timestamp"],
                    "debug_modes": {
                        "listen_only": False,
                        "think_only": False,
                        "no_remember": True,
                    },
                },
            )
            assert seed_response.status_code == 200, (
                "typed Brain seed chat failed: "
                f"HTTP {seed_response.status_code}"
            )
            seed_response_payload = seed_response.json()
            assert isinstance(seed_response_payload, Mapping)
            seed_rows = await database.conversation_history.find(
                {
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "platform_message_id": seed_platform_message_id,
                    "role": "user",
                    "global_user_id": global_user_id,
                },
                {"_id": 0, "body_text": 1},
            ).to_list(length=None)
            assert len(seed_rows) == 1
            assert seed_rows[0].get("body_text") == seed_text

            from agentic_resolver import AgenticResolverRuntime

            runtime = AgenticResolverRuntime.from_environment(
                data_root=tmp_path.resolve(),
            )
            objective_ref = f"e2e-brain-{suffix}"
            authority = runtime.new_runtime_authority(
                objective_ref=objective_ref,
                brain_conversation_ref=brain_conversation_ref,
                service_scope={
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "global_user_id": global_user_id,
                },
                audience={"kind": "operator", "operation": objective_ref},
                interaction_issuer="kazusa-brain",
            )
            resolution_thread_id = str(authority["resolution_thread_id"])
            segment_id = str(authority["segment_id"])
            activation_id = str(authority["activation_id"])
            lease_epoch = int(authority["lease_epoch"])
            pwsh_command = (
                f"Get-Content -LiteralPath '{outside_file.resolve().as_posix()}'"
            )
            objective = (
                f"Run exactly this read-only native pwsh command: {pwsh_command}. "
                "Invoke pwsh and let the native tool policy and Brain handle "
                "any required approval; do not substitute a terminal "
                "approval_required request. After authorization, report the "
                f"nonce marker {marker} and submit."
            )
            intake = AgenticResolverRuntime.build_intake(
                authority,
                objective=objective,
                facts=[],
            )
            first = await runtime.resolve(intake.to_dict())
            first_result = first.to_dict()
            assert first_result.get("kind") == "checkpointed", json.dumps(
                first_result,
                sort_keys=True,
            )

            interaction_collection = database[
                dsh_interactions.DSH_INTERACTIONS_COLLECTION
            ]
            interaction_rows = await interaction_collection.find(
                {
                    "brain_conversation_ref": brain_conversation_ref,
                    "global_user_id": global_user_id,
                },
                {"_id": 0},
            ).to_list(length=None)
            assert len(interaction_rows) == 1
            initial_row = dict(interaction_rows[0])
            interaction_id_value = initial_row.get("interaction_id")
            assert isinstance(interaction_id_value, str)
            interaction_id = interaction_id_value
            request_identity = initial_row.get("request_identity")
            assert isinstance(request_identity, Mapping)
            assert request_identity.get("kind") == "approval"
            assert request_identity.get("tool_name") == "pwsh"
            assert request_identity.get("resolution_thread_id") == (
                resolution_thread_id
            )
            assert request_identity.get("segment_id") == segment_id
            assert request_identity.get("activation_id") == activation_id
            assert request_identity.get("lease_epoch") == lease_epoch
            assert request_identity.get("brain_conversation_ref") == (
                brain_conversation_ref
            )
            assert request_identity.get("platform") == platform
            assert request_identity.get("platform_channel_id") == (
                platform_channel_id
            )
            assert request_identity.get("global_user_id") == global_user_id

            assert initial_row.get("resolution_thread_id") == (
                resolution_thread_id
            )
            assert initial_row.get("segment_id") == segment_id
            assert initial_row.get("activation_id") == activation_id
            assert initial_row.get("lease_epoch") == lease_epoch
            assert initial_row.get("status") == "delivered"
            decision = initial_row.get("decision")
            assert isinstance(decision, Mapping)
            assert decision.get("decision") == "relay_to_user"
            assert initial_row.get("relay_mode") == "approval"
            response_goal = initial_row.get("response_goal")
            assert isinstance(response_goal, str) and response_goal.strip()
            delivered_message_id = initial_row.get(
                "delivered_platform_message_id"
            )
            assert (
                isinstance(delivered_message_id, str)
                and delivered_message_id.strip()
            )
            deliveries = adapter.delivery_payloads()
            assert len(deliveries) == 1
            assert deliveries[0].get("message_id") == delivered_message_id
            assert deliveries[0].get("channel_id") == platform_channel_id
            assert adapter.capability_payloads()

            reply_text = "I approve this exact read once."
            reply_response = await client.post(
                f"{brain_url}/chat",
                json={
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                    "channel_type": "private",
                    "platform_message_id": reply_platform_message_id,
                    "platform_user_id": platform_user_id,
                    "platform_bot_id": platform_bot_id,
                    "display_name": display_name,
                    "channel_name": f"DSH E2E {suffix[:8]}",
                    "content_type": "text",
                    "message_envelope": {
                        "body_text": reply_text,
                        "raw_wire_text": reply_text,
                        "mentions": [],
                        "reply": {
                            "platform_message_id": delivered_message_id,
                            "platform_user_id": platform_bot_id,
                            "global_user_id": character_global_user_id,
                            "display_name": "Kazusa",
                            "excerpt": str(response_goal)[:500],
                            "derivation": "platform_native",
                        },
                        "attachments": [],
                        "addressed_to_global_user_ids": [
                            character_global_user_id,
                        ],
                        "broadcast": False,
                    },
                    "local_timestamp": build_turn_clock()["local_timestamp"],
                    "debug_modes": {
                        "listen_only": False,
                        "think_only": False,
                        "no_remember": True,
                    },
                },
            )
            assert reply_response.status_code == 200, (
                "typed Brain reply chat failed: "
                f"HTTP {reply_response.status_code}"
            )
            reply_response_payload = reply_response.json()
            assert isinstance(reply_response_payload, Mapping)

        assert interaction_id is not None
        updated_row = await interaction_collection.find_one(
            {"interaction_id": interaction_id},
            {"_id": 0},
        )
        assert isinstance(updated_row, Mapping)
        assert updated_row.get("status") == "replied"
        assert updated_row.get("resolution_thread_id") == (
            resolution_thread_id
        )
        assert updated_row.get("segment_id") == segment_id
        assert updated_row.get("activation_id") == activation_id
        assert updated_row.get("lease_epoch") == lease_epoch
        reply_result = updated_row.get("reply_result")
        assert isinstance(reply_result, Mapping)
        assert reply_result.get("decision") == "allow_once"
        assert updated_row.get("reply_platform_message_id") == (
            reply_platform_message_id
        )
        assert updated_row.get("grant_status") == "consumed"
        grant = updated_row.get("grant")
        assert isinstance(grant, Mapping)
        assert grant.get("grant_status") == "consumed"

        continuation = updated_row.get("result")
        assert isinstance(continuation, Mapping)
        assert continuation.get("resolution_thread_id") == (
            resolution_thread_id
        )
        assert continuation.get("segment_id") == segment_id
        continuation_exhaust = continuation.get("exhaust")
        assert isinstance(continuation_exhaust, Mapping)
        assert continuation_exhaust.get("kind") == "terminal"
        terminal = continuation_exhaust.get("terminal")
        assert isinstance(terminal, Mapping)
        assert terminal.get("status") in {"resolved", "partial"}

        exact_events = _read_exact_dsh_events(
            resolution_thread_id=resolution_thread_id,
            segment_id=segment_id,
        )
        asked_events = [
            event
            for _, event_type, event in exact_events
            if event_type == "approval/asked"
        ]
        decided_events = [
            event
            for _, event_type, event in exact_events
            if event_type == "approval/decided"
        ]
        assert len(asked_events) == 2
        assert len(decided_events) == 2
        decisions_by_id = {
            event.get("id"): event.get("outcome")
            for event in decided_events
        }
        asked_call_ids = []
        for asked in asked_events:
            asked_id = asked.get("id")
            assert isinstance(asked_id, str) and asked_id.strip()
            assert decisions_by_id.get(asked_id) in {
                "cancelled",
                "allowed-once",
            }
            call_id = asked.get("callId")
            assert isinstance(call_id, str) and call_id.strip()
            asked_call_ids.append(call_id)
        assert [decisions_by_id[asked.get("id")] for asked in asked_events] == [
            "cancelled",
            "allowed-once",
        ]
        assert len(set(asked_call_ids)) == 2
        assert {
            asked.get("id") for asked in asked_events
        } == set(decisions_by_id)
        assert all(
            event.get("outcome") != "unavailable"
            for event in decided_events
        )
        terminal_events = []
        for _, event_type, event in exact_events:
            if event_type != "tool/result":
                continue
            meta = event.get("meta")
            if not isinstance(meta, Mapping):
                continue
            receipt = meta.get("kazusa")
            if isinstance(receipt, Mapping) and receipt.get("kind") == (
                "terminal_resolution_v2"
            ):
                terminal_events.append(event)
        assert len(terminal_events) == 1
        checkpoint_end_events = []
        for _, event_type, event in exact_events:
            if event_type != "turn/end":
                continue
            reason = event.get("reason")
            if not isinstance(reason, Mapping):
                continue
            nested_reason = reason.get("reason")
            if (
                reason.get("kind") == "aborted"
                and isinstance(nested_reason, Mapping)
                and nested_reason.get("kind") == "hook"
                and nested_reason.get("reason") == "checkpoint"
            ):
                checkpoint_end_events.append(event)
        assert len(checkpoint_end_events) == 1

        terminal_text = json.dumps(terminal, ensure_ascii=False)
        native_pwsh_call_ids = {
            event.get("callId")
            for _, event_type, event in exact_events
            if event_type == "tool/call"
            and event.get("name") == "pwsh"
            and isinstance(event.get("callId"), str)
        }
        assert set(asked_call_ids).issubset(native_pwsh_call_ids)
        native_pwsh_result_has_marker = any(
            marker in json.dumps(event, ensure_ascii=False)
            for _, event_type, event in exact_events
            if event_type == "tool/result"
            for item in (
                event.get("message", {}).get("content", [])
                if isinstance(event.get("message"), Mapping)
                and isinstance(event.get("message", {}).get("content"), list)
                else []
            )
            if isinstance(item, Mapping)
            and item.get("toolCallId") in native_pwsh_call_ids
        )
        assert marker in terminal_text or native_pwsh_result_has_marker

        reply_rows = await database.conversation_history.find(
            {
                "platform": platform,
                "platform_channel_id": platform_channel_id,
                "platform_message_id": reply_platform_message_id,
                "role": "user",
                "global_user_id": global_user_id,
            },
            {"_id": 1},
        ).to_list(length=None)
        assert len(reply_rows) == 1
        interaction_rows_after_reply = await interaction_collection.find(
            {
                "brain_conversation_ref": brain_conversation_ref,
                "global_user_id": global_user_id,
            },
            {"_id": 0},
        ).to_list(length=None)
        assert len(interaction_rows_after_reply) == 2
        retry_rows = [
            row
            for row in interaction_rows_after_reply
            if row.get("interaction_id") != interaction_id
        ]
        assert len(retry_rows) == 1
        retry_decision = retry_rows[0].get("decision")
        assert isinstance(retry_decision, Mapping)
        assert retry_rows[0].get("status") == "decided"
        assert retry_decision.get("decision") == "allow_once"
    finally:
        if database is not None:
            conversation_rows = await database.conversation_history.find(
                {
                    "platform": platform,
                    "platform_channel_id": platform_channel_id,
                },
                {"_id": 1},
            ).to_list(length=None)
            created_conversation_ids = [
                row["_id"]
                for row in conversation_rows
                if "_id" in row
                and row["_id"] not in baseline_conversation_ids
            ]
            if created_conversation_ids:
                await database.conversation_history.delete_many({
                    "_id": {"$in": created_conversation_ids},
                })
            interaction_collection = database[
                dsh_interactions.DSH_INTERACTIONS_COLLECTION
            ]
            if global_user_id is not None:
                await interaction_collection.delete_many({
                    "brain_conversation_ref": brain_conversation_ref,
                    "global_user_id": global_user_id,
                })
            if resolution_thread_id is not None:
                await resolution_threads.delete_thread(resolution_thread_id)
            await database.user_profiles.update_one(
                {"global_user_id": character_global_user_id},
                {
                    "$pull": {
                        "platform_accounts": {
                            "platform": platform,
                            "platform_user_id": platform_bot_id,
                        },
                    },
                },
            )
            if profile_created and global_user_id is not None:
                await database.user_profiles.delete_one({
                    "global_user_id": global_user_id,
                })
        await db_client.close_db()
        db_client.MONGODB_DB_NAME = original_database_name
        if original_guard is None:
            os.environ.pop("KAZUSA_TEST_DB_GUARD", None)
        else:
            os.environ["KAZUSA_TEST_DB_GUARD"] = original_guard
        outside_file.unlink(missing_ok=True)
        adapter.stop()
