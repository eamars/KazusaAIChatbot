"""Persistent length-prefixed semantic worker used by the DSH sidecar."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import sqlite3
import struct
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
    CALL_SCHEMA_VERSION,
    SemanticActivationAuthorityV1,
    SignedSemanticCallV1,
    authenticate_semantic_call,
)
from kazusa_ai_chatbot.dsh_tool_gateway.catalog import semantic_catalog_digest
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
)

MAX_FRAME_BYTES = 32 * 1024
WORKER_HEALTH_CONTROL = "health"
WORKER_HEALTH_SCHEMA_VERSION = "kazusa_semantic_worker_health.v1"


@dataclass(frozen=True, slots=True)
class SemanticWorkerCall:
    """Minimal worker call shape for injected deterministic handlers."""

    call_id: str
    payload: dict[str, Any]


class SemanticOutcomeOwner(Protocol):
    """Durable owner for semantic call outcomes across worker restarts."""

    async def lookup(
        self,
        call_id: str,
        payload_digest: str,
    ) -> tuple[Literal["missing", "committed", "uncertain", "mismatch"], dict[str, Any] | None]:
        """Return the durable state for one call identity."""

    async def commit(
        self,
        call_id: str,
        payload_digest: str,
        result: dict[str, Any],
    ) -> None:
        """Persist one exact committed result."""

    async def mark_uncertain(self, call_id: str, payload_digest: str) -> None:
        """Persist that a mutation outcome cannot be safely replayed."""


class InMemorySemanticOutcomeOwner:
    """Explicit test-only outcome owner used by deterministic worker tests."""

    def __init__(self) -> None:
        self._rows: dict[str, tuple[str, str, dict[str, Any] | None]] = {}

    async def lookup(
        self,
        call_id: str,
        payload_digest: str,
    ) -> tuple[Literal["missing", "committed", "uncertain", "mismatch"], dict[str, Any] | None]:
        row = self._rows.get(call_id)
        if row is None:
            return "missing", None
        stored_digest, status, result = row
        if stored_digest != payload_digest:
            return "mismatch", None
        if status == "committed":
            return "committed", None if result is None else dict(result)
        return "uncertain", None

    async def commit(
        self,
        call_id: str,
        payload_digest: str,
        result: dict[str, Any],
    ) -> None:
        self._rows[call_id] = (payload_digest, "committed", dict(result))

    async def mark_uncertain(self, call_id: str, payload_digest: str) -> None:
        self._rows[call_id] = (payload_digest, "uncertain", None)


class SQLiteSemanticOutcomeOwner:
    """Durable semantic outcome and replay owner used by the worker process."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        outcome_path = Path(path)
        if not outcome_path.is_absolute():
            raise ValueError("semantic outcome path must be absolute")
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        self._path = outcome_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self._path), timeout=5.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_outcomes (
                identity_key TEXT PRIMARY KEY,
                payload_digest TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_replays (
                replay_key TEXT PRIMARY KEY,
                token_id TEXT NOT NULL,
                nonce TEXT NOT NULL,
                call_id TEXT NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        connection.close()

    async def lookup(
        self,
        call_id: str,
        payload_digest: str,
    ) -> tuple[Literal["missing", "committed", "uncertain", "mismatch"], dict[str, Any] | None]:
        return await asyncio.to_thread(
            self._lookup_sync,
            call_id,
            payload_digest,
        )

    def _lookup_sync(
        self,
        identity_key: str,
        payload_digest: str,
    ) -> tuple[Literal["missing", "committed", "uncertain", "mismatch"], dict[str, Any] | None]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT payload_digest, status, result_json FROM semantic_outcomes "
                "WHERE identity_key = ?",
                (identity_key,),
            ).fetchone()
        finally:
            connection.close()
        if row is None:
            return "missing", None
        stored_digest, status, result_json = row
        if stored_digest != payload_digest:
            return "mismatch", None
        if status == "uncertain":
            return "uncertain", None
        if status != "committed" or not isinstance(result_json, str):
            return "uncertain", None
        try:
            result = json.loads(result_json)
        except (TypeError, ValueError):
            return "uncertain", None
        if not isinstance(result, Mapping):
            return "uncertain", None
        return "committed", dict(result)

    async def commit(
        self,
        call_id: str,
        payload_digest: str,
        result: dict[str, Any],
    ) -> None:
        await asyncio.to_thread(
            self._commit_sync,
            call_id,
            payload_digest,
            result,
        )

    def _commit_sync(
        self,
        identity_key: str,
        payload_digest: str,
        result: dict[str, Any],
    ) -> None:
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_digest, status FROM semantic_outcomes "
                "WHERE identity_key = ?",
                (identity_key,),
            ).fetchone()
            if row is not None:
                stored_digest, status = row
                if stored_digest != payload_digest:
                    raise ValueError("semantic outcome identity was reused")
                if status in {"uncertain", "committed"}:
                    connection.commit()
                    return
                connection.execute(
                    "UPDATE semantic_outcomes SET status = ?, result_json = ? "
                    "WHERE identity_key = ?",
                    ("committed", encoded, identity_key),
                )
            else:
                connection.execute(
                    "INSERT INTO semantic_outcomes "
                    "(identity_key, payload_digest, status, result_json) "
                    "VALUES (?, ?, ?, ?)",
                    (identity_key, payload_digest, "committed", encoded),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    async def mark_uncertain(self, call_id: str, payload_digest: str) -> None:
        await asyncio.to_thread(
            self._mark_uncertain_sync,
            call_id,
            payload_digest,
        )

    def _mark_uncertain_sync(self, identity_key: str, payload_digest: str) -> None:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT payload_digest, status FROM semantic_outcomes "
                "WHERE identity_key = ?",
                (identity_key,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO semantic_outcomes "
                    "(identity_key, payload_digest, status, result_json) "
                    "VALUES (?, ?, ?, NULL)",
                    (identity_key, payload_digest, "uncertain"),
                )
            elif row[0] != payload_digest:
                raise ValueError("semantic outcome identity was reused")
            elif row[1] != "committed":
                connection.execute(
                    "UPDATE semantic_outcomes SET status = ? WHERE identity_key = ?",
                    ("uncertain", identity_key),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def consume(self, authority: Any, call_id: str) -> None:
        """Persist a replay identity before semantic dispatch."""
        token_id = getattr(authority, "token_id", None)
        nonce = getattr(authority, "nonce", None)
        if (
            not isinstance(token_id, str)
            or not isinstance(nonce, str)
            or not isinstance(call_id, str)
            or not token_id
            or not nonce
            or not call_id
        ):
            raise ValueError("semantic replay identity is incomplete")
        replay_key = hashlib.sha256(
            json.dumps(
                [token_id, nonce, call_id],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM semantic_replays "
                "WHERE token_id = ? AND nonce = ? AND call_id = ?",
                (token_id, nonce, call_id),
            ).fetchone()
            if existing is not None:
                raise ValueError("semantic call was replayed")
            try:
                connection.execute(
                    "INSERT INTO semantic_replays "
                    "(replay_key, token_id, nonce, call_id) VALUES (?, ?, ?, ?)",
                    (replay_key, token_id, nonce, call_id),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("semantic call was replayed") from exc
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


class SemanticWorker:
    """Replay-safe worker wrapper around a semantic dispatcher."""

    def __init__(
        self,
        *,
        handler: Callable[[Any], Awaitable[Any] | Any],
        outcome_owner: SemanticOutcomeOwner | None = None,
        preflight: Callable[[Mapping[str, Any]], Awaitable[Any] | Any] | None = None,
        replay_consumer: Callable[[Mapping[str, Any]], Awaitable[Any] | Any] | None = None,
    ) -> None:
        self._handler = handler
        self._outcome_owner = outcome_owner
        self._preflight = preflight
        self._replay_consumer = replay_consumer
        self._committed: dict[str, tuple[str, dict[str, Any]]] = {}
        self._uncertain: set[str] = set()

    async def handle_mapping(self, frame: Mapping[str, Any]) -> dict[str, Any]:
        """Handle one JSON object while replaying committed call outcomes."""

        call_id = frame.get("call_id")
        payload = frame.get("payload")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("worker.call_id is required")
        if not isinstance(payload, Mapping):
            raise ValueError("worker.payload must be an object")
        payload_dict = dict(payload)
        embedded_call_id = payload_dict.get("call_id")
        if embedded_call_id is not None and embedded_call_id != call_id:
            return _authority_invalid_result()
        if self._preflight is not None:
            try:
                preflight_result = self._preflight(payload_dict)
                if inspect.isawaitable(preflight_result):
                    await preflight_result
            except (TypeError, ValueError):
                return _authority_invalid_result()
        identity_key = _outcome_identity(call_id, payload_dict)
        payload_digest = _digest(_replay_payload(payload_dict))
        if self._outcome_owner is not None:
            state, durable = await self._outcome_owner.lookup(
                identity_key,
                payload_digest,
            )
            if state == "mismatch":
                raise ValueError("worker outcome identity was reused with a different payload")
            if state == "committed" and durable is not None:
                return dict(durable)
            if state == "uncertain":
                return _uncertain_result()
        else:
            previous = self._committed.get(identity_key)
            if previous is not None:
                if previous[0] != payload_digest:
                    raise ValueError("worker outcome identity was reused with a different payload")
                return dict(previous[1])
            if identity_key in self._uncertain:
                return _uncertain_result()
        if self._replay_consumer is not None:
            try:
                replay_result = self._replay_consumer(payload_dict)
                if inspect.isawaitable(replay_result):
                    await replay_result
            except (TypeError, ValueError):
                return _replay_denied_result()
        call = SemanticWorkerCall(call_id=call_id, payload=payload_dict)
        try:
            result = self._handler(call)
            if inspect.isawaitable(result):
                result = await result
        except Exception:  # noqa: BLE001 - handler faults fail closed
            if self._outcome_owner is not None:
                await self._outcome_owner.mark_uncertain(identity_key, payload_digest)
            else:
                self._uncertain.add(identity_key)
            return _unavailable_result("SEMANTIC_WORKER_UNAVAILABLE")
        if isinstance(result, KazusaSemanticCapabilityResultV1):
            output = result.to_dict()
        elif isinstance(result, Mapping):
            output = dict(result)
        else:
            if self._outcome_owner is not None:
                await self._outcome_owner.mark_uncertain(identity_key, payload_digest)
            else:
                self._uncertain.add(identity_key)
            raise TypeError("worker handler returned an invalid result")
        if (
            _is_mutation_payload(payload_dict)
            and output.get("status") == "unavailable"
        ):
            if self._outcome_owner is not None:
                await self._outcome_owner.mark_uncertain(identity_key, payload_digest)
            else:
                self._uncertain.add(identity_key)
            return dict(output)
        if self._outcome_owner is not None:
            await self._outcome_owner.commit(identity_key, payload_digest, output)
        else:
            self._committed[identity_key] = (payload_digest, output)
        return dict(output)

    async def handle_call(
        self,
        call: SignedSemanticCallV1,
        dispatch: Callable[[SignedSemanticCallV1], Awaitable[KazusaSemanticCapabilityResultV1]],
    ) -> KazusaSemanticCapabilityResultV1:
        """Dispatch a signed semantic call with idempotent result replay."""

        frame = {"call_id": call.call_id, "payload": call.to_dict()}
        result = await self.handle_mapping(frame)
        return KazusaSemanticCapabilityResultV1.from_mapping(result)


async def serve(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    handler: Callable[[Mapping[str, Any]], Awaitable[Mapping[str, Any]]],
    *,
    outcome_owner: SemanticOutcomeOwner,
) -> None:
    """Serve length-prefixed JSON frames until EOF."""

    async def invoke(call: SemanticWorkerCall) -> Mapping[str, Any]:
        return await handler(call.payload)

    worker = SemanticWorker(handler=invoke, outcome_owner=outcome_owner)
    try:
        while True:
            length_bytes = await reader.readexactly(4)
            length = struct.unpack(">I", length_bytes)[0]
            if length <= 0 or length > MAX_FRAME_BYTES:
                raise ValueError("worker frame length is outside the bound")
            payload = await reader.readexactly(length)
            value = json.loads(payload.decode("utf-8"))
            if not isinstance(value, Mapping):
                raise ValueError("worker frame must be an object")
            result = await worker.handle_mapping(value)
            await write_frame(writer, result)
    except asyncio.IncompleteReadError:
        return
    finally:
        writer.close()
        await writer.wait_closed()


async def write_frame(writer: asyncio.StreamWriter, value: Mapping[str, Any]) -> None:
    """Write one bounded length-prefixed JSON frame."""

    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds the body limit")
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


def _digest(value: Mapping[str, Any]) -> str:
    """Return a deterministic payload digest."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _uncertain_result() -> dict[str, Any]:
    """Return the fixed safe result for an uncertain prior mutation."""

    return KazusaSemanticCapabilityResultV1.failure(
        "unavailable",
        "OPERATION_OUTCOME_UNCERTAIN",
        "The prior mutation outcome is uncertain.",
    ).to_dict()


def _unavailable_result(code: str) -> dict[str, Any]:
    """Return a fixed safe worker-unavailable result."""

    return KazusaSemanticCapabilityResultV1.failure(
        "unavailable",
        code,
        "The semantic worker is unavailable.",
    ).to_dict()


_MUTATION_OPERATIONS = frozenset({
    "kazusa_remember_information",
    "kazusa_revise_memory",
    "kazusa_change_memory_lifecycle",
})


def _is_mutation_payload(payload: Mapping[str, Any]) -> bool:
    """Identify calls whose uncertain result must fail closed."""

    return (
        payload.get("operation") in _MUTATION_OPERATIONS
        and isinstance(payload.get("idempotency_key"), str)
        and bool(payload.get("idempotency_key"))
    )


def _outcome_identity(call_id: str, payload: Mapping[str, Any]) -> str:
    """Return a restart-stable result identity for one transport frame."""

    if _is_mutation_payload(payload):
        return f"mutation:{payload['idempotency_key']}"
    return f"call:{call_id}"


def _replay_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Remove transport-only fields from the semantic result digest."""

    replay_payload = dict(payload)
    replay_payload.pop("call_id", None)
    replay_payload.pop("signature", None)
    return replay_payload


def _authority_invalid_result() -> dict[str, Any]:
    """Return the fixed safe result for an unauthenticated semantic frame."""

    return KazusaSemanticCapabilityResultV1.failure(
        "denied",
        "SEMANTIC_AUTHORITY_INVALID",
        "The semantic authority is invalid.",
    ).to_dict()


def _replay_denied_result() -> dict[str, Any]:
    """Return the fixed safe result for a consumed semantic call identity."""

    return KazusaSemanticCapabilityResultV1.failure(
        "denied",
        "SEMANTIC_AUTHORITY_REPLAYED",
        "The semantic call has already been consumed.",
    ).to_dict()


def _parse_signed_call(value: object) -> SignedSemanticCallV1:
    """Parse one exact signed semantic call received over the worker wire."""

    if not isinstance(value, Mapping):
        raise ValueError("semantic call must be an object")
    expected = {
        "schema_version",
        "call_id",
        "operation",
        "arguments",
        "arguments_digest",
        "issued_reference_digest",
        "idempotency_key",
        "authority",
        "signature",
    }
    if set(value) != expected:
        raise ValueError("semantic call has unknown or missing fields")
    if value["schema_version"] != CALL_SCHEMA_VERSION:
        raise ValueError("semantic call schema is unsupported")
    call_id = value["call_id"]
    operation = value["operation"]
    arguments_digest = value["arguments_digest"]
    issued_reference_digest = value["issued_reference_digest"]
    signature = value["signature"]
    if any(
        not isinstance(item, str) or not item
        for item in (
            call_id,
            operation,
            arguments_digest,
            issued_reference_digest,
            signature,
        )
    ):
        raise ValueError("semantic call identity is incomplete")
    arguments = value["arguments"]
    if not isinstance(arguments, Mapping):
        raise ValueError("semantic call arguments must be an object")
    idempotency_key = value["idempotency_key"]
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not idempotency_key
    ):
        raise ValueError("semantic call idempotency key is invalid")
    authority = SemanticActivationAuthorityV1.from_mapping(value["authority"])
    return SignedSemanticCallV1(
        call_id=call_id,
        operation=operation,
        arguments=dict(arguments),
        authority=authority,
        arguments_digest=arguments_digest,
        issued_reference_digest=issued_reference_digest,
        idempotency_key=idempotency_key,
        signature=signature,
    )


def _build_production_dispatcher(
    authority: SemanticActivationAuthorityV1,
    *,
    secret: bytes,
) -> Any:
    """Construct the approved semantic services for one authenticated call."""

    from kazusa_ai_chatbot.db.users import list_users_by_display_name
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.conversation import (
        ConversationSemanticService,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.dispatch import SemanticCapabilityDispatcher
    from kazusa_ai_chatbot.dsh_tool_gateway.media import MediaSemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.memory import MemorySemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.people import PeopleSemanticService
    from kazusa_ai_chatbot.dsh_tool_gateway.recall_calendar import (
        RecallCalendarSemanticService,
    )

    scope = dict(authority.service_scope)
    codec = OpaqueReferenceCodec(secret=secret)

    async def find_people(
        value: str,
        *,
        operator: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        return await list_users_by_display_name(
            value,
            operator=operator,
            source="both",
            platform=scope["platform"],
            platform_channel_id=scope["platform_channel_id"],
            limit=limit,
        )

    return SemanticCapabilityDispatcher(
        conversation=ConversationSemanticService(codec=codec),
        memory=MemorySemanticService(
            codec=codec,
            source_global_user_id=scope["global_user_id"],
        ),
        people=PeopleSemanticService(codec=codec, find=find_people),
        recall_calendar=RecallCalendarSemanticService(codec=codec),
        media=MediaSemanticService(
            scope=(
                scope["platform"],
                scope["platform_channel_id"],
                scope["global_user_id"],
            ),
            codec=codec,
        ),
        expected_catalog_digest=semantic_catalog_digest(),
    )


async def _production_handler(
    call: SemanticWorkerCall,
    *,
    secret: bytes,
) -> dict[str, Any]:
    """Authenticate and dispatch one call through the production gateway."""

    signed_call = _parse_signed_call(call.payload)
    if signed_call.call_id != call.call_id:
        raise ValueError("semantic call id does not match worker frame")
    dispatcher = _build_production_dispatcher(signed_call.authority, secret=secret)
    result = await dispatcher.dispatch(signed_call)
    return result.to_dict()


def _production_preflight(
    *,
    secret: bytes,
) -> Callable[[Mapping[str, Any]], None]:
    """Build the authentication fence for production worker calls."""

    def check(payload: Mapping[str, Any]) -> SignedSemanticCallV1:
        call = _parse_signed_call(payload)
        return authenticate_semantic_call(call, secret=secret)

    return check


def _production_replay_consumer(
    *,
    secret: bytes,
    replay_owner: SQLiteSemanticOutcomeOwner,
) -> Callable[[Mapping[str, Any]], None]:
    """Build the durable replay consumer used immediately before dispatch."""

    def consume(payload: Mapping[str, Any]) -> None:
        call = _parse_signed_call(payload)
        authenticate_semantic_call(call, secret=secret)
        replay_owner.consume(call.authority, call.call_id)

    return consume


async def _read_stdio_exact(stream: Any, length: int) -> bytes | None:
    """Read exactly one bounded number of bytes from a blocking stdio stream."""

    chunks: list[bytes] = []
    remaining = length
    while remaining:
        chunk = await asyncio.to_thread(stream.read, remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


async def _write_stdio_frame(stream: Any, value: Mapping[str, Any]) -> None:
    """Write one bounded result frame to the worker stdout pipe."""

    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_FRAME_BYTES:
        raise ValueError("worker frame exceeds the body limit")
    frame = struct.pack(">I", len(payload)) + payload
    await asyncio.to_thread(stream.write, frame)
    await asyncio.to_thread(stream.flush)


def _worker_health_response(frame: Mapping[str, Any]) -> dict[str, Any]:
    """Answer the hidden framed worker liveness handshake."""

    request_id = frame.get("request_id")
    if (
        frame.get("control") != WORKER_HEALTH_CONTROL
        or set(frame) != {"control", "request_id"}
        or not isinstance(request_id, str)
        or not request_id
    ):
        return {
            "control": WORKER_HEALTH_CONTROL,
            "request_id": request_id if isinstance(request_id, str) else "invalid",
            "schema_version": WORKER_HEALTH_SCHEMA_VERSION,
            "status": "unavailable",
        }
    return {
        "control": WORKER_HEALTH_CONTROL,
        "request_id": request_id,
        "schema_version": WORKER_HEALTH_SCHEMA_VERSION,
        "status": "ready",
        "protocol": "length-prefixed-json",
    }


async def run_stdio_worker() -> None:
    """Run the production length-prefixed worker used by the DSH sidecar."""

    import sys

    secret_text = os.environ.get("KAZUSA_DSH_TOOL_GATEWAY_SECRET")
    outcome_path = os.environ.get("KAZUSA_DSH_SEMANTIC_OUTCOME_PATH")
    if not secret_text:
        raise RuntimeError("KAZUSA_DSH_TOOL_GATEWAY_SECRET is required")
    if not outcome_path:
        raise RuntimeError("KAZUSA_DSH_SEMANTIC_OUTCOME_PATH is required")
    secret = secret_text.encode("utf-8")
    owner = SQLiteSemanticOutcomeOwner(outcome_path)
    worker = SemanticWorker(
        handler=lambda call: _production_handler(call, secret=secret),
        outcome_owner=owner,
        preflight=_production_preflight(secret=secret),
        replay_consumer=_production_replay_consumer(
            secret=secret,
            replay_owner=owner,
        ),
    )
    reader = sys.stdin.buffer
    writer = sys.stdout.buffer
    while True:
        header = await _read_stdio_exact(reader, 4)
        if header is None:
            return
        length = struct.unpack(">I", header)[0]
        if length <= 0 or length > MAX_FRAME_BYTES:
            return
        encoded = await _read_stdio_exact(reader, length)
        if encoded is None:
            return
        try:
            frame = json.loads(encoded.decode("utf-8"))
            if not isinstance(frame, Mapping):
                raise ValueError("worker frame must be an object")
            if frame.get("control") == WORKER_HEALTH_CONTROL:
                result = _worker_health_response(frame)
            else:
                result = await worker.handle_mapping(frame)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
            result = _authority_invalid_result()
        await _write_stdio_frame(writer, result)


if __name__ == "__main__":
    asyncio.run(run_stdio_worker())
