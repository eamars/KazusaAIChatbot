"""Deterministic Brain interaction service and process-boundary tests."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tests.test_dsh_brain_interaction_contracts import _request_mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = str(PROJECT_ROOT).replace("\\", "/")
_CONTINUATION_SECRET = b"semantic-continuation-secret"


@pytest.mark.asyncio
async def test_dsh_cognition_state_contains_canonical_empty_rag_projection(
    monkeypatch,
) -> None:
    """Direct DSH cognition receives the same required RAG shape as chat."""

    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )

    async def character_profile():
        return {"name": "Kazusa", "global_user_id": "character-1"}

    async def user_profile(_global_user_id):
        return {"display_name": "User"}

    async def cognition_state(*_args, **_kwargs):
        return {}

    async def conversation_history(**_kwargs):
        return []

    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        character_profile,
    )
    monkeypatch.setattr(service_module, "get_user_profile", user_profile)
    monkeypatch.setattr(
        service_module,
        "get_user_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_character_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        conversation_history,
    )
    monkeypatch.setattr(
        service_module,
        "_action_availability_runtime_for_target",
        lambda **_kwargs: {},
    )
    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    state = await service_module._build_dsh_cognition_state(request)
    rag_result = state["rag_result"]
    assert rag_result["answer"] == ""
    assert rag_result["conversation_evidence"] == []
    assert rag_result["memory_evidence"] == []
    assert rag_result["user_image"]["user_memory_context"]
    assert state["debug_modes"] == {
        "think_only": False,
        "no_remember": True,
        "no_visual_directives": True,
    }
    episode = state["cognitive_episode"]
    assert episode["trigger_source"] == "self_cognition"
    assert episode["privacy_scope"] == "private"
    assert episode["target_scope"]["channel_type"] == "private"
    assert episode["target_scope"]["current_global_user_id"] == (
        request.global_user_id
    )
    semantic_percepts = [
        percept
        for percept in episode["percepts"]
        if percept["percept_kind"] == "dsh_interaction_context"
    ]
    assert len(semantic_percepts) == 1
    semantic_text = semantic_percepts[0]["content"]["semantic_text"]
    assert request.transient_detail not in semantic_text
    assert "runtime-authored system observation" in semantic_text
    assert "not a user-authored request" in semantic_text
    assert "user authorization" in semantic_text
    assert "user requested" not in semantic_text
    assert "user granted" not in semantic_text
    assert state["user_input"] == semantic_text
    assert state["decontextualized_input"] == semantic_text

    captured_pending: list[dict[str, object]] = []

    async def fake_run_dsh_cognition(
        state,
        *,
        pending_interaction,
        services,
    ):
        del state, services
        captured_pending.append(dict(pending_interaction))
        return {
            "decision": "reject",
            "answer": None,
            "response_goal": None,
            "relay_mode": None,
            "reason": "the runtime event is not user authorization",
        }

    monkeypatch.setattr(
        service_module,
        "run_dsh_interaction_cognition",
        fake_run_dsh_cognition,
    )
    await service_module._production_dsh_judge(request, {})
    assert captured_pending[0]["transient_detail"] == request.transient_detail


@pytest.mark.asyncio
async def test_dsh_cognition_reply_state_uses_only_user_reply(monkeypatch) -> None:
    """A matched reply is the sole user-authored episode percept."""

    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )

    async def character_profile():
        return {"name": "Kazusa", "global_user_id": "character-1"}

    async def user_profile(_global_user_id):
        return {"display_name": "User"}

    async def cognition_state(*_args, **_kwargs):
        return {}

    async def conversation_history(**_kwargs):
        return []

    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        character_profile,
    )
    monkeypatch.setattr(service_module, "get_user_profile", user_profile)
    monkeypatch.setattr(
        service_module,
        "get_user_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_character_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        conversation_history,
    )
    monkeypatch.setattr(
        service_module,
        "_action_availability_runtime_for_target",
        lambda **_kwargs: {},
    )

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    reply_text = "I approve this exact operation once."
    state = await service_module._build_dsh_cognition_state(
        request,
        user_reply_text=reply_text,
    )

    episode = state["cognitive_episode"]
    assert episode["trigger_source"] == "user_message"
    assert episode["privacy_scope"] == "conversation"
    dialog_percepts = [
        percept
        for percept in episode["percepts"]
        if percept["source_kind"] == "dialog"
    ]
    assert len(dialog_percepts) == 1
    assert dialog_percepts[0]["content"] == {"text": reply_text}
    assert state["user_input"] == reply_text
    assert state["decontextualized_input"] == reply_text
    assert request.transient_detail not in json.dumps(episode)
    assert request.transient_detail not in state["user_input"]
    assert request.transient_detail not in state["decontextualized_input"]


@pytest.mark.asyncio
async def test_dsh_cognition_state_projects_bounded_channel_history(
    monkeypatch,
) -> None:
    """Direct DSH cognition receives bounded chronological channel context."""

    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.config import (
        CHAT_HISTORY_RECENT_LIMIT,
        CONVERSATION_HISTORY_LIMIT,
    )
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )

    history_rows = []
    for index in range(CHAT_HISTORY_RECENT_LIMIT + 1):
        role = "assistant" if index % 2 else "user"
        display_name = "" if role == "assistant" else f"User {index}"
        history_rows.append({
            "body_text": f"history-body-{index}",
            "display_name": display_name,
            "platform_message_id": f"history-message-{index}",
            "platform_user_id": f"platform-user-{index}",
            "global_user_id": f"global-user-{index}",
            "role": role,
            "addressed_to_global_user_ids": [],
            "mentions": [],
            "broadcast": False,
            "timestamp": f"2026-08-28T00:00:0{index}Z",
        })
    history_calls = []

    async def character_profile():
        return {"name": "Kazusa", "global_user_id": "character-1"}

    async def user_profile(_global_user_id):
        return {"display_name": "User"}

    async def cognition_state(*_args, **_kwargs):
        return {}

    async def conversation_history(**kwargs):
        history_calls.append(kwargs)
        return history_rows

    monkeypatch.setattr(
        service_module,
        "_load_latest_character_profile_snapshot",
        character_profile,
    )
    monkeypatch.setattr(service_module, "get_user_profile", user_profile)
    monkeypatch.setattr(
        service_module,
        "get_user_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_character_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(
        service_module,
        "get_conversation_history",
        conversation_history,
    )
    monkeypatch.setattr(
        service_module,
        "_action_availability_runtime_for_target",
        lambda **_kwargs: {},
    )

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    state = await service_module._build_dsh_cognition_state(request)

    assert history_calls == [{
        "platform": request.platform,
        "platform_channel_id": request.platform_channel_id,
        "limit": CONVERSATION_HISTORY_LIMIT,
    }]
    assert len(state["chat_history_wide"]) == len(history_rows)
    expected_recent = state["chat_history_wide"][-CHAT_HISTORY_RECENT_LIMIT:]
    assert state["chat_history_recent"] == expected_recent

    conversation_evidence = state["rag_result"]["conversation_evidence"]
    assert len(conversation_evidence) == CHAT_HISTORY_RECENT_LIMIT
    evidence_text = "\n".join(conversation_evidence)
    assert "history-body-0" not in evidence_text
    assert "User 2: history-body-2" in evidence_text
    assert "Kazusa: history-body-1" in evidence_text
    assert "Kazusa: history-body-3" in evidence_text
    for index in range(1, CHAT_HISTORY_RECENT_LIMIT + 1):
        assert f"history-body-{index}" in evidence_text
    evidence_positions = [
        evidence_text.index(f"history-body-{index}")
        for index in range(1, CHAT_HISTORY_RECENT_LIMIT + 1)
    ]
    assert evidence_positions == sorted(evidence_positions)
    assert set(state["rag_result"]) == {
        "answer",
        "user_image",
        "user_memory_unit_candidates",
        "character_image",
        "third_party_profiles",
        "memory_evidence",
        "recall_evidence",
        "conversation_evidence",
        "external_evidence",
        "supervisor_trace",
    }


@pytest.mark.asyncio
async def test_production_dsh_deliver_passes_explicit_dialog_scope_and_receipt(
    monkeypatch,
) -> None:
    """A DSH relay uses the shared dialog contract and durable dispatch receipt."""

    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )

    debug_modes = {
        "think_only": False,
        "no_remember": True,
        "no_visual_directives": True,
    }
    chat_history_wide = [{
        "role": "user",
        "body_text": "Earlier channel context",
    }]
    chat_history_recent = [{
        "role": "assistant",
        "body_text": "Most recent channel context",
    }]
    dialog_states = []
    dispatch_calls = []
    registry = object()

    async def cognition_state(_request):
        return {
            "character_profile": {
                "name": "Kazusa",
                "global_user_id": "character-1",
            },
            "debug_modes": dict(debug_modes),
            "chat_history_wide": chat_history_wide,
            "chat_history_recent": chat_history_recent,
        }

    async def dialog_agent(state):
        dialog_states.append(state)
        return {"final_dialog": ["Please confirm the requested action."]}

    async def send_message(payload, context, adapters):
        dispatch_calls.append((payload, context, adapters))
        return {
            "adapter_message_id": "platform-message-1",
            "delivery_tracking_id": "delivery-1",
        }

    monkeypatch.setattr(
        service_module,
        "_build_dsh_cognition_state",
        cognition_state,
    )
    monkeypatch.setattr(service_module, "dialog_agent", dialog_agent)
    monkeypatch.setattr(service_module, "handle_send_message", send_message)
    monkeypatch.setattr(service_module, "_adapter_registry", registry)

    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())
    receipt = await service_module._production_dsh_deliver(
        {
            "response_goal": "Ask the user whether to continue.",
            "relay_mode": "approval",
        },
        request,
    )

    assert len(dialog_states) == 1
    assert dialog_states[0]["dialog_usage_mode"] == "dsh_relay_visible"
    assert dialog_states[0]["debug_modes"] == debug_modes
    assert dialog_states[0]["chat_history_wide"] == chat_history_wide
    assert dialog_states[0]["chat_history_recent"] == chat_history_recent
    assert len(dispatch_calls) == 1
    assert dispatch_calls[0][2] is registry
    assert dispatch_calls[0][0]["text"] == (
        "Please confirm the requested action."
    )
    assert receipt["platform_message_id"] == "platform-message-1"
    assert receipt["delivery_tracking_id"] == "delivery-1"
    assert receipt["adapter"] == request.platform
    assert isinstance(receipt["delivered_at"], str)
    assert receipt["delivered_at"]


def _canonical_continuation_issuer(request, row, grant):
    """Issue the shared activation envelope for deterministic test owners."""

    from kazusa_ai_chatbot.dsh_tool_gateway.authority import (
        SemanticActivationAuthorityV1,
        issue_activation_token,
    )
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest

    del row
    issued_at = grant.issued_at if grant is not None else request.issued_at
    issued = datetime.fromisoformat(issued_at.replace("Z", "+00:00"))
    expires = issued + timedelta(minutes=5)
    scope = {
        "platform": request.platform,
        "platform_channel_id": request.platform_channel_id,
        "global_user_id": request.global_user_id,
    }
    authority = SemanticActivationAuthorityV1(
        activation_id=request.activation_id,
        lease_epoch=request.lease_epoch,
        resolution_thread_id=request.resolution_thread_id,
        segment_id=request.segment_id,
        brain_conversation_ref=request.brain_conversation_ref,
        service_scope=scope,
        scope_fingerprint=content_digest(scope),
        audience_fingerprint=request.audience_fingerprint,
        workspace_root=_WORKSPACE_ROOT,
        route_digest=request.model_route_digest,
        catalog_digest=request.catalog_digest,
        profile_version=request.profile_version,
        model_route_digest=request.model_route_digest,
        workspace_fingerprint=content_digest({"workspace_root": _WORKSPACE_ROOT}),
        issued_reference_digest=request.issued_reference_digest,
        policy_epoch=request.policy_epoch,
        interaction_issuer=request.issuer,
        issued_at=issued_at,
        expires_at=expires.isoformat().replace("+00:00", "Z"),
        token_id=f"tok_{request.interaction_id}",
        nonce=f"nonce_{request.interaction_id}",
    )
    return issue_activation_token(
        authority,
        secret=_CONTINUATION_SECRET,
        now=issued_at,
    )


def _run_brain_child(
    request: dict[str, object],
    judge_result: dict[str, object],
) -> dict[str, object]:
    """Run the TypeScript provider against a fresh FastAPI Brain process."""

    brain_script = """
import asyncio
import json
import pathlib
import sys

from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService
from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
from kazusa_ai_chatbot import service as service_module
from uvicorn import Config, Server


async def main():
    request = DshBrainInteractionRequestV1.from_mapping(json.loads(sys.argv[1]))
    result_value = json.loads(sys.argv[2])
    ready_path = pathlib.Path(sys.argv[3])
    port = int(sys.argv[4])
    secret = b"process-boundary-secret"

    async def judge(request, context):
        del context
        result = dict(result_value)
        result.update({
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
        })
        return result

    async def deliver(response_goal, request):
        del response_goal, request
        return {
            "platform_message_id": "platform-message-1",
            "delivered_at": "2026-08-28T00:00:00Z",
        }

    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=InMemoryInteractionStore(),
        context_provider=lambda request: {
            "workspace_fingerprint": request.workspace_fingerprint,
            "policy_epoch": request.policy_epoch,
        },
        deliver=deliver,
    )
    service_module.configure_dsh_interaction_service(service)
    server = Server(Config(
        service_module.app,
        host="127.0.0.1",
        port=port,
        lifespan="off",
        log_config=None,
        log_level="critical",
    ))
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        if server_task.done():
            await server_task
        await asyncio.sleep(0.01)
    ready_path.write_text("ready", encoding="utf-8")
    await server_task


asyncio.run(main())
"""
    environment = os.environ.copy()
    source_root = str(PROJECT_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        item for item in (source_root, environment.get("PYTHONPATH")) if item
    )
    with tempfile.TemporaryDirectory(prefix="dsh-brain-boundary-") as temporary:
        ready_path = Path(temporary) / "ready"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = int(probe.getsockname()[1])
        brain = subprocess.Popen(
            [
                sys.executable,
                "-c",
                brain_script,
                json.dumps(request, ensure_ascii=False),
                json.dumps(judge_result, ensure_ascii=False),
                str(ready_path),
                str(port),
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        try:
            deadline = time.monotonic() + 10
            while not ready_path.exists() and time.monotonic() < deadline:
                if brain.poll() is not None:
                    stderr = brain.stderr.read() if brain.stderr is not None else ""
                    raise AssertionError(
                        f"Brain process exited before startup: {stderr}"
                    )
                time.sleep(0.025)
            assert ready_path.exists(), "Brain process did not become ready"
            request_json = json.dumps(request, ensure_ascii=False)
            endpoint_json = json.dumps(f"http://127.0.0.1:{port}")
            module_path_json = json.dumps(
                str(
                    PROJECT_ROOT
                    / "sidecars"
                    / "dsh_resolution"
                    / "dist"
                    / "src"
                    / "brain_interaction.js"
                )
            )
            node_script = (
                """
import { pathToFileURL } from "node:url";

const request = """
                + request_json
                + """;
const endpoint = """
                + endpoint_json
                + """;
const modulePath = """
                + module_path_json
                + """;
const { createBrainInteractionProvider } = await import(pathToFileURL(modulePath).href);
let brainResponse = null;
const provider = createBrainInteractionProvider({
  secret: "process-boundary-secret",
  request: async (value) => {
    const response = await fetch(`${endpoint}/runtime/dsh/interactions`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer process-boundary-secret",
      },
      body: JSON.stringify(value),
    });
    if (!response.ok) throw new Error(`Brain interaction failed: ${response.status}`);
    brainResponse = await response.json();
    return brainResponse;
  },
  checkpoint: async (value) => {
    const response = await fetch(`${endpoint}/runtime/dsh/interactions/checkpoint`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: "Bearer process-boundary-secret",
      },
      body: JSON.stringify(value),
    });
    if (!response.ok) throw new Error(`Brain checkpoint failed: ${response.status}`);
    return await response.json();
  },
});
const result = await provider.handle(request);
process.stdout.write(JSON.stringify({ provider_result: result, brain_response: brainResponse }));
"""
            )
            completed = subprocess.run(
                ["node", "--input-type=module", "-e", node_script],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
                env=environment,
            )
            assert completed.returncode == 0, completed.stderr
            return json.loads(completed.stdout)
        finally:
            brain.terminate()
            try:
                brain.wait(timeout=5)
            except subprocess.TimeoutExpired:
                brain.kill()
                brain.wait(timeout=5)


@pytest.mark.asyncio
async def test_signed_loopback_interaction_returns_immediate_decision_or_durable_checkpoint_required() -> None:
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore

    secret = b"brain-secret"
    request = sign_request(
        DshBrainInteractionRequestV1.from_mapping(_request_mapping()),
        secret=secret,
    )

    async def judge(request, context):
        del context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "answer",
            "answer": "answered",
            "reason": "context",
            "response_goal": None,
            "relay_mode": None,
        }

    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=InMemoryInteractionStore(),
    )
    result = await service.handle_signed(request)
    assert result["decision"] == "answer"
    assert result["answer"] == "answered"
    assert result["checkpoint_required"] is False


@pytest.mark.asyncio
async def test_immediate_allow_once_consumes_durable_grant_and_replay_is_rejected() -> None:
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"immediate-approval-secret"
    value = _request_mapping()
    value.update({"kind": "approval", "tool_name": "pwsh"})
    request = DshBrainInteractionRequestV1.from_mapping(value)
    signed = sign_request(request, secret=secret)
    judge_calls = 0

    async def judge(request, context):
        nonlocal judge_calls
        del context
        judge_calls += 1
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "allow_once",
            "answer": None,
            "response_goal": None,
            "relay_mode": None,
            "reason": "the native approval is allowed once",
        }

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=store,
    )
    first = await service.handle_signed(signed)
    replay = await service.handle_signed(signed)
    assert first["decision"] == "allow_once"
    assert first["grant"]["grant_status"] == "consumed"
    assert first["grant"]["activation_id"] == request.activation_id
    assert first["grant"]["lease_epoch"] == request.lease_epoch
    assert replay["decision"] == "reject"
    assert "grant" not in replay
    assert judge_calls == 1
    row = await store.get(request.interaction_id)
    assert row is not None
    assert row["grant_status"] == "consumed"
    assert row["grant"]["grant_status"] == "consumed"


@pytest.mark.asyncio
async def test_body_limit_rejects_before_fastapi_pydantic_materialization() -> None:
    from kazusa_ai_chatbot import service as service_module

    events: list[dict[str, object]] = []
    delivered = False

    async def receive() -> dict[str, object]:
        return {
            "type": "http.request",
            "body": b"x" * (32 * 1024 + 1),
            "more_body": False,
        }

    async def send(message: dict[str, object]) -> None:
        nonlocal delivered
        events.append(message)
        if message["type"] == "http.response.body":
            delivered = True

    await service_module.app(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/runtime/dsh/interactions",
            "raw_path": b"/runtime/dsh/interactions",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("127.0.0.1", 8000),
        },
        receive,
        send,
    )
    assert delivered is True
    assert events[0]["status"] == 413
    assert b"DSH_INTERACTION_BODY_TOO_LARGE" in events[1]["body"]


def test_brain_service_exposes_only_versioned_internal_dsh_request_and_response_models() -> None:
    from kazusa_ai_chatbot.brain_service.contracts import (
        DshBrainInteractionRequestV1,
        DshBrainInteractionResponseV1,
    )

    assert DshBrainInteractionRequestV1.model_fields.keys() >= {
        "schema_version", "platform", "platform_channel_id", "global_user_id",
    }
    assert DshBrainInteractionResponseV1.model_fields.keys() >= {
        "schema_version", "decision", "request_digest",
    }


def _loopback_request(*, secret: str) -> object:
    from starlette.requests import Request

    return Request({
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/runtime/dsh/interactions",
        "raw_path": b"/runtime/dsh/interactions",
        "query_string": b"",
        "headers": [
            (b"authorization", f"Bearer {secret}".encode("ascii")),
            (b"content-length", b"1024"),
        ],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    })


@pytest.mark.asyncio
async def test_runtime_dsh_interaction_endpoint_authenticates_canonical_body_and_returns_versioned_dto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.brain_service.contracts import (
        DshBrainInteractionRequestV1 as RequestDTO,
    )
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"endpoint-secret"
    request = DshBrainInteractionRequestV1.from_mapping(_request_mapping())

    async def judge(request, context):
        del context
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "answer",
            "answer": "The typed Brain answer.",
            "response_goal": None,
            "relay_mode": None,
            "reason": "bounded context",
        }

    interaction_service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=InMemoryInteractionStore(),
    )
    monkeypatch.setattr(
        service_module,
        "_dsh_interaction_service",
        interaction_service,
    )
    signed = sign_request(request, secret=secret)
    response = await service_module.runtime_dsh_interaction(
        RequestDTO.model_validate(signed.to_dict()),
        _loopback_request(secret="endpoint-secret"),
    )
    assert response.schema_version == "dsh_brain_interaction.v1"
    assert response.decision == "answer"
    assert response.answer == "The typed Brain answer."

    with pytest.raises(service_module.HTTPException) as error:
        await service_module.runtime_dsh_interaction(
            RequestDTO.model_validate(signed.to_dict()),
            _loopback_request(secret="wrong-secret"),
        )
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_runtime_dsh_checkpoint_endpoint_replays_only_durable_delivery_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot import service as service_module
    from kazusa_ai_chatbot.brain_service.contracts import (
        DshBrainInteractionCheckpointV1,
        DshBrainInteractionRequestV1 as RequestDTO,
    )
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"checkpoint-secret"
    value = _request_mapping()
    value["kind"] = "approval"
    value["tool_name"] = "pwsh"
    request = DshBrainInteractionRequestV1.from_mapping(value)

    async def judge(request, context):
        del context
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "relay_to_user",
            "answer": None,
            "response_goal": "Ask the user whether to continue.",
            "relay_mode": "approval",
            "reason": "approval requires user context",
        }

    async def deliver(goal, request):
        del goal, request
        return {
            "platform_message_id": "platform-message-checkpoint",
            "delivered_at": "2026-08-29T00:00:00Z",
        }

    interaction_service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=InMemoryInteractionStore(),
        deliver=deliver,
    )
    monkeypatch.setattr(
        service_module,
        "_dsh_interaction_service",
        interaction_service,
    )
    signed = sign_request(request, secret=secret)
    first = await service_module.runtime_dsh_interaction(
        RequestDTO.model_validate(signed.to_dict()),
        _loopback_request(secret="checkpoint-secret"),
    )
    assert first.checkpoint_required is True
    checkpoint = DshBrainInteractionCheckpointV1.model_validate({
        **signed.to_dict(),
        "response_goal": "Ask the user whether to continue.",
        "relay_mode": "approval",
    })
    replayed = await service_module.runtime_dsh_interaction_checkpoint(
        checkpoint,
        _loopback_request(secret="checkpoint-secret"),
    )
    assert replayed.checkpoint_required is True
    assert replayed.interaction_id == request.interaction_id


async def _exercise_service_relay_roundtrip() -> None:
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import DshBrainInteractionRequestV1
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService
    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore

    secret = b"brain-secret"
    value = _request_mapping()
    value["kind"] = "approval"
    value["tool_name"] = "pwsh"
    request = sign_request(
        DshBrainInteractionRequestV1.from_mapping(value),
        secret=secret,
    )
    delivered: list[dict[str, object]] = []
    continued: list[dict[str, object]] = []

    async def judge(request, context):
        del context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "relay_to_user",
            "answer": None,
            "reason": "the approval needs user context",
            "response_goal": "Ask the user whether this exact workspace command is allowed.",
            "relay_mode": "approval",
        }

    async def deliver(response_goal, request):
        del request
        delivered.append(response_goal)
        return {"platform_message_id": "message-1", "delivered_at": "2026-08-28T00:00:00Z"}

    async def continue_resolution(**kwargs):
        continued.append(kwargs)
        return {"status": "continued", "thread": kwargs["resolution_thread_id"]}

    async def reply_judge(pending, reply_context):
        del reply_context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": pending["interaction_id"],
            "request_digest": pending["request_digest"],
            "decision": "allow_once",
            "reason": "the semantic reply authorizes the matching operation",
            "answer": None,
        }

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        reply_judge=reply_judge,
        interaction_store=store,
        deliver=deliver,
        continue_resolution=continue_resolution,
        context_provider=lambda request: {
            "workspace_fingerprint": request.workspace_fingerprint,
            "policy_epoch": "kazusa-resolver-standard-v2",
        },
        issue_continuation_authority=_canonical_continuation_issuer,
    )
    first = await service.handle_signed(request)
    assert first["decision"] == "relay_to_user"
    assert first["checkpoint_required"] is True
    assert first["delivered_platform_message_id"] == "message-1"
    row = await store.get(request.interaction_id)
    assert row is not None
    assert row["delivered_platform_message_id"] == "message-1"
    assert row["delivery_receipt"]
    assert delivered == [{
        "response_goal": "Ask the user whether this exact workspace command is allowed.",
        "relay_mode": "approval",
    }]

    resumed = await service.handle_user_reply(
        platform="debug",
        platform_channel_id="channel-1",
        global_user_id="user-1",
        reply_to_platform_message_id="message-1",
        reply_platform_message_id="reply-1",
        reply_text="yes, go ahead",
        now="2026-08-28T00:00:05Z",
    )
    assert resumed["status"] == "continued"
    assert resumed["resolution_thread_id"] == request.resolution_thread_id
    assert resumed["segment_id"] == request.segment_id
    assert resumed["activation_id"] == request.activation_id
    assert resumed["lease_epoch"] == request.lease_epoch
    assert continued[0]["resolution_thread_id"] == request.resolution_thread_id
    assert continued[0]["segment_id"] == request.segment_id
    assert "reply_text" not in continued[0]
    assert "decision" in continued[0]
    final_row = await store.get(request.interaction_id)
    assert final_row is not None
    assert final_row["status"] == "replied"
    assert final_row["result"] == resumed


@pytest.mark.asyncio
async def test_service_relay_uses_cognition_dialog_dispatcher_then_resumes_after_normal_chat_commit() -> None:
    await _exercise_service_relay_roundtrip()


@pytest.mark.asyncio
async def test_continuation_lineage_conflict_is_rejected_before_reply_commit() -> None:
    """Continuation output cannot override the durable interaction lineage."""

    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"continuation-lineage-secret"
    request_value = _request_mapping()
    request_value.update({
        "interaction_id": "continuation-lineage-conflict",
        "kind": "approval",
        "tool_name": "pwsh",
    })
    request = sign_request(
        DshBrainInteractionRequestV1.from_mapping(request_value),
        secret=secret,
    )

    async def judge(request, context):
        del context
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "relay_to_user",
            "answer": None,
            "response_goal": "Ask whether the exact operation may continue.",
            "relay_mode": "approval",
            "reason": "the operation needs user approval",
        }

    async def deliver(response_goal, request):
        del response_goal, request
        return {
            "platform_message_id": "continuation-lineage-message",
            "delivered_at": "2026-08-29T00:00:00Z",
        }

    async def reply_judge(pending, reply_context):
        del reply_context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": pending["interaction_id"],
            "request_digest": pending["request_digest"],
            "decision": "allow_once",
            "answer": None,
            "reason": "the typed reply authorizes this exact operation",
        }

    async def continue_resolution(**kwargs):
        del kwargs
        return {
            "status": "continued",
            "resolution_thread_id": "different-resolution-thread",
        }

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=store,
        reply_judge=reply_judge,
        deliver=deliver,
        continue_resolution=continue_resolution,
        issue_continuation_authority=_canonical_continuation_issuer,
    )
    first = await service.handle_signed(request)
    assert first["checkpoint_required"] is True

    with pytest.raises(
        ValueError,
        match="continuation lineage mismatch: resolution_thread_id",
    ):
        await service.handle_user_reply(
            platform="debug",
            platform_channel_id="channel-1",
            global_user_id="user-1",
            reply_to_platform_message_id="continuation-lineage-message",
            reply_platform_message_id="continuation-lineage-reply",
            reply_text="yes",
            now=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        )

    row = await store.get(request.interaction_id)
    assert row is not None
    assert row["status"] == "continuation_pending"
    assert row["grant_status"] == "available"
    assert row["reply_result"]["decision"] == "allow_once"
    assert row["result"]["status"] == "continuation_pending"
    assert "resolution_thread_id" not in row["result"]


@pytest.mark.asyncio
async def test_relayed_grant_waits_for_matching_fresh_approval_request() -> None:
    """A reply grants one exact native retry before cognition is consulted."""

    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"grant-lifecycle-secret"
    source_value = _request_mapping()
    source_value.update({
        "interaction_id": "relay-source",
        "dsh_call_id": "native-call-1",
        "kind": "approval",
        "tool_name": "pwsh",
        "nonce": "native-nonce-1",
    })
    source = sign_request(
        DshBrainInteractionRequestV1.from_mapping(source_value),
        secret=secret,
    )
    judge_calls: list[str] = []

    async def judge(request, context):
        del context
        judge_calls.append(request.interaction_id)
        decision = (
            "relay_to_user"
            if request.interaction_id == source.interaction_id
            else "reject"
        )
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": decision,
            "answer": None,
            "response_goal": (
                "Ask whether the exact native operation may continue."
                if decision == "relay_to_user"
                else None
            ),
            "relay_mode": "approval" if decision == "relay_to_user" else None,
            "reason": "the operation requires the matching approval context",
        }

    async def deliver(response_goal, request):
        del response_goal, request
        return {
            "platform_message_id": "grant-lifecycle-message",
            "delivered_at": "2026-08-29T00:00:00Z",
        }

    async def reply_judge(pending, reply_context):
        del reply_context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": pending["interaction_id"],
            "request_digest": pending["request_digest"],
            "decision": "allow_once",
            "answer": None,
            "reason": "the typed reply authorizes this exact operation",
        }

    async def continue_resolution(**kwargs):
        del kwargs
        return {"status": "continued"}

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=store,
        reply_judge=reply_judge,
        deliver=deliver,
        continue_resolution=continue_resolution,
        issue_continuation_authority=_canonical_continuation_issuer,
    )
    first = await service.handle_signed(source)
    assert first["checkpoint_required"] is True
    reply_now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    await service.handle_user_reply(
        platform="debug",
        platform_channel_id="channel-1",
        global_user_id="user-1",
        reply_to_platform_message_id="grant-lifecycle-message",
        reply_platform_message_id="grant-lifecycle-reply",
        reply_text="I approve this exact operation.",
        now=reply_now,
    )
    source_row = await store.get(source.interaction_id)
    assert source_row is not None
    assert source_row["grant_status"] == "available"
    assert source_row["grant"]["grant_status"] == "available"
    assert source_row["grant"]["activation_id"] == source.activation_id
    assert source_row["grant"]["lease_epoch"] == source.lease_epoch
    assert source_row["continuation_authority_token"]

    mismatch_value = dict(source_value)
    mismatch_value.update({
        "interaction_id": "retry-mismatch",
        "dsh_call_id": "native-call-mismatch",
        "nonce": "native-nonce-mismatch",
        "arguments_digest": "sha256:different-executable-operation",
    })
    mismatch = sign_request(
        DshBrainInteractionRequestV1.from_mapping(mismatch_value),
        secret=secret,
    )
    mismatch_result = await service.handle_signed(mismatch)
    assert mismatch_result["decision"] == "reject"
    assert judge_calls == [source.interaction_id, mismatch.interaction_id]
    source_row = await store.get(source.interaction_id)
    assert source_row is not None
    assert source_row["grant_status"] == "available"

    stale_activation_value = dict(source_value)
    stale_activation_value.update({
        "interaction_id": "retry-stale-activation",
        "activation_id": "activation-stale",
        "dsh_call_id": "native-call-stale-activation",
        "nonce": "native-nonce-stale-activation",
    })
    stale_activation = sign_request(
        DshBrainInteractionRequestV1.from_mapping(stale_activation_value),
        secret=secret,
    )
    stale_activation_result = await service.handle_signed(stale_activation)
    assert stale_activation_result["decision"] == "reject"
    source_row = await store.get(source.interaction_id)
    assert source_row is not None
    assert source_row["grant_status"] == "available"

    stale_lease_value = dict(source_value)
    stale_lease_value.update({
        "interaction_id": "retry-stale-lease",
        "lease_epoch": source.lease_epoch + 1,
        "dsh_call_id": "native-call-stale-lease",
        "nonce": "native-nonce-stale-lease",
    })
    stale_lease = sign_request(
        DshBrainInteractionRequestV1.from_mapping(stale_lease_value),
        secret=secret,
    )
    stale_lease_result = await service.handle_signed(stale_lease)
    assert stale_lease_result["decision"] == "reject"
    source_row = await store.get(source.interaction_id)
    assert source_row is not None
    assert source_row["grant_status"] == "available"

    retry_value = dict(source_value)
    retry_value.update({
        "interaction_id": "retry-exact",
        "dsh_call_id": "native-call-2",
        "nonce": "native-nonce-2",
    })
    retry = sign_request(
        DshBrainInteractionRequestV1.from_mapping(retry_value),
        secret=secret,
    )
    retry_result = await service.handle_signed(retry)
    assert retry_result["decision"] == "allow_once"
    assert retry_result["interaction_id"] == retry.interaction_id
    assert retry_result["grant"]["grant_status"] == "consumed"
    assert retry_result["grant"]["activation_id"] == retry.activation_id
    assert retry_result["grant"]["lease_epoch"] == retry.lease_epoch
    assert judge_calls == [
        source.interaction_id,
        mismatch.interaction_id,
        stale_activation.interaction_id,
        stale_lease.interaction_id,
    ]

    source_row = await store.get(source.interaction_id)
    assert source_row is not None
    assert source_row["grant_status"] == "consumed"
    assert source_row["grant"]["grant_status"] == "consumed"
    retry_row = await store.get(retry.interaction_id)
    assert retry_row is not None
    assert retry_row["status"] == "decided"
    assert retry_row["decision_state"] == "allow_once"


@pytest.mark.asyncio
async def test_reply_approval_reconciles_after_consumption_or_continuation_transport_loss() -> None:
    """An available approval keeps its continuation retryable after transport loss."""

    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"reply-reconcile-secret"
    request_value = _request_mapping()
    request_value.update({"kind": "approval", "tool_name": "pwsh"})
    request = sign_request(
        DshBrainInteractionRequestV1.from_mapping(request_value),
        secret=secret,
    )
    callback_calls = 0

    async def judge(request, context):
        del context
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "relay_to_user",
            "answer": None,
            "response_goal": "Ask whether the command may continue.",
            "relay_mode": "approval",
            "reason": "the operation needs user approval",
        }

    async def deliver(response_goal, request):
        del response_goal, request
        return {
            "platform_message_id": "reconcile-message",
            "delivered_at": "2026-08-28T00:00:00Z",
        }

    async def reply_judge(pending, reply_context):
        del reply_context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": pending["interaction_id"],
            "request_digest": pending["request_digest"],
            "decision": "allow_once",
            "answer": None,
            "reason": "the typed reply authorizes this exact operation",
        }

    async def continue_resolution(**kwargs):
        nonlocal callback_calls
        callback_calls += 1
        if callback_calls == 1:
            raise RuntimeError("simulated continuation transport loss")
        return {"status": "continued", "thread": kwargs["resolution_thread_id"]}

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=store,
        reply_judge=reply_judge,
        deliver=deliver,
        continue_resolution=continue_resolution,
        issue_continuation_authority=_canonical_continuation_issuer,
    )
    first = await service.handle_signed(request)
    assert first["checkpoint_required"] is True

    with pytest.raises(RuntimeError, match="continuation transport loss"):
        await service.handle_user_reply(
            platform="debug",
            platform_channel_id="channel-1",
            global_user_id="user-1",
            reply_to_platform_message_id="reconcile-message",
            reply_platform_message_id="reply-reconcile",
            reply_text="yes",
            now="2026-08-28T00:00:05Z",
        )
    after_transport_loss = await store.get(request.interaction_id)
    assert after_transport_loss is not None
    assert after_transport_loss["status"] == "continuation_pending"
    assert after_transport_loss["grant_status"] == "available"
    assert after_transport_loss["grant"]["grant_status"] == "available"
    assert after_transport_loss["continuation_authority_token"]

    resumed = await service.handle_user_reply(
        platform="debug",
        platform_channel_id="channel-1",
        global_user_id="user-1",
        reply_to_platform_message_id="reconcile-message",
        reply_platform_message_id="reply-reconcile",
        reply_text="yes",
        now="2026-08-28T00:00:07Z",
    )
    assert resumed["status"] == "continued"
    row = await store.get(request.interaction_id)
    assert row is not None
    assert row["status"] == "replied"
    assert row["grant_status"] == "available"
    assert row["grant"]["grant_status"] == "available"
    assert callback_calls == 2


@pytest.mark.asyncio
async def test_reply_approval_requires_canonical_issuer_before_grant_consumption() -> None:
    """An unavailable continuation issuer leaves the durable grant untouched."""

    from kazusa_ai_chatbot.db.dsh_interactions import InMemoryInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.auth import sign_request
    from kazusa_ai_chatbot.dsh_interaction.contracts import (
        DshBrainInteractionRequestV1,
    )
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    secret = b"issuer-required-secret"
    request_value = _request_mapping()
    request_value.update({"kind": "approval", "tool_name": "pwsh"})
    request = sign_request(
        DshBrainInteractionRequestV1.from_mapping(request_value),
        secret=secret,
    )

    async def judge(request, context):
        del context
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "relay_to_user",
            "answer": None,
            "response_goal": "Ask whether the command may continue.",
            "relay_mode": "approval",
            "reason": "the operation needs user approval",
        }

    async def deliver(response_goal, request):
        del response_goal, request
        return {
            "platform_message_id": "issuer-message",
            "delivered_at": "2026-08-28T00:00:00Z",
        }

    async def reply_judge(pending, reply_context):
        del reply_context
        return {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": pending["interaction_id"],
            "request_digest": pending["request_digest"],
            "decision": "allow_once",
            "answer": None,
            "reason": "the typed reply authorizes this exact operation",
        }

    async def continue_resolution(**kwargs):
        del kwargs
        return {"status": "continued"}

    store = InMemoryInteractionStore()
    service = BrainInteractionService(
        secret=secret,
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=store,
        reply_judge=reply_judge,
        deliver=deliver,
        continue_resolution=continue_resolution,
    )
    await service.handle_signed(request)
    with pytest.raises(ValueError, match="canonical continuation authority issuer"):
        await service.handle_user_reply(
            platform="debug",
            platform_channel_id="channel-1",
            global_user_id="user-1",
            reply_to_platform_message_id="issuer-message",
            reply_platform_message_id="issuer-reply",
            reply_text="yes",
            now="2026-08-28T00:00:05Z",
        )
    row = await store.get(request.interaction_id)
    assert row is not None
    assert row["status"] == "delivered"
    assert row["grant_status"] is None
    assert row["grant"] is None
    assert row["continuation_authority_token"] is None


@pytest.mark.asyncio
async def test_lifespan_constructs_durable_brain_owner_after_graph_and_adapter_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Startup readiness reflects the production Brain owner composition."""

    from kazusa_ai_chatbot.db.dsh_interactions import MongoInteractionStore
    from kazusa_ai_chatbot.dsh_interaction.decision import BrainDecisionEngine
    from kazusa_ai_chatbot.dsh_interaction.service import BrainInteractionService

    import kazusa_ai_chatbot.service as service_module

    previous_service = service_module._dsh_interaction_service
    previous_runtime = service_module._dsh_resolver_runtime
    service_module._dsh_interaction_service = None
    service_module._dsh_resolver_runtime = None
    factory_observations: list[dict[str, object]] = []
    index_calls: list[str] = []

    async def no_op(*args, **kwargs):
        del args, kwargs

    def sync_no_op(*args, **kwargs):
        del args, kwargs

    async def startup_profile():
        return (
            {"name": "Kazusa", "global_user_id": "character"},
            {},
        )

    async def latest_profile():
        return {"name": "Kazusa", "global_user_id": "character"}

    async def post_commit(**kwargs):
        del kwargs
        return {"failed_count": 0}

    async def ensure_indexes():
        index_calls.append("indexes")

    class FakeResolverRuntime:
        @classmethod
        def from_environment(cls):
            factory_observations.append({
                "owner": "resolver",
                "adapter_ready": service_module._adapter_registry is not None,
            })
            return cls()

    async def judge(request, context):
        del context
        return {
            "schema_version": request.schema_version,
            "interaction_id": request.interaction_id,
            "request_digest": request.request_digest,
            "kind": request.kind,
            "decision": "reject",
            "answer": None,
            "response_goal": None,
            "relay_mode": None,
            "reason": "test owner",
        }

    owner = BrainInteractionService(
        secret=b"lifespan-brain-secret",
        judge=BrainDecisionEngine(judge=judge),
        interaction_store=MongoInteractionStore(),
    )

    def from_environment(**kwargs):
        factory_observations.append({
            "owner": "brain",
            "adapter_ready": service_module._adapter_registry is not None,
            "judge_injected": kwargs.get("judge") is not None,
            "deliver_injected": kwargs.get("deliver") is not None,
            "continuation_injected": kwargs.get("continue_resolution") is not None,
        })
        return owner

    monkeypatch.setenv("KAZUSA_DSH_BRAIN_SHARED_SECRET", "lifespan-brain-secret")
    monkeypatch.setenv("KAZUSA_DSH_TOOL_GATEWAY_SECRET", "lifespan-tool-secret")
    monkeypatch.setattr(service_module, "AgenticResolverRuntime", FakeResolverRuntime)
    monkeypatch.setattr(
        service_module.BrainInteractionService,
        "from_environment",
        staticmethod(from_environment),
    )
    monkeypatch.setattr(service_module, "build_cognition_core_services", lambda: object())
    monkeypatch.setattr(service_module, "db_bootstrap", no_op)
    monkeypatch.setattr(service_module, "close_db", no_op)
    monkeypatch.setattr(service_module, "_load_startup_character_profile", startup_profile)
    monkeypatch.setattr(service_module, "_load_latest_character_profile_snapshot", latest_profile)
    monkeypatch.setattr(
        service_module,
        "reconcile_identity_growth_post_commit",
        post_commit,
    )
    monkeypatch.setattr(service_module, "_hydrate_media_descriptor_cache", no_op)
    monkeypatch.setattr(service_module, "_build_graph", lambda: object())
    monkeypatch.setattr(service_module, "ensure_dsh_interaction_indexes", ensure_indexes)
    monkeypatch.setattr(service_module.mcp_manager, "start", no_op)
    monkeypatch.setattr(service_module.mcp_manager, "stop", no_op)
    monkeypatch.setattr(service_module, "_ensure_chat_input_worker_started", sync_no_op)
    monkeypatch.setattr(service_module, "_stop_chat_input_worker", no_op)
    monkeypatch.setattr(service_module, "render_llm_route_table", lambda: "route")
    monkeypatch.setattr(service_module, "CALENDAR_SCHEDULER_ENABLED", False)
    monkeypatch.setattr(service_module, "SELF_COGNITION_ENABLED", False)
    monkeypatch.setattr(service_module, "BACKGROUND_WORK_WORKER_ENABLED", False)
    monkeypatch.setattr(service_module, "REFLECTION_CYCLE_ENABLED", False)
    monkeypatch.setattr(
        service_module.event_logging,
        "record_resource_health_event",
        no_op,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_process_event",
        no_op,
    )
    monkeypatch.setattr(
        service_module.event_logging,
        "record_runtime_error_event",
        no_op,
    )

    try:
        async with service_module.lifespan(service_module.app):
            assert service_module._adapter_registry is not None
            assert service_module._dsh_interaction_service is owner
            health = service_module._dsh_interaction_health()
            assert health.status == "ready"
            assert health.configured is True
            assert health.durable_store is True
            assert health.cognition_judge is True
        assert index_calls == ["indexes"]
        assert factory_observations == [
            {
                "owner": "resolver",
                "adapter_ready": True,
            },
            {
                "owner": "brain",
                "adapter_ready": True,
                "judge_injected": True,
                "deliver_injected": True,
                "continuation_injected": True,
            },
        ]
    finally:
        service_module._dsh_interaction_service = previous_service
        service_module._dsh_resolver_runtime = previous_runtime


def test_real_sidecar_question_is_answered_by_brain_without_user_delivery() -> None:
    """A fresh sidecar process receives an immediate Brain answer."""

    result = _run_brain_child(
        _request_mapping(),
        {
            "decision": "answer",
            "answer": "The Brain has enough context.",
            "reason": "context",
            "response_goal": None,
            "relay_mode": None,
        },
    )
    assert result["provider_result"] == {
        "kind": "answer",
        "answer": "The Brain has enough context.",
    }
    assert result["brain_response"]["decision"] == "answer"
    assert result["brain_response"]["answer"] == "The Brain has enough context."


def test_real_sidecar_outside_workspace_retry_consumes_one_brain_grant() -> None:
    """A fresh sidecar process returns one authority grant, never two."""

    request = _request_mapping()
    request["kind"] = "approval"
    request["tool_name"] = "pwsh"
    result = _run_brain_child(
        request,
        {
            "schema_version": "dsh_brain_interaction.v1",
            "interaction_id": request["interaction_id"],
            "decision": "allow_once",
            "answer": None,
            "reason": "matching workspace approval",
            "response_goal": None,
            "relay_mode": None,
        },
    )
    assert result["provider_result"] == {"kind": "allow_once"}
    assert result["brain_response"]["decision"] == "allow_once"
    assert result["brain_response"]["grant"]["grant_status"] == "consumed"


@pytest.mark.asyncio
async def test_signed_media_inspection_resolves_only_exact_scoped_cache_ref() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.media import (
        MediaSemanticService,
        issue_attached_media_reference,
    )

    codec = OpaqueReferenceCodec(b"media-secret")
    scope = ("debug", "channel-1", "user-1")

    async def inspect(request):
        del request
        return {"status": "ok", "answer": "a cup", "evidence_boundary_notes": []}

    service = MediaSemanticService(
        scope=scope,
        codec=codec,
        get_media=lambda received_scope, cache_ref: (
            {"content_type": "image/png", "base64_data": "AA=="}
            if received_scope == scope and cache_ref == "cache-1"
            else None
        ),
        inspect=inspect,
    )
    valid = issue_attached_media_reference(
        codec=codec,
        scope=scope,
        cache_ref="cache-1",
    )
    result = await service.inspect_attached_media(
        attached_media_ref=valid,
        question="What is shown?",
    )
    assert result.status == "ok"
    foreign = issue_attached_media_reference(
        codec=codec,
        scope=("debug", "channel-2", "user-1"),
        cache_ref="cache-1",
    )
    denied = await service.inspect_attached_media(
        attached_media_ref=foreign,
        question="What is shown?",
    )
    assert denied.status == "denied"
    assert denied.error is not None
    assert denied.error.code == "MEDIA_SCOPE_MISMATCH"


@pytest.mark.asyncio
async def test_relay_checkpoints_delivers_matches_reply_and_resumes_same_thread() -> None:
    """The exact cross-boundary node executes the full relay behavior."""

    await _exercise_service_relay_roundtrip()
