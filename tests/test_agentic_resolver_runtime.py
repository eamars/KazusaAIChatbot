"""Standalone public runtime tests."""

from __future__ import annotations

import pytest

from agentic_resolver import AgenticResolverRuntime
from agentic_resolver.contracts import DSHResolutionExhaustV2
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import content_digest


class FakeController:
    async def resolve(self, intake: dict[str, object]) -> dict[str, object]:
        del intake
        return {"kind": "checkpointed", "checkpoint": {"reason": "requested"}}


class EmptyCheckpointController:
    """Return the sidecar shape used when a relay checkpoint has no payload."""

    async def resolve(self, intake: dict[str, object]) -> dict[str, object]:
        del intake
        return {"kind": "checkpointed", "checkpoint": {}}


@pytest.mark.asyncio
async def test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust() -> None:
    runtime = AgenticResolverRuntime(FakeController())
    exhaust = await runtime.resolve({"schema_version": "dsh_resolution_intake.v2"})
    assert isinstance(exhaust, DSHResolutionExhaustV2)
    assert exhaust.kind == "checkpointed"


@pytest.mark.asyncio
async def test_open_carries_admission_sequence_into_empty_checkpoint_identity(
    monkeypatch,
    tmp_path,
) -> None:
    """An empty sidecar checkpoint keeps the already-admitted typed reference."""

    monkeypatch.setenv(
        "AGENTIC_RESOLVER_LLM_BASE_URL", "http://127.0.0.1:8080/v1"
    )
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_API_KEY", "route-secret")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_MODEL", "qwen27b-5090")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS", "50176")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS", "8192")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_THINKING_ENABLED", "true")
    monkeypatch.setenv("AGENTIC_RESOLVER_WORKSPACE_ROOT", str(tmp_path.resolve()))
    monkeypatch.setenv("KAZUSA_DSH_TOOL_GATEWAY_SECRET", "semantic-secret")

    continuation_ref = {
        "source_episode_id": "runtime-episode",
        "source_message_id": "runtime-message",
        "branch_id": "ordinary_response",
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": "runtime-goal",
        },
    }
    facts = [f"fact-{index}" for index in range(10)]
    execution_context = {
        "brain_conversation_ref": "chat:debug:runtime",
        "platform": "debug",
        "channel_id": "runtime-channel",
        "requester_global_user_id": "runtime-user",
        "goal_continuation_ref": continuation_ref,
    }
    start_spec = {
        "model_facts": facts,
        "model_facts_digest": content_digest(facts),
        "objective_ref": content_digest(continuation_ref),
    }
    admitted_references: list[dict[str, object]] = []

    async def before_resolve(reference: dict[str, object]) -> None:
        admitted_references.append(dict(reference))

    runtime = AgenticResolverRuntime(EmptyCheckpointController())
    exhaust = await runtime.open(
        task_session_id="session-runtime",
        operation_generation=0,
        request={"semantic_goal": "Resolve the runtime test goal."},
        execution_context=execution_context,
        start_spec=start_spec,
        before_resolve=before_resolve,
    )

    assert admitted_references[0]["last_committed_seq"] == 0
    assert exhaust.to_dict()["identity"]["last_committed_seq"] == 0


def test_runtime_builds_v2_authority_from_canonical_project_route_and_workspace(
    monkeypatch,
    tmp_path,
) -> None:
    """Runtime authority uses the project route and canonical workspace fields."""

    monkeypatch.setenv(
        "AGENTIC_RESOLVER_LLM_BASE_URL", "http://localhost:8080/v1"
    )
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_API_KEY", "route-secret")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_MODEL", "qwen27b-5090")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_CONTEXT_WINDOW_TOKENS", "50176")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_MAX_COMPLETION_TOKENS", "8192")
    monkeypatch.setenv("AGENTIC_RESOLVER_LLM_THINKING_ENABLED", "true")
    monkeypatch.setenv("AGENTIC_RESOLVER_WORKSPACE_ROOT", str(tmp_path.resolve()))
    monkeypatch.setenv("KAZUSA_DSH_SIDECAR_URL", "http://127.0.0.1:8081/rpc")
    monkeypatch.setenv("KAZUSA_DSH_RPC_TOKEN", "rpc-secret")
    monkeypatch.setenv("KAZUSA_DSH_TOOL_GATEWAY_SECRET", "semantic-secret")

    runtime = AgenticResolverRuntime.from_environment(data_root=tmp_path.resolve())
    authority = runtime.new_runtime_authority(
        objective_ref="runtime-v2",
        brain_conversation_ref="chat:debug:runtime-v2",
        service_scope={
            "platform": "debug",
            "platform_channel_id": "channel",
            "global_user_id": "user",
        },
        audience={"kind": "operator"},
        interaction_issuer="kazusa-brain",
    )
    assert authority["resolver_profile_version"] == "kazusa-resolver-standard-v2"
    assert authority["model_route"] == "qwen27b-5090"
    assert authority["workspace_root"] == str(tmp_path.resolve()).replace("\\", "/")
    assert authority["model_route_digest"].startswith("sha256:")
    assert authority["semantic_catalog_digest"].startswith("sha256:")
    assert authority["semantic_tool_authority"]["token"].startswith("ksa1.")
    assert authority["brain_conversation_ref"] == "chat:debug:runtime-v2"
    assert authority["interaction_authority"]["scope_fingerprint"].startswith("sha256:")
    assert authority["interaction_authority"]["audience_fingerprint"].startswith("sha256:")
    assert authority["activation_id"].startswith("act_")
    assert "max_model_steps" not in authority
    assert "max_tool_calls" not in authority
    assert "max_tool_bytes" not in authority
