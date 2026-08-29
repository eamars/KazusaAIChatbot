"""Standalone public runtime tests."""

from __future__ import annotations

import inspect

import pytest

from agentic_resolver import AgenticResolverRuntime
from agentic_resolver.contracts import DSHResolutionExhaustV2


class FakeController:
    async def resolve(self, intake: dict[str, object]) -> dict[str, object]:
        del intake
        return {"kind": "checkpointed", "checkpoint": {"reason": "requested"}}


@pytest.mark.asyncio
async def test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust() -> None:
    runtime = AgenticResolverRuntime(FakeController())
    exhaust = await runtime.resolve({"schema_version": "dsh_resolution_intake.v2"})
    assert isinstance(exhaust, DSHResolutionExhaustV2)
    assert exhaust.kind == "checkpointed"


def test_runtime_has_no_brain_task_resolution_rag_or_coding_import_edge() -> None:
    source = inspect.getsource(__import__("agentic_resolver.runtime", fromlist=["*"]))
    forbidden = ("brain_service", "task_resolution", ".rag", "coding_agent")
    assert all(name not in source for name in forbidden)


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
