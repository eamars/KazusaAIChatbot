"""Standalone public runtime tests."""

from __future__ import annotations

import inspect

import pytest

from agentic_resolver import AgenticResolverRuntime
from agentic_resolver.contracts import DSHResolutionExhaustV1


class FakeController:
    async def resolve(self, intake: dict[str, object]) -> dict[str, object]:
        del intake
        return {"kind": "checkpointed", "checkpoint": {"reason": "requested"}}


@pytest.mark.asyncio
async def test_resolve_preserves_standalone_entrypoint_and_returns_typed_exhaust() -> None:
    runtime = AgenticResolverRuntime(FakeController())
    exhaust = await runtime.resolve({"schema_version": "dsh_resolution_intake.v1"})
    assert isinstance(exhaust, DSHResolutionExhaustV1)
    assert exhaust.kind == "checkpointed"


def test_runtime_has_no_brain_task_resolution_rag_or_coding_import_edge() -> None:
    source = inspect.getsource(__import__("agentic_resolver.runtime", fromlist=["*"]))
    forbidden = ("brain_service", "task_resolution", ".rag", "coding_agent")
    assert all(name not in source for name in forbidden)
