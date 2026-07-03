"""Tests for generic background-work router contracts."""

from __future__ import annotations

import importlib


def test_router_normalizer_excludes_worker_local_payload() -> None:
    """Router output should route to a worker without worker-local params."""

    router = importlib.import_module("kazusa_ai_chatbot.background_work.router")
    normalize = getattr(router, "normalize_background_work_router_output")

    decision = normalize({
        "action": "execute",
        "worker": "text_artifact",
        "task": "Generate a Fibonacci function snippet.",
        "reason": "The task is bounded text artifact work.",
        "work_kind": "coding_snippet",
        "tool_args": {"path": "fibonacci.py"},
        "adapter_id": "debug-adapter",
    })

    assert decision == {
        "action": "execute",
        "worker": "text_artifact",
        "reason": "The task is bounded text artifact work.",
    }


def test_router_normalizer_accepts_enabled_coding_agent_worker() -> None:
    """Router validation should use enabled workers, not one hardcoded name."""

    router = importlib.import_module("kazusa_ai_chatbot.background_work.router")
    normalize = getattr(router, "normalize_background_work_router_output")

    decision = normalize(
        {
            "action": "execute",
            "worker": "coding_agent",
            "reason": "The task asks for bounded source-code reading.",
        },
        enabled_workers={"text_artifact", "coding_agent"},
    )

    assert decision == {
        "action": "execute",
        "worker": "coding_agent",
        "reason": "The task asks for bounded source-code reading.",
    }


def test_background_work_router_decision_is_route_only() -> None:
    """Router decisions must contain only route fields: action, worker, reason.
    No worker-facing task string or task parameters."""

    router = importlib.import_module("kazusa_ai_chatbot.background_work.router")
    normalize = getattr(router, "normalize_background_work_router_output")

    decision = normalize({
        "action": "execute",
        "worker": "text_artifact",
        "task": "Generate a Fibonacci function snippet.",
        "reason": "The task is bounded text artifact work.",
    })

    assert "action" in decision
    assert "worker" in decision
    assert "reason" in decision
    assert "task" not in decision, (
        "router decision must not contain a worker-facing 'task' field; "
        "routers only route, they do not generate worker parameters"
    )


def test_router_payload_contains_worker_descriptions_not_job_mechanics() -> None:
    """Router prompt payload should describe workers without runtime internals."""

    router = importlib.import_module("kazusa_ai_chatbot.background_work.router")
    build_payload = getattr(router, "build_background_work_router_payload")

    payload = build_payload(
        task_brief="Generate a Fibonacci function snippet.",
        source_summary="The user asked for a simple Fibonacci generator.",
        worker_descriptions={
            "text_artifact": (
                "Bounded text artifacts including code snippets, rewrites, "
                "and summaries."
            ),
        },
        max_output_chars=3000,
    )

    serialized = repr(payload).lower()

    assert "text_artifact" in serialized
    assert "fibonacci" in serialized
    for forbidden in (
        "lease",
        "retry",
        "adapter",
        "platform_channel_id",
        "job_id",
        "mongodb",
        "work_kind",
        "coding_snippet",
    ):
        assert forbidden not in serialized
