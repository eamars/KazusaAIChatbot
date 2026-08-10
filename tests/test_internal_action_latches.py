"""Focused tests for durable action-latch continuation contracts."""

from __future__ import annotations

import importlib


def test_internal_action_latch_schema_contains_fixed_lifecycle_fields() -> None:
    """The latch schema must expose the bounded retry and lease state."""

    schemas_module = importlib.import_module(
        "kazusa_ai_chatbot.db.schemas",
    )
    annotations = schemas_module.InternalActionLatchV1.__annotations__

    assert annotations["max_attempts"]
    assert annotations["claim_token"]
    assert annotations["consumed_episode_id"]
    assert annotations["purge_after"]


def test_internal_action_latch_repository_exposes_atomic_lifecycle_api() -> None:
    """The repository must own issue, claim, retry, consume, and expiry."""

    module = importlib.import_module(
        "kazusa_ai_chatbot.db.internal_action_latches",
    )

    for function_name in (
        "issue_internal_action_latch",
        "claim_due_internal_action_latch",
        "release_internal_action_latch",
        "consume_internal_action_latch",
        "fail_internal_action_latch",
        "expire_due_internal_action_latches",
    ):
        assert callable(getattr(module, function_name))
