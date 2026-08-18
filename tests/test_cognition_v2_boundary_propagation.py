"""Fail-closed propagation at the Cognition V2 commit boundary."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import kazusa_ai_chatbot.nodes.persona_supervisor2_cognition as cognition_module
from kazusa_ai_chatbot.cognition_core_v2.contracts import CognitionExecutionError
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from tests.test_cognition_chain_connector_mapping import (
    _core_output,
    _global_state,
)


@pytest.mark.asyncio
async def test_boundary_rejection_stops_before_action_resolver_delivery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed user-state CAS prevents downstream effectful surfaces."""

    previous = build_acquaintance_user_state(
        global_user_id="boundary-propagation-user",
        updated_at="2026-08-18T00:00:00Z",
    )
    downstream_calls: list[str] = []

    async def reject_commit(*_args: object, **_kwargs: object) -> bool:
        """Represent a stale complete-state boundary."""

        return False

    async def record_commit(*_args: object, **_kwargs: object) -> None:
        """Keep telemetry outside this propagation assertion."""

    def downstream(*_args: object, **_kwargs: object) -> None:
        """Record any attempted action, resolver, or delivery handoff."""

        downstream_calls.append("effectful_boundary")

    monkeypatch.setattr(cognition_module, "run_cognition", AsyncMock(
        return_value=_core_output(),
    ))
    monkeypatch.setattr(
        cognition_module,
        "get_user_cognition_state",
        AsyncMock(return_value=previous),
    )
    monkeypatch.setattr(
        cognition_module,
        "get_character_cognition_state",
        AsyncMock(return_value=build_character_production_state(
            updated_at="2026-07-14T00:00:00Z",
        )),
    )
    monkeypatch.setattr(
        cognition_module,
        "record_continuity_boundary_event",
        AsyncMock(),
    )
    monkeypatch.setattr(
        cognition_module,
        "compare_and_replace_user_cognition_state",
        reject_commit,
    )
    monkeypatch.setattr(
        cognition_module,
        "_record_state_commit_event",
        record_commit,
    )
    monkeypatch.setattr(
        cognition_module,
        "_materialize_v2_action_requests",
        downstream,
    )

    with pytest.raises(CognitionExecutionError, match="version conflict"):
        await cognition_module.call_cognition_subgraph(
            _global_state(),
            commit=True,
        )

    assert downstream_calls == []
