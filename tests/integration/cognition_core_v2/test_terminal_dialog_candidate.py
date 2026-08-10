"""Cross-boundary terminal dialog verification tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.cognition_episode import (
    CURRENT_CHARACTER_ROLE,
    CURRENT_USER_ROLE,
    NO_ROLE,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DialogGenerationContractError,
    dialog_generator,
)
from tests.unit.nodes.dialog_fixtures import build_dialog_state
from tests.cognition_core_v2_test_helpers import canonical_episode


@pytest.mark.asyncio
async def test_terminal_dialog_candidate_opposite_polarity_is_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The terminal candidate never reaches delivery after polarity failure."""

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {"content": json.dumps({"final_dialog": ["Opposite direction."]})},
    )())
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "score": 0.1,
                "hard_errors": ["polarity mismatch"],
            })
        },
    )())
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {"content": '{"score": 1.0, "issues": []}'},
    )())
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    with pytest.raises(DialogGenerationContractError):
        await dialog_generator(build_dialog_state())

    assert generator_llm.ainvoke.await_count == 3


@pytest.mark.asyncio
async def test_reward_offer_required_selection_delivers_visible_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The selected reward operation reaches role verification and delivery."""

    input_operation = {
        "operation": "the character chooses a reward",
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": NO_ROLE,
        "embedded_target_role": NO_ROLE,
    }
    selected_operation = {
        **input_operation,
        "operation": "the user gives the selected reward to the character",
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }
    state = build_dialog_state()
    episode = canonical_episode(
        episode_id="reward-offer-terminal-dialog",
        content="the user offers a reward and asks the character to choose",
        metadata={"response_operation": input_operation},
    )
    state["cognitive_episode"] = episode
    surface_input = state["text_surface_input_v2"]
    surface_input["episode"] = episode
    surface_input["intention"][
        "selected_response_operation"
    ] = selected_operation
    surface_input["selected_response_operation"] = selected_operation

    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({
            "final_dialog": [
                "I choose the reward; please give it to me.",
            ],
        }),
    ))
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"score": 1.0, "hard_errors": []}),
    ))
    role_llm = MagicMock()
    role_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"score": 1.0, "violations": []}),
    ))
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=SimpleNamespace(
        content=json.dumps({"score": 1.0, "issues": []}),
    ))
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        surface_llm,
    )

    result = await dialog_generator(state)

    assert result["final_dialog"] == [
        "I choose the reward; please give it to me.",
    ]
    role_payload = json.loads(role_llm.ainvoke.await_args.args[0][1].content)
    assert role_payload["required_role_operations"] == [{
        "response_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_owner_role": CURRENT_CHARACTER_ROLE,
        "selection_required": True,
        "embedded_actor_role": CURRENT_USER_ROLE,
        "embedded_target_role": CURRENT_CHARACTER_ROLE,
    }]
