"""Direct ownership tests for terminal dialog verification."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import (
    DialogGenerationContractError,
    dialog_generator,
)
from tests.unit.nodes.dialog_fixtures import build_dialog_state


@pytest.mark.asyncio
async def test_terminal_candidate_opposite_polarity_is_withheld(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An opposite-polarity terminal candidate is withheld after rejection."""

    invalid_dialog = "Ask me what to do next; I will follow your choice."
    generator_llm = MagicMock()
    generator_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {"content": json.dumps({"final_dialog": [invalid_dialog]})},
    )())
    semantic_llm = MagicMock()
    semantic_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {
            "content": json.dumps({
                "aligned": False,
                "hard_errors": ["Subject reversal remains."],
            })
        },
    )())
    surface_llm = MagicMock()
    surface_llm.ainvoke = AsyncMock(return_value=type(
        "Response",
        (),
        {"content": '{"aligned": true, "issues": []}'},
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

    with pytest.raises(
        DialogGenerationContractError,
        match="terminal verification",
    ):
        await dialog_generator(build_dialog_state())

    assert generator_llm.ainvoke.await_count == 3
    assert semantic_llm.ainvoke.await_count == 3
    assert surface_llm.ainvoke.await_count == 3


def test_dialog_agent_exposes_owned_contract() -> None:
    """Keep terminal dialog generation attached to this source owner."""

    assert callable(dialog_generator)
