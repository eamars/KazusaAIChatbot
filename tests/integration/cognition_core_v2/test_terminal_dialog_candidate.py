"""Cross-boundary terminal dialog verification tests."""

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
                "aligned": False,
                "hard_errors": ["polarity mismatch"],
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
