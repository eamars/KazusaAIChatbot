"""Direct ownership tests for terminal dialog generation."""

from __future__ import annotations

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from tests.unit.nodes.dialog_fixtures import build_dialog_state


def test_dialog_agent_exposes_owned_contract() -> None:
    """Keep terminal dialog generation attached to this source owner."""

    assert callable(dialog_generator)


def test_validated_dialog_messages_collapses_blank_line_runs() -> None:
    """Collapse internal blank lines while preserving message boundaries."""

    value = {
        "final_dialog": [
            "first\n\nsecond\n\nthird\n\nfourth\n\nfifth",
            "single\nline",
        ],
    }

    validated_messages = dialog_module._validated_dialog_messages(value)

    assert validated_messages == [
        "first\nsecond\nthird\nfourth\nfifth",
        "single\nline",
    ]
