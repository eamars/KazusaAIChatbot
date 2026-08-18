"""Public database facade ownership tests."""

from __future__ import annotations

from kazusa_ai_chatbot import db


def test_user_cognition_compare_and_replace_is_public() -> None:
    """Expose the canonical user-state CAS operation from ``db``."""

    assert callable(db.compare_and_replace_user_cognition_state)
