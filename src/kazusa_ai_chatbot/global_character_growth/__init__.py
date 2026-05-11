"""Public facade for reflection-driven global character growth."""

from __future__ import annotations

from kazusa_ai_chatbot.global_character_growth.context import (
    build_global_character_growth_context,
)
from kazusa_ai_chatbot.global_character_growth.runner import (
    run_global_character_growth_pass,
)

__all__ = [
    "run_global_character_growth_pass",
    "build_global_character_growth_context",
]
