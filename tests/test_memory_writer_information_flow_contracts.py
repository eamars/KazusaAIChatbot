"""Static information-flow checks for memory-writer prompt contracts."""

from __future__ import annotations

import inspect

from kazusa_ai_chatbot.consolidation import (
    memory_units as memory_units_module,
)






def test_memory_unit_extractor_uses_projected_payload_before_llm_call() -> None:
    """Extractor call site should project speaker metadata before JSON payload."""

    source = inspect.getsource(memory_units_module.extract_memory_unit_candidates)

    assert "project_memory_unit_extractor_prompt_payload" in source
    assert "character_profile" in source
    assert "json.dumps(payload" in source
