"""Static information-flow checks for memory-writer prompt contracts."""

from __future__ import annotations

import inspect

from kazusa_ai_chatbot import memory_writer_prompt_projection as projection_module
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_consolidator_memory_units as memory_units_module,
)


def test_projection_module_exposes_stage_specific_functions() -> None:
    """Projection should stay stage-owned and require the profile name."""

    public_names = {
        name
        for name in dir(projection_module)
        if name.startswith("project_")
    }

    assert public_names == {
        "project_character_image_prompt_payload",
        "project_memory_unit_extractor_prompt_payload",
        "project_memory_unit_rewrite_prompt_payload",
        "project_reflection_promotion_prompt_payload",
        "project_relationship_prompt_payload",
    }
    for name in public_names:
        signature = inspect.signature(getattr(projection_module, name))
        assert "character_name" in signature.parameters


def test_projection_module_has_no_semantic_user_input_helpers() -> None:
    """Projection code should not classify or rewrite user-input meaning."""

    source = inspect.getsource(projection_module)

    for forbidden in (
        "_filter_",
        "_classify_",
        "_infer_",
        "_normalize_",
        "_score_",
        "_validate_",
        "_route_",
        "_reclassify_",
        "_decide_",
        "import re",
        "from re ",
        "openai",
        "anthropic",
        "embed",
    ):
        assert forbidden not in source


def test_memory_unit_extractor_uses_projected_payload_before_llm_call() -> None:
    """Extractor call site should project speaker metadata before JSON payload."""

    source = inspect.getsource(memory_units_module.extract_memory_unit_candidates)

    assert "project_memory_unit_extractor_prompt_payload" in source
    assert "character_profile" in source
    assert "json.dumps(payload" in source
