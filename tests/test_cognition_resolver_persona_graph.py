"""Persona graph V2 resolver ownership tests."""

import inspect

from kazusa_ai_chatbot.nodes import persona_supervisor2 as persona_module


def test_persona_graph_has_one_v2_resolver_path() -> None:
    """The persona graph enters the V2 resolver loop without a V1 switch."""

    source = inspect.getsource(persona_module.stage_1_goal_resolver)

    assert "call_v2_resolver_loop" in source
    assert not hasattr(persona_module, "ENABLE_COGNITION_RESOLVER")
    assert not hasattr(persona_module, "call_rag_supervisor2")
