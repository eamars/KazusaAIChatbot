"""Direct ownership tests for bounded cognition graph semantics."""

from __future__ import annotations

from kazusa_ai_chatbot import service


def test_cognition_graph_projects_subjective_semantics_without_stage_topology() -> None:
    """The operator graph shows semantic products without internal call nodes."""

    output = {
        "schema_version": "cognition_output.v3",
        "appraisals": [],
        "active_character_goal": {
            "goal_kind": "clarify",
            "intent": "understand the unfamiliar object",
            "reason": "the observation does not identify it",
            "cause_summary": "an unfamiliar object appeared",
        },
        "private_monologue": (
            "I am curious, but I do not want to fake recognition."
        ),
        "response_plan": {
            "goal_resolution": "answerable_now",
            "response_goal": "ask what the object is",
            "action_requests": [],
            "resolver_requests": [],
            "epistemic_boundary": (
                "Describe only visible form; keep identity unknown."
            ),
        },
        "affect_projection": [],
        "cause_provenance": [],
    }

    nodes, _edges = service._graph_cognition_nodes(output)
    by_category = {node["category"]: node for node in nodes}

    assert by_category["active_character_goal"]["detail"][
        "private_monologue"
    ] == output["private_monologue"]
    assert by_category["response_plan"]["detail"][
        "epistemic_boundary"
    ] == output["response_plan"]["epistemic_boundary"]
    assert not {"A1", "A2", "G", "P"}.intersection(
        node["id"] for node in nodes
    )
