"""V2 conversation-progress cognition and surface ownership tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    DEFAULT_BRANCH_DEFINITIONS,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.goal_cognition import (
    run_goal_cognition,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    project_state_for_prompt,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    CONTENT_PLAN_SYSTEM_PROMPT,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_input_from_global_state,
)
from llm_test_helpers import make_llm_call_config
from tests.cognition_core_v2_test_helpers import canonical_episode


NOW = "2026-07-15T00:00:00Z"


class _GoalCaptureLLM:
    """Capture one goal prompt and return a complete speech bid."""

    def __init__(self) -> None:
        self.payload: dict[str, Any] = {}

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        del config
        self.payload = json.loads(str(getattr(messages[-1], "content", "{}")))
        result = {
            "intention": "advance the unresolved thread",
            "desired_outcome": "answer the current blocker directly",
            "concrete_detail": "avoid repeating reassurance",
            "reason": "the conversation continuity names an open blocker",
            "private_monologue": "I should address the missing point directly.",
            "target_role_handles": [],
            "evidence_handles": ["e1"],
            "expected_consequences": ["the conversation advances"],
            "confidence": "high",
        }
        return SimpleNamespace(content=json.dumps(result))


def _progress() -> dict[str, Any]:
    """Build a prompt-safe progress document plus ignored operational data."""

    return {
        "status": "active",
        "episode_label": "slides help",
        "continuity": "same_episode",
        "turn_count": 8,
        "conversation_mode": "task_support",
        "episode_phase": "stuck_loop",
        "topic_momentum": "stable",
        "current_thread": "the missing third contribution point",
        "user_goal": "finish the contribution slide",
        "current_blocker": "the third point overlaps the second",
        "user_state_updates": [],
        "assistant_moves": ["reassurance"],
        "overused_moves": ["reassurance"],
        "open_loops": [{
            "text": "provide a distinct third contribution",
            "age_hint": "~5m ago",
        }],
        "resolved_threads": [],
        "avoid_reopening": [{
            "text": "do not restart the outline",
            "age_hint": "~20m ago",
        }],
        "emotional_trajectory": "tired but still engaged",
        "next_affordances": ["give one concrete distinct angle"],
        "progression_guidance": "address the missing point directly",
        "platform_channel_id": "SECRET_CHANNEL_ID",
        "source_row_ids": ["SECRET_ROW_ID"],
    }


def _payload() -> dict[str, Any]:
    """Build a V2 connector payload carrying bounded continuity."""

    character_state = build_character_production_state(updated_at=NOW)
    return build_cognition_input_from_global_state(
        {
            "cognitive_episode": canonical_episode(
                episode_id="conversation-progress",
                content="What should the missing third point be?",
                current_global_user_id="progress-user",
            ),
            "global_user_id": "progress-user",
            "user_input": "What should the missing third point be?",
            "decontexualized_input": "The participant asks for the missing point.",
            "conversation_progress": _progress(),
            "user_multimedia_input": [],
            "rag_result": {"memory_evidence": []},
        },
        mutable_state=build_acquaintance_user_state(
            global_user_id="progress-user",
            updated_at=NOW,
        ),
        character_state=character_state,
    )


def test_content_plan_owns_visible_conversation_progression() -> None:
    """Topic and progression wording remain in the semantic surface stage."""

    prompt = CONTENT_PLAN_SYSTEM_PROMPT.casefold()

    assert "实际会说出或发送的内容" in prompt
    assert "角色判断" in prompt
    assert "这个场景" in prompt
    assert "不写最终对话" in prompt


def test_connector_projects_allowlisted_bounded_conversation_progress() -> None:
    """Continuity reaches SceneContextV2 without operational identifiers."""

    scene_context = _payload()["scene_context"]
    scene = scene_context["conversation_continuity"]

    assert "missing third contribution point" in scene
    assert "avoid repeating: reassurance" in scene
    assert len(scene) <= 500
    assert "SECRET_CHANNEL_ID" not in scene
    assert "SECRET_ROW_ID" not in scene
    assert "turn_count" not in scene


@pytest.mark.asyncio
async def test_goal_branch_receives_conversation_progress_before_surface() -> None:
    """Cognition consumes continuity before selecting the visible response goal."""

    payload = _payload()
    projection = project_state_for_prompt(
        payload["mutable_state"],
        character_constraints=payload["character_constraints"],
        evidence=payload["evidence"],
    )
    context = facade._branch_context(
        projection,
        payload["mutable_state"],
        payload["evidence"],
        scene_context=payload["scene_context"],
        private_continuity_context=payload["private_continuity_context"],
    )
    llm = _GoalCaptureLLM()
    config = make_llm_call_config("conversation_progress_goal")
    services = CognitionCoreServicesV2(
        llm=llm,
        appraisal_config=config,
        goal_cognition_config=config,
        collapse_config=config,
        action_selection_config=config,
    )

    bid = await run_goal_cognition(
        DEFAULT_BRANCH_DEFINITIONS["ordinary_response"],
        {
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal:ordinary-response",
        },
        context,
        payload["evidence"],
        services,
    )

    conversation_continuity = llm.payload["semantic_context"]["scene_context"][
        "conversation_continuity"
    ]
    assert "the third point overlaps the second" in conversation_continuity
    assert llm.payload["evidence"][0]["source_kind"] == "episode"
    assert bid["concrete_detail"] == "avoid repeating reassurance"
