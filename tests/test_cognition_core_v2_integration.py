"""Checkpoint E integration tests for V2 facade and surface handoff."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v2 import facade as facade_module
from kazusa_ai_chatbot.cognition_core_v2 import (
    run_cognition,
    run_text_surface_planning,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.db.character import (
    get_character_cognition_state,
    replace_character_cognition_state,
)
from kazusa_ai_chatbot.db.users import (
    get_user_cognition_state,
    replace_user_cognition_state,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionCoreServicesV2,
    CognitionExecutionError,
    EVIDENCE_SOURCE_QUESTION_IDS,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    build_acquaintance_user_state,
    build_character_production_state,
)
from kazusa_ai_chatbot.cognition_core_v2.workspace import collapse_bids

from llm_test_helpers import make_llm_call_config
from tests.cognition_core_v2_test_helpers import (
    canonical_episode,
    canonical_identity_context,
)
from tests.live_llm_mongo import live_db, seed_shared_documents, unique_owner_id


NOW = "2026-07-14T00:00:00Z"


class _ScriptedLLM:
    """Return exact contract-shaped responses for each V2 stage."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.human_calls: list[str] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> SimpleNamespace:
        stage_name = getattr(config, "stage_name", "")
        system = str(getattr(messages[0], "content", ""))
        human = str(getattr(messages[-1], "content", "{}"))
        payload = json.loads(human)
        if stage_name == "v2_content":
            result = {
                "content_plan": "bounded content-plan guidance",
                "content_requirements": ["preserve actor direction"],
                "delivery_profile": {
                    "lexical_register": "plain",
                    "sentence_shape": "concise",
                    "rhythm": "steady",
                    "hesitation": "light",
                    "punctuation": "restrained",
                },
            }
        elif stage_name == "v2_preference":
            result = {
                "visible_boundaries": ["bounded visible boundary"],
                "addressee_plan": ["bounded addressee plan"],
            }
        elif stage_name == "v2_visual":
            result = {"visual_directives": "bounded visual directives"}
        elif "selected_evidence_handles" in system:
            question = payload["question"]
            result = {
                "question_id": question["question_id"],
                "proposition": None,
                "delta": None,
            }
        elif "private_monologue" in system:
            result = {
                "intention": "acknowledge the grounded episode",
                "desired_outcome": "maintain a coherent exchange",
                "concrete_detail": "use the current episode only",
                "reason": "the episode supplies bounded evidence",
                "private_monologue": "I want to answer this carefully.",
                "target_role_handles": [],
                "evidence_handles": ["e1"],
                "expected_consequences": ["preserve continuity"],
                "confidence": "high",
            }
        elif "primary_bid_handle" in system:
            handles = sorted(payload["bids"])
            result = {
                "primary_bid_handle": handles[0],
                "supporting_bid_handles": handles[1:],
                "suppressed_bid_handles": [],
            }
        elif "action_requests" in system:
            result = {
                "action_requests": [],
                "resolver_requests": [],
                "goal_resolution": "answerable_now",
                "resolver_pending_resolution": None,
                "resolver_goal_progress": None,
            }
        else:
            raise AssertionError("unexpected V2 model stage")
        self.calls.append(system)
        self.human_calls.append(human)
        return SimpleNamespace(content=json.dumps(result))


def _core_services(llm: _ScriptedLLM) -> CognitionCoreServicesV2:
    """Build all core stage bindings from one scripted model."""

    return CognitionCoreServicesV2(
        llm=llm,
        appraisal_event_agency_config=make_llm_call_config("v2_appraisal"),
        appraisal_relationship_social_config=make_llm_call_config(
            "v2_appraisal"
        ),
        appraisal_moral_identity_config=make_llm_call_config("v2_appraisal"),
        appraisal_goal_threat_outcome_config=make_llm_call_config(
            "v2_appraisal"
        ),
        appraisal_epistemic_comparison_memory_config=make_llm_call_config(
            "v2_appraisal"
        ),
        appraisal_existential_drive_config=make_llm_call_config(
            "v2_appraisal"
        ),
        goal_ordinary_response_config=make_llm_call_config("v2_goal"),
        goal_active_branch_config=make_llm_call_config("v2_goal"),
        workspace_collapse_config=make_llm_call_config("v2_collapse"),
        action_planning_config=make_llm_call_config("v2_route"),
        action_authorization_config=make_llm_call_config("v2_route"),
        resolver_authorization_config=make_llm_call_config("v2_route"),
    )


def _surface_services(llm: _ScriptedLLM) -> TextSurfaceServicesV2:
    """Build both text-surface stage bindings."""

    return TextSurfaceServicesV2(
        llm=llm,
        content_plan_config=make_llm_call_config("v2_content"),
        preference_config=make_llm_call_config("v2_preference"),
    )


def _visual_services(llm: _ScriptedLLM) -> VisualSurfaceServicesV2:
    """Build the terminal visual-surface binding."""

    return VisualSurfaceServicesV2(
        llm=llm,
        visual_config=make_llm_call_config("v2_visual"),
    )


def _input(
    *,
    episode_id: str = "e-integration",
    mutable_state: dict[str, object] | None = None,
    state_scope: str = "user",
    trigger_source: str = "user_message",
) -> dict[str, object]:
    """Build one evidence-grounded user episode."""

    character = build_character_production_state(updated_at=NOW)
    state = mutable_state
    if state is None:
        state = build_acquaintance_user_state(
            global_user_id="integration-user",
            updated_at=NOW,
        )
    return {
        "schema_version": "cognition_core_input.v2",
        "episode": canonical_episode(
            episode_id=episode_id,
            trigger_source=trigger_source,
            content="private evidence-grounded exchange",
            current_global_user_id=str(
                state.get("owner_user_id", "integration-user")
            ),
        ),
        "state_scope": state_scope,
        "mutable_state": state,
        "character_constraints": {
            "drives": character["drives"],
            "standards": character["standards"],
            "meaning_state": character["meaning_state"],
            "personality_judgment": {
                "logic": "analytical",
                "defense": "reserved",
                "quirks": "occasional hesitation",
                "taboos": "preserve character agency",
            },
        },
        "character_identity_context": canonical_identity_context(),
        "evidence": [{
            "evidence_handle": "e1",
            "evidence_ref": {
                "source_kind": "episode",
                "source_id": f"episode:{episode_id}",
                "occurred_at": NOW,
                "semantic_summary": "the user supplied a direct bounded episode",
            },
            "semantic_text": "the user supplied a direct bounded episode",
            "visible_to": list(EVIDENCE_SOURCE_QUESTION_IDS["episode"]),
        }],
        "direct_facts": [],
        "available_actions": [],
        "available_resolver_capabilities": [],
        "resolver_context": "resolver_status=idle",
        "scene_context": {
            "channel_scope": "private",
            "character_role": "companion",
            "semantic_scene": "private evidence-grounded exchange",
            "conversation_continuity": "Continue the current exchange.",
            "semantic_temporal_context": "immediate",
        },
        "private_continuity_context": "I remain attentive to this exchange.",
    }


@pytest.mark.asyncio
async def test_private_residual_and_group_guidance_reach_only_owned_stages(
) -> None:
    """Goal and action stages receive only their authorized restored context."""

    payload = _input()
    payload["past_dialog_cognition_context"] = "PAST_DIALOG_PRIVATE_SENTINEL"
    payload["group_engagement_action_context"] = {
        "engagement_guidelines": ["GROUP_ENGAGEMENT_SENTINEL"],
        "confidence": "high",
    }
    llm = _ScriptedLLM()

    await run_cognition(payload, _core_services(llm))

    rendered_calls = [json.loads(call) for call in llm.human_calls]
    goal_calls = [
        call for call in rendered_calls
        if "semantic_context" in call and "branch" in call
    ]
    action_calls = [
        call for call in rendered_calls
        if "action_handles" in call and "resolver_handles" in call
    ]
    appraisal_calls = [
        call for call in rendered_calls
        if "question" in call
    ]
    workspace_calls = [
        call for call in rendered_calls
        if "bids" in call and "primary_bid_handle" not in call
        and "action_handles" not in call
    ]

    assert goal_calls
    assert action_calls
    for call in goal_calls:
        context = call["semantic_context"]
        assert context["past_dialog_cognition_context"] == (
            "PAST_DIALOG_PRIVATE_SENTINEL"
        )
        assert context["group_engagement_action_context"] == {
            "engagement_guidelines": ["GROUP_ENGAGEMENT_SENTINEL"],
            "confidence": "high",
        }
    assert action_calls[0]["group_engagement_action_context"] == {
        "engagement_guidelines": ["GROUP_ENGAGEMENT_SENTINEL"],
        "confidence": "high",
    }
    assert "PAST_DIALOG_PRIVATE_SENTINEL" not in json.dumps(
        action_calls,
        ensure_ascii=False,
    )
    assert "PAST_DIALOG_PRIVATE_SENTINEL" not in json.dumps(
        appraisal_calls,
        ensure_ascii=False,
    )
    assert "GROUP_ENGAGEMENT_SENTINEL" not in json.dumps(
        appraisal_calls,
        ensure_ascii=False,
    )
    assert "PAST_DIALOG_PRIVATE_SENTINEL" not in json.dumps(
        workspace_calls,
        ensure_ascii=False,
    )
    assert "GROUP_ENGAGEMENT_SENTINEL" not in json.dumps(
        workspace_calls,
        ensure_ascii=False,
    )


async def _prepare_db_user(
    live_db: object,
    request: pytest.FixtureRequest,
) -> tuple[str, dict[str, object]]:
    """Create one owner-isolated user row from the checked-in seed."""

    await seed_shared_documents(live_db)
    owner_id = unique_owner_id(request.node.nodeid)
    seed = await live_db.user_profiles.find_one(
        {"global_user_id": "seed-s2-acquaintance"},
    )
    if seed is None:
        raise AssertionError("the shared acquaintance seed is missing")
    state = deepcopy(seed["cognition_state"])
    state["owner_user_id"] = owner_id
    state["relationship"]["other_user_id"] = owner_id
    state["relationship"]["relationship_id"] = (
        f"relationship:user:{owner_id}"
    )
    await live_db.user_profiles.insert_one({
        "global_user_id": owner_id,
        "cognition_state": state,
    })
    return owner_id, await get_user_cognition_state(owner_id)


async def _run_db_user_smoke(
    live_db: object,
    request: pytest.FixtureRequest,
    *,
    trigger_source: str,
) -> None:
    """Exercise one user-scoped facade run and durable replacement reload."""

    owner_id, state = await _prepare_db_user(live_db, request)
    try:
        output = await run_cognition(
            _input(
                episode_id=f"db-smoke-{trigger_source}",
                mutable_state=state,
                trigger_source=trigger_source,
            ),
            _core_services(_ScriptedLLM()),
        )
        replacement = output["state_update"]["replacement_state"]
        await replace_user_cognition_state(owner_id, replacement)
        reloaded = await get_user_cognition_state(owner_id)
        assert output["schema_version"] == "cognition_core_output.v2"
        assert output["state_update"]["state_scope"] == "user"
        assert reloaded["owner_user_id"] == owner_id
        assert reloaded == replacement
    finally:
        await live_db.user_profiles.delete_one(
            {"global_user_id": owner_id},
        )


@pytest.mark.asyncio
async def test_v2_facade_commits_before_surface_and_preserves_complete_bid() -> None:
    """The core returns one committed state update and complete admitted bid."""

    llm = _ScriptedLLM()
    output = await run_cognition(_input(), _core_services(llm))

    assert output["state_update"]["state_scope"] == "user"
    assert output["intention"]["route"] == "speech"
    assert output["admitted_bid"]["reason"] == "the episode supplies bounded evidence"
    assert output["diagnostics"]["completed_branch_count"] >= 1
    assert output["state_update"]["replacement_state"]["state_scope"] == "user"


@pytest.mark.asyncio
async def test_v2_facade_projects_persistent_goal_to_entity_ref() -> None:
    """Persistent goal state crosses the branch boundary as an entity ref."""

    state = build_acquaintance_user_state(
        global_user_id="integration-user",
        updated_at=NOW,
    )
    state["goals"] = [{
        "entity_id": "goal:ordinary_response:user:episode:prior",
        "description": "continue the grounded conversation",
        "salience": 70,
        "role_refs": [],
        "evidence_refs": [{
            "source_kind": "episode",
            "source_id": "episode:prior",
            "occurred_at": NOW,
            "semantic_summary": "a prior turn established the response goal",
        }],
        "created_at": NOW,
        "updated_at": NOW,
        "status": "pursuing",
        "goal_kind": "ordinary_response",
        "importance": 70,
        "progress": 20,
        "obstruction": 0,
        "expected_success": 70,
        "controllability": 70,
        "recoverability": 60,
        "urgency": 50,
    }]

    output = await run_cognition(
        _input(mutable_state=state),
        _core_services(_ScriptedLLM()),
    )

    assert output["admitted_bid"]["goal_ref"] == {
        "scope": "user",
        "kind": "goal",
        "entity_id": "goal:ordinary_response:user:episode:prior",
    }


@pytest.mark.asyncio
async def test_canonical_private_source_has_no_legacy_output_mode() -> None:
    """Private delivery policy stays outside the canonical episode envelope."""

    llm = _ScriptedLLM()
    payload = _input(trigger_source="internal_thought")

    output = await run_cognition(payload, _core_services(llm))

    assert "output_mode" not in payload["episode"]
    assert output["intention"] == {
        "selected_branch_id": "ordinary_response",
        "route": "silence",
        "intention": "acknowledge the grounded episode",
        "target_roles": [],
        "reason": "the episode supplies bounded evidence",
    }
    assert output["admitted_bid"]["branch_id"] == "ordinary_response"
    assert any(
        "action_requests" in call
        for call in llm.calls
    )


@pytest.mark.asyncio
async def test_v2_surface_receives_semantic_handoff_only() -> None:
    """The four surface stages emit a bounded plan from non-private fields."""

    llm = _ScriptedLLM()
    input_payload = {
        "schema_version": "text_surface_input.v2",
        "episode": canonical_episode(
            episode_id="v2-surface-integration",
            content="private scene",
        ),
        "intention": {
            "route": "speech",
            "intention": "acknowledge the episode",
            "target_roles": [],
            "reason": "grounded evidence",
        },
        "goal_resolution": "answerable_now",
        "primary_bid": {
            "motive": "continuity",
            "intention": "acknowledge the episode",
            "desired_outcome": "maintain exchange",
            "permitted_detail": "current episode only",
            "target_summaries": [],
            "expected_consequences": ["preserve continuity"],
        },
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "neutral",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "calm and concise",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": "Concise clauses with light hesitation.",
        },
        "visual_character_context": "reserved, analytical, and warm",
    }

    output = await run_text_surface_planning(input_payload, _surface_services(llm))

    assert output["schema_version"] == "text_surface_output.v2"
    assert output["content_plan"] == "bounded content-plan guidance"
    assert output["visible_boundaries"] == ["bounded visible boundary"]
    assert output["addressee_plan"] == ["bounded addressee plan"]
    assert len(llm.calls) == 2
    rendered_prompts = "\n".join(llm.human_calls)
    for raw_value in (
        "v2-surface-integration",
        "percept:v2-surface-integration",
        "channel-test",
        "platform-user-test",
        "message-test",
    ):
        assert raw_value not in rendered_prompts
    assert '"target_scope"' not in rendered_prompts
    assert '"origin_metadata"' not in rendered_prompts
    assert '"storage_timestamp_utc"' not in rendered_prompts

    visual_output = await run_visual_surface_planning(
        input_payload,
        _visual_services(llm),
    )
    assert visual_output["visual_directives"] == "bounded visual directives"
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_v2_text_surface_never_invokes_terminal_visual_stage() -> None:
    """Text planning stays independent from the terminal visual branch."""

    llm = _ScriptedLLM()
    episode = canonical_episode(
        episode_id="v2-surface-no-visual",
        content="private scene",
    )
    episode["origin_metadata"]["debug_modes"] = {
        "no_visual_directives": True,
    }
    input_payload = {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": "acknowledge the episode",
            "target_roles": [],
            "reason": "grounded evidence",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "neutral",
            "intensity": "restrained",
            "directness": "balanced",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": "calm and concise",
        "character_expression_context": {
            "tempo": "steady",
            "linguistic_texture": "Concise clauses with light hesitation.",
        },
        "visual_character_context": "reserved, analytical, and warm",
    }

    output = await run_text_surface_planning(
        input_payload,
        _surface_services(llm),
    )

    assert len(llm.calls) == 2
    assert "pacing_guidance" not in output


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_test_database_private_chat_smoke(
    live_db: object,
    request: pytest.FixtureRequest,
) -> None:
    """Verify the private-chat user state path against the test database."""

    await _run_db_user_smoke(
        live_db,
        request,
        trigger_source="user_message",
    )


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_test_database_resolver_recurrence_smoke(
    live_db: object,
    request: pytest.FixtureRequest,
) -> None:
    """Verify recurrence preserves the originating user state owner."""

    await _run_db_user_smoke(
        live_db,
        request,
        trigger_source="user_message",
    )


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_test_database_self_cognition_smoke(live_db: object) -> None:
    """Verify character-scoped cognition persists and restores its singleton."""

    await seed_shared_documents(live_db)
    snapshot = await get_character_cognition_state()
    try:
        output = await run_cognition(
            _input(
                episode_id="db-smoke-self-cognition",
                mutable_state=deepcopy(snapshot),
                state_scope="character",
                trigger_source="internal_thought",
            ),
            _core_services(_ScriptedLLM()),
        )
        replacement = output["state_update"]["replacement_state"]
        await replace_character_cognition_state(replacement)
        reloaded = await get_character_cognition_state()
        assert output["schema_version"] == "cognition_core_output.v2"
        assert output["state_update"]["state_scope"] == "character"
        assert reloaded == replacement
    finally:
        await replace_character_cognition_state(snapshot)


@pytest.mark.live_db
@pytest.mark.asyncio
async def test_test_database_accepted_task_result_smoke(
    live_db: object,
    request: pytest.FixtureRequest,
) -> None:
    """Verify accepted-task results use the user-scoped cognition path."""

    await _run_db_user_smoke(
        live_db,
        request,
        trigger_source="tool_result",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("recover_on_second", [True, False])
async def test_appraisal_retry_or_omission_preserves_cognition(
    recover_on_second: bool,
) -> None:
    """One appraisal can recover or omit after two bounded attempts."""

    class _AppraisalFailureLLM(_ScriptedLLM):
        def __init__(self) -> None:
            super().__init__()
            self.target_calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            stage_name = getattr(config, "stage_name", "")
            payload = json.loads(str(getattr(messages[1], "content", "{}")))
            question = payload.get("question", {})
            if (
                stage_name == "v2_appraisal"
                and question.get("question_id") == "q:event_agency"
            ):
                self.target_calls += 1
                if not recover_on_second or self.target_calls < 2:
                    return SimpleNamespace(content='{"invalid": true}')
                result = {
                    "question_id": question["question_id"],
                    "proposition": None,
                    "delta": None,
                }
                return SimpleNamespace(
                    content=json.dumps(result, ensure_ascii=False),
                )
            return await super().ainvoke(messages, config=config)

    llm = _AppraisalFailureLLM()

    output = await run_cognition(_input(), _core_services(llm))

    assert output["schema_version"] == "cognition_core_output.v2"
    assert llm.target_calls == 2


@pytest.mark.asyncio
async def test_appraisal_internal_invariant_propagates() -> None:
    """An unexpected appraisal exception reaches the fatal pipeline boundary."""

    async def _raise_internal_invariant() -> dict[str, object]:
        raise AssertionError("appraisal task invariant failed")

    task = asyncio.create_task(_raise_internal_invariant())

    with pytest.raises(AssertionError, match="appraisal task invariant failed"):
        await facade_module._collect_appraisals(
            [task],
            [{"question_id": "q:event_agency"}],
        )


@pytest.mark.asyncio
async def test_appraisal_task_cancellation_propagates() -> None:
    """A cancelled appraisal task preserves graph cancellation semantics."""

    task = asyncio.create_task(asyncio.sleep(0))
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await facade_module._collect_appraisals(
            [task],
            [{"question_id": "q:event_agency"}],
        )


@pytest.mark.asyncio
async def test_goal_structure_recovers_on_third_attempt() -> None:
    """The required goal branch can recover on its final local attempt."""

    class _GoalRecoveryLLM(_ScriptedLLM):
        def __init__(self) -> None:
            super().__init__()
            self.goal_calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            if getattr(config, "stage_name", "") == "v2_goal":
                self.goal_calls += 1
                if self.goal_calls < 3:
                    return SimpleNamespace(content='{"invalid": true}')
                result = {
                    "intention": "acknowledge the grounded episode",
                    "desired_outcome": "maintain a coherent exchange",
                    "concrete_detail": "use the current episode only",
                    "reason": "the episode supplies bounded evidence",
                    "private_monologue": "I want to answer this carefully.",
                    "target_role_handles": [],
                    "evidence_handles": ["e1"],
                    "expected_consequences": ["preserve continuity"],
                    "confidence": "high",
                }
                return SimpleNamespace(
                    content=json.dumps(result, ensure_ascii=False),
                )
            return await super().ainvoke(messages, config=config)

    llm = _GoalRecoveryLLM()

    output = await run_cognition(_input(), _core_services(llm))

    assert output["admitted_bid"]["branch_id"] == "ordinary_response"
    assert llm.goal_calls == 3


@pytest.mark.asyncio
async def test_required_goal_invalid_evidence_reaches_action_planning(
) -> None:
    """A complete invalid-handle bid continues as a degraded cognition bid."""

    class _GoalDegradedLLM(_ScriptedLLM):
        def __init__(self) -> None:
            super().__init__()
            self.goal_calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            if getattr(config, "stage_name", "") == "v2_goal":
                self.goal_calls += 1
                result = {
                    "selection_kind": "choice",
                    "selection": "acknowledge the grounded episode",
                    "reason": "the episode supplies bounded evidence",
                    "private_monologue": "I can answer this directly.",
                    "target_role_handles": [],
                    "evidence_handles": ["e1", "r1"],
                    "expected_consequences": ["preserve continuity"],
                    "confidence": "high",
                }
                return SimpleNamespace(
                    content=json.dumps(result, ensure_ascii=False),
                )
            return await super().ainvoke(messages, config=config)

    payload = _input()
    payload["evidence"][0]["semantic_text"] = json.dumps({
        "role_explicit_content": "The character must answer.",
        "response_operation": {
            "operation": "The character answers the current question.",
            "response_owner_role": "current character",
            "selection_owner_role": "current character",
            "selection_required": True,
            "embedded_actor_role": "current user",
            "embedded_target_role": "current character",
        },
    })
    llm = _GoalDegradedLLM()

    output = await run_cognition(payload, _core_services(llm))

    assert llm.goal_calls == 3
    assert output["admitted_bid"]["branch_id"] == "ordinary_response"
    assert output["admitted_bid"]["evidence_handles"] == ["e1"]


@pytest.mark.asyncio
async def test_required_goal_exhaustion_requests_clean_graph_retry() -> None:
    """A required zero-valid branch remains a typed pre-commit retry case."""

    class _GoalExhaustionLLM(_ScriptedLLM):
        def __init__(self) -> None:
            super().__init__()
            self.goal_calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            if getattr(config, "stage_name", "") == "v2_goal":
                self.goal_calls += 1
                return SimpleNamespace(content='{"invalid": true}')
            return await super().ainvoke(messages, config=config)

    llm = _GoalExhaustionLLM()

    with pytest.raises(CognitionExecutionError) as error_info:
        await run_cognition(_input(), _core_services(llm))

    error = error_info.value
    assert error.safe_checkpoint == "pre_state_commit"
    assert error.retryable is True
    assert error.attempt_count == 3
    assert llm.goal_calls == 3


def _workspace_bid(branch_id: str) -> dict[str, object]:
    """Build one complete model-authored bid for collapse fallback tests."""

    return {
        "branch_id": branch_id,
        "goal_ref": {
            "scope": "user",
            "kind": "goal",
            "entity_id": branch_id,
        },
        "intention": f"advance {branch_id}",
        "desired_outcome": "preserve a grounded response",
        "concrete_detail": "use only current evidence",
        "reason": "the current episode supports this bid",
        "private_monologue": "I should answer deliberately.",
        "target_roles": [],
        "evidence_handles": ["e1"],
        "expected_consequences": ["the response remains coherent"],
        "confidence": "high",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("recover_on_third", [True, False])
async def test_workspace_retry_or_branch_order_fallback(
    recover_on_third: bool,
) -> None:
    """Collapse recovers on attempt three or selects the first valid branch."""

    class _WorkspaceLLM:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(
            self,
            messages: list[object],
            *,
            config: object,
        ) -> SimpleNamespace:
            del messages, config
            self.calls += 1
            if recover_on_third and self.calls == 3:
                return SimpleNamespace(content=json.dumps({
                    "primary_bid_handle": "b2",
                    "supporting_bid_handles": [],
                    "suppressed_bid_handles": ["b1"],
                }))
            return SimpleNamespace(content='{"invalid": true}')

    llm = _WorkspaceLLM()
    services = SimpleNamespace(llm=llm, workspace_collapse_config=object())

    output = await collapse_bids(
        [
            _workspace_bid("ordinary_response"),
            _workspace_bid("relationship_connection"),
        ],
        services,
    )

    expected_primary = (
        "relationship_connection"
        if recover_on_third
        else "ordinary_response"
    )
    assert output["primary_branch_id"] == expected_primary
    assert llm.calls == 3
