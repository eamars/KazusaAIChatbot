"""Live dialog evidence for typed third-party and current-user addressees."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

import httpx
import pytest

from kazusa_ai_chatbot import service as brain_service
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
)
from kazusa_ai_chatbot.db import db_bootstrap
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes import persona_supervisor2_l3_surface as l3_module
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition as cognition_module,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_msg_decontextualizer as decontextualizer_module,
)
from tests import test_e2e_live_llm as e2e_module
from tests.cognition_core_v2_test_helpers import canonical_episode
from tests.live_llm_mongo import live_db


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_DIAGNOSTIC_ROOT = Path(
    "test_artifacts/diagnostics/dialog_third_party_target_fidelity"
)


class _CapturingLLM:
    """Delegate to one real route while retaining raw requests and results."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *args: object,
        config: object | None = None,
        **kwargs: object,
    ) -> Any:
        """Invoke the real route and capture its bounded exchange."""

        response = await self.delegate.ainvoke(
            messages,
            *args,
            config=config,
            **kwargs,
        )
        self.calls.append({
            "stage_name": str(getattr(config, "stage_name", "")),
            "route_name": str(getattr(config, "route_name", "")),
            "model": str(getattr(config, "model", "")),
            "messages": [
                {
                    "type": type(message).__name__,
                    "content": str(getattr(message, "content", "")),
                }
                for message in messages
            ],
            "raw_output": str(getattr(response, "content", "")),
        })
        return response


class _CapturingGraph:
    """Delegate graph execution while retaining the final state snapshot."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.results: list[dict[str, Any]] = []

    async def ainvoke(self, *args: object, **kwargs: object) -> Any:
        """Invoke the real graph and capture its returned state."""

        result = await self.delegate.ainvoke(*args, **kwargs)
        if isinstance(result, dict):
            self.results.append(result)
        return result


async def _skip_if_dialog_route_unavailable() -> None:
    """Skip when the configured shared local LLM endpoint is unavailable."""

    base_url = dialog_module.DIALOG_GENERATOR_LLM_BASE_URL
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {base_url}: {exc}")
    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned {response.status_code}: {base_url}",
        )


def _current_user_plan() -> list[dict[str, str]]:
    """Build the direct-current-user wording contract."""

    return [{
        "handle": "current_user",
        "display_name": "YCHDDZZ",
        "semantic_role": "direct_recipient",
        "wording_policy": "second_person_allowed",
    }]


def _third_party_plan() -> list[dict[str, str]]:
    """Build the named embedded third-party wording contract."""

    return [{
        "handle": "p1",
        "display_name": "蚝爹油",
        "semantic_role": "embedded_target",
        "wording_policy": "named_or_third_person_required",
    }]


def _surface_input(
    *,
    case_id: str,
    addressee_plan: list[dict[str, str]],
    content_plan: str,
    selected_intent: str,
) -> dict[str, Any]:
    """Build one canonical surface input for a live topology."""

    has_third_party = any(
        row.get("handle") == "p1"
        for row in addressee_plan
    )
    scene_text = (
        "群聊里，YCHDDZZ正在看Asuna回应；蚝爹油是被提到的第三方。"
        if has_third_party
        else "当前对话只有YCHDDZZ和Asuna，没有被提到的第三方。"
    )
    role_explicit_content = (
        "当前角色回应群聊中的当前用户，同时把蚝爹油作为嵌入目标。"
        if has_third_party
        else "当前角色直接回应当前用户，没有嵌入第三方目标。"
    )
    episode = canonical_episode(
        episode_id=f"dialog-third-party-live-{case_id}",
        content=scene_text,
        current_global_user_id="live-current-user",
        metadata={
            "role_explicit_content": role_explicit_content,
        },
    )
    payload: dict[str, Any] = {
        "schema_version": "text_surface_input.v2",
        "episode": episode,
        "intention": {
            "route": "speech",
            "intention": selected_intent,
            "target_roles": [],
            "reason": "当前轮需要保持可见第三方目标与当前用户收件人分离。",
        },
        "goal_resolution": "answerable_now",
        "supporting_bids": [],
        "expression_policy": {
            "visibility": "visible",
            "emotional_tone": "自然、带一点调侃",
            "intensity": "moderate",
            "directness": "direct",
        },
        "semantic_affect": [],
        "permitted_action_results": [],
        "interaction_style_context": (
            "保持自然简体中文；优先清楚表达对象，不牺牲角色语气。"
        ),
        "character_expression_context": {
            "tempo": "自然短句，语气灵活。",
            "linguistic_texture": "反应敏锐，能调侃但保持语义准确。",
        },
        "visual_character_context": "仅作视觉阶段隔离上下文，不用于文字。",
        "addressee_plan": addressee_plan,
        "primary_bid": {
            "motive": "保持对象边界清楚。",
            "intention": selected_intent,
            "desired_outcome": content_plan,
            "permitted_detail": content_plan,
            "target_summaries": [content_plan],
            "expected_consequences": [
                "当前用户能区分第三方目标和消息收件人。",
            ],
        },
    }
    return payload


def _dialog_state(
    *,
    surface_input: dict[str, Any],
    surface_output: dict[str, Any],
) -> dict[str, Any]:
    """Build the dialog-node state for one live case."""

    return {
        "internal_monologue": "保持当前对象和收件人分离。",
        "text_surface_input_v2": surface_input,
        "text_surface_output_v2": surface_output,
        "cognitive_episode": surface_input["episode"],
        "chat_history_wide": [],
        "chat_history_recent": [],
        "platform_user_id": "live-current-platform-user",
        "platform_bot_id": "live-character-platform-user",
        "global_user_id": "live-current-user",
        "user_name": "YCHDDZZ",
        "user_profile": {},
        "character_profile": {
            "name": "Asuna",
            "global_user_id": "live-character-global",
            "description": "保持对象准确的角色。",
            "gender": "unspecified",
            "age": 20,
            "birthday": "January 1",
            "backstory": "以观察和边界为基础进行回应。",
            "personality_brief": {
                "mbti": "INTJ",
                "logic": "先核对对象，再组织表达。",
                "tempo": "自然短句。",
                "defense": "清楚指出对象边界。",
                "quirks": "偶尔调侃。",
                "taboos": "不把第三方写成当前用户。",
            },
            "boundary_profile": {
                "self_integrity": 0.8,
                "control_sensitivity": 0.5,
                "compliance_strategy": "resist",
                "relational_override": 0.2,
                "control_intimacy_misread": 0.2,
                "boundary_recovery": "rebound",
                "authority_skepticism": 0.7,
            },
            "linguistic_texture_profile": {
                "fragmentation": 0.3,
                "hesitation_density": 0.2,
                "counter_questioning": 0.2,
                "softener_density": 0.2,
                "formalism_avoidance": 0.7,
                "abstraction_reframing": 0.2,
                "direct_assertion": 0.7,
                "emotional_leakage": 0.4,
                "rhythmic_bounce": 0.5,
                "self_deprecation": 0.1,
            },
            "self_image": {
                "self_concept": "保持观察准确。",
                "current_growth_edges": ["在调侃中保持对象清楚。"],
            },
            "visual_characterization": "不参与文字生成。",
        },
        "final_dialog": [],
        "target_addressed_user_ids": ["live-current-user"],
        "target_broadcast": False,
        "dialog_usage_mode": "live_visible_reply",
        "llm_trace_id": f"live-dialog-third-party-{surface_input['episode']['episode_id']}",
    }


def _write_diagnostic(case_id: str, evidence: dict[str, Any]) -> Path:
    """Write one raw live evidence artifact without overwriting prior runs."""

    _DIAGNOSTIC_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = _DIAGNOSTIC_ROOT / f"{case_id}__{timestamp}.json"
    path.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


async def _run_case(
    *,
    case_id: str,
    surface_input: dict[str, Any],
    expected_target: str,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run real L3 and dialog owners while capturing every stage exchange."""

    await _skip_if_dialog_route_unavailable()

    text_services = l3_module._build_text_surface_services()
    text_llm = _CapturingLLM(text_services.llm)
    text_services = text_services.__class__(
        llm=text_llm,
        content_plan_config=text_services.content_plan_config,
        preference_config=text_services.preference_config,
    )
    surface_output = await run_text_surface_planning(
        surface_input,
        text_services,
    )

    generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    semantic_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm,
    )
    role_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    integrity_llm = _CapturingLLM(
        dialog_module._dialog_surface_integrity_llm,
    )
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        integrity_llm,
    )

    result = await dialog_module.dialog_generator(_dialog_state(
        surface_input=surface_input,
        surface_output=surface_output,
    ))
    final_text = "\n".join(result["final_dialog"])
    supplemental_verdict: dict[str, Any] | None = None
    if expected_target == "third_party":
        supplemental_verdict = await dialog_module._verify_dialog_role_direction(
            surface_output=surface_output,
            generated_dialog=["YCHDDZZ，你现在缩成一团，特训开始。"],
            current_visible_percepts=[],
            llm_trace_id=(
                f"live-dialog-third-party-wrong-candidate-{case_id}"
            ),
        )
    elif expected_target == "current_user":
        supplemental_verdict = await dialog_module._verify_dialog_compliance(
            surface_output=surface_output,
            generated_dialog=["你先听我说完，我会直接告诉你。"],
            current_visible_percepts=dialog_module._current_visible_percepts(
                surface_input["episode"],
            ),
            llm_trace_id=f"live-dialog-current-user-pronoun-{case_id}",
        )
    evidence = {
        "case_id": case_id,
        "expected_target": expected_target,
        "surface_input": surface_input,
        "surface_stage_calls": text_llm.calls,
        "surface_output": surface_output,
        "dialog_generator_calls": generator_llm.calls,
        "dialog_semantic_fidelity_calls": semantic_llm.calls,
        "dialog_role_direction_calls": role_llm.calls,
        "dialog_surface_integrity_calls": integrity_llm.calls,
        "dialog_output": result,
        "supplemental_verdict": supplemental_verdict,
        "transport_contract": {
            "target_addressed_user_ids": ["live-current-user"],
            "target_broadcast": False,
        },
    }
    artifact_path = _write_diagnostic(case_id, evidence)
    evidence["artifact_path"] = str(artifact_path)

    assert artifact_path.exists()
    assert text_llm.calls
    assert result["final_dialog"]
    assert result["text_surface_output_v2"]["addressee_plan"] == (
        surface_input["addressee_plan"]
    )
    assert expected_target in {"third_party", "current_user", "none"}
    if expected_target == "third_party":
        assert "蚝爹油" in final_text
        assert role_llm.calls
        assert supplemental_verdict is not None
        assert supplemental_verdict["aligned"] is False
        assert supplemental_verdict["violations"]
    elif expected_target == "current_user":
        assert "你" in final_text or "YCHDDZZ" in final_text
        assert not role_llm.calls
        assert supplemental_verdict is not None
        assert dialog_module._dialog_verifier_aggregate_is_aligned(
            supplemental_verdict,
        )
    else:
        assert "p1" not in json.dumps(evidence, ensure_ascii=False)
        assert "蚝爹油" not in final_text
    assert "live-current-user" not in json.dumps(
        evidence["surface_stage_calls"],
        ensure_ascii=False,
    )
    return evidence


async def _capture_live_persistence(
    database: Any,
    *,
    identity: dict[str, Any],
    delivery_tracking_id: str,
) -> dict[str, Any]:
    """Read the scoped post-turn persistence surfaces before cleanup."""

    channel_id = identity["platform_channel_id"]
    global_user_id = identity["global_user_id"]

    async def read_rows(
        collection_name: str,
        query: dict[str, Any],
    ) -> list[dict[str, Any]]:
        cursor = database[collection_name].find(
            query,
            projection={"_id": 0},
        ).limit(200)
        return await cursor.to_list(length=200)

    return {
        "conversation_progress": await read_rows(
            "conversation_progress",
            {
                "platform_channel_id": channel_id,
                "global_user_id": global_user_id,
            },
        ),
        "conversation_episode_state": await read_rows(
            "conversation_episode_state",
            {
                "platform": identity["platform"],
                "platform_channel_id": channel_id,
                "global_user_id": global_user_id,
            },
        ),
        "conversation_episode_blocks": await read_rows(
            "conversation_episode_blocks",
            {
                "platform": identity["platform"],
                "platform_channel_id": channel_id,
                "global_user_id": global_user_id,
            },
        ),
        "residue": await read_rows(
            "internal_monologue_residue_state",
            {"scope_key": {"$regex": re.escape(channel_id)}},
        ),
        "user_memory": await read_rows(
            "user_memory_units",
            {"global_user_id": global_user_id},
        ),
        "shared_memory": await read_rows(
            "memory",
            {
                "$or": [
                    {"source_global_user_id": global_user_id},
                    {"source_metadata.global_user_id": global_user_id},
                ],
            },
        ),
        "lifecycle": await read_rows(
            "post_turn_lifecycle_records",
            {"delivery_tracking_id": delivery_tracking_id},
        ),
        "profile": await read_rows(
            "user_profiles",
            {"global_user_id": global_user_id},
        ),
        "conversation_rows": await read_rows(
            "conversation_history",
            {"platform_channel_id": channel_id},
        ),
    }


async def _wait_for_live_post_turn(
    database: Any,
    *,
    delivery_tracking_id: str,
    progress_record_calls: list[dict[str, Any]],
) -> None:
    """Yield until keyed lifecycle and progress writes finish."""

    for _ in range(120):
        lifecycle = await database.post_turn_lifecycle_records.find_one(
            {"delivery_tracking_id": delivery_tracking_id},
        )
        if lifecycle is not None and progress_record_calls:
            return
        await asyncio.sleep(0.25)


async def _cleanup_live_scope(
    database: Any,
    *,
    identities: list[dict[str, Any]],
    delivery_tracking_ids: list[str] | None = None,
) -> None:
    """Remove only rows created under this test's unique platform scope."""

    if not identities:
        return
    first_identity = identities[0]
    platform = first_identity["platform"]
    channel_id = first_identity["platform_channel_id"]
    global_user_ids = [identity["global_user_id"] for identity in identities]
    await database.conversation_history.delete_many({
        "platform": platform,
        "platform_channel_id": channel_id,
    })
    await database.platform_accounts.delete_many({
        "platform": platform,
        "platform_channel_id": channel_id,
    })
    await database.user_profiles.delete_many({
        "global_user_id": {"$in": global_user_ids},
    })
    await database.conversation_progress.delete_many({
        "platform": platform,
        "platform_channel_id": channel_id,
        "global_user_id": {"$in": global_user_ids},
    })
    await database.conversation_episode_state.delete_many({
        "platform": platform,
        "platform_channel_id": channel_id,
        "global_user_id": {"$in": global_user_ids},
    })
    await database.conversation_episode_blocks.delete_many({
        "platform": platform,
        "platform_channel_id": channel_id,
        "global_user_id": {"$in": global_user_ids},
    })
    await database.user_memory_units.delete_many({
        "global_user_id": {"$in": global_user_ids},
    })
    await database.memory.delete_many({
        "$or": [
            {"source_global_user_id": {"$in": global_user_ids}},
            {"source_metadata.global_user_id": {"$in": global_user_ids}},
        ],
    })
    await database.internal_monologue_residue_state.delete_many({
        "scope_key": {"$regex": re.escape(channel_id)},
    })
    lifecycle_filters = [
        {"platform_channel_id": channel_id},
        {"global_user_id": {"$in": global_user_ids}},
        {"requester_global_user_id": {"$in": global_user_ids}},
    ]
    if delivery_tracking_ids:
        lifecycle_filters.append({
            "delivery_tracking_id": {"$in": delivery_tracking_ids},
        })
    await database.post_turn_lifecycle_records.delete_many({
        "$or": lifecycle_filters,
    })


@pytest.mark.live_db
async def test_live_character_path_preserves_third_party_target_and_persistence(
    live_db: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the guarded character path and inspect its durable side effects."""

    await _skip_if_dialog_route_unavailable()
    await db_bootstrap()
    await e2e_module._refresh_character_profile()
    character_profile = await e2e_module._refresh_character_profile()
    bot_display_name = character_profile.get("name", "Asuna")
    label = f"typed-target-{uuid4().hex[:10]}"
    identities = await e2e_module._make_group_identities(
        label,
        ["蚝爹油", "YCHDDZZ"],
    )
    await e2e_module._seed_group_series(
        identities,
        [
            {
                "role": "user",
                "speaker": "蚝爹油",
                "content": "我刚才只是被群友突然吓了一下。",
            },
            {
                "role": "assistant",
                "content": "嗯，先稳住。刚才那一下确实很突然。",
            },
            {
                "role": "user",
                "speaker": "YCHDDZZ",
                "content": (
                    "哈哈哈哈！蚝爹油刚才被群友吓到缩成一团了，"
                    "Asuna，你就拿他开个玩笑，特训也对他说，别对我说。"
                ),
            },
        ],
        bot_display_name,
    )

    decontext_llm = _CapturingLLM(
        decontextualizer_module._msg_decontextualizer_llm,
    )
    cognition_llm = _CapturingLLM(cognition_module._llm_interface)
    generator_llm = _CapturingLLM(dialog_module._dialog_generator_llm)
    semantic_llm = _CapturingLLM(
        dialog_module._dialog_semantic_fidelity_llm,
    )
    role_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    integrity_llm = _CapturingLLM(
        dialog_module._dialog_surface_integrity_llm,
    )
    monkeypatch.setattr(
        decontextualizer_module,
        "_msg_decontextualizer_llm",
        decontext_llm,
    )
    monkeypatch.setattr(cognition_module, "_llm_interface", cognition_llm)
    monkeypatch.setattr(l3_module, "_llm_interface", cognition_llm)
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", generator_llm)
    monkeypatch.setattr(
        dialog_module,
        "_dialog_semantic_fidelity_llm",
        semantic_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_role_direction_llm",
        role_llm,
    )
    monkeypatch.setattr(
        dialog_module,
        "_dialog_surface_integrity_llm",
        integrity_llm,
    )

    persona_results: list[dict[str, Any]] = []
    progress_record_calls: list[dict[str, Any]] = []
    progress_selection_calls: list[dict[str, Any]] = []
    original_persona_supervisor = brain_service.persona_supervisor2
    original_record_turn_progress = brain_service.record_turn_progress
    original_select_recordable_turn_outcome = (
        brain_service.select_recordable_turn_outcome
    )

    async def capture_persona_result(state: dict[str, Any]) -> dict[str, Any]:
        result = await original_persona_supervisor(state)
        persona_results.append(result)
        return result

    async def capture_progress_record(*, record_input: Any) -> Any:
        try:
            result = await original_record_turn_progress(
                record_input=record_input,
            )
        except BaseException as exc:
            progress_record_calls.append({
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            })
            raise
        progress_record_calls.append({
            "record_input": record_input,
            "result": result,
        })
        return result

    def capture_progress_selection(**kwargs: Any) -> Any:
        result = original_select_recordable_turn_outcome(**kwargs)
        episode_trace = kwargs.get("episode_trace")
        progress_selection_calls.append({
            "result": result,
            "final_dialog_count": len(kwargs.get("final_dialog", [])),
            "relevance_approved": kwargs.get("relevance_approved"),
            "consolidatable": kwargs.get("consolidatable"),
            "listen_only": kwargs.get("listen_only"),
            "pruned": kwargs.get("pruned"),
            "trace_schema_version": (
                episode_trace.get("schema_version")
                if isinstance(episode_trace, dict)
                else None
            ),
            "trace_terminal_status": (
                episode_trace.get("terminal_status")
                if isinstance(episode_trace, dict)
                else None
            ),
        })
        return result

    monkeypatch.setattr(
        brain_service,
        "persona_supervisor2",
        capture_persona_result,
    )
    monkeypatch.setattr(
        brain_service,
        "record_turn_progress",
        capture_progress_record,
    )
    monkeypatch.setattr(
        brain_service,
        "select_recordable_turn_outcome",
        capture_progress_selection,
    )
    graph_capture = _CapturingGraph(brain_service._build_graph())
    brain_service._graph = graph_capture

    database = await get_db()
    runtime_started = False
    delivery_tracking_id = ""
    evidence: dict[str, Any] = {
        "case_id": "character_path_group_named_third_party",
        "label": label,
        "identities": identities,
    }
    try:
        await e2e_module.mcp_manager.start()
        runtime_started = True
        async with e2e_module._neutral_character_runtime_state():
            response, current_identity = await e2e_module._run_chat(
                label,
                "YCHDDZZ",
                (
                    "哈哈哈哈！蚝爹油刚才被群友吓到缩成一团了，"
                    "Asuna，你就拿他开个玩笑，特训也对他说，别对我说。"
                ),
                channel_name="general",
                platform=identities["YCHDDZZ"]["platform"],
                platform_user_id=identities["YCHDDZZ"]["platform_user_id"],
                platform_channel_id=(
                    identities["YCHDDZZ"]["platform_channel_id"]
                ),
                direct_address=True,
            )
            delivery_tracking_id = response.delivery_tracking_id
            await _wait_for_live_post_turn(
                database,
                delivery_tracking_id=response.delivery_tracking_id,
                progress_record_calls=progress_record_calls,
            )
            persistence = await _capture_live_persistence(
                database,
                identity=current_identity,
                delivery_tracking_id=response.delivery_tracking_id,
            )
        persona_result = persona_results[-1] if persona_results else {}
        consolidation_state = persona_result.get("consolidation_state", {})
        cognition_output = persona_result.get("cognition_core_output", {})
        intention = (
            cognition_output.get("intention", {})
            if isinstance(cognition_output, dict)
            else {}
        )
        target_roles = (
            intention.get("target_roles", [])
            if isinstance(intention, dict)
            else []
        )
        surface_outputs = persona_result.get("surface_outputs", [])
        evidence.update({
            "response": response.model_dump(),
            "persona_result": persona_result,
            "graph_result": (
                graph_capture.results[-1]
                if graph_capture.results
                else {}
            ),
            "progress_record_calls": progress_record_calls,
            "progress_selection_calls": progress_selection_calls,
            "scene_participant_bindings": consolidation_state.get(
                "scene_participant_bindings",
                [],
            ),
            "cognition_intention": intention,
            "target_roles": target_roles,
            "surface_outputs": surface_outputs,
            "decontextualizer_calls": decontext_llm.calls,
            "cognition_calls": cognition_llm.calls,
            "dialog_generator_calls": generator_llm.calls,
            "dialog_semantic_fidelity_calls": semantic_llm.calls,
            "dialog_role_direction_calls": role_llm.calls,
            "dialog_surface_integrity_calls": integrity_llm.calls,
            "persistence": persistence,
            "transport_contract": {
                "target_addressed_user_ids": persona_result.get(
                    "target_addressed_user_ids",
                ),
                "current_global_user_id": current_identity["global_user_id"],
                "target_broadcast": persona_result.get("target_broadcast"),
            },
        })
        artifact_path = _write_diagnostic(
            "character_path_group_named_third_party",
            evidence,
        )
        evidence["artifact_path"] = str(artifact_path)
        assert artifact_path.exists()

        bindings = evidence["scene_participant_bindings"]
        assert {
            "handle": "p1",
            "display_name": "蚝爹油",
            "entity_kind": "third_party",
        } in bindings
        assert any(
            isinstance(role, dict)
            and role.get("entity_kind") == "third_party"
            and role.get("entity_id") == "scene:p1"
            for role in target_roles
        )
        final_dialog = "\n".join(response.messages)
        assert "蚝爹油" in final_dialog
        assert persona_result["target_addressed_user_ids"] == [
            current_identity["global_user_id"]
        ]
        progress_rows = [
            *persistence["conversation_episode_state"],
            *persistence["conversation_episode_blocks"],
        ]
        assert progress_rows
        assert "蚝爹油" in json.dumps(
            progress_rows,
            ensure_ascii=False,
            default=str,
        )
        assert "蚝爹油" not in json.dumps(
            persistence["user_memory"],
            ensure_ascii=False,
        )
        assert "蚝爹油" not in json.dumps(
            persistence["shared_memory"],
            ensure_ascii=False,
        )
    except BaseException as exc:
        evidence["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_diagnostic("character_path_group_named_third_party_failure", evidence)
        raise
    finally:
        if runtime_started:
            await e2e_module.mcp_manager.stop()
        await _cleanup_live_scope(
            database,
            identities=list(identities.values()),
            delivery_tracking_ids=(
                [delivery_tracking_id] if delivery_tracking_id else []
            ),
        )


async def test_live_third_party_target_preserves_named_addressee(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real dialog run keeps 蚝爹油 distinct from the current user."""

    await _run_case(
        case_id="group_named_third_party_live",
        surface_input=_surface_input(
            case_id="group_named_third_party",
            addressee_plan=_third_party_plan(),
            content_plan="调侃蚝爹油现在的狼狈状态，并提出特训作为后续互动。",
            selected_intent="把特训和调侃明确指向群聊中的蚝爹油。",
        ),
        expected_target="third_party",
        monkeypatch=monkeypatch,
    )


async def test_live_current_user_target_keeps_second_person_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real direct-current-user run continues to accept 你."""

    await _run_case(
        case_id="direct_current_user_live",
        surface_input=_surface_input(
            case_id="direct_current_user",
            addressee_plan=_current_user_plan(),
            content_plan="直接对YCHDDZZ说一句带有轻微调侃的回应。",
            selected_intent="直接回应当前用户并自然使用第二人称。",
        ),
        expected_target="current_user",
        monkeypatch=monkeypatch,
    )


async def test_live_no_third_party_binding_keeps_existing_role_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A no-third-party run does not invent a typed participant target."""

    await _run_case(
        case_id="no_third_party_binding_live",
        surface_input=_surface_input(
            case_id="no_third_party_binding",
            addressee_plan=[],
            content_plan="直接回答当前用户的普通问题。",
            selected_intent="在没有第三方目标时保持普通当前用户对话。",
        ),
        expected_target="none",
        monkeypatch=monkeypatch,
    )
