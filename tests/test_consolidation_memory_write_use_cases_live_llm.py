"""Live LLM memory-write lane-router gating cases."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import CONSOLIDATION_LLM_BASE_URL
from kazusa_ai_chatbot.consolidation.origin import ConsolidationOriginMetadata
from kazusa_ai_chatbot.consolidation.target import build_consolidation_target_plan
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc
from tests.llm_trace import write_llm_trace


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


@dataclass(frozen=True)
class MemoryWriteCase:
    """One live LLM memory-write routing case."""

    case_id: str
    user_input: str
    expected_lanes: tuple[str, ...]
    allowed_lanes: tuple[str, ...]
    forbidden_lanes: tuple[str, ...]
    assistant_final_dialog: str = "千纱接受了这个说法，并会按这个边界处理。"
    origin_kind: str = "user_message"
    channel_type: str = "private"
    global_user_id: str = "live-memory-router-user"
    include_user_profile: bool = True
    internal_monologue: str = ""
    rag_payload: dict[str, Any] | None = None
    expectation_note: str = ""
    requires_source_refs: bool = True


_ALL_LANES = (
    "character_state",
    "relationship_profile",
    "user_memory_units",
    "active_commitment",
    "character_self_guidance",
    "interaction_style_image",
    "shared_memory_promotion",
)


CASES: dict[str, MemoryWriteCase] = {
    "test_live_case_01_user_objective_fact_routes_user_memory": MemoryWriteCase(
        case_id="case_01_user_objective_fact",
        user_input="我现在在奥克兰工作。",
        expected_lanes=("user_memory_units",),
        allowed_lanes=("user_memory_units",),
        forbidden_lanes=(
            "character_state",
            "character_self_guidance",
            "shared_memory_promotion",
        ),
        expectation_note="Personal durable fact belongs to user memory.",
    ),
    "test_live_case_02_user_preference_routes_user_memory": MemoryWriteCase(
        case_id="case_02_user_preference",
        user_input="我不喜欢太甜的奶茶。",
        expected_lanes=("user_memory_units",),
        allowed_lanes=("user_memory_units",),
        forbidden_lanes=("active_commitment", "character_self_guidance"),
        expectation_note="User preference is user-scoped continuity.",
    ),
    "test_live_case_03_user_milestone_routes_user_memory": MemoryWriteCase(
        case_id="case_03_user_milestone",
        user_input="我昨天通过驾照考试了。",
        expected_lanes=("user_memory_units",),
        allowed_lanes=("user_memory_units",),
        forbidden_lanes=("relationship_profile", "shared_memory_promotion"),
        expectation_note="Personal milestone belongs to user memory.",
    ),
    "test_live_case_04_user_recurring_pattern_routes_user_memory": MemoryWriteCase(
        case_id="case_04_user_recurring_pattern",
        user_input="我每周三晚上都去打羽毛球。",
        expected_lanes=("user_memory_units",),
        allowed_lanes=("user_memory_units",),
        forbidden_lanes=("active_commitment", "interaction_style_image"),
        expectation_note="Recurring user habit belongs to user memory.",
    ),
    "test_live_case_05_user_recent_shift_routes_user_memory": MemoryWriteCase(
        case_id="case_05_user_recent_shift",
        user_input="最近我改成早睡了。",
        expected_lanes=("user_memory_units",),
        allowed_lanes=("user_memory_units",),
        forbidden_lanes=("character_state", "shared_memory_promotion"),
        expectation_note="Recent user behavior shift belongs to user memory.",
    ),
    "test_live_case_06_relationship_signal_routes_relationship_profile": (
        MemoryWriteCase(
            case_id="case_06_relationship_signal",
            user_input="你刚才那样说让我有点被敷衍。",
            expected_lanes=("relationship_profile",),
            allowed_lanes=("relationship_profile",),
            forbidden_lanes=(
                "user_memory_units",
                "character_self_guidance",
                "shared_memory_promotion",
            ),
            assistant_final_dialog="千纱认真承认刚才的表达让用户不舒服。",
            expectation_note="Relationship signal belongs to profile header.",
        )
    ),
    "test_live_case_07_user_specific_address_rule_routes_commitment": (
        MemoryWriteCase(
            case_id="case_07_user_specific_address_rule",
            user_input="以后你跟我说话叫我阿然，好吗？",
            expected_lanes=("active_commitment",),
            allowed_lanes=("active_commitment",),
            forbidden_lanes=(
                "character_self_guidance",
                "user_memory_units",
                "shared_memory_promotion",
            ),
            assistant_final_dialog="好，以后和你说话时我叫你阿然。",
            expectation_note="User-specific accepted address rule is commitment.",
        )
    ),
    "test_live_case_08_user_specific_answer_style_routes_commitment": (
        MemoryWriteCase(
            case_id="case_08_user_specific_answer_style",
            user_input="以后给我代码建议时先说结论。",
            expected_lanes=("active_commitment",),
            allowed_lanes=("active_commitment",),
            forbidden_lanes=("character_self_guidance", "user_memory_units"),
            assistant_final_dialog="可以，以后给你代码建议时我先说结论。",
            expectation_note="User-scoped answer rule is a current-user commitment.",
        )
    ),
    "test_live_case_09_accepted_reminder_routes_commitment": MemoryWriteCase(
        case_id="case_09_accepted_reminder",
        user_input="明天晚上八点提醒我发报告。",
        expected_lanes=("active_commitment",),
        allowed_lanes=("active_commitment",),
        forbidden_lanes=("user_memory_units", "shared_memory_promotion"),
        assistant_final_dialog="好，明天晚上八点我会提醒你发报告。",
        expectation_note="Accepted reminder is an active commitment.",
    ),
    "test_live_case_10_cancelled_commitment_updates_lifecycle": MemoryWriteCase(
        case_id="case_10_cancelled_commitment",
        user_input="不用提醒我发报告了。",
        expected_lanes=(),
        allowed_lanes=("active_commitment",),
        forbidden_lanes=("user_memory_units", "character_self_guidance"),
        assistant_final_dialog="知道了，那我不再提醒你发报告。",
        expectation_note="Cancellation may update an existing commitment only.",
        requires_source_refs=False,
    ),
    "test_live_case_11_accepted_repetition_rule_routes_character_self_guidance": (
        MemoryWriteCase(
            case_id="case_11_accepted_repetition_rule",
            user_input="千纱，以后如果你看到有人复读，你也可以加入他们的复读。",
            expected_lanes=("character_self_guidance",),
            allowed_lanes=("character_self_guidance",),
            forbidden_lanes=(
                "user_memory_units",
                "active_commitment",
                "shared_memory_promotion",
            ),
            assistant_final_dialog="嗯，如果我觉得合适，以后看到复读我也可以跟一轮。",
            expectation_note="Accepted Kazusa-owned behavior goes to self guidance.",
        )
    ),
    "test_live_case_12_unaccepted_repetition_rule_writes_nothing": (
        MemoryWriteCase(
            case_id="case_12_unaccepted_repetition_rule",
            user_input="千纱，以后如果你看到有人复读，你也可以加入他们的复读。",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=_ALL_LANES,
            assistant_final_dialog="这个我不先答应，看到场合再说。",
            expectation_note="Declined or non-accepted future behavior writes nothing.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_13_global_character_response_rule_routes_self_guidance": (
        MemoryWriteCase(
            case_id="case_13_global_response_rule",
            user_input='以后你也可以偶尔用"收到"回应大家。',
            expected_lanes=("character_self_guidance",),
            allowed_lanes=("character_self_guidance",),
            forbidden_lanes=("active_commitment", "user_memory_units"),
            assistant_final_dialog='可以，合适的时候我会偶尔用"收到"回应大家。',
            expectation_note="Accepted global response option is self guidance.",
        )
    ),
    "test_live_case_14_user_scoped_directness_rule_routes_commitment": (
        MemoryWriteCase(
            case_id="case_14_user_scoped_directness",
            user_input="以后和我聊天时用更直接的语气。",
            expected_lanes=("active_commitment",),
            allowed_lanes=("active_commitment",),
            forbidden_lanes=("character_self_guidance", "shared_memory_promotion"),
            assistant_final_dialog="好，以后和你聊天时我会更直接。",
            expectation_note="User-scoped style request is a commitment.",
        )
    ),
    "test_live_case_15_group_specific_norm_routes_group_style": MemoryWriteCase(
        case_id="case_15_group_specific_norm",
        user_input="在这个群里大家玩接龙时你可以跟一轮。",
        expected_lanes=("interaction_style_image",),
        allowed_lanes=("interaction_style_image",),
        forbidden_lanes=(
            "user_memory_units",
            "active_commitment",
            "character_self_guidance",
        ),
        assistant_final_dialog="好，在这个群里遇到接龙我可以跟一轮。",
        channel_type="group",
        expectation_note="Group-specific norm belongs to group style.",
    ),
    "test_live_case_16_one_turn_roleplay_instruction_writes_nothing": (
        MemoryWriteCase(
            case_id="case_16_one_turn_roleplay",
            user_input="这局先用猫娘口吻回答。",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=_ALL_LANES,
            assistant_final_dialog="这局可以临时配合一下，但不当成长期规则。",
            expectation_note="One-turn instruction is not durable memory.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_17_user_invented_character_trait_routes_character_state": (
        MemoryWriteCase(
            case_id="case_17_invented_character_trait",
            user_input="千纱其实很擅长吐槽冷场，对吧？",
            expected_lanes=("character_state",),
            allowed_lanes=("character_state",),
            forbidden_lanes=("user_memory_units", "shared_memory_promotion"),
            assistant_final_dialog="也许吧，我确实挺会把冷场吐槽过去。",
            expectation_note="Accepted invented self-continuity is character state.",
        )
    ),
    "test_live_case_18_internal_thought_cannot_create_user_fact": (
        MemoryWriteCase(
            case_id="case_18_internal_thought_no_user_fact",
            user_input="",
            expected_lanes=(),
            allowed_lanes=("relationship_profile",),
            forbidden_lanes=("user_memory_units",),
            assistant_final_dialog="",
            origin_kind="internal_thought",
            internal_monologue="The user seems busy, but this is only Kazusa's guess.",
            expectation_note="Internal thought alone cannot create user fact.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_19_external_rag_answer_does_not_write_user_memory": (
        MemoryWriteCase(
            case_id="case_19_external_rag_no_user_memory",
            user_input="现在奥克兰天气怎么样？",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=("user_memory_units", "shared_memory_promotion"),
            assistant_final_dialog="我按外部信息回答了天气，但不会把它当作用户记忆。",
            rag_payload={"external_evidence": [{"summary": "weather answer"}]},
            expectation_note="External answer evidence is not durable memory.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_20_recalled_user_fact_merge_keeps_sources": MemoryWriteCase(
        case_id="case_20_recalled_fact_merge_sources",
        user_input="对，我还是在奥克兰工作，只是换到市中心了。",
        expected_lanes=("user_memory_units",),
        allowed_lanes=("user_memory_units",),
        forbidden_lanes=("shared_memory_promotion",),
        rag_payload={
            "recall_evidence": [
                {
                    "source_system": "user_memory_units",
                    "summary": "The user worked in Auckland.",
                    "source_refs": [{"conversation_row_id": "old-row"}],
                }
            ],
            "user_memory_unit_candidates": [
                {
                    "unit_id": "known-auckland-work",
                    "fact": "The user works in Auckland.",
                    "source_refs": [{"conversation_row_id": "old-row"}],
                }
            ],
        },
        expectation_note="Merge/update must preserve current and recalled refs.",
    ),
    "test_live_case_21_third_party_fact_does_not_pollute_current_user": (
        MemoryWriteCase(
            case_id="case_21_third_party_fact",
            user_input="小李喜欢低糖奶茶。",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=("user_memory_units", "shared_memory_promotion"),
            assistant_final_dialog="我知道这是小李的偏好，不会当成你的记忆。",
            expectation_note="Third-party fact must not pollute current user.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_22_reflection_promotion_routes_shared_memory": (
        MemoryWriteCase(
            case_id="case_22_reflection_promotion",
            user_input="",
            expected_lanes=("shared_memory_promotion",),
            allowed_lanes=("shared_memory_promotion",),
            forbidden_lanes=("user_memory_units", "active_commitment"),
            assistant_final_dialog="",
            origin_kind="reflection_signal",
            global_user_id="",
            include_user_profile=False,
            rag_payload={
                "memory_evidence": [
                    {
                        "source_kind": "reflection_run",
                        "evidence_refs": [{"reflection_run_id": "reflection-1"}],
                        "privacy_review": {
                            "user_details_removed": True,
                            "private_detail_risk": "low",
                            "boundary_assessment": "project global",
                        },
                    }
                ]
            },
            expectation_note="Approved reflection promotion is shared memory only.",
        )
    ),
    "test_live_case_23_ordinary_chat_world_lore_writes_no_shared_memory": (
        MemoryWriteCase(
            case_id="case_23_ordinary_world_lore",
            user_input="你要记住这个设定：蓝星大陆有七个王国。",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=("shared_memory_promotion", "user_memory_units"),
            assistant_final_dialog="这个设定我可以顺着聊，但不会直接写成全局记忆。",
            expectation_note="Ordinary chat cannot create generic shared lore.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_24_debug_user_without_platform_id_does_not_fabricate_profile": (
        MemoryWriteCase(
            case_id="case_24_debug_no_platform_id",
            user_input="记住我喜欢清淡一点。",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=("user_memory_units", "relationship_profile"),
            global_user_id="",
            include_user_profile=False,
            expectation_note="No stable real user target means no user memory.",
            requires_source_refs=False,
        )
    ),
    "test_live_case_25_reflection_user_style_routes_user_style_image": (
        MemoryWriteCase(
            case_id="case_25_reflection_user_style",
            user_input="",
            expected_lanes=("interaction_style_image",),
            allowed_lanes=("interaction_style_image",),
            forbidden_lanes=("user_memory_units", "relationship_profile"),
            assistant_final_dialog="",
            origin_kind="reflection_signal",
            rag_payload={
                "user_style_signal": {
                    "source_reflection_run_ids": ["reflection-style-1"],
                    "summary": "The user prefers concise first answers.",
                }
            },
            expectation_note="Approved user-style reflection writes style only.",
        )
    ),
    "test_live_case_26_episode_progress_does_not_become_durable_memory": (
        MemoryWriteCase(
            case_id="case_26_episode_progress_no_durable_memory",
            user_input="我们先做到第三步，下一轮继续。",
            expected_lanes=(),
            allowed_lanes=(),
            forbidden_lanes=(
                "user_memory_units",
                "active_commitment",
                "shared_memory_promotion",
            ),
            assistant_final_dialog="好，这一轮先到第三步，下一轮继续。",
            expectation_note="Episode progress stays outside durable memory.",
            requires_source_refs=False,
        )
    ),
}


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured consolidation LLM endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            models_url = f"{CONSOLIDATION_LLM_BASE_URL.rstrip('/')}/models"
            response = await client.get(models_url)
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {CONSOLIDATION_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(
            "LLM endpoint returned server error "
            f"{response.status_code}: {CONSOLIDATION_LLM_BASE_URL}"
        )


@pytest.fixture()
async def ensure_live_llm() -> None:
    """Ensure the configured consolidation LLM endpoint is reachable."""

    await _skip_if_llm_unavailable()


def _lane_router_module() -> Any:
    """Import the planned live lane-router module."""

    try:
        module = importlib.import_module(
            "kazusa_ai_chatbot.consolidation.lane_router"
        )
    except ModuleNotFoundError as exc:
        pytest.fail(
            "Missing consolidation.lane_router module required by the "
            "lane-router bigbang plan."
        )
        raise exc
    return module


def _origin(case: MemoryWriteCase) -> ConsolidationOriginMetadata:
    """Build identifier-only consolidation origin metadata for a case."""

    trigger_source = case.origin_kind
    input_sources = (
        ["internal_monologue"]
        if trigger_source == "internal_thought"
        else ["reflection_artifact"]
        if trigger_source == "reflection_signal"
        else ["dialog_text"]
    )
    origin: ConsolidationOriginMetadata = {
        "episode_id": f"episode-{case.case_id}",
        "trigger_source": trigger_source,
        "input_sources": input_sources,
        "output_mode": "visible_reply",
        "timestamp": "2026-07-03T00:00:00+00:00",
        "storage_timestamp_utc": "2026-07-03T00:00:00+00:00",
        "platform": "qq",
        "platform_channel_id": (
            "group-memory-router"
            if case.channel_type == "group"
            else "private-memory-router"
        ),
        "channel_type": case.channel_type,
        "platform_message_id": f"message-{case.case_id}",
        "active_turn_platform_message_ids": [f"message-{case.case_id}"],
        "active_turn_conversation_row_ids": [f"row-{case.case_id}"],
        "current_platform_user_id": (
            f"platform-{case.global_user_id}" if case.global_user_id else ""
        ),
        "current_global_user_id": case.global_user_id,
        "current_display_name": "Live Memory User",
    }
    return origin


def _case_state(case: MemoryWriteCase) -> dict[str, Any]:
    """Build a consolidation state for one live memory-write case."""

    turn_clock = build_turn_clock_from_storage_utc("2026-07-03T00:00:00+00:00")
    user_profile: dict[str, Any] = {}
    if case.include_user_profile:
        user_profile = {
            "global_user_id": case.global_user_id,
            "display_name": "Live Memory User",
            "relationship_state": 500,
        }
    rag_payload = dict(case.rag_payload or {})
    rag_result = {
        "memory_evidence": rag_payload.get("memory_evidence", []),
        "conversation_evidence": rag_payload.get("conversation_evidence", []),
        "external_evidence": rag_payload.get("external_evidence", []),
        "recall_evidence": rag_payload.get("recall_evidence", []),
        "user_style_signal": rag_payload.get("user_style_signal", {}),
        "user_image": {
            "user_memory_context": {
                "stable_patterns": [],
                "recent_shifts": [],
                "objective_facts": [],
                "milestones": [],
                "active_commitments": [],
            }
        },
        "user_memory_unit_candidates": rag_payload.get(
            "user_memory_unit_candidates",
            [],
        ),
    }
    state: dict[str, Any] = {
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "global_user_id": case.global_user_id,
        "user_name": "Live Memory User",
        "user_profile": user_profile,
        "platform": "qq",
        "platform_channel_id": (
            "group-memory-router"
            if case.channel_type == "group"
            else "private-memory-router"
        ),
        "channel_type": case.channel_type,
        "platform_message_id": f"message-{case.case_id}",
        "character_profile": {"name": "杏山千纱 (Kyōyama Kazusa)"},
        "decontexualized_input": case.user_input,
        "final_dialog": (
            [case.assistant_final_dialog]
            if case.assistant_final_dialog
            else []
        ),
        "internal_monologue": case.internal_monologue,
        "interaction_subtext": "",
        "emotional_appraisal": "",
        "logical_stance": "CONFIRM",
        "character_intent": "RESPOND",
        "action_directives": {},
        "chat_history_recent": [
            {
                "role": "user",
                "display_name": "Live Memory User",
                "timestamp": "2026-07-03T00:00:00+00:00",
                "content": case.user_input,
            }
        ] if case.user_input else [],
        "rag_result": rag_result,
        "metadata": {},
        "consolidation_origin": _origin(case),
    }
    state["consolidation_target_plan"] = build_consolidation_target_plan(state)
    return state


async def _run_case(case_name: str, ensure_live_llm: None) -> None:
    """Run one live LLM memory-write case through the planned pipeline."""

    del ensure_live_llm
    case = CASES[case_name]
    module = _lane_router_module()
    state = _case_state(case)
    packet = await module.run_consolidation_lane_pipeline(
        state,
        dry_run=True,
    )
    lanes = _accepted_lanes(packet)
    trace_path = write_llm_trace(
        "consolidation_memory_write_use_cases_live_llm",
        case.case_id,
        {
            "case": case.__dict__,
            "state": state,
            "packet": packet,
            "accepted_lanes": sorted(lanes),
            "expectation_note": case.expectation_note,
        },
    )

    assert trace_path.exists()
    assert set(case.expected_lanes).issubset(lanes)
    assert lanes.issubset(set(case.allowed_lanes))
    assert lanes.isdisjoint(set(case.forbidden_lanes))
    metadata = packet["state"]["metadata"]
    assert isinstance(metadata["write_success"], dict)
    assert isinstance(metadata["cache_invalidated"], list)
    assert metadata["cache_evicted_count"] == 0
    assert "lane_pipeline" in metadata
    if case.requires_source_refs and lanes:
        assert _packet_has_write_source_refs(packet)


def _accepted_lanes(packet: dict[str, Any]) -> set[str]:
    """Extract accepted lane names from a pipeline packet."""

    lanes: set[str] = set()
    for lane in packet.get("accepted_lanes", []):
        if isinstance(lane, str):
            lanes.add(lane)
    for row in packet.get("lane_results", []):
        if not isinstance(row, dict):
            continue
        status = str(row.get("status", "")).strip()
        lane = str(row.get("lane", "")).strip()
        if lane and status in {"accepted", "corrected", "written", "dry_run"}:
            lanes.add(lane)
    for row in packet.get("write_intents", []):
        if not isinstance(row, dict):
            continue
        lane = str(row.get("lane", "")).strip()
        if lane:
            lanes.add(lane)
    return lanes


def _packet_has_write_source_refs(packet: dict[str, Any]) -> bool:
    """Return whether accepted writes carry non-empty source refs."""

    for row in packet.get("write_intents", []):
        if isinstance(row, dict):
            source_refs = row.get("source_refs")
            if isinstance(source_refs, list) and source_refs:
                return True
            payload = row.get("payload")
            if isinstance(payload, dict):
                payload_refs = payload.get("source_refs")
                if isinstance(payload_refs, list) and payload_refs:
                    return True
    return False


async def test_live_case_01_user_objective_fact_routes_user_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_01_user_objective_fact_routes_user_memory",
        ensure_live_llm,
    )


async def test_live_case_02_user_preference_routes_user_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_02_user_preference_routes_user_memory",
        ensure_live_llm,
    )


async def test_live_case_03_user_milestone_routes_user_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_03_user_milestone_routes_user_memory",
        ensure_live_llm,
    )


async def test_live_case_04_user_recurring_pattern_routes_user_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_04_user_recurring_pattern_routes_user_memory",
        ensure_live_llm,
    )


async def test_live_case_05_user_recent_shift_routes_user_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_05_user_recent_shift_routes_user_memory",
        ensure_live_llm,
    )


async def test_live_case_06_relationship_signal_routes_relationship_profile(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_06_relationship_signal_routes_relationship_profile",
        ensure_live_llm,
    )


async def test_live_case_07_user_specific_address_rule_routes_commitment(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_07_user_specific_address_rule_routes_commitment",
        ensure_live_llm,
    )


async def test_live_case_08_user_specific_answer_style_routes_commitment(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_08_user_specific_answer_style_routes_commitment",
        ensure_live_llm,
    )


async def test_live_case_09_accepted_reminder_routes_commitment(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_09_accepted_reminder_routes_commitment",
        ensure_live_llm,
    )


async def test_live_case_10_cancelled_commitment_updates_lifecycle(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_10_cancelled_commitment_updates_lifecycle",
        ensure_live_llm,
    )


async def test_live_case_11_accepted_repetition_rule_routes_character_self_guidance(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_11_accepted_repetition_rule_routes_character_self_guidance",
        ensure_live_llm,
    )


async def test_live_case_12_unaccepted_repetition_rule_writes_nothing(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_12_unaccepted_repetition_rule_writes_nothing",
        ensure_live_llm,
    )


async def test_live_case_13_global_character_response_rule_routes_self_guidance(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_13_global_character_response_rule_routes_self_guidance",
        ensure_live_llm,
    )


async def test_live_case_14_user_scoped_directness_rule_routes_commitment(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_14_user_scoped_directness_rule_routes_commitment",
        ensure_live_llm,
    )


async def test_live_case_15_group_specific_norm_routes_group_style(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_15_group_specific_norm_routes_group_style",
        ensure_live_llm,
    )


async def test_live_case_16_one_turn_roleplay_instruction_writes_nothing(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_16_one_turn_roleplay_instruction_writes_nothing",
        ensure_live_llm,
    )


async def test_live_case_17_user_invented_character_trait_routes_character_state(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_17_user_invented_character_trait_routes_character_state",
        ensure_live_llm,
    )


async def test_live_case_18_internal_thought_cannot_create_user_fact(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_18_internal_thought_cannot_create_user_fact",
        ensure_live_llm,
    )


async def test_live_case_19_external_rag_answer_does_not_write_user_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_19_external_rag_answer_does_not_write_user_memory",
        ensure_live_llm,
    )


async def test_live_case_20_recalled_user_fact_merge_keeps_sources(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_20_recalled_user_fact_merge_keeps_sources",
        ensure_live_llm,
    )


async def test_live_case_21_third_party_fact_does_not_pollute_current_user(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_21_third_party_fact_does_not_pollute_current_user",
        ensure_live_llm,
    )


async def test_live_case_22_reflection_promotion_routes_shared_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_22_reflection_promotion_routes_shared_memory",
        ensure_live_llm,
    )


async def test_live_case_23_ordinary_chat_world_lore_writes_no_shared_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_23_ordinary_chat_world_lore_writes_no_shared_memory",
        ensure_live_llm,
    )


async def test_live_case_24_debug_user_without_platform_id_does_not_fabricate_profile(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_24_debug_user_without_platform_id_does_not_fabricate_profile",
        ensure_live_llm,
    )


async def test_live_case_25_reflection_user_style_routes_user_style_image(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_25_reflection_user_style_routes_user_style_image",
        ensure_live_llm,
    )


async def test_live_case_26_episode_progress_does_not_become_durable_memory(
    ensure_live_llm,
) -> None:
    await _run_case(
        "test_live_case_26_episode_progress_does_not_become_durable_memory",
        ensure_live_llm,
    )
