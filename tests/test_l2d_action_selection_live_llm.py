"""Real LLM routing checks for L2d action selection."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.config import COGNITION_LLM_BASE_URL
from kazusa_ai_chatbot.db import close_db, get_character_profile
from kazusa_ai_chatbot.cognition_chain_core.stages import l2d as l2d
from kazusa_ai_chatbot.cognition_chain_core.stages.l2d import (
    build_action_selection_payload_text,
    select_semantic_actions,
)
from kazusa_ai_chatbot.nodes import (
    persona_supervisor2_cognition_actions as action_connector,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_cognition_chain_services,
)
from kazusa_ai_chatbot.self_cognition import runner as self_cognition_runner
from kazusa_ai_chatbot.self_cognition.sources import (
    collect_active_commitment_cases,
)
from tests.l2d_action_selection_cases import (
    compare_action_specs_to_expectations,
    load_l2d_routing_case_set,
    select_l2d_routing_case,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_CASE_FILE_ENV = "L2D_LIVE_CASE_FILE"
_CASE_ID_ENV = "L2D_LIVE_CASE_ID"
REAL_COMMITMENT_CASE_LIMIT = 1
_FORBIDDEN_ACTION_SPEC_FRAGMENTS = (
    "background_artifact_request",
    "handler_id",
    "credentials",
    "api_key",
    "mongodb",
    "mongo",
    "collection",
    "self_cognition_action_attempts",
    "work_kind",
    "task_type",
    "coding_snippet",
    "text_rewrite",
)


async def test_l2d_live_case_against_frozen_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one frozen upstream case through L2d and compare route shape."""

    await _skip_if_llm_unavailable()
    case_file = _configured_case_file()
    case_id = _configured_case_id()
    case_set = load_l2d_routing_case_set(case_file)
    case = select_l2d_routing_case(case_set, case_id)
    frozen_state = case["frozen_l2d_state"]
    prompt_payload = build_action_selection_payload_text(frozen_state)

    action_selection_llm = build_cognition_chain_services().action_selection_llm
    capturing_llm = _CapturingLLM(action_selection_llm)
    monkeypatch.setattr(l2d, "_action_selection_llm", capturing_llm)
    result = await select_semantic_actions(frozen_state)
    raw_output = capturing_llm.raw_output
    raw_parsed_output = parse_llm_json_output(raw_output)
    action_specs = action_connector.materialize_semantic_action_requests(
        result.get("semantic_action_requests", []),
        frozen_state,
    )
    report = compare_action_specs_to_expectations(case, action_specs)
    leakage_errors = _action_spec_leakage_errors(action_specs)
    background_specs = [
        action_spec for action_spec in action_specs
        if action_spec.get("kind") == "background_work_request"
    ]
    trace_path = write_llm_trace(
        "l2d_action_selection_live_llm",
        case_id,
        {
            "case_id": case_id,
            "case_file": str(case_file),
            "source_kind": case["source_kind"],
            "historical_comparison": case["historical_comparison"],
            "prompt_payload": prompt_payload,
            "raw_model_output": raw_output,
            "raw_parsed_output": raw_parsed_output,
            "parsed_output": result,
            "comparison_report": report,
            "leakage_errors": leakage_errors,
            "background_work_specs": background_specs,
            "judgment": "manual_review_required_for_l2d_route_quality",
        },
    )
    logger.info(
        f"L2D_ACTION_SELECTION_LIVE case={case_id} "
        f"trace={trace_path} report={json.dumps(report, ensure_ascii=True)}"
    )

    assert len(action_specs) <= 3
    assert leakage_errors == []
    assert report["ok"] is True, report["errors"]
    assert len(background_specs) == 1
    task_brief = background_specs[0]["params"]["task_brief"]
    assert isinstance(task_brief, str)
    assert task_brief.strip()


@pytest.mark.live_db
async def test_l2d_live_routes_real_active_commitment_lifecycle_update() -> None:
    """Run L2d against one production active commitment lifecycle case."""

    await _skip_if_llm_unavailable()
    case = await _load_real_active_commitment_case()
    frozen_state = _lifecycle_update_state_from_case(case)
    prompt_payload = build_action_selection_payload_text(frozen_state)
    rag_result = frozen_state["rag_result"]
    user_image = rag_result["user_image"]
    user_memory_context = user_image["user_memory_context"]
    active_commitments = user_memory_context["active_commitments"]
    active_commitment = active_commitments[0]

    result = await select_semantic_actions(frozen_state)
    action_specs = result["action_specs"]
    observed_kinds = [
        action_spec["kind"]
        for action_spec in action_specs
        if isinstance(action_spec, dict)
    ]
    lifecycle_specs = [
        action_spec for action_spec in action_specs
        if action_spec["kind"] == "memory_lifecycle_update"
    ]
    leakage_errors = _action_spec_leakage_errors(action_specs)
    trace_path = write_llm_trace(
        "l2d_memory_lifecycle_live_llm",
        case["case_id"],
        {
            "case_id": case["case_id"],
            "source_kind": "production_active_commitment",
            "prompt_payload": prompt_payload,
            "selected_commitment": {
                "fact": active_commitment["fact"],
                "due_at": active_commitment.get("due_at"),
                "due_state": active_commitment.get("due_state"),
            },
            "parsed_output": result,
            "observed_kinds": observed_kinds,
            "leakage_errors": leakage_errors,
            "judgment": (
                "manual_review_required_for_memory_lifecycle_route_quality"
            ),
        },
    )
    logger.info(
        f"L2D_MEMORY_LIFECYCLE_LIVE case={case['case_id']} "
        f"trace={trace_path} kinds={json.dumps(observed_kinds)}"
    )

    assert "active_commitment_clues" in prompt_payload
    assert active_commitment["unit_id"] not in prompt_payload
    assert "speak" not in observed_kinds
    assert len(lifecycle_specs) == 1
    lifecycle_spec = lifecycle_specs[0]
    assert lifecycle_spec["visibility"] == "private"
    assert lifecycle_spec["target"]["target_kind"] == "cognitive_episode"
    assert lifecycle_spec["target"]["target_id"] is None
    assert lifecycle_spec["target"]["owner"] == "memory_lifecycle_specialist"
    assert lifecycle_spec["params"]["review_kind"] == (
        "active_commitment_lifecycle"
    )
    assert lifecycle_spec["params"]["detail"]
    assert set(lifecycle_spec["params"]) == {"review_kind", "detail"}
    serialized_spec = json.dumps(lifecycle_spec, ensure_ascii=False)
    assert active_commitment["unit_id"] not in serialized_spec
    assert "lifecycle_decision" not in serialized_spec
    assert leakage_errors == []


async def _skip_if_llm_unavailable() -> None:
    """Skip when the configured cognition endpoint is unavailable."""

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{COGNITION_LLM_BASE_URL.rstrip('/')}/models"
            )
    except httpx.HTTPError as exc:
        pytest.skip(f"LLM endpoint is unavailable: {COGNITION_LLM_BASE_URL}; {exc}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{COGNITION_LLM_BASE_URL}"
        )


def _configured_case_file() -> Path:
    """Read the selected frozen case file path from the environment."""

    raw_path = os.environ.get(_CASE_FILE_ENV)
    if raw_path is None or not raw_path.strip():
        pytest.skip(f"{_CASE_FILE_ENV} is required for one-case live L2d runs")
    case_file = Path(raw_path)
    if not case_file.exists():
        pytest.skip(f"{_CASE_FILE_ENV} does not exist: {case_file}")
    return case_file


def _configured_case_id() -> str:
    """Read the selected frozen case id from the environment."""

    case_id = os.environ.get(_CASE_ID_ENV)
    if case_id is None or not case_id.strip():
        pytest.skip(f"{_CASE_ID_ENV} is required for one-case live L2d runs")
    return case_id


async def _load_real_active_commitment_case() -> dict:
    """Read one active commitment case from the configured production database."""

    now = datetime.now(timezone.utc)
    try:
        character_profile = await get_character_profile()
        cases = await collect_active_commitment_cases(
            now=now,
            character_profile=character_profile,
            max_cases=REAL_COMMITMENT_CASE_LIMIT,
        )
    except PyMongoError as exc:
        pytest.skip(f"MongoDB unavailable for live lifecycle test: {exc}")
    finally:
        await close_db()

    if not cases:
        pytest.skip("No production active commitment case is available")
    case = cases[0]
    return case


def _lifecycle_update_state_from_case(case: dict) -> dict:
    """Build a frozen L2d state where the character chose commitment closure."""

    rendered_packet = (
        '自我认知检查到一个生产数据库中的有效承诺已经长期过期。'
        '当前决定不再等待自然触发，而是关闭这条承诺。'
    )
    state = self_cognition_runner._build_cognition_state(
        case,
        rendered_packet,
    )
    state.update({
        "decontexualized_input": rendered_packet,
        "logical_stance": "CONFIRM",
        "character_intent": "DISMISS",
        "judgment_note": (
            '角色已经决定将这个长期过期承诺标记为 abandoned，'
            '不再让它继续占据开放承诺列表。'
        ),
        "internal_monologue": (
            '这个承诺已经长期 past_due，继续等待只会让待办失真。'
            '我现在选择放弃这条承诺，并把它作为私有记忆生命周期更新处理。'
        ),
        "emotional_appraisal": '平静、明确，没有对外表达压力',
        "interaction_subtext": '自我维护承诺状态，不需要打扰用户',
        "boundary_core_assessment": {
            "boundary_issue": "none",
            "acceptance": "allow",
            "stance_bias": "confirm",
        },
        "social_distance": '私有自检',
        "emotional_intensity": '低',
        "vibe_check": '安静维护',
        "relational_dynamic": '关系稳定，当前只做后台承诺整理',
        "conversation_progress": {
            "source": "active_commitment_due_check",
            "current_thread": '私有承诺生命周期复核',
            "next_affordances": [
                '如果角色决定放弃承诺，关闭该承诺而不生成文字表层',
            ],
        },
    })
    return state


def _action_spec_leakage_errors(action_specs: list[dict]) -> list[str]:
    """Return prompt-safety errors found in action-spec output."""

    serialized = json.dumps(action_specs, ensure_ascii=False).lower()
    errors = []
    for fragment in _FORBIDDEN_ACTION_SPEC_FRAGMENTS:
        if fragment in serialized:
            errors.append(f"forbidden runtime fragment leaked: {fragment}")
    return errors


class _CapturingLLM:
    """Capture raw LLM output while preserving the production call path."""

    def __init__(self, inner_llm: object) -> None:
        self._inner_llm = inner_llm
        self.raw_output = ""

    async def ainvoke(self, messages: object) -> object:
        """Call the wrapped LLM and store the raw message content."""

        response = await self._inner_llm.ainvoke(messages)
        self.raw_output = str(response.content)
        return_value = response
        return return_value
