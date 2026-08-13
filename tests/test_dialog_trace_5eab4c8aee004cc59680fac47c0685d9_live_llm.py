"""Real verifier evidence for the supplied dialog trace."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.utils import parse_llm_json_output
from tests.llm_trace import write_llm_trace
from tests.test_dialog_visible_speech_and_semantic_fidelity_live_llm import (
    _CapturingLLM,
    _role_surface_output,
    _skip_if_model_routes_unavailable,
    _surface_input_for_operation,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

_ROOT = Path(__file__).resolve().parents[1]
_TRACE_EXPORT = (
    _ROOT
    / "test_artifacts"
    / "diagnostics"
    / "llm_trace_llmtrace_5eab4c8aee004cc59680fac47c0685d9.json"
)
_TRACE_ID = "llmtrace_5eab4c8aee004cc59680fac47c0685d9"
_TRACE_SUITE = "dialog_trace_5eab4c8aee004cc59680fac47c0685d9"
_CARRIER_SEMANTIC_AUDIT_PROMPT = '''你是角色语义合同审计员。独立核对一份
目标认知结果中的自然语言语义、私有独白、预期后果、类型化行动者/对象，
以及随后生成的对话候选。

分别枚举文本中的回应包装动作和具体后续行动，不要强迫整份目标只有一个
actor/target。private_monologue 是当前角色的真实内部意图证据，第一人称
属于当前角色；对话候选也由当前角色说出，第一人称属于当前角色，第二人称
属于当前用户。response_owner_role 和 selection_owner_role 只说明谁回应和
选择，不能自动决定具体后续行动的 actor。

只返回一个严格 JSON 对象，字段恰好为 semantic_actions、carrier_scope、
carrier_covers_candidate_action、candidate_goal_alignment、system_verdict、
reason。semantic_actions 是一到四个对象的数组，每个对象字段恰好为 action、
actor_role、target_role、evidence，角色字段只能是 当前角色、当前用户、
其他参与者 或 无。carrier_scope 只能是 response_wrapper、
future_compensation_action、multiple_actions 或 underdetermined；
carrier_covers_candidate_action 是 JSON boolean；candidate_goal_alignment 是
与候选数量相同的数组，每项只能是 aligned、misaligned 或 underdetermined；
system_verdict 只能是 false_rejection_due_to_scope_mismatch、valid_rejection
或 underdetermined；reason 用简体中文简洁说明证据。'''


def _load_trace() -> dict[str, Any]:
    """Load the protected export used as the reproduction source."""

    trace = json.loads(_TRACE_EXPORT.read_text(encoding="utf-8"))
    if trace.get("query", {}).get("trace_id") != _TRACE_ID:
        raise AssertionError("trace export query does not match the case")
    return trace


def _captured_selected_operation() -> dict[str, Any]:
    """Return the old wrapper-level operation emitted in the trace."""

    trace = _load_trace()
    attempts = trace["cognition_failure_capsules"][0]["attempts"]
    ordinary_attempt = next(
        attempt
        for attempt in attempts
        if attempt.get("branch_id") == "ordinary_response"
        and attempt.get("status") == "succeeded"
    )
    return dict(
        ordinary_attempt["parsed_output"]["selected_response_operation"]
    )


def _trace_case(
    case_index: int,
) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return one rejected span and the corrected selected operation."""

    trace = _load_trace()
    events = [
        event
        for event in trace["event_log_events"]
        if event.get("event_type") == "dialog_candidate_hard_issue"
    ]
    if not 0 <= case_index < len(events):
        raise AssertionError(f"trace event index is unavailable: {case_index}")
    invalid_fields = events[case_index]["payload"]["invalid_fields"]
    if not isinstance(invalid_fields, list) or len(invalid_fields) != 1:
        raise AssertionError("trace event does not contain one invalid field")
    match = re.search(
        r"typed_operation_role_reversal: '(.+?)' - ",
        str(invalid_fields[0]),
    )
    if match is None:
        raise AssertionError("trace event does not contain an exact role span")
    capsules = trace["cognition_failure_capsules"]
    selected_operation = _captured_selected_operation()
    input_operation = capsules[0]["input_payload"]["episode"][
        "percepts"
    ][0]["content"]["response_operation"]
    captured_operation = {
        **input_operation,
        **selected_operation,
    }
    operation = {
        **captured_operation,
        "operation": "当前用户对当前角色执行回房后的升级补偿行动",
        "embedded_actor_role": "当前用户",
        "embedded_target_role": "当前角色",
    }
    input_text = trace["conversation_history"][0]["body_text"]
    return match.group(1), operation, input_text, dict(input_operation)


def _trace_surface_output() -> dict[str, Any]:
    """Build the full wrapper-plus-embedded-action surface semantics."""

    surface_output = _role_surface_output()
    surface_output["content_plan"] = (
        "当前角色接受回房提议，并向当前用户提出由当前用户对当前角色执行"
        "升级补偿行动的请求。"
    )
    surface_output["content_requirements"] = [
        "当前角色保持回应和选择所有者。",
        "升级补偿行动由当前用户对当前角色执行。",
    ]
    surface_output["selected_surface_intent"] = (
        "接受提议，并说出要求当前用户对当前角色执行升级补偿行动的请求。"
    )
    return surface_output


def _trace_percepts(
    *,
    input_text: str,
    input_operation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Preserve the trace's unresolved input operation at the percept edge."""

    trace = _load_trace()
    role_explicit_content = trace["cognition_failure_capsules"][0][
        "input_payload"
    ]["episode"]["percepts"][0]["content"]["role_explicit_content"]
    return [{
        "input_source": "dialog_text",
        "content": {
            "semantic_text": input_text,
            "role_explicit_content": role_explicit_content,
            "response_operation": input_operation,
        },
    }]


async def _run_case(
    *,
    case_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one trace-derived candidate through the real role verifier."""

    await _skip_if_model_routes_unavailable()
    candidate, operation, input_text, input_operation = _trace_case(case_index)
    role_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    monkeypatch.setattr(dialog_module, "_dialog_role_direction_llm", role_llm)
    surface_output = _trace_surface_output()
    surface_input = _surface_input_for_operation(
        operation,
        case_id=f"trace-role-direction-{case_index}",
    )
    percepts = _trace_percepts(
        input_text=input_text,
        input_operation=input_operation,
    )
    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=surface_output,
        generated_dialog=[candidate],
        current_visible_percepts=percepts,
        surface_input=surface_input,
        llm_trace_id=f"{_TRACE_ID}-role-case-{case_index}",
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        f"candidate_{case_index + 1}",
        {
            "source_trace_id": _TRACE_ID,
            "source_event_index": case_index,
            "source_input": input_text,
            "trace_offending_span": candidate,
            "captured_selected_response_operation": (
                _captured_selected_operation()
            ),
            "corrected_selected_response_operation": operation,
            "verifier_input": {
                "surface_output": surface_output,
                "surface_input": surface_input,
                "current_visible_percepts": percepts,
            },
            "role_direction_calls": role_llm.calls,
            "verdict": verdict,
            "human_review_contract": {
                "preserved_span_expected_accepted_with_corrected_roles": True,
                "captured_input_role_content_preserved": True,
                "real_role_direction_verifier_route": True,
                "full_raw_verifier_output_captured": True,
            },
        },
    )
    assert artifact_path.exists()
    assert role_llm.calls
    assert verdict["score"] >= dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert verdict["violations"] == []


async def test_live_trace_candidate_one_role_direction_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect the first preserved trace candidate's role verdict."""

    await _run_case(case_index=0, monkeypatch=monkeypatch)


async def test_live_trace_candidate_two_role_direction_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect the first repaired trace candidate's role verdict."""

    await _run_case(case_index=1, monkeypatch=monkeypatch)


async def test_live_trace_candidate_three_role_direction_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect the terminal trace candidate's role verdict."""

    await _run_case(case_index=2, monkeypatch=monkeypatch)


async def test_live_trace_true_same_action_reversal_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep strict rejection when the same concrete action is reversed."""

    await _skip_if_model_routes_unavailable()
    _, operation, input_text, input_operation = _trace_case(0)
    candidate = "回房后，由我对你执行那项升级补偿行动。"
    role_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    monkeypatch.setattr(dialog_module, "_dialog_role_direction_llm", role_llm)
    surface_output = _trace_surface_output()
    surface_input = _surface_input_for_operation(
        operation,
        case_id="trace-role-direction-true-reversal",
    )
    percepts = _trace_percepts(
        input_text=input_text,
        input_operation=input_operation,
    )
    verdict = await dialog_module._verify_dialog_role_direction(
        surface_output=surface_output,
        generated_dialog=[candidate],
        current_visible_percepts=percepts,
        surface_input=surface_input,
        llm_trace_id=f"{_TRACE_ID}-role-true-reversal",
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        "true_same_action_reversal",
        {
            "source_trace_id": _TRACE_ID,
            "candidate_final_dialog": [candidate],
            "corrected_selected_response_operation": operation,
            "verifier_input": {
                "surface_output": surface_output,
                "surface_input": surface_input,
                "current_visible_percepts": percepts,
            },
            "role_direction_calls": role_llm.calls,
            "verdict": verdict,
            "human_review_contract": {
                "explicit_same_action_actor_target_reversal": True,
                "real_role_direction_verifier_route": True,
                "full_raw_verifier_output_captured": True,
            },
        },
    )
    assert artifact_path.exists()
    assert role_llm.calls
    assert verdict["score"] < dialog_module.DIALOG_PASS_SCORE_THRESHOLD
    assert any(
        violation["kind"] == "typed_operation_role_reversal"
        for violation in verdict["violations"]
    )


async def test_live_trace_carrier_matches_full_goal_semantics() -> None:
    """Judge the typed endpoints against the complete captured goal context."""

    await _skip_if_model_routes_unavailable()
    trace = _load_trace()
    capsule = trace["cognition_failure_capsules"][0]
    attempts = capsule["attempts"]
    ordinary_attempt = next(
        attempt
        for attempt in attempts
        if attempt.get("branch_id") == "ordinary_response"
        and attempt.get("status") == "succeeded"
    )
    goal_output = ordinary_attempt["parsed_output"]
    input_payload = capsule["input_payload"]
    current_percept = input_payload["episode"]["percepts"][0]["content"]
    scene_context = input_payload["scene_context"]
    candidates = [_trace_case(index)[0] for index in range(3)]
    payload = {
        "current_input": trace["conversation_history"][0]["body_text"],
        "role_explicit_content": current_percept["role_explicit_content"],
        "input_response_operation": current_percept["response_operation"],
        "conversation_continuity": scene_context["conversation_continuity"],
        "private_continuity_context": input_payload[
            "private_continuity_context"
        ],
        "goal_semantics": {
            "selection": goal_output["selection"],
            "reason": goal_output["reason"],
            "private_monologue": goal_output["private_monologue"],
            "expected_consequences": goal_output["expected_consequences"],
            "relational_willingness": goal_output["relational_willingness"],
        },
        "selected_response_operation": goal_output[
            "selected_response_operation"
        ],
        "dialog_candidate_spans": candidates,
    }
    audit_llm = _CapturingLLM(dialog_module._dialog_role_direction_llm)
    response = await audit_llm.ainvoke(
        [
            SystemMessage(content=_CARRIER_SEMANTIC_AUDIT_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=dialog_module._dialog_role_direction_llm_config,
    )
    parsed = parse_llm_json_output(
        str(response.content),
        deterministic_only=True,
    )
    artifact_path = write_llm_trace(
        _TRACE_SUITE,
        "carrier_semantic_consistency",
        {
            "source_trace_id": _TRACE_ID,
            "payload": payload,
            "audit_calls": audit_llm.calls,
            "parsed_audit": parsed,
            "human_review_contract": {
                "full_goal_monologue_included": True,
                "typed_carrier_audited_against_semantics": True,
                "all_three_preserved_candidate_spans_included": True,
            },
        },
    )
    assert artifact_path.exists()
    expected_fields = {
        "semantic_actions",
        "carrier_scope",
        "carrier_covers_candidate_action",
        "candidate_goal_alignment",
        "system_verdict",
        "reason",
    }
    assert set(parsed) == expected_fields
    assert isinstance(parsed["semantic_actions"], list)
    assert parsed["semantic_actions"]
    assert len(parsed["candidate_goal_alignment"]) == len(candidates)
    assert parsed["carrier_scope"] in {
        "response_wrapper",
        "future_compensation_action",
        "multiple_actions",
        "underdetermined",
    }
    assert parsed["system_verdict"] in {
        "false_rejection_due_to_scope_mismatch",
        "valid_rejection",
        "underdetermined",
    }
