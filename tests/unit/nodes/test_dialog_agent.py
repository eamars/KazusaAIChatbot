"""Direct ownership tests for terminal dialog generation."""

from __future__ import annotations

import inspect
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,
    reset_protected_chain_records,
    snapshot_protected_chain_records,
)
from kazusa_ai_chatbot.cognition_episode import (
    build_goal_continuation_ref,
    build_tool_result_episode,
)
from kazusa_ai_chatbot.nodes import dialog_agent as dialog_module
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_generator
from tests.unit.nodes.dialog_fixtures import build_dialog_state


class _SequencedLLM:
    """Return deterministic dialog products or provider failures."""

    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[object] = []

    async def ainvoke(self, messages: object, *, config: object) -> object:
        del config
        self.calls.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, str):
            content = outcome
        else:
            content = json.dumps(outcome, ensure_ascii=False)
        return SimpleNamespace(content=content)


def _patch_dialog_recorders(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Capture dialog trace, contract, and quality events without persistence."""

    trace_events: list[dict[str, object]] = []
    contract_events: list[dict[str, object]] = []
    quality_events: list[dict[str, object]] = []

    async def record_trace(**kwargs: object) -> dict[str, object]:
        trace_events.append(kwargs)
        return {}

    async def record_contract(**kwargs: object) -> dict[str, object]:
        contract_events.append(kwargs)
        return {}

    async def record_quality(**kwargs: object) -> dict[str, object]:
        quality_events.append(kwargs)
        return {}

    monkeypatch.setattr(
        dialog_module.llm_tracing,
        "record_llm_trace_step",
        record_trace,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_llm_stage_event",
        record_trace,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_model_contract_event",
        record_contract,
    )
    monkeypatch.setattr(
        dialog_module.event_logging,
        "record_dialog_quality_event",
        record_quality,
    )
    return trace_events, contract_events, quality_events


def _source_dialog_state(
    source_url: str,
    *,
    artifact_text: str | None = None,
) -> dict[str, object]:
    """Build dialog state whose completed tool evidence carries one URL."""

    state = build_dialog_state()
    created_at = "2026-07-14T00:00:00Z"
    continuation_ref = build_goal_continuation_ref(
        source_episode_id="dialog-source-evidence",
        source_message_id="dialog-source-message",
        branch_id="ordinary_response",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "v2-test-user",
        },
    )
    episode = build_tool_result_episode(
        result={
            "schema_version": "tool_result_ready.v1",
            "task_id": "dialog-source-task",
            "task_kind": "background_work",
            "semantic_summary": "Completed source evidence.",
            "artifact_text": artifact_text or source_url,
            "failure_text": "",
            "completed_at": created_at,
            "target_scope": {
                "platform": "debug",
                "platform_channel_id": "channel-test",
                "channel_type": "private",
                "current_platform_user_id": "platform-user-test",
                "current_global_user_id": "v2-test-user",
                "current_display_name": "Test User",
                "target_addressed_user_ids": ["v2-test-user"],
                "target_broadcast": False,
            },
            "evidence_refs": [],
            "result_ref": "dialog-source-result",
            "goal_continuation_ref": continuation_ref,
        },
        evidence_refs=[],
        local_time_context={
            "current_local_datetime": "2026-07-14 12:00",
            "current_local_weekday": "Tuesday",
        },
        created_at=created_at,
    )
    state["cognitive_episode"] = episode
    return state


def test_dialog_prompt_is_a_complete_literal_with_owned_authority() -> None:
    """Keep dialog authority local to its complete immutable prompt."""

    source = inspect.getsource(dialog_module)
    prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT

    assert "VISIBLE_CONTENT_AUTHORITY_GUIDANCE" not in source
    assert "_V2_DIALOG_GENERATOR_PROMPT_TEMPLATE" not in source
    assert "_DIALOG_REPAIR_INSTRUCTION" not in source
    assert ".format(" not in source
    assert prompt.count("可见语义的选择权属于") == 1
    assert "dialog 必须服从已选 content_plan" in prompt
    assert "# 输出格式" not in prompt
    assert "字段必须恰好是 final_dialog" not in prompt


@pytest.mark.asyncio
async def test_dialog_agent_exposes_owned_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep terminal dialog generation attached to this source owner."""

    assert callable(dialog_generator)
    diagnostic = {
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "dialog_generation",
        "error_code": "dialog_source_url_degraded",
        "attempt_count": 3,
        "safe_checkpoint": "post_cognition_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }

    async def fake_generator(state: dict[str, object]) -> dict[str, object]:
        return {
            "final_dialog": ["retained"],
            "text_surface_output_v2": state["text_surface_output_v2"],
            "attempt_diagnostics": [diagnostic],
        }

    monkeypatch.setattr(dialog_module, "dialog_generator", fake_generator)
    state = build_dialog_state()
    state.pop("text_surface_input")
    result = await dialog_module.dialog_agent(state)

    assert result["attempt_diagnostics"] == [diagnostic]


def test_dialog_prompt_prioritizes_epistemic_boundary() -> None:
    """Keep P-owned assertion authority above lower surface plan fields."""

    prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT

    assert "epistemic_boundary" in prompt
    assert "它的权威高于" in prompt
    assert "未观察到的特征不能用来排除" in prompt
    assert "从句、前提句、原因连接和反问" in prompt
    assert "输出前逐句检查可见断言" in prompt
    assert "不用动作舞台提示、拟声" in prompt
    assert "低于 permitted_action_results 的事实权威" in prompt
    assert "action_kind=speak 只授权说出或发送 final_dialog 的文字" in prompt
    assert "同一类型、同一效果的 executed 行精确支持" in prompt
    assert "# 语义审计" in prompt
    assert "对未来外部效果的具体承诺也属于行动主张" in prompt
    assert "pending、scheduled 或 executed 行" in prompt


def test_dialog_creative_expansion_cannot_add_unselected_stance_or_relationship_payoff() -> None:
    """Keep wording creativity inside the selected semantic surface."""

    prompt = dialog_module._V2_DIALOG_GENERATOR_PROMPT

    assert "创造性展开不得增加 content_plan 未选择的立场" in prompt
    assert "不得把已表达的关系性回应模式改写成新的主要收束" in prompt


def test_validated_dialog_messages_collapses_blank_line_runs() -> None:
    """Collapse internal blank lines while preserving message boundaries."""

    value = {
        "final_dialog": [
            "first\n\nsecond\n\nthird\n\nfourth\n\nfifth",
            "single\nline",
        ],
    }

    validated_messages = dialog_module._validated_dialog_messages(value)

    assert validated_messages == [
        "first\nsecond\nthird\nfourth\nfifth",
        "single\nline",
    ]


@pytest.mark.asyncio
async def test_dialog_retry_prompt_carries_rejected_candidate_and_contract_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A structural rejection receives one bounded content-repair payload."""

    _, _, quality_events = _patch_dialog_recorders(monkeypatch)
    fake_llm = _SequencedLLM([
        {"final_dialog": "invalid string candidate"},
        {"final_dialog": ["repaired answer"]},
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_generator(build_dialog_state())

    assert result["final_dialog"] == ["repaired answer"]
    assert len(fake_llm.calls) == 2
    first_messages = fake_llm.calls[0]
    second_messages = fake_llm.calls[1]
    assert first_messages[0].content == second_messages[0].content
    first_payload = json.loads(first_messages[1].content)
    second_payload = json.loads(second_messages[1].content)
    assert set(second_payload) == set(first_payload) | {"contract_repair"}
    assert first_payload["output_contract"] == (
        dialog_module._DIALOG_OUTPUT_CONTRACT
    )
    assert second_payload["output_contract"] == first_payload["output_contract"]
    repair = second_payload["contract_repair"]
    assert set(repair) == {
        "reason",
        "contract_error",
        "invalid_candidate",
    }
    assert repair["contract_error"] == "dialog output messages are invalid"
    assert '"final_dialog": "invalid string candidate"' in (
        repair["invalid_candidate"]
    )
    assert "repair_instruction" not in json.dumps(second_payload)
    assert "guidance" not in json.dumps(first_payload)
    assert "instruction" not in json.dumps(first_payload)
    assert quality_events[0]["quality_status"] == "passed"


@pytest.mark.asyncio
async def test_missing_required_source_url_is_appended_without_regeneration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing required source token is repaired in the first attempt."""

    _, contract_events, quality_events = _patch_dialog_recorders(monkeypatch)
    source_url = "https://allowed.example/source"
    fake_llm = _SequencedLLM([
        {"final_dialog": ["answer without a source"]},
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_generator(_source_dialog_state(source_url))

    assert result["final_dialog"] == [
        f"answer without a source\n{source_url}",
    ]
    assert len(fake_llm.calls) == 1
    assert contract_events == [{
        "component": dialog_module.DIALOG_COMPONENT,
        "stage_name": "dialog_source_url_fidelity",
        "violation_kind": "source_url_fidelity",
        "missing_fields": [],
        "invalid_fields": [],
        "repair_used": True,
        "status": "normalized",
        "correlation_id": "visible-speech-test",
    }]
    assert quality_events[0]["quality_status"] == "passed"
    assert quality_events[0]["retry_count"] == 0


@pytest.mark.asyncio
async def test_unexpected_source_url_is_removed_before_degradation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsupported source token is removed with an empty allowed set."""

    trace_events, contract_events, quality_events = _patch_dialog_recorders(
        monkeypatch
    )
    fake_llm = _SequencedLLM([
        {"final_dialog": [
            "answer https://unexpected.example/source",
        ]},
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    token = bind_protected_chain_records(run_id="dialog-normalization-test")
    try:
        result = await dialog_generator(build_dialog_state())
        protected_records = snapshot_protected_chain_records()
    finally:
        reset_protected_chain_records(token)

    assert result["final_dialog"] == [
        "answer",
    ]
    assert len(fake_llm.calls) == 1
    assert contract_events[0]["status"] == "normalized"
    assert contract_events[0]["correlation_id"] == "visible-speech-test"
    assert any(
        row.get("parse_status") == "normalized"
        and row.get("status") == "succeeded"
        for row in trace_events
    )
    assert [row["status"] for row in protected_records] == ["parsed"]
    assert protected_records[0]["parse_status"] == "normalized"
    assert "attempt_diagnostics" not in result
    assert quality_events[0]["quality_status"] == "passed"


@pytest.mark.asyncio
async def test_dialog_delivers_newest_retained_candidate_after_structural_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The newest structurally valid source-fidelity candidate is retained."""

    _, _, quality_events = _patch_dialog_recorders(monkeypatch)
    source_url = "https://allowed.example/source"
    fake_llm = _SequencedLLM([
        RuntimeError("provider unavailable"),
        {"unexpected": "field"},
        {"final_dialog": ["c" * dialog_module.DIALOG_CANDIDATE_MAX_CHARS]},
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_generator(_source_dialog_state(source_url))

    assert result["final_dialog"] == [
        "c" * dialog_module.DIALOG_CANDIDATE_MAX_CHARS,
    ]
    assert len(fake_llm.calls) == 3
    assert quality_events[0]["quality_status"] == "accepted_degraded"
    assert quality_events[0]["failure_codes"] == ["source_url_fidelity"]
    assert quality_events[0]["retry_count"] == 2
    assert result["attempt_diagnostics"] == [{
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "dialog_generation",
        "error_code": "dialog_source_url_degraded",
        "attempt_count": 3,
        "safe_checkpoint": "post_cognition_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }]


@pytest.mark.asyncio
async def test_dialog_projects_content_plan_when_no_candidate_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural exhaustion projects the validated upstream content plan."""

    _, _, quality_events = _patch_dialog_recorders(monkeypatch)
    state = build_dialog_state()
    surface_output = deepcopy(state["text_surface_output_v2"])
    assert isinstance(surface_output, dict)
    surface_output["content_plan"] = "https://unusable.example/plan"
    surface_output["selected_surface_intent"] = "Use the selected intent."
    state["text_surface_output_v2"] = surface_output
    fake_llm = _SequencedLLM([
        {"unexpected": "field"},
        {"unexpected": "field"},
        {"unexpected": "field"},
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_generator(state)

    assert result["final_dialog"] == [
        "Use the selected intent.",
    ]
    assert len(fake_llm.calls) == 3
    assert quality_events[0]["quality_status"] == "accepted_degraded"
    assert quality_events[0]["failure_codes"] == [
        "deterministic_surface_projection",
    ]
    for messages in fake_llm.calls[1:]:
        payload = json.loads(messages[1].content)
        assert set(payload["contract_repair"]) == {
            "reason",
            "contract_error",
            "invalid_candidate",
        }
    assert result["attempt_diagnostics"] == [{
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "dialog_generation",
        "error_code": "dialog_surface_projection_degraded",
        "attempt_count": 3,
        "safe_checkpoint": "post_cognition_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }]


@pytest.mark.asyncio
async def test_dialog_never_raises_on_provider_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider exhaustion still returns a non-empty deterministic surface."""

    _, _, quality_events = _patch_dialog_recorders(monkeypatch)
    fake_llm = _SequencedLLM([
        RuntimeError("provider unavailable"),
        RuntimeError("provider unavailable"),
        RuntimeError("provider unavailable"),
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_generator(build_dialog_state())

    assert result["final_dialog"] == [
        "Answer the current request by inference.",
    ]
    assert len(fake_llm.calls) == 3
    assert quality_events[0]["quality_status"] == "accepted_degraded"
    assert quality_events[0]["failure_codes"] == [
        "deterministic_surface_projection",
    ]
    assert result["attempt_diagnostics"] == [{
        "schema_version": "episode_attempt_diagnostic.v1",
        "stage": "dialog_generation",
        "error_code": "dialog_surface_projection_degraded",
        "attempt_count": 3,
        "safe_checkpoint": "post_cognition_commit",
        "retryable": False,
        "final_status": "accepted_degraded",
    }]
    for messages in fake_llm.calls[1:]:
        payload = json.loads(messages[1].content)
        assert payload["contract_repair"]["reason"] == "no_candidate"
        assert payload["contract_repair"]["contract_error"] == ""
        assert payload["contract_repair"]["invalid_candidate"] == ""


@pytest.mark.asyncio
async def test_oversized_visible_percepts_bound_url_scan_without_failing_dialog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized evidence is bounded during URL extraction and cannot fail dialog."""

    _, _, quality_events = _patch_dialog_recorders(monkeypatch)
    source_url = "https://late.example/source"
    fake_llm = _SequencedLLM([
        {"final_dialog": ["answer"]},
    ])
    monkeypatch.setattr(dialog_module, "_dialog_generator_llm", fake_llm)

    result = await dialog_generator(_source_dialog_state(
        source_url,
        artifact_text=(
            "x" * dialog_module._DIALOG_VISIBLE_PERCEPT_SCAN_MAX_CHARS
            + source_url
        ),
    ))

    assert result["final_dialog"] == ["answer"]
    assert len(fake_llm.calls) == 1
    assert quality_events[0]["quality_status"] == "passed"
