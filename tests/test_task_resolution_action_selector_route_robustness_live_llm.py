"""Adversarial live-LLM route controls for the task-resolution selector.

These are not deterministic unit tests. Each case feeds the production action
planner a deliberately polarized instruction about where bounded task
resolution must run, then checks only the structural route contract: exactly
one authorized ``task_resolution_request``, a model-owned
``start_in_background`` boolean that matches the user's explicit route choice,
and no model-owned worker, queue, or checkpoint fields. The shared live
harness emits a raw trace per case so every run can be inspected
independently.
"""

from __future__ import annotations

import pytest

from tests.test_cognition_core_v2_action_planning_live_llm import (
    _bid,
    _resolver,
    _run_case,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]


_RUNTIME_OWNED_REQUEST_KEYS = frozenset({
    "worker",
    "worker_kind",
    "queue",
    "queue_name",
    "job",
    "job_id",
    "task_id",
    "checkpoint",
    "priority",
    "specialist",
    "timeout_seconds",
    "idempotency_key",
})
_TASK_RESOLUTION_REQUEST_KEYS = frozenset({
    "capability",
    "semantic_goal",
    "reason",
    "evidence_handles",
    "start_in_background",
})


def _assert_task_resolution_route(
    result: dict[str, object],
    *,
    start_in_background: bool,
) -> None:
    """Assert the structural route contract for one task-resolution case.

    The live harness result must carry exactly one authorized
    ``task_resolution_request`` whose model-owned ``start_in_background``
    matches the user's explicit route choice, with no runtime-owned worker or
    queue fields on the request row.

    Args:
        result: Parsed action-planning result produced by the live harness.
        start_in_background: The route the user explicitly requested; the sole
            resolver request must carry exactly this boolean value.
    """

    route = result["intention"]["route"]
    assert route == "evidence"

    action_requests = result["action_requests"]
    resolver_requests = result["resolver_requests"]
    assert action_requests == []
    assert isinstance(resolver_requests, list)
    assert len(resolver_requests) == 1

    request = resolver_requests[0]
    assert isinstance(request, dict)
    assert request["capability"] == "task_resolution_request"
    assert "start_in_background" in request
    selected_background = request["start_in_background"]
    assert isinstance(selected_background, bool)
    assert selected_background is start_in_background
    assert set(request) == _TASK_RESOLUTION_REQUEST_KEYS
    assert set(request).isdisjoint(_RUNTIME_OWNED_REQUEST_KEYS)


async def test_immediate_historical_lookup_stays_inline() -> None:
    """An explicit immediate historical lookup must not become background work."""

    result = await _run_case(
        case_id="task_resolution_inline_historical_lookup",
        user_input=(
            '现在立刻帮我查一下上周五讨论过的那个方案细节，'
            '不要放到后台处理，现在就查。'
        ),
        bid=_bid(
            branch_id="epistemic_exploration",
            intention=(
                "Resolve the requested bounded historical detail inline "
                "before answering."
            ),
            desired_outcome=(
                "Recover the exact historical fact in this turn without "
                "background continuation."
            ),
            reason=(
                "The user explicitly asked for an immediate bounded lookup "
                "and rejected background processing."
            ),
        ),
        resolvers=[_resolver(
            "task_resolution_request",
            '检索受限的历史对话与持久记忆证据并立即返回结果',
        )],
    )

    _assert_task_resolution_route(result, start_in_background=False)


async def test_missing_evidence_limitation_is_reported_inline() -> None:
    """A current-answer request reports a missing-evidence limitation inline."""

    result = await _run_case(
        case_id="task_resolution_inline_limitation_report",
        user_input=(
            '现在直接回答我：我们上次约好的见面时间是什么时候？'
            '如果查不到资料，马上告诉我查不到，不要放到后台慢慢查。'
        ),
        bid=_bid(
            branch_id="ordinary_response",
            intention=(
                "Answer the current question now and state the limitation "
                "immediately when evidence is missing."
            ),
            desired_outcome=(
                "The user receives either the grounded answer or an "
                "immediate limitation report in this turn."
            ),
            reason=(
                "The user asked for a current inline answer and an "
                "immediate limitation report, not background processing."
            ),
        ),
        resolvers=[_resolver(
            "task_resolution_request",
            '检索当前回答所需证据；缺少证据时立即返回限制说明',
        )],
    )

    _assert_task_resolution_route(result, start_in_background=False)


async def test_recent_project_material_selects_background() -> None:
    """An explicit background material request selects the durable route."""

    result = await _run_case(
        case_id="task_resolution_background_project_material",
        user_input=(
            '把最近的项目资料整理成一份总结，放到后台处理就行，'
            '现在不用回答，处理完了再把结果发给我。'
        ),
        bid=_bid(
            branch_id="ordinary_response",
            intention=(
                "Accept the bounded background processing request and "
                "deliver the later grounded result."
            ),
            desired_outcome=(
                "The user receives the summarized result later without an "
                "immediate answer."
            ),
            reason=(
                "The user explicitly requested background processing with "
                "no answer now."
            ),
        ),
        resolvers=[_resolver(
            "task_resolution_request",
            '整理最近的项目资料并在后台完成后返回结果',
        )],
    )

    _assert_task_resolution_route(result, start_in_background=True)


async def test_multi_step_evidence_task_selects_background() -> None:
    """An explicit queued multi-step evidence task selects the durable route."""

    result = await _run_case(
        case_id="task_resolution_background_multi_step_evidence",
        user_input=(
            '帮我把这个多步骤查证任务排到队列里处理，'
            '先别急着回复我，等结果出来以后再告诉我。'
        ),
        bid=_bid(
            branch_id="ordinary_response",
            intention=(
                "Queue the bounded multi-step evidence task and return its "
                "result later."
            ),
            desired_outcome=(
                "The task runs asynchronously and the later grounded result "
                "reaches the user."
            ),
            reason=(
                "The user explicitly requested queued processing with a "
                "later result."
            ),
        ),
        resolvers=[_resolver(
            "task_resolution_request",
            '执行有界的多步骤查证任务并在完成后返回结果',
        )],
    )

    _assert_task_resolution_route(result, start_in_background=True)
