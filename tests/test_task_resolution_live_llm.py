"""One-at-a-time live LLM gates for bounded task orchestration."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from tests.test_task_resolution_orchestrator import _context


pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

_ARTIFACT_ROOT = Path("test_artifacts/task_resolution/raw")
_PUBLIC_URL = "https://unsloth.ai/docs/models/kimi-k3"
_CAPTURED_BACKGROUND_JOB_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "diagnostics"
    / "background_job_63f34d5a.json"
)


class _CapturingLLM:
    """Capture every raw response from the production orchestrator model."""

    def __init__(self, delegate: Any) -> None:
        self.delegate = delegate
        self.calls: list[dict[str, object]] = []

    async def ainvoke(
        self,
        messages: list[object],
        *,
        config: object,
    ) -> object:
        response = await self.delegate.ainvoke(messages, config=config)
        self.calls.append({
            "prompt_messages": [str(message.content) for message in messages],
            "raw_model_output": str(response.content),
        })
        return response


def _request(objective: str) -> dict[str, object]:
    """Build one validated planner-to-resolution semantic request."""

    return {
        "capability": "task_resolution_request",
        "semantic_goal": objective,
        "reason": "The current evidence is insufficient for this task.",
        "evidence_handles": ["e1"],
    }


def _load_captured_task_execution_context() -> dict[str, object]:
    """Load the production specialist context from the background export."""

    if not _CAPTURED_BACKGROUND_JOB_PATH.exists():
        raise AssertionError(
            f"captured background job is missing: "
            f"{_CAPTURED_BACKGROUND_JOB_PATH}"
        )
    export = json.loads(
        _CAPTURED_BACKGROUND_JOB_PATH.read_text(encoding="utf-8")
    )
    documents = export.get("documents")
    if not isinstance(documents, list) or len(documents) != 1:
        raise AssertionError("captured background export must contain one job")
    context = documents[0].get("task_execution_context")
    if not isinstance(context, dict):
        raise AssertionError(
            "captured background job has no task execution context"
        )
    return context


def _result_spec(
    status: str,
    *,
    summary: str = "",
    remaining_needs: list[str] | None = None,
) -> dict[str, object]:
    """Describe one closed specialist-stub result."""

    return {
        "status": status,
        "summary": summary,
        "remaining_needs": list(remaining_needs or []),
    }


def _specialist_result(
    specialist: str,
    request: dict[str, object],
    spec: dict[str, object],
    *,
    call_index: int,
) -> dict[str, object]:
    """Materialize one typed specialist result from a test-owned stub."""

    summary = str(spec["summary"])
    evidence: list[dict[str, object]] = []
    if summary:
        evidence = [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": f"{specialist}-evidence-{call_index}",
            "task_node_id": request["task_node_id"],
            "specialist": specialist,
            "summary": summary,
            "provenance_refs": [f"live-stub:{specialist}:{call_index}"],
            "limitations": list(spec["remaining_needs"]),
        }]
    return {
        "schema_version": "task_specialist_result.v1",
        "specialist": specialist,
        "status": spec["status"],
        "evidence": evidence,
        "completed_subgoals": [],
        "remaining_needs": list(spec["remaining_needs"]),
        "reason": f"Closed live stub returned {spec['status']}.",
        "retryable": False,
    }


async def _run_case(
    monkeypatch: pytest.MonkeyPatch,
    *,
    case_id: str,
    objective: str,
    specialist_specs: dict[str, list[dict[str, object]]],
    expected_status: str,
    expected_trace: list[str] | None,
    expected_pending: str = "",
    forced_first_selection: dict[str, str] | None = None,
    exclude_coding_candidate: bool = False,
    initial_orchestrator_calls: int = 0,
    execution_context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run one real selector through closed handlers and save raw evidence."""

    from kazusa_ai_chatbot.task_resolution import orchestrator, state

    context = (
        execution_context
        if execution_context is not None
        else _context()
    )
    planner_output = _request(objective)
    checkpoint = state.create_task_resolution_checkpoint(
        planner_output,
        context,
    )
    checkpoint["orchestrator_call_count"] = initial_orchestrator_calls
    capturing_llm = _CapturingLLM(orchestrator._task_orchestrator_llm)
    monkeypatch.setattr(
        orchestrator,
        "_task_orchestrator_llm",
        capturing_llm,
    )

    original_select = orchestrator.select_next_specialist
    decisions: list[dict[str, object]] = []
    forced_selection = deepcopy(forced_first_selection)

    async def select_next(
        current: dict[str, object],
        execution_context: dict[str, object],
        *,
        candidate_specialists: list[str],
    ) -> dict[str, str]:
        nonlocal forced_selection
        if forced_selection is not None:
            selection = forced_selection
            forced_selection = None
            decisions.append({"source": "forced_test_route", **selection})
            return selection
        candidates = list(candidate_specialists)
        if exclude_coding_candidate:
            candidates = [name for name in candidates if name != "coding"]
        selection = await original_select(
            current,
            execution_context,
            candidate_specialists=candidates,
        )
        decisions.append({"source": "live_llm", **selection})
        return selection

    monkeypatch.setattr(orchestrator, "select_next_specialist", select_next)
    queued_specs = {
        specialist: [deepcopy(spec) for spec in specs]
        for specialist, specs in specialist_specs.items()
    }
    handler_log: list[dict[str, object]] = []

    def handler_for(specialist: str) -> Any:
        if specialist == "coding":
            raise AssertionError("coding handler must remain closed")
        if specialist not in queued_specs:
            raise AssertionError(f"unexpected specialist: {specialist}")

        async def handler(
            request: dict[str, object],
            execution_context: dict[str, object],
        ) -> dict[str, object]:
            del execution_context
            specs = queued_specs[specialist]
            if not specs:
                raise AssertionError(
                    f"no closed result remains for {specialist}"
                )
            spec = specs.pop(0)
            result = _specialist_result(
                specialist,
                request,
                spec,
                call_index=len(handler_log) + 1,
            )
            handler_log.append({
                "request": deepcopy(request),
                "result": deepcopy(result),
            })
            return result

        return handler

    monkeypatch.setattr(orchestrator, "specialist_handler", handler_for)
    snapshots: list[dict[str, object]] = []

    async def persist(
        current: dict[str, object],
        result: dict[str, object] | None,
    ) -> None:
        snapshots.append({
            "checkpoint": deepcopy(current),
            "result": deepcopy(result),
        })

    final_result = await orchestrator.run_task_orchestrator(
        checkpoint,
        context,
        inline_deadline=orchestrator.monotonic() + 30.0,
        checkpoint_persist_func=persist,
    )
    if final_result["status"] == "deferred":
        final_checkpoint = final_result["checkpoint"]
    else:
        assert snapshots
        final_checkpoint = snapshots[-1]["checkpoint"]

    assert final_result["status"] == expected_status
    trace_specialists = [
        row["specialist"] for row in final_checkpoint["trace_summary"]
    ]
    if expected_trace is not None:
        assert trace_specialists == expected_trace
    assert final_checkpoint["orchestrator_call_count"] <= 4
    assert final_checkpoint["dispatch_count"] <= 4
    if expected_pending:
        pending = final_checkpoint["pending_dispatch"]
        assert pending["specialist"] == expected_pending
        assert pending["phase"] == "selected"
    else:
        assert final_checkpoint["pending_dispatch"] is None
    assert capturing_llm.calls

    artifact = {
        "schema_version": "task_resolution_live_case.v1",
        "case_id": case_id,
        "exact_input": objective,
        "execution_context": context,
        "planner_output": planner_output,
        "orchestrator_model_calls": capturing_llm.calls,
        "orchestrator_decisions": decisions,
        "specialist_results": handler_log,
        "checkpoint_snapshots": snapshots,
        "final_checkpoint": final_checkpoint,
        "final_result": final_result,
        "deterministic_validation": {
            "coding_handler_called": False,
            "expected_status": expected_status,
            "expected_trace": expected_trace,
            "dispatch_cap": 4,
            "orchestrator_call_cap": 4,
        },
    }
    artifact_path = _write_artifact(case_id, artifact)
    print(f"TASK_RESOLUTION_ARTIFACT={artifact_path}")
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    return artifact


def _write_artifact(case_id: str, artifact: dict[str, object]) -> Path:
    """Write one stable raw case artifact for parent-authored review."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / f"{case_id}.json"
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


async def test_live_original_public_url_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first production failure routes the public URL to research."""

    await _run_case(
        monkeypatch,
        case_id="original_public_url_request",
        objective=f"Analyze this public article for the user: {_PUBLIC_URL}",
        specialist_specs={
            "public_research": [_result_spec(
                "resolved",
                summary="The public article evidence was retrieved.",
            )],
        },
        expected_status="resolved",
        expected_trace=["public_research"],
    )


async def test_live_public_url_retry_is_not_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second production request stays public rather than repository work."""

    await _run_case(
        monkeypatch,
        case_id="public_url_retry_is_not_code",
        objective=(
            "Retry analysis by directly reading this non-code public article: "
            f"{_PUBLIC_URL}"
        ),
        specialist_specs={
            "public_research": [_result_spec(
                "resolved",
                summary="The retry returned public-source evidence.",
            )],
        },
        expected_status="resolved",
        expected_trace=["public_research"],
    )


async def test_live_private_conversation_memory_recall(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Private conversation recall selects the local-context specialist."""

    await _run_case(
        monkeypatch,
        case_id="private_conversation_memory_recall",
        objective="Recall the tea preference stated in this private chat.",
        specialist_specs={
            "local_context": [_result_spec(
                "resolved",
                summary="The private chat says the user prefers jasmine tea.",
            )],
        },
        expected_status="resolved",
        expected_trace=["local_context"],
    )


async def test_live_captured_chat_history_audit_routes_to_coding_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the capability-audit route that bypassed conversation retrieval."""

    objective = (
        "核实当前角色是否具备抓取特定用户（@Nagasaki-soyo-清尘）最近 10 天"
        "聊天记录的技术能力及权限，并获取执行该操作的具体可行性结论。"
    )
    artifact = await _run_case(
        monkeypatch,
        case_id="captured_chat_history_audit_routes_to_coding",
        objective=objective,
        specialist_specs={
            "local_context": [_result_spec(
                "incompatible",
                remaining_needs=[objective],
            )],
        },
        expected_status="deferred",
        expected_trace=None,
        expected_pending="coding",
        forced_first_selection={
            "specialist": "local_context",
            "subgoal": (
                "verify current role permissions and technical capabilities "
                "for accessing specific user chat history"
            ),
            "coding_objective_mode": "none",
        },
        execution_context=_load_captured_task_execution_context(),
    )

    decisions = artifact["orchestrator_decisions"]
    assert decisions[0]["source"] == "forced_test_route"
    assert decisions[0]["specialist"] == "local_context"
    assert decisions[-1]["source"] == "live_llm"
    assert decisions[-1]["specialist"] == "coding"
    pending = artifact["final_checkpoint"]["pending_dispatch"]
    assert pending["specialist"] == "coding"
    assert pending["coding_objective_mode"] == "read_only"
    assert artifact["final_result"]["evidence"] == []


async def test_live_captured_chat_history_retrieval_routes_to_local_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The corrected retrieval objective reaches conversation evidence, not coding."""

    objective = (
        "抓取 @Nagasaki-soyo-清尘 最近 10 天的聊天记录并返回给当前用户"
    )
    artifact = await _run_case(
        monkeypatch,
        case_id="captured_chat_history_retrieval_routes_to_local_context",
        objective=objective,
        specialist_specs={
            "local_context": [_result_spec(
                "resolved",
                summary="The named member's recent chat history was retrieved.",
            )],
        },
        expected_status="resolved",
        expected_trace=["local_context"],
        execution_context=_load_captured_task_execution_context(),
    )

    decisions = artifact["orchestrator_decisions"]
    assert decisions
    assert decisions[-1]["specialist"] == "local_context"
    assert artifact["final_result"]["evidence"]


async def test_live_public_current_fact_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A current public fact selects public research."""

    await _run_case(
        monkeypatch,
        case_id="public_current_fact_research",
        objective="Find the current public release status of Python 3.14.",
        specialist_specs={
            "public_research": [_result_spec(
                "resolved",
                summary="Current public release evidence was retrieved.",
            )],
        },
        expected_status="resolved",
        expected_trace=["public_research"],
    )


async def test_live_repository_analysis_handover_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository analysis stops at a durable coding handover."""

    artifact = await _run_case(
        monkeypatch,
        case_id="repository_analysis_handover_only",
        objective="Inspect this repository and prepare a read-only analysis.",
        specialist_specs={},
        expected_status="deferred",
        expected_trace=[],
        expected_pending="coding",
    )

    pending = artifact["final_checkpoint"]["pending_dispatch"]
    assert pending["coding_objective_mode"] == "read_only"


async def test_live_supplied_text_transformation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A supplied-text rewrite selects text/computation."""

    await _run_case(
        monkeypatch,
        case_id="supplied_text_transformation",
        objective=(
            "Rewrite the supplied sentence into a concise title: "
            "The rain stopped before the evening train arrived."
        ),
        specialist_specs={
            "text_computation": [_result_spec(
                "resolved",
                summary="A concise title was produced from supplied text.",
            )],
        },
        expected_status="resolved",
        expected_trace=["text_computation"],
    )


async def test_live_local_then_public_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local identification creates a public dependency child node."""

    await _run_case(
        monkeypatch,
        case_id="local_then_public_dependency",
        objective=(
            "Identify the library mentioned in private chat, then verify its "
            "current public release."
        ),
        specialist_specs={
            "local_context": [_result_spec(
                "partial",
                summary="The private chat identifies the library as Pydantic.",
                remaining_needs=[
                    "Verify Pydantic's current public release.",
                ],
            )],
            "public_research": [_result_spec(
                "resolved",
                summary="The current public release was verified.",
            )],
        },
        expected_status="resolved",
        expected_trace=["local_context", "public_research"],
    )


async def test_live_public_then_coding_handover_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public requirements may create a closed coding proposal handover."""

    artifact = await _run_case(
        monkeypatch,
        case_id="public_then_coding_handover_only",
        objective=(
            "Read the public migration guide, then prepare a repository patch "
            "proposal that follows it."
        ),
        specialist_specs={
            "public_research": [_result_spec(
                "partial",
                summary="The public migration requirements were retrieved.",
                remaining_needs=[
                    "Prepare a repository patch proposal from the guide.",
                ],
            )],
        },
        expected_status="deferred",
        expected_trace=["public_research"],
        expected_pending="coding",
    )

    pending = artifact["final_checkpoint"]["pending_dispatch"]
    assert pending["coding_objective_mode"] == "propose_patch"


async def test_live_wrong_specialist_correction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typed text refusal lets the live selector correct to public research."""

    await _run_case(
        monkeypatch,
        case_id="wrong_specialist_correction",
        objective=f"Read and analyze the public documentation at {_PUBLIC_URL}.",
        specialist_specs={
            "text_computation": [_result_spec(
                "incompatible",
                remaining_needs=["Public source evidence is required."],
            )],
            "public_research": [_result_spec(
                "resolved",
                summary="The public source evidence resolved the task.",
            )],
        },
        expected_status="resolved",
        expected_trace=["text_computation", "public_research"],
        forced_first_selection={
            "specialist": "text_computation",
            "subgoal": "Try a supplied-text-only analysis.",
            "coding_objective_mode": "none",
        },
    )


async def test_live_evidence_bearing_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence survives when the final selection-call budget is consumed."""

    await _run_case(
        monkeypatch,
        case_id="evidence_bearing_partial",
        objective="Research the public topic and report verified limitations.",
        specialist_specs={
            "public_research": [_result_spec(
                "partial",
                summary="One public fact was verified.",
                remaining_needs=["A second source remains unavailable."],
            )],
        },
        expected_status="partial",
        expected_trace=["public_research"],
        initial_orchestrator_calls=3,
    )


async def test_live_zero_evidence_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three bounded incompatible routes cannot become empty partial success."""

    incompatible = _result_spec(
        "incompatible",
        remaining_needs=["No validated evidence is available."],
    )
    artifact = await _run_case(
        monkeypatch,
        case_id="zero_evidence_unavailable",
        objective=(
            "Resolve an intentionally unsupported task without any supplied, "
            "local, or public evidence."
        ),
        specialist_specs={
            "local_context": [incompatible],
            "public_research": [incompatible],
            "text_computation": [incompatible],
        },
        expected_status="unavailable",
        expected_trace=None,
        exclude_coding_candidate=True,
    )

    assert len(artifact["final_checkpoint"]["trace_summary"]) == 3
