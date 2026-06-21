from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _repository(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "acceptance_repo"
    _write(
        repo_root / "src" / "orders" / "service.py",
        [
            "from orders.payments import PaymentGateway",
            "",
            "class OrderService:",
            "    def submit_order(self, payload: dict) -> dict:",
            "        payment = PaymentGateway().charge(payload)",
            "        return {'status': payment['status']}",
        ],
    )
    _write(
        repo_root / "src" / "orders" / "payments.py",
        [
            "class PaymentGateway:",
            "    def charge(self, payload: dict) -> dict:",
            "        return {'status': 'charged', 'id': payload['id']}",
        ],
    )
    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "acceptance-repo",
        "source_url": "https://github.com/fixture/acceptance-repo",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "c" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "github-fixture-acceptance-repo-main",
        "dirty_state": "clean",
    }
    return repository


def _scope() -> dict[str, Any]:
    scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://fixture/acceptance-repo",
        "requested_ref": None,
        "interpretation": "entire repository",
    }
    return scope


def _request(tmp_path: Path, question: str) -> dict[str, Any]:
    request = {
        "question": question,
        "repository": _repository(tmp_path),
        "source_scope": _scope(),
        "preferred_language": "English",
        "max_answer_chars": 1800,
    }
    return request


def _evidence() -> list[dict[str, Any]]:
    rows = [
        {
            "path": "src/orders/service.py",
            "line_start": 4,
            "line_end": 5,
            "symbol_or_topic": "OrderService.submit_order",
            "excerpt": "payment = PaymentGateway().charge(payload)",
            "reason": "Matched order submission call.",
        },
        {
            "path": "src/orders/payments.py",
            "line_start": 1,
            "line_end": 2,
            "symbol_or_topic": "PaymentGateway.charge",
            "excerpt": "class PaymentGateway:\n    def charge(self, payload: dict) -> dict:",
            "reason": "Matched payment gateway definition.",
        },
    ]
    return rows


def _decision(status: str = "sufficient") -> dict[str, Any]:
    decision = {
        "status": status,
        "intent": "pipeline_or_data_flow",
        "required_slots": ["entry point", "downstream call"],
        "assignments": [
            {
                "assignment_id": "order-flow",
                "role": "call-flow reader",
                "scope": {
                    "kind": "file",
                    "values": ["src/orders/service.py"],
                },
                "questions": ["What call flow is visible?"],
                "required_slots": ["entry point", "downstream call"],
            }
        ],
        "missing_slots": [],
    }
    return decision


def _report() -> dict[str, Any]:
    report = {
        "assignment_id": "order-flow",
        "status": "succeeded",
        "files_read": ["src/orders/service.py", "src/orders/payments.py"],
        "facts": [
            {
                "kind": "call_flow",
                "summary": "OrderService.submit_order calls PaymentGateway.charge.",
                "evidence_refs": ["src/orders/service.py:4"],
            }
        ],
        "evidence": _evidence(),
        "open_questions": [],
        "discovered_symbols": ["OrderService", "PaymentGateway"],
        "candidate_next_hops": [
            {
                "reason": "Bounded evidence imports related modules.",
                "scope": {
                    "kind": "search",
                    "values": ["orders.payments"],
                },
            }
        ],
    }
    return report


def _patch_reading_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decision: dict[str, Any],
    report: dict[str, Any] | None = None,
    answer: str = "OrderService.submit_order calls PaymentGateway.charge.",
    patch_synthesis: bool = True,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import supervisor
    from kazusa_ai_chatbot.coding_agent.code_reading import product_manager
    from kazusa_ai_chatbot.coding_agent.code_reading import programmer
    from kazusa_ai_chatbot.coding_agent.code_reading import synthesizer

    reports = [] if report is None else [report]

    def fake_pm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return decision

    def fake_programmer(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if report is None:
            raise AssertionError("programmer should not run for this case")
        return report

    def fake_synthesis(*_args: Any, **_kwargs: Any) -> tuple[str, list[str]]:
        return answer, []

    monkeypatch.setattr(
        product_manager,
        "decide_reading_work",
        fake_pm,
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "decide_reading_work",
        fake_pm,
        raising=False,
    )
    monkeypatch.setattr(
        programmer,
        "run_programmer_assignment",
        fake_programmer,
        raising=False,
    )
    monkeypatch.setattr(
        supervisor,
        "run_programmer_assignment",
        fake_programmer,
        raising=False,
    )
    if patch_synthesis:
        monkeypatch.setattr(
            synthesizer,
            "synthesize_from_programmer_reports",
            fake_synthesis,
            raising=False,
        )
        monkeypatch.setattr(
            supervisor,
            "synthesize_from_programmer_reports",
            fake_synthesis,
            raising=False,
        )
    monkeypatch.setattr(
        supervisor,
        "selected_evidence_from_reports",
        lambda _reports: _evidence() if reports else [],
        raising=False,
    )


def test_public_run_uses_pm_programmer_reports_for_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    _patch_reading_pipeline(
        monkeypatch,
        decision=_decision(),
        report=_report(),
    )

    result = run(_request(tmp_path, "How does order submission call payment?"))

    assert result["status"] == "succeeded"
    assert "PaymentGateway.charge" in result["answer_text"]
    assert result["evidence"]
    assert "workspace" not in repr(result)
    assert "cache_key" not in repr(result)


def test_public_run_returns_needs_user_input_for_overloaded_pm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    _patch_reading_pipeline(
        monkeypatch,
        decision={
            "status": "overloaded",
            "intent": "architecture_overview",
            "required_slots": ["multiple subsystems"],
            "assignments": [],
            "missing_slots": ["narrower scope"],
        },
        report=None,
    )

    result = run(_request(tmp_path, "Explain every subsystem in this repository."))

    assert result["status"] == "needs_user_input"
    assert "narrower" in " ".join(result["limitations"]).casefold()
    assert result["evidence"] == []


def test_public_run_allows_third_bounded_programmer_wave(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run
    from kazusa_ai_chatbot.coding_agent.code_reading import supervisor

    assignment = _decision()["assignments"][0]
    decisions = [
        {
            **_decision("need_programmers"),
            "assignments": [{**assignment, "assignment_id": "wave-1"}],
        },
        {
            **_decision("need_programmers"),
            "assignments": [{**assignment, "assignment_id": "wave-2"}],
        },
        {
            **_decision("need_programmers"),
            "assignments": [{**assignment, "assignment_id": "wave-3"}],
        },
        {**_decision(), "assignments": []},
    ]
    reports = []

    def fake_pm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return decisions.pop(0)

    def fake_programmer(
        _repository_arg: dict[str, Any],
        assignment_arg: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        report = {**_report(), "assignment_id": assignment_arg["assignment_id"]}
        reports.append(report)
        return report

    monkeypatch.setattr(supervisor, "decide_reading_work", fake_pm)
    monkeypatch.setattr(supervisor, "run_programmer_assignment", fake_programmer)
    monkeypatch.setattr(
        supervisor,
        "synthesize_from_programmer_reports",
        lambda *_args, **_kwargs: ("Third wave evidence was synthesized.", []),
    )

    result = run(_request(tmp_path, "How does the bounded flow finish?"))

    assert result["status"] == "succeeded"
    assert len(reports) == 3
    assert "Third wave" in result["answer_text"]


def test_public_run_trims_assignments_to_remaining_report_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run
    from kazusa_ai_chatbot.coding_agent.code_reading import supervisor

    assignment = _decision()["assignments"][0]
    decisions = [
        {
            **_decision("need_programmers"),
            "assignments": [
                {**assignment, "assignment_id": "wave-1-a"},
                {**assignment, "assignment_id": "wave-1-b"},
                {**assignment, "assignment_id": "wave-1-c"},
            ],
        },
        {
            **_decision("need_programmers"),
            "assignments": [
                {**assignment, "assignment_id": "wave-2-a"},
                {**assignment, "assignment_id": "wave-2-b"},
            ],
        },
        {
            **_decision("need_programmers"),
            "assignments": [
                {**assignment, "assignment_id": "wave-3-a"},
                {**assignment, "assignment_id": "wave-3-b"},
            ],
        },
        {**_decision(), "assignments": []},
    ]
    reports = []

    def fake_pm(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return decisions.pop(0)

    def fake_programmer(
        _repository_arg: dict[str, Any],
        assignment_arg: dict[str, Any],
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        report = {**_report(), "assignment_id": assignment_arg["assignment_id"]}
        reports.append(report)
        return report

    monkeypatch.setattr(supervisor, "decide_reading_work", fake_pm)
    monkeypatch.setattr(supervisor, "run_programmer_assignment", fake_programmer)
    monkeypatch.setattr(
        supervisor,
        "synthesize_from_programmer_reports",
        lambda *_args, **_kwargs: ("Budget-trimmed evidence was synthesized.", []),
    )

    result = run(_request(tmp_path, "How does the bounded flow finish?"))

    assert result["status"] == "succeeded"
    assert len(reports) == supervisor.MAX_PROGRAMMER_REPORTS_PER_PM
    assert reports[-1]["assignment_id"] == "wave-3-a"
    assert "Budget-trimmed" in result["answer_text"]


def test_public_run_blocks_ungrounded_identifier_in_final_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import run

    _patch_reading_pipeline(
        monkeypatch,
        decision=_decision(),
        report=_report(),
        patch_synthesis=False,
    )
    from kazusa_ai_chatbot.coding_agent.code_reading import synthesizer

    monkeypatch.setattr(
        synthesizer._synthesis_llm,
        "invoke",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=(
                '{"answer_text":"OrderService.submit_order calls '
                'ShadowLedger.persist.","limitations":[]}'
            )
        ),
    )

    result = run(_request(tmp_path, "How does order submission call payment?"))

    assert result["status"] == "succeeded"
    assert "ShadowLedger" not in result["answer_text"]
    assert result["limitations"]


@pytest.mark.asyncio
async def test_direct_answer_preserves_reading_limitations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import answer_code_question
    from kazusa_ai_chatbot.coding_agent import code_fetching

    repository = _repository(tmp_path)

    async def fake_fetch(_request: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "succeeded",
            "message": "fixture resolved",
            "repository": repository,
            "source_scope": _scope(),
            "limitations": ["Fetching limitation."],
            "trace_summary": ["fetch:fixture"],
        }
        return result

    def fake_reading_run(_request: dict[str, Any]) -> dict[str, Any]:
        result = {
            "status": "needs_user_input",
            "answer_text": "Please narrow the requested source scope.",
            "evidence": [],
            "limitations": ["Reading limitation."],
            "trace_summary": ["reading:test"],
        }
        return result

    monkeypatch.setattr(code_fetching, "run", fake_fetch)
    monkeypatch.setattr(
        "kazusa_ai_chatbot.coding_agent.supervisor.code_reading.run",
        fake_reading_run,
    )

    response = await answer_code_question(
        {
            "question": "Explain order submission.",
            "source_url": repository["source_url"],
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "needs_user_input"
    assert response["answer_text"] == "Please narrow the requested source scope."
    assert response["limitations"] == [
        "Fetching limitation.",
        "Reading limitation.",
    ]
    assert "workspace" not in repr(response)
    assert "cache_key" not in repr(response)
