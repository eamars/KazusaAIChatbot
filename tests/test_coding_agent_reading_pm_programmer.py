from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _source_scope(
    kind: str = "repository",
    repo_relative_path: str | None = None,
) -> dict[str, Any]:
    scope = {
        "kind": kind,
        "repo_relative_path": repo_relative_path,
        "source_url": "local://fixture/repo",
        "requested_ref": None,
        "interpretation": "fixture scope",
    }
    return scope


def _assignment(
    *,
    kind: str,
    values: list[str],
) -> dict[str, Any]:
    assignment = {
        "assignment_id": "bounded-reader",
        "role": "interface reader",
        "scope": {
            "kind": kind,
            "values": values,
        },
        "questions": ["What source facts answer the local question?"],
        "required_slots": ["definition", "behavior"],
    }
    return assignment


def _repository(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "contract_repo"
    (repo_root / "src" / "orders").mkdir(parents=True)
    (repo_root / "src" / "orders" / "service.py").write_text(
        "\n".join(
            [
                "class OrderService:",
                "    def submit_order(self, payload: dict) -> dict:",
                "        return {'status': 'submitted', 'order_id': payload['id']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "contract-repo",
        "source_url": "https://github.com/fixture/contract-repo",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "a" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "github-fixture-contract-repo-main",
        "dirty_state": "clean",
    }
    return repository


def test_contracts_define_simplified_pm_programmer_shapes() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.models import (
        ReadingPMDecision,
        ReadingPMInput,
        ReadingProgrammerTask,
        ReadingProgrammerReport,
    )

    assert {
        "question",
        "repository_summary",
        "source_scope",
        "repo_map_summary",
        "previous_reports",
    } <= set(ReadingPMInput.__annotations__)
    assert {
        "status",
        "intent",
        "required_slots",
        "assignments",
        "missing_slots",
    } <= set(ReadingPMDecision.__annotations__)
    assert {
        "assignment_id",
        "role",
        "scope",
        "questions",
        "required_slots",
    } == set(ReadingProgrammerTask.__annotations__)
    assert {
        "assignment_id",
        "status",
        "files_read",
        "facts",
        "evidence",
        "open_questions",
        "discovered_symbols",
        "candidate_next_hops",
    } == set(ReadingProgrammerReport.__annotations__)


def test_reading_pm_retries_when_assignment_cap_is_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading import product_manager
    from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
        decide_reading_work,
    )

    def _decision_with_assignment_count(count: int) -> str:
        assignments = [
            {
                "assignment_id": f"reader_{index}",
                "role": "source reader",
                "scope": {
                    "kind": "search",
                    "values": [f"term_{index}"],
                },
                "questions": ["Read bounded source evidence."],
                "required_slots": [f"slot_{index}"],
            }
            for index in range(count)
        ]
        return product_manager.json.dumps({
            "status": "need_programmers",
            "intent": "architecture_overview",
            "required_slots": ["overview"],
            "assignments": assignments,
            "missing_slots": [],
        })

    responses = [
        _decision_with_assignment_count(4),
        _decision_with_assignment_count(2),
    ]
    calls = []

    def _fake_invoke(*args: Any, **kwargs: Any) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(content=responses[len(calls) - 1])

    monkeypatch.setattr(product_manager._reading_pm_llm, "invoke", _fake_invoke)
    trace: dict[str, Any] = {}

    decision = decide_reading_work(
        {
            "question": "Explain the local flow from source evidence.",
            "repository_summary": {"repo": "fixture"},
            "source_scope": _source_scope(),
            "repo_map_summary": {"files": ["src/flow.py"]},
            "previous_reports": [],
        },
        trace=trace,
    )

    assert decision["status"] == "need_programmers"
    assert len(decision["assignments"]) == 2
    assert len(calls) == 2
    assert len(trace["attempts"]) == 2


def test_assignment_validation_accepts_bounded_simplified_scope() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
        validate_programmer_assignment,
    )

    validate_programmer_assignment(
        _assignment(kind="file", values=["src/orders/service.py"]),
        _source_scope(),
    )


@pytest.mark.parametrize(
    "assignment",
    [
        _assignment(kind="file", values=[]),
        _assignment(kind="repository", values=["."]),
        {
            "assignment_id": "missing-questions",
            "role": "reader",
            "scope": {
                "kind": "file",
                "values": ["src/orders/service.py"],
            },
            "questions": [],
            "required_slots": ["definition"],
        },
    ],
)
def test_assignment_validation_rejects_unbounded_or_invalid_scope(
    assignment: dict[str, Any],
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
        validate_programmer_assignment,
    )

    with pytest.raises(ValueError):
        validate_programmer_assignment(assignment, _source_scope())


def test_assignment_validation_rejects_source_scope_escape() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.product_manager import (
        validate_programmer_assignment,
    )

    assignment = _assignment(kind="file", values=["src/orders/service.py"])

    with pytest.raises(ValueError, match="source scope"):
        validate_programmer_assignment(
            assignment,
            _source_scope("directory", "src/payments"),
        )


def test_programmer_report_uses_simplified_memory_shape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.programmer import (
        run_programmer_assignment,
    )
    from kazusa_ai_chatbot.coding_agent.code_reading import programmer

    repository = _repository(tmp_path)
    monkeypatch.setattr(
        programmer._programmer_llm,
        "invoke",
        lambda *_args, **_kwargs: SimpleNamespace(
            content=(
                '{"status":"succeeded","facts":[{"kind":"behavior",'
                '"summary":"OrderService.submit_order returns submitted '
                'order state.","evidence_refs":["src/orders/service.py:1-3"]}],'
                '"open_questions":[]}'
            )
        ),
    )
    report = run_programmer_assignment(
        repository,
        _assignment(kind="file", values=["src/orders/service.py"]),
        _source_scope(),
        max_files=6,
        max_excerpt_chars=12000,
    )

    assert {
        "assignment_id",
        "status",
        "files_read",
        "facts",
        "evidence",
        "open_questions",
        "discovered_symbols",
        "candidate_next_hops",
    } == set(report)
    assert report["status"] == "succeeded"
    assert report["files_read"] == ["src/orders/service.py"]
    assert report["facts"]
    assert report["evidence"]
    assert "workspace" not in repr(report)
    assert "cache_key" not in repr(report)


def test_reading_overload_preserves_existing_evidence() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.supervisor import (
        MAX_PROGRAMMERS_PER_WAVE,
        _overload_result_if_any,
    )

    evidence = [{
        "path": "src/runtime.py",
        "line_start": 1,
        "line_end": 4,
        "symbol_or_topic": "runtime owner",
        "excerpt": "def handle() -> None:\n    pass\n",
        "reason": "Shows the current owner.",
    }]
    existing_reports = [{
        "assignment_id": "runtime",
        "status": "succeeded",
        "files_read": ["src/runtime.py"],
        "facts": [],
        "evidence": evidence,
        "open_questions": [],
        "discovered_symbols": [],
        "candidate_next_hops": [],
    }]
    decision = {
        "status": "need_programmers",
        "intent": "read more",
        "assignments": [
            {
                "assignment_id": f"extra_{index}",
                "role": "code_reader",
                "scope": {"kind": "repository", "values": ["runtime"]},
                "questions": ["Read one extra slice."],
                "required_slots": ["runtime"],
            }
            for index in range(MAX_PROGRAMMERS_PER_WAVE + 1)
        ],
        "missing_slots": [],
    }

    result = _overload_result_if_any(decision, existing_reports, [])

    assert result is not None
    assert result["status"] == "needs_user_input"
    assert result["evidence"] == evidence


def test_code_reading_llm_routes_require_full_pm_and_programmer_settings() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.llm_config import (
        resolve_code_reading_llm_settings,
    )

    settings = resolve_code_reading_llm_settings(
        {
            "CODING_AGENT_PM_LLM_BASE_URL": "http://pm.example/v1",
            "CODING_AGENT_PM_LLM_API_KEY": "pm-key",
            "CODING_AGENT_PM_LLM_MODEL": "pm-model",
            "CODING_AGENT_PROGRAMMER_LLM_BASE_URL": "http://programmer.example/v1",
            "CODING_AGENT_PROGRAMMER_LLM_API_KEY": "programmer-key",
            "CODING_AGENT_PROGRAMMER_LLM_MODEL": "programmer-model",
        }
    )

    assert settings["pm"]["route_name"] == "CODING_AGENT_PM_LLM"
    assert settings["pm"]["base_url"] == "http://pm.example/v1"
    assert settings["pm"]["api_key"] == "pm-key"
    assert settings["pm"]["model"] == "pm-model"
    assert settings["programmer"]["route_name"] == "CODING_AGENT_PROGRAMMER_LLM"
    assert settings["programmer"]["base_url"] == "http://programmer.example/v1"
    assert settings["programmer"]["api_key"] == "programmer-key"
    assert settings["programmer"]["model"] == "programmer-model"


def test_code_reading_pm_route_is_reused_for_synthesis() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.llm_config import (
        resolve_code_reading_llm_settings,
    )

    settings = resolve_code_reading_llm_settings(
        {
            "CODING_AGENT_PM_LLM_BASE_URL": "http://pm.example/v1",
            "CODING_AGENT_PM_LLM_API_KEY": "pm-key",
            "CODING_AGENT_PM_LLM_MODEL": "pm-model",
            "CODING_AGENT_PROGRAMMER_LLM_BASE_URL": "http://programmer.example/v1",
            "CODING_AGENT_PROGRAMMER_LLM_API_KEY": "programmer-key",
            "CODING_AGENT_PROGRAMMER_LLM_MODEL": "programmer-model",
        }
    )

    assert settings["synthesis"] == settings["pm"]


def test_code_reading_llm_route_partial_configuration_fails_fast() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.llm_config import (
        resolve_code_reading_llm_settings,
    )

    with pytest.raises(ValueError, match="CODING_AGENT_PM_LLM"):
        resolve_code_reading_llm_settings(
            {
                "CODING_AGENT_PM_LLM_BASE_URL": "http://pm.example/v1",
                "CODING_AGENT_PROGRAMMER_LLM_BASE_URL": "http://programmer.example/v1",
                "CODING_AGENT_PROGRAMMER_LLM_API_KEY": "programmer-key",
                "CODING_AGENT_PROGRAMMER_LLM_MODEL": "programmer-model",
            }
        )

    with pytest.raises(ValueError, match="CODING_AGENT_PROGRAMMER_LLM"):
        resolve_code_reading_llm_settings(
            {
                "CODING_AGENT_PM_LLM_BASE_URL": "http://pm.example/v1",
                "CODING_AGENT_PM_LLM_API_KEY": "pm-key",
                "CODING_AGENT_PM_LLM_MODEL": "pm-model",
                "CODING_AGENT_PROGRAMMER_LLM_BASE_URL": "http://programmer.example/v1",
            }
        )
