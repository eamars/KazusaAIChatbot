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
        PMDecision,
        PMInput,
        ProgrammerAssignment,
        ProgrammerReport,
    )

    assert {
        "question",
        "repository_summary",
        "source_scope",
        "repo_map_summary",
        "previous_reports",
    } <= set(PMInput.__annotations__)
    assert {
        "status",
        "intent",
        "required_slots",
        "assignments",
        "missing_slots",
    } <= set(PMDecision.__annotations__)
    assert {
        "assignment_id",
        "role",
        "scope",
        "questions",
        "required_slots",
    } == set(ProgrammerAssignment.__annotations__)
    assert {
        "assignment_id",
        "status",
        "files_read",
        "facts",
        "evidence",
        "open_questions",
    } == set(ProgrammerReport.__annotations__)


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
    } == set(report)
    assert report["status"] == "succeeded"
    assert report["files_read"] == ["src/orders/service.py"]
    assert report["facts"]
    assert report["evidence"]
    assert "workspace" not in repr(report)
    assert "cache_key" not in repr(report)


def test_coding_agent_llm_route_absent_falls_back_to_background() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.llm_config import (
        resolve_coding_agent_llm_settings,
    )

    settings = resolve_coding_agent_llm_settings(
        {
            "BACKGROUND_WORK_LLM_BASE_URL": "http://background.example/v1",
            "BACKGROUND_WORK_LLM_API_KEY": "background-key",
            "BACKGROUND_WORK_LLM_MODEL": "background-model",
            "BACKGROUND_WORK_LLM_MAX_COMPLETION_TOKENS": "1200",
            "BACKGROUND_WORK_LLM_THINKING_ENABLED": "false",
        }
    )

    assert settings["route_name"] == "CODING_AGENT_LLM"
    assert settings["fallback_route_name"] == "BACKGROUND_WORK_LLM"
    assert settings["uses_fallback"] is True
    assert settings["base_url"] == "http://background.example/v1"
    assert settings["api_key"] == "background-key"
    assert settings["model"] == "background-model"


def test_coding_agent_llm_route_complete_configuration_is_used() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.llm_config import (
        resolve_coding_agent_llm_settings,
    )

    settings = resolve_coding_agent_llm_settings(
        {
            "CODING_AGENT_LLM_BASE_URL": "http://coding.example/v1",
            "CODING_AGENT_LLM_API_KEY": "coding-key",
            "CODING_AGENT_LLM_MODEL": "coding-model",
            "BACKGROUND_WORK_LLM_BASE_URL": "http://background.example/v1",
            "BACKGROUND_WORK_LLM_API_KEY": "background-key",
            "BACKGROUND_WORK_LLM_MODEL": "background-model",
        }
    )

    assert settings["route_name"] == "CODING_AGENT_LLM"
    assert settings["base_url"] == "http://coding.example/v1"
    assert settings["api_key"] == "coding-key"
    assert settings["model"] == "coding-model"


def test_coding_agent_llm_route_partial_configuration_fails_fast() -> None:
    from kazusa_ai_chatbot.coding_agent.code_reading.llm_config import (
        resolve_coding_agent_llm_settings,
    )

    with pytest.raises(ValueError, match="CODING_AGENT_LLM"):
        resolve_coding_agent_llm_settings(
            {
                "CODING_AGENT_LLM_BASE_URL": "http://coding.example/v1",
                "BACKGROUND_WORK_LLM_BASE_URL": "http://background.example/v1",
                "BACKGROUND_WORK_LLM_API_KEY": "background-key",
                "BACKGROUND_WORK_LLM_MODEL": "background-model",
            }
        )
