import re
from pathlib import Path
from typing import Any

import pytest


def _repository(tmp_path: Path) -> dict[str, Any]:
    repo_root = tmp_path / "edit_repo"
    repo_root.mkdir()
    repository = {
        "provider": "github",
        "owner": "fixture",
        "repo": "editor",
        "source_url": "https://github.com/fixture/editor",
        "requested_ref": None,
        "resolved_ref": "main",
        "current_commit": "c" * 40,
        "default_branch": "main",
        "local_root": str(repo_root),
        "storage_kind": "existing_local_checkout",
        "managed_checkout": False,
        "workspace_root": str(tmp_path / "workspace"),
        "cache_key": "github-fixture-editor-main",
        "dirty_state": "clean",
    }
    return repository


def _source_scope() -> dict[str, Any]:
    scope = {
        "kind": "repository",
        "repo_relative_path": None,
        "source_url": "local://fixture/editor",
        "requested_ref": None,
        "interpretation": "fixture repository",
    }
    return scope


def _fetching_result(tmp_path: Path) -> dict[str, Any]:
    result = {
        "status": "succeeded",
        "message": "resolved",
        "repository": _repository(tmp_path),
        "source_scope": _source_scope(),
        "limitations": ["fetch note"],
        "trace_summary": ["fetch:resolved"],
    }
    return result


def _reading_result() -> dict[str, Any]:
    result = {
        "status": "succeeded",
        "answer_text": "The selected module owns the current behavior.",
        "evidence": [
            {
                "path": "src/module.py",
                "line_start": 1,
                "line_end": 3,
                "symbol_or_topic": "current behavior",
                "excerpt": "def current_value() -> int:",
                "reason": "Shows the behavior to change.",
            }
        ],
        "limitations": ["reading note"],
        "trace_summary": ["reading:evidence"],
    }
    return result


def test_file_agent_resolves_paths_and_cross_file_imports(tmp_path: Path) -> None:
    from kazusa_ai_chatbot.coding_agent.file_agent import (
        resolve_writing_file_demands,
    )

    repository = _repository(tmp_path)
    repo_root = Path(repository["local_root"])
    source_dir = repo_root / "src" / "pkg"
    test_dir = repo_root / "tests"
    source_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    (source_dir / "service.py").write_text("class ServiceRuntime:\n    pass\n")
    (source_dir / "state.py").write_text("class StateTracker:\n    pass\n")
    (test_dir / "test_service.py").write_text("def test_service():\n    pass\n")
    owner_candidates = [
        {
            "path": "src/pkg/service.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 2,
            "symbols": ["ServiceRuntime"],
            "exception_types": [],
            "feature_markers": ["python_source"],
            "reasons": ["runtime source candidate"],
            "evidence_refs": ["src/pkg/service.py:1-2"],
        },
        {
            "path": "src/pkg/state.py",
            "role": "runtime",
            "line_start": 1,
            "line_end": 2,
            "symbols": ["StateTracker"],
            "exception_types": [],
            "feature_markers": ["python_source"],
            "reasons": ["runtime source candidate"],
            "evidence_refs": ["src/pkg/state.py:1-2"],
        },
        {
            "path": "tests/test_service.py",
            "role": "test",
            "line_start": 1,
            "line_end": 2,
            "symbols": ["test_service"],
            "exception_types": [],
            "feature_markers": ["test_source"],
            "reasons": ["test source candidate"],
            "evidence_refs": ["tests/test_service.py:1-2"],
        },
    ]
    file_demands = [
        {
            "demand_id": "buffer_module",
            "role": "buffer module",
            "purpose": "Provide a reusable work buffer.",
            "file_kind": "new",
            "interface_contract": {
                "component": "WorkBuffer",
                "exports": ["WorkBuffer"],
                "inputs": ["work items"],
                "outputs": ["buffered work"],
                "invariants": ["stable ordering"],
            },
            "integration_contract": {
                "provides_to_pm": ["WorkBuffer"],
                "consumes_from": [],
            },
            "change_goal": "Create the work buffer module.",
            "work_instructions": ["Implement the buffer module."],
            "required_slots": ["WorkBuffer"],
            "validation_expectations": ["module import succeeds"],
        },
        {
            "demand_id": "service_integration",
            "role": "service integration",
            "purpose": "Connect ServiceRuntime to the work buffer.",
            "file_kind": "existing",
            "preferred_path": "src/pkg/service.py",
            "interface_contract": {
                "component": "ServiceRuntime",
                "exports": ["ServiceRuntime"],
                "inputs": ["buffered work"],
                "outputs": ["service result"],
                "invariants": ["service behavior remains bounded"],
            },
            "integration_contract": {
                "provides_to_pm": ["service result"],
                "consumes_from": ["buffer_module"],
            },
            "change_goal": "Update the service integration.",
            "work_instructions": ["Use the work buffer from buffer_module."],
            "required_slots": ["ServiceRuntime"],
            "validation_expectations": ["service import succeeds"],
        },
        {
            "demand_id": "state_update",
            "role": "state update",
            "purpose": "Connect StateTracker to the service integration result.",
            "file_kind": "existing",
            "preferred_path": "src/pkg/state.py",
            "interface_contract": {
                "component": "StateTracker",
                "exports": ["StateTracker"],
                "inputs": ["service result"],
                "outputs": ["state marker"],
                "invariants": ["state remains monotonic"],
            },
            "integration_contract": {
                "provides_to_pm": ["state marker"],
                "consumes_from": ["service_integration"],
            },
            "change_goal": "Update state tracking.",
            "work_instructions": ["Use the service integration output."],
            "required_slots": ["StateTracker"],
            "validation_expectations": ["state import succeeds"],
        },
        {
            "demand_id": "behavior_tests",
            "role": "behavior tests",
            "purpose": "Cover buffer and service behavior.",
            "file_kind": "test",
            "preferred_path": "tests/test_service.py",
            "interface_contract": {
                "component": "WorkBufferTests",
                "exports": ["test_work_buffer"],
                "inputs": ["buffered work"],
                "outputs": ["test result"],
                "invariants": ["tests do not use live services"],
            },
            "integration_contract": {
                "provides_to_pm": ["test coverage"],
                "consumes_from": ["buffer_module", "service_integration"],
            },
            "change_goal": "Add focused behavior tests.",
            "work_instructions": ["Test the work buffer and service integration."],
            "required_slots": ["test_work_buffer"],
            "validation_expectations": ["pytest can collect the tests"],
        },
    ]

    resolution = resolve_writing_file_demands(
        mode="edit_existing_repository",
        repository=repository,
        source_scope=_source_scope(),
        owner_candidates=owner_candidates,
        file_demands=file_demands,
    )

    assert resolution["status"] == "accepted"
    contracts_by_id = {
        contract["file_contract_id"]: contract
        for contract in resolution["file_contracts"]
    }
    assert contracts_by_id["buffer_module"]["owned_path"] == (
        "src/pkg/work_buffer.py"
    )
    assert contracts_by_id["service_integration"]["owned_path"] == (
        "src/pkg/service.py"
    )
    assert contracts_by_id["state_update"]["owned_path"] == "src/pkg/state.py"
    assert contracts_by_id["behavior_tests"]["owned_path"] == "tests/test_service.py"
    assert contracts_by_id["service_integration"]["cross_file_imports"] == [
        "from pkg.work_buffer import WorkBuffer"
    ]
    assert contracts_by_id["state_update"]["cross_file_imports"] == [
        "from pkg.service import ServiceRuntime"
    ]
    assert contracts_by_id["behavior_tests"]["cross_file_imports"] == [
        "from pkg.work_buffer import WorkBuffer",
        "from pkg.service import ServiceRuntime",
    ]


def _writing_result(mode: str = "edit_existing_repository") -> dict[str, Any]:
    result = {
        "status": "succeeded",
        "mode": mode,
        "answer_text": "Proposed one limited patch.",
        "patch_artifacts": [
            {
                "artifact_id": "main",
                "base": "repository",
                "diff_text": (
                    "diff --git a/src/module.py b/src/module.py\n"
                    "--- a/src/module.py\n"
                    "+++ b/src/module.py\n"
                    "@@ -1 +1 @@\n"
                    "-VALUE = 1\n"
                    "+VALUE = 2\n"
                ),
                "files": ["src/module.py"],
                "summary": "Updates one value.",
            }
        ],
        "created_files": [],
        "changed_files": [],
        "external_evidence_requests": [],
        "external_evidence": [],
        "validation": {
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": ["src/module.py"],
        },
        "session": {
            "session_id": "session-one",
            "public_handle": "writing-session-one",
            "invalidated_previous": False,
        },
        "limitations": ["writing note"],
        "trace_summary": ["writing:done"],
        "trace": {
            "pm_initial": {
                "raw_output": "{\"status\":\"need_module_pms\"}",
                "parsed_output": {"status": "need_module_pms"},
            }
        },
    }
    return result


async def test_propose_code_change_uses_fetching_reading_then_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    calls: dict[str, list[dict[str, Any]]] = {
        "fetching": [],
        "reading": [],
        "writing": [],
    }
    call_order: list[str] = []
    expected_fetching_result = _fetching_result(tmp_path)

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        call_order.append("fetching")
        calls["fetching"].append(request)
        return expected_fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        call_order.append("reading")
        calls["reading"].append(request)
        return _reading_result()

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        call_order.append("writing")
        calls["writing"].append(request)
        assert request["reading_result"] == _reading_result()
        return _writing_result()

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small behavior update.",
            "repo_hint": "fixture/editor",
            "workspace_root": str(tmp_path / "workspace"),
            "max_artifact_chars": 4000,
        }
    )

    assert response["status"] == "succeeded"
    assert response["mode"] == "edit_existing_repository"
    assert response["repository"] is not None
    assert response["repository"]["repo"] == "editor"
    assert response["evidence"] == [
        {
            "path": "src/module.py",
            "line_start": 1,
            "line_end": 3,
            "symbol_or_topic": "source evidence",
            "excerpt": "[source excerpt omitted from patch proposal response]",
            "reason": "Selected by the read-only evidence survey.",
        }
    ]
    assert response["patch_artifacts"]
    assert response["validation"]["status"] == "succeeded"
    assert response["trace"]["pm_initial"]["parsed_output"] == {
        "status": "need_module_pms"
    }
    assert calls["fetching"]
    assert calls["reading"]
    assert len(calls["writing"]) == 1
    assert call_order == ["fetching", "reading", "writing"]
    writing_request = calls["writing"][0]
    assert writing_request["repository"] == expected_fetching_result["repository"]
    assert writing_request["source_scope"] == _source_scope()
    assert writing_request["reading_result"] == _reading_result()
    assert writing_request["workspace_root"] == str(tmp_path / "workspace")
    reading_request = calls["reading"][0]
    assert reading_request["question"] != "Propose a small behavior update."
    assert "Read-only repository evidence survey" in reading_request["question"]
    assert "User request:" in reading_request["question"]
    assert "Propose a small behavior update." in reading_request["question"]
    assert "edit_repo" not in repr(response)
    assert "workspace" not in repr(response)
    assert "cache_key" not in repr(response)


async def test_propose_code_change_allows_followup_source_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    expected_fetching_result = _fetching_result(tmp_path)
    calls: dict[str, list[dict[str, Any]]] = {
        "reading": [],
        "writing": [],
    }

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        return expected_fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["reading"].append(request)
        result = _reading_result()
        row = result["evidence"][0]
        row["path"] = f"src/module_{len(calls['reading'])}.py"
        row["excerpt"] = f"def current_value_{len(calls['reading'])}() -> int:"
        result["answer_text"] = f"Reading pass {len(calls['reading'])}."
        return result

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["writing"].append(request)
        evidence_count = 0
        if request["reading_result"] is not None:
            evidence_count = len(request["reading_result"]["evidence"])
        if evidence_count < 2:
            result = _writing_result()
            result["status"] = "need_reading"
            result["answer_text"] = "More source evidence is required."
            result["patch_artifacts"] = []
            result["validation"] = {
                "status": "failed",
                "parsed": False,
                "sandbox_applied": False,
                "errors": [],
                "warnings": [],
                "files": [],
            }
            result["trace_summary"] = ["writing:need_reading"]
            result["reading_requests"] = [{
                "request_id": f"source-evidence-{evidence_count + 1}",
                "task": "Inspect the current runtime owner and companion tests.",
                "reason": "The writing PM needs another bounded source slice.",
                "required_slots": [
                    "Current runtime owner.",
                    "Companion validation path.",
                ],
            }]
            return result
        return _writing_result()

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small behavior update.",
            "repo_hint": "fixture/editor",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert len(calls["reading"]) == 2
    assert len(calls["writing"]) == 2
    final_writing_request = calls["writing"][-1]
    assert len(final_writing_request["reading_result"]["evidence"]) == 2
    assert "reading_merge:evidence=2" in response["trace_summary"]


async def test_propose_code_change_stops_repeated_source_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    expected_fetching_result = _fetching_result(tmp_path)
    calls: dict[str, list[dict[str, Any]]] = {
        "reading": [],
        "writing": [],
    }

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        return expected_fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["reading"].append(request)
        result = _reading_result()
        row = result["evidence"][0]
        row["path"] = f"src/module_{len(calls['reading'])}.py"
        row["excerpt"] = f"def current_value_{len(calls['reading'])}() -> int:"
        result["answer_text"] = f"Reading pass {len(calls['reading'])}."
        return result

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["writing"].append(request)
        result = _writing_result()
        result["status"] = "need_reading"
        result["answer_text"] = "More source evidence is required."
        result["patch_artifacts"] = []
        result["validation"] = {
            "status": "failed",
            "parsed": False,
            "sandbox_applied": False,
            "errors": [],
            "warnings": [],
            "files": [],
        }
        result["trace_summary"] = ["writing:need_reading"]
        result["reading_requests"] = [{
            "request_id": f"source-evidence-{len(calls['writing'])}",
            "task": "Inspect another bounded source owner.",
            "reason": "The writing PM needs another source slice.",
            "required_slots": ["Current runtime owner."],
        }]
        return result

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small behavior update.",
            "repo_hint": "fixture/editor",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "failed"
    assert len(calls["reading"]) == 2
    assert len(calls["writing"]) == 2
    first_state = calls["writing"][0]["supervisor_evidence_state"]
    second_state = calls["writing"][1]["supervisor_evidence_state"]
    assert first_state["remaining_reading_attempts"] == 1
    assert first_state["merged_reading_evidence_count"] == 1
    assert second_state["remaining_reading_attempts"] == 0
    assert second_state["merged_reading_evidence_count"] == 2
    assert second_state["completed_reading_requests"]
    assert "reading_budget:exhausted evidence=2" in response["trace_summary"]
    assert response["validation"]["errors"] == [
        "Writing PM requested source reading after supervisor reading budget was exhausted."
    ]


async def test_propose_code_change_continues_with_partial_reading_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    expected_fetching_result = _fetching_result(tmp_path)
    calls: dict[str, list[dict[str, Any]]] = {
        "reading": [],
        "writing": [],
    }

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        return expected_fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["reading"].append(request)
        result = _reading_result()
        result["status"] = "needs_user_input"
        result["answer_text"] = "Existing source evidence was found with one gap."
        result["limitations"] = ["Future artifact evidence is not present."]
        result["trace_summary"] = ["reading:partial"]
        return result

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["writing"].append(request)
        assert request["reading_result"]["status"] == "needs_user_input"
        assert request["reading_result"]["evidence"]
        return _writing_result()

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small behavior update.",
            "repo_hint": "fixture/editor",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert len(calls["reading"]) == 1
    assert len(calls["writing"]) == 1
    assert "reading_partial:status=needs_user_input evidence=1" in (
        response["trace_summary"]
    )


async def test_propose_code_change_new_project_skips_fetching_and_reading(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    calls: dict[str, list[dict[str, Any]]] = {
        "fetching": [],
        "reading": [],
        "writing": [],
    }

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["fetching"].append(request)
        raise AssertionError("Fetching should not run for source-free writing.")

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["reading"].append(request)
        raise AssertionError("Reading should not run for source-free writing.")

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["writing"].append(request)
        result = _writing_result(mode="create_new_project")
        result["repository"] = None
        result["created_files"] = [
            {
                "path": "src/app.py",
                "role": "application module",
            }
        ]
        return result

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Create a small source package from scratch.",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert response["mode"] == "create_new_project"
    assert response["repository"] is None
    assert response["evidence"] == []
    assert response["created_files"]
    assert calls["fetching"] == []
    assert calls["reading"] == []
    assert calls["writing"]
    assert calls["writing"][0]["repository"] is None
    assert calls["writing"][0]["reading_result"] is None


async def test_propose_code_change_runs_external_evidence_from_supervisor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change
    from kazusa_ai_chatbot.coding_agent import supervisor

    calls: dict[str, list[dict[str, Any]]] = {
        "writing": [],
        "external": [],
    }

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["writing"].append(request)
        if not request.get("external_evidence"):
            result = _writing_result(mode="create_new_project")
            result["status"] = "need_external_evidence"
            result["answer_text"] = "External evidence is required."
            result["patch_artifacts"] = []
            result["external_evidence_requests"] = [{
                "request_id": "docs",
                "task": "Find public documentation for the requested API.",
                "reason": "The patch needs current public API evidence.",
            }]
            result["validation"] = {
                "status": "failed",
                "parsed": False,
                "sandbox_applied": False,
                "errors": [],
                "warnings": [],
                "files": [],
            }
            result["trace_summary"] = ["writing:need_external_evidence"]
            return result

        result = _writing_result(mode="create_new_project")
        result["external_evidence"] = request["external_evidence"]
        result["trace_summary"] = ["writing:done"]
        return result

    async def fake_collect_external_evidence(
        requests: list[dict[str, str]],
        *,
        trace_summary: list[str],
    ) -> list[dict[str, object]]:
        calls["external"].append({"requests": requests})
        trace_summary.append("external_evidence request=docs resolved=True")
        evidence = [{
            "request_id": "docs",
            "task": requests[0]["task"],
            "resolved": True,
            "result": "official reference summary",
        }]
        return evidence

    monkeypatch.setattr(code_writing, "run", fake_writing_run)
    monkeypatch.setattr(
        supervisor,
        "collect_external_evidence",
        fake_collect_external_evidence,
    )

    response = await propose_code_change(
        {
            "question": "Create a small source package from scratch.",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert response["status"] == "succeeded"
    assert response["mode"] == "create_new_project"
    assert len(calls["writing"]) == 2
    assert calls["external"]
    assert calls["writing"][1]["external_evidence"] == [{
        "request_id": "docs",
        "task": "Find public documentation for the requested API.",
        "resolved": True,
        "result": "official reference summary",
    }]
    assert response["external_evidence"] == calls["writing"][1]["external_evidence"]


async def test_propose_code_change_redacts_public_write_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    expected_fetching_result = _fetching_result(tmp_path)
    reading_result = _reading_result()
    reading_result["evidence"][0]["excerpt"] = (
        "Safety text mentions .git internals and .env files; "
        "os.environ remains normal code."
    )

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        return expected_fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        return reading_result

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        result = _writing_result()
        result["answer_text"] = (
            "Patch notes mention .git internals and .env files; "
            "os.environ remains normal code."
        )
        return result

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small behavior update.",
            "repo_hint": "fixture/editor",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    public_text = repr(response)
    assert re.search(r"(?<![A-Za-z0-9_])\.git(?![A-Za-z0-9_])", public_text) is None
    assert re.search(r"(?<![A-Za-z0-9_])\.env(?![A-Za-z0-9_])", public_text) is None
    assert "os.environ" in public_text
    assert "[git-internal]" in public_text
    assert "[environment-file]" in public_text
    assert reading_result["evidence"][0]["excerpt"] not in public_text


async def test_propose_code_change_fails_closed_without_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    calls: dict[str, list[dict[str, Any]]] = {
        "fetching": [],
        "writing": [],
    }

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["fetching"].append(request)
        raise AssertionError("Fetching should not run without storage.")

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        calls["writing"].append(request)
        raise AssertionError("Writing should not run without storage.")

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small change.",
        }
    )

    assert response["status"] == "failed"
    assert response["patch_artifacts"] == []
    assert response["validation"]["status"] == "failed"
    assert response["limitations"]
    assert calls["fetching"] == []
    assert calls["writing"] == []
    assert "local_root" not in repr(response)
    assert "cache_key" not in repr(response)


def test_write_response_sanitizes_trace_cache_metadata() -> None:
    from kazusa_ai_chatbot.coding_agent.supervisor import _write_response

    response = _write_response(
        status="succeeded",
        mode="edit_existing_repository",
        answer_text="Proposed patch.",
        repository=None,
        source_scope=None,
        evidence=[],
        patch_artifacts=[],
        created_files=[],
        changed_files=[],
        validation={
            "status": "succeeded",
            "parsed": True,
            "sandbox_applied": True,
            "errors": [],
            "warnings": [],
            "files": [],
        },
        external_evidence=[],
        session=None,
        limitations=[],
        trace_summary=["writing_validation:status=succeeded"],
        trace={
            "programmer": {
                "raw_output": "build_initializer_cache_key uses cache_key text",
                "parsed_output": {"cache_key": "internal"},
            },
        },
    )

    assert "cache_key" not in repr(response)
    assert "[cache-key]" in repr(response)


def test_initial_write_reading_prompt_requests_interface_context() -> None:
    from kazusa_ai_chatbot.coding_agent.supervisor import (
        _initial_reading_question_for_write_request,
    )

    prompt = _initial_reading_question_for_write_request({
        "question": "Add runtime reporting with tests.",
    })

    assert "caller/import sites" in prompt
    assert "limited interface update" in prompt
    assert "Do not create or describe implementation changes" in prompt


async def test_propose_code_change_answer_reflects_merged_limitations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import code_writing
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    expected_fetching_result = _fetching_result(tmp_path)

    async def fake_fetching_run(request: dict[str, Any]) -> dict[str, Any]:
        return expected_fetching_result

    def fake_reading_run(request: dict[str, Any]) -> dict[str, Any]:
        result = _reading_result()
        result["limitations"] = ["Reading found missing proof."]
        return result

    def fake_writing_run(request: dict[str, Any]) -> dict[str, Any]:
        assert request["reading_result"] is not None
        result = _writing_result()
        result["answer_text"] = (
            "Patch proposal validated. "
            "There are no reported limitations or missing information."
        )
        result["limitations"] = []
        return result

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_writing, "run", fake_writing_run)

    response = await propose_code_change(
        {
            "question": "Propose a small behavior update.",
            "repo_hint": "fixture/editor",
            "workspace_root": str(tmp_path / "workspace"),
        }
    )

    assert "no reported limitations" not in response["answer_text"].lower()
    assert "Reading found missing proof." in response["answer_text"]
