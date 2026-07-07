from pathlib import Path
from typing import Any

import pytest


@pytest.mark.asyncio
async def test_source_backed_proposal_routes_through_modifying(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from kazusa_ai_chatbot.coding_agent import code_fetching
    from kazusa_ai_chatbot.coding_agent import code_modifying
    from kazusa_ai_chatbot.coding_agent import code_reading
    from kazusa_ai_chatbot.coding_agent import propose_code_change

    repo_root = tmp_path / "repo"
    workspace_root = tmp_path / "workspace"
    repo_root.mkdir()
    workspace_root.mkdir()
    target_path = repo_root / "app.py"
    target_path.write_text("VALUE = 1\n", encoding="utf-8")

    calls: list[str] = []

    async def fake_fetching_run(_request: dict[str, Any]) -> dict[str, Any]:
        calls.append("fetching")
        return {
            "status": "succeeded",
            "message": "resolved",
            "repository": {
                "provider": "github",
                "owner": "fixture",
                "repo": "source-backed",
                "source_url": "local://fixture/source-backed",
                "requested_ref": None,
                "resolved_ref": "local",
                "current_commit": "local-sha256:test",
                "default_branch": "main",
                "local_root": str(repo_root),
                "storage_kind": "existing_local_checkout",
                "managed_checkout": False,
                "workspace_root": str(workspace_root),
                "cache_key": None,
                "dirty_state": "clean",
            },
            "source_scope": {
                "kind": "repository",
                "repo_relative_path": None,
                "source_url": "local://fixture/source-backed",
                "requested_ref": None,
                "interpretation": "Local fixture source.",
            },
            "limitations": [],
            "trace_summary": ["fetching:succeeded"],
        }

    def fake_reading_run(_request: dict[str, Any]) -> dict[str, Any]:
        calls.append("reading")
        return {
            "status": "succeeded",
            "answer_text": "VALUE is defined in app.py.",
            "evidence": [
                {
                    "path": "app.py",
                    "line_start": 1,
                    "line_end": 1,
                    "symbol_or_topic": "VALUE",
                    "excerpt": "VALUE = 1",
                    "reason": "Defines the value to modify.",
                }
            ],
            "limitations": [],
            "trace_summary": ["reading:succeeded"],
        }

    async def fake_modifying_run(_request: dict[str, Any]) -> dict[str, Any]:
        calls.append("modifying")
        return {
            "status": "succeeded",
            "mode": "edit_existing_repository",
            "answer_text": "Proposed VALUE update.",
            "modification_artifacts": [
                {
                    "artifact_id": "artifact-1",
                    "status": "succeeded",
                    "task_id": "task-1",
                    "target_path": "app.py",
                    "evidence_ids": ["ev-1"],
                    "operation_kind": "replace",
                    "exact_anchor": "VALUE = 1\n",
                    "replacement_or_insert_content": "VALUE = 2\n",
                    "operation_summary": "Update value.",
                    "risk_notes": [],
                    "tests_or_docs_to_update": [],
                }
            ],
            "created_files": [],
            "changed_files": [
                {
                    "path": "app.py",
                    "change_type": "modify",
                    "summary": "Update value.",
                }
            ],
            "limitations": [],
            "trace_summary": ["modifying:succeeded"],
        }

    monkeypatch.setattr(code_fetching, "run", fake_fetching_run)
    monkeypatch.setattr(code_reading, "run", fake_reading_run)
    monkeypatch.setattr(code_modifying, "run", fake_modifying_run)

    response = await propose_code_change({
        "question": "Update VALUE to 2.",
        "local_root_hint": str(repo_root),
        "workspace_root": str(workspace_root),
        "preferred_language": "English",
        "max_answer_chars": 1200,
        "max_artifact_chars": 4000,
    })

    assert response["status"] == "succeeded"
    assert response["mode"] == "edit_existing_repository"
    assert calls == ["fetching", "reading", "modifying"]
    assert response["validation"]["sandbox_applied"] is True
    assert response["changed_files"] == [
        {
            "path": "app.py",
            "change_type": "modify",
            "summary": "Update value.",
        }
    ]
    assert "writing:existing_source_rejected" not in response["trace_summary"]
    assert str(repo_root.resolve()) not in repr(response)
    assert str(workspace_root.resolve()) not in repr(response)
