"""Dry-run CLI contract tests for self-cognition case files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.self_cognition import models
from scripts import run_self_cognition_dry_run


def _private_case() -> dict[str, Any]:
    case = {
        "case_name": models.CASE_PRIVATE_NO_ACTION,
        "case_id": "private-no-action-001",
        "idle_timestamp": "2026-05-10T00:30:00+00:00",
        "last_evidence_timestamp": "2026-05-10T00:20:00+00:00",
        "trigger_kind": models.TRIGGER_RECENT_DIRECT_DIALOG_REVIEW,
        "semantic_due_state": None,
        "actionability": "no_action_needed",
        "target_scope": {
            "platform": "qq",
            "platform_channel_id": "673225019",
            "channel_type": "private",
            "user_id": "673225019",
        },
        "source_refs": [
            {
                "source_kind": "conversation_window",
                "source_id": "private-window-001",
                "summary": "Recent private chat is settled.",
            }
        ],
        "visible_context": [
            {
                "role": "user",
                "text": "The recent exchange already reached closure.",
                "timestamp": "2026-05-10T00:20:00+00:00",
            }
        ],
    }
    return case


def test_dry_run_cli_loads_case_file_and_uses_output_dir(
    monkeypatch,
    tmp_path,
) -> None:
    case_file = tmp_path / "case.json"
    output_dir = tmp_path / "output"
    case_file.write_text(json.dumps(_private_case()), encoding="utf-8")
    captured: dict[str, Any] = {}

    def fake_run_self_cognition_case(
        case: dict[str, Any],
        output_path: Path,
    ) -> dict[str, str]:
        captured["case"] = case
        captured["output_path"] = output_path
        output_path.mkdir(parents=True, exist_ok=True)
        trace_path = output_path / models.ARTIFACT_LOOP_TRACE
        trace_path.write_text("trace body", encoding="utf-8")
        return {models.ARTIFACT_LOOP_TRACE: str(trace_path)}

    monkeypatch.setattr(
        run_self_cognition_dry_run,
        "run_self_cognition_case",
        fake_run_self_cognition_case,
    )

    exit_code = run_self_cognition_dry_run.main(
        [
            "--case-file",
            str(case_file),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code == 0
    assert captured["case"]["case_name"] == models.CASE_PRIVATE_NO_ACTION
    assert captured["output_path"] == output_dir
    assert (output_dir / models.ARTIFACT_LOOP_TRACE).exists()


def test_dry_run_cli_rejects_missing_case_file_before_output_dir(
    tmp_path,
) -> None:
    output_dir = tmp_path / "output"

    exit_code = run_self_cognition_dry_run.main(
        [
            "--case-file",
            str(tmp_path / "missing.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code != 0
    assert not output_dir.exists()


def test_dry_run_cli_rejects_malformed_case_file_before_output_dir(
    tmp_path,
) -> None:
    case_file = tmp_path / "case.json"
    output_dir = tmp_path / "output"
    case_file.write_text("{not json", encoding="utf-8")

    exit_code = run_self_cognition_dry_run.main(
        [
            "--case-file",
            str(case_file),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code != 0
    assert not output_dir.exists()


def test_dry_run_cli_rejects_unknown_case_before_output_dir(
    tmp_path,
) -> None:
    case_file = tmp_path / "case.json"
    output_dir = tmp_path / "output"
    case_file.write_text(
        json.dumps({"case_name": "unsupported_case"}),
        encoding="utf-8",
    )

    exit_code = run_self_cognition_dry_run.main(
        [
            "--case-file",
            str(case_file),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert exit_code != 0
    assert not output_dir.exists()
