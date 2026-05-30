"""Artifact writers for the goal resolver POC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.goal_resolver_poc.models import (
    CASEBOOK_ARTIFACT,
    SUMMARY_ARTIFACT,
)


def write_json(path: Path, payload: object) -> None:
    """Write stable UTF-8 JSON."""

    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        default=str,
    )
    path.write_text(f"{rendered}\n", encoding="utf-8")


def write_casebook(output_dir: Path, cases: list[dict[str, Any]]) -> Path:
    """Write the POC casebook artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / CASEBOOK_ARTIFACT
    write_json(path, cases)
    return path


def write_run_artifact(output_dir: Path, run: dict[str, Any]) -> Path:
    """Write one run artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(run.get("case_id") or "case")
    path = output_dir / f"goal_resolver_run_{case_id}.json"
    write_json(path, run)
    return path


def write_summary(output_dir: Path, summary: dict[str, Any]) -> Path:
    """Write the evaluation summary artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / SUMMARY_ARTIFACT
    write_json(path, summary)
    return path
