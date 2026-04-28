"""Durable trace helpers for real LLM tests."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRACE_DIR = Path(__file__).resolve().parents[1] / "test_artifacts" / "llm_traces"


def write_llm_trace(test_name: str, case_id: str, payload: dict[str, Any]) -> Path:
    """Write one inspectable real-LLM trace artifact.

    Args:
        test_name: Name of the test or graph/node contract under inspection.
        case_id: Stable identifier for the scenario.
        payload: JSON-serializable trace data, including input, output, and
            judgment notes when available.

    Returns:
        Path to the trace file written for post-run inspection.
    """
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_filename(f"{test_name}__{case_id}")
    path = _unique_trace_path(safe_name)
    document = {
        "test_name": test_name,
        "case_id": case_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _unique_trace_path(safe_name: str) -> Path:
    """Return a trace path that preserves any existing artifact.

    Args:
        safe_name: Filesystem-safe trace filename stem.

    Returns:
        New trace path, using the stable name when free and a timestamp suffix
        when the stable name already exists.
    """

    path = _TRACE_DIR / f"{safe_name}.json"
    if not path.exists():
        return path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return _TRACE_DIR / f"{safe_name}__{timestamp}.json"


def _safe_filename(value: str) -> str:
    """Return a compact filesystem-safe trace filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._")[:160] or "llm_trace"
