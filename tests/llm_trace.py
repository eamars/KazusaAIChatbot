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
    path = _TRACE_DIR / f"{safe_name}.json"
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


def _safe_filename(value: str) -> str:
    """Return a compact filesystem-safe trace filename stem."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._")[:160] or "llm_trace"
