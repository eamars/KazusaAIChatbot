"""Deterministic bounded rendering for controller context."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping


MAX_CONTEXT_CHARS = 50_000
MAX_GOAL_CHARS = 12_000
MAX_WORKING_NOTES_CHARS = 8_000
MAX_OBSERVATION_CHARS = 8_000
MAX_EVIDENCE_ROWS = 20
MAX_EVIDENCE_TEXT_CHARS = 4_000
MAX_LIST_ITEMS = 20
MAX_LIST_ITEM_CHARS = 2_000

_COMMON_EVIDENCE_KEYS = {
    "repo_path",
    "path",
    "target_path",
    "start_line",
    "end_line",
    "symbol",
    "kind",
    "status",
    "tool",
    "content_sha256",
    "excerpt",
    "candidate_revision",
    "stdout_excerpt",
    "stderr_excerpt",
    "limitations",
    "user_answer",
}
_WINDOWS_ABSOLUTE_PATH = re.compile(r"(?i)(?:[a-z]:[\\/][^\s\"']+)")
_POSIX_ABSOLUTE_PATH = re.compile(r"(?<![\w.])/(?:[^\s\"']+/)+[^\s\"']+")


def render_controller_context(
    *,
    goal: str,
    acceptance_criteria: list[str] | None = None,
    capabilities: list[str],
    source_identity_digest: str = "",
    candidate_revision: int = 0,
    changed_paths: list[str] | None = None,
    current_failure: Mapping[str, object] | None = None,
    working_notes: str,
    observations: list[Mapping[str, object]],
) -> str:
    """Render bounded semantic context without host or execution internals."""

    bounded_paths = _bounded_strings(
        (changed_paths or [])[-MAX_LIST_ITEMS:],
        MAX_LIST_ITEMS,
        256,
    )
    payload: dict[str, object] = {
        "goal": goal[:MAX_GOAL_CHARS],
        "acceptance_criteria": _bounded_strings(
            acceptance_criteria or [],
            16,
            1000,
        ),
        "capabilities": _bounded_strings(capabilities, 16, 64),
        "source_available": bool(source_identity_digest),
        "candidate_revision": candidate_revision,
        "changed_paths": bounded_paths,
        "changed_path_count": len(changed_paths or []),
        "current_failure": _project_current_failure(current_failure),
        "working_notes": _safe_model_text(
            working_notes,
        )[:MAX_WORKING_NOTES_CHARS],
        "observations": [],
    }
    retained_observations: list[dict[str, object]] = []
    for observation in reversed(observations):
        projected = _project_observation(observation)
        candidate_observations = [projected, *retained_observations]
        payload["observations"] = candidate_observations
        rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(rendered) > MAX_CONTEXT_CHARS:
            payload["observations"] = retained_observations
            break
        retained_observations = candidate_observations
    rendered_context = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(rendered_context) > MAX_CONTEXT_CHARS:
        raise ValueError("required controller context exceeds the context budget")
    return rendered_context


def _project_observation(
    observation: Mapping[str, object],
) -> dict[str, object]:
    """Project one durable observation into the model-facing semantic schema."""

    projected: dict[str, object] = {}
    for key in (
        "sequence",
        "action_sequence",
        "outcome",
        "kind",
        "candidate_revision",
        "cursor",
    ):
        value = observation.get(key)
        if isinstance(value, str):
            projected[key] = value[:MAX_EVIDENCE_TEXT_CHARS]
        elif isinstance(value, int) and not isinstance(value, bool):
            projected[key] = value
    summary = observation.get("summary")
    if isinstance(summary, str):
        projected["summary"] = _safe_model_text(summary)[:MAX_EVIDENCE_TEXT_CHARS]
    evidence = observation.get("evidence")
    if isinstance(evidence, list):
        projected["evidence"] = [
            _project_evidence(row)
            for row in evidence[:MAX_EVIDENCE_ROWS]
            if isinstance(row, Mapping)
        ]
    rendered = json.dumps(projected, ensure_ascii=False, sort_keys=True)
    if len(rendered) <= MAX_OBSERVATION_CHARS:
        return projected
    projected["evidence"] = _trim_projected_evidence(projected.get("evidence"))
    rendered = json.dumps(projected, ensure_ascii=False, sort_keys=True)
    if len(rendered) > MAX_OBSERVATION_CHARS:
        projected.pop("evidence", None)
        projected["summary"] = str(projected.get("summary", ""))[:1000]
    return projected


def _project_evidence(row: Mapping[str, object]) -> dict[str, object]:
    """Keep semantic evidence while excluding workspace and identity internals."""

    projected: dict[str, object] = {}
    for key in _COMMON_EVIDENCE_KEYS:
        value = row.get(key)
        if isinstance(value, str):
            projected[key] = _safe_model_text(value)[:MAX_EVIDENCE_TEXT_CHARS]
        elif isinstance(value, int) and not isinstance(value, bool):
            projected[key] = value
        elif key == "limitations" and isinstance(value, list):
            projected[key] = _bounded_strings(
                value,
                16,
                1000,
                sanitize=True,
            )
    content = row.get("content")
    if isinstance(content, str):
        projected["content"] = _safe_model_text(
            content,
        )[:MAX_EVIDENCE_TEXT_CHARS]
    patch_operation = row.get("patch_operation")
    if isinstance(patch_operation, Mapping):
        projected["patch_operation"] = {
            key: _safe_model_text(str(patch_operation[key]))
            for key in ("kind", "path", "target_path", "content_sha256")
            if isinstance(patch_operation.get(key), str)
        }
    return projected


def _project_current_failure(
    failure: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Keep the one current failure semantic and hide effect identities."""

    if failure is None:
        return None
    projected: dict[str, object] = {}
    kind = failure.get("kind")
    summary = failure.get("summary")
    candidate_revision = failure.get("candidate_revision")
    if isinstance(kind, str):
        projected["kind"] = kind[:128]
    if isinstance(summary, str):
        projected["summary"] = _safe_model_text(summary)[:MAX_EVIDENCE_TEXT_CHARS]
    if isinstance(candidate_revision, int) and not isinstance(
        candidate_revision,
        bool,
    ):
        projected["candidate_revision"] = candidate_revision
    return projected or None


def _safe_model_text(value: str) -> str:
    """Redact absolute host paths from otherwise prompt-safe evidence."""

    without_windows_paths = _WINDOWS_ABSOLUTE_PATH.sub(
        "<managed_path>",
        value,
    )
    safe_text = _POSIX_ABSOLUTE_PATH.sub("<managed_path>", without_windows_paths)
    return safe_text


def _trim_projected_evidence(value: object) -> list[dict[str, object]]:
    """Shrink large evidence while preserving the newest semantic rows."""

    if not isinstance(value, list):
        return []
    rows = [row for row in value if isinstance(row, dict)]
    return rows[:4]


def _bounded_strings(
    values: list[object],
    max_items: int,
    max_chars: int,
    *,
    sanitize: bool = False,
) -> list[str]:
    """Return a bounded string-only list for prompt rendering."""

    bounded: list[str] = []
    for value in values[:max_items]:
        if not isinstance(value, str):
            continue
        bounded_value = _safe_model_text(value) if sanitize else value
        bounded.append(bounded_value[:max_chars])
    return bounded
