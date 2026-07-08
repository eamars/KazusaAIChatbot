"""Workspace-local ledger persistence for durable coding-agent runs."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kazusa_ai_chatbot.coding_agent.coding_run.models import (
    CodingRunEvent,
    CodingRunLedger,
    CodingRunResponse,
)
from kazusa_ai_chatbot.coding_agent.tools.paths import (
    PathSafetyError,
    ensure_path_inside,
)

RUN_SCHEMA_VERSION = 1
RUN_ROOT_DIR_NAME = "coding_runs"
RUN_FILE_NAME = "run.json"
EVENT_FILE_NAME = "events.jsonl"
RUN_ID_RE = re.compile(r"^[a-f0-9]{32}$")
GIT_INTERNAL_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.git(?![A-Za-z0-9_])")
ENV_FILE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])\.env(?![A-Za-z0-9_])")
CACHE_KEY_TOKEN_RE = re.compile(r"cache_key", re.IGNORECASE)
SECRET_LIKE_RE = re.compile(
    r"(?i)(token|password|credential|secret)[A-Za-z0-9_]*=\S+",
)


@dataclass(frozen=True)
class CodingRunPaths:
    """Resolved workspace paths for one durable coding-agent run."""

    workspace_root: Path
    run_root: Path
    run_dir: Path
    run_file: Path
    event_file: Path


def new_run_id() -> str:
    """Create an opaque id suitable for a workspace run directory."""

    run_id = uuid.uuid4().hex
    return run_id


def utc_timestamp() -> str:
    """Return a stable UTC timestamp for ledger updates and events."""

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return timestamp


def build_run_paths(
    *,
    workspace_root_text: str,
    run_id: str,
    create: bool,
) -> CodingRunPaths | str:
    """Resolve all ledger paths and enforce workspace containment.

    Args:
        workspace_root_text: Caller-owned coding workspace root.
        run_id: Durable run identifier.
        create: Whether missing directories should be created.

    Returns:
        A path bundle or a public error string.
    """

    if not workspace_root_text.strip():
        return "Coding run requires a workspace root."
    if not RUN_ID_RE.fullmatch(run_id):
        return "Coding run id is invalid or missing."

    workspace_root = Path(workspace_root_text).expanduser().resolve(strict=False)
    try:
        if create:
            workspace_root.mkdir(parents=True, exist_ok=True)
        if not workspace_root.exists() or not workspace_root.is_dir():
            return "Coding run workspace root is missing."
        run_root = ensure_path_inside(
            workspace_root / RUN_ROOT_DIR_NAME,
            workspace_root,
        )
        run_dir = ensure_path_inside(run_root / run_id, workspace_root)
        run_file = ensure_path_inside(run_dir / RUN_FILE_NAME, workspace_root)
        event_file = ensure_path_inside(run_dir / EVENT_FILE_NAME, workspace_root)
        if create:
            run_root.mkdir(parents=True, exist_ok=True)
            run_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, PathSafetyError) as exc:
        error = f"Coding run workspace cannot be prepared: {exc}"
        return error

    paths = CodingRunPaths(
        workspace_root=workspace_root,
        run_root=run_root,
        run_dir=run_dir,
        run_file=run_file,
        event_file=event_file,
    )
    return paths


def write_ledger(paths: CodingRunPaths, ledger: CodingRunLedger) -> None:
    """Persist the canonical JSON ledger with an atomic local replacement."""

    ledger["updated_at"] = utc_timestamp()
    content = json.dumps(
        ledger,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    temp_path = ensure_path_inside(
        paths.run_dir / f"{RUN_FILE_NAME}.{uuid.uuid4().hex}.tmp",
        paths.workspace_root,
    )
    temp_path.write_text(f"{content}\n", encoding="utf-8")
    temp_path.replace(paths.run_file)


def load_ledger(paths: CodingRunPaths) -> CodingRunLedger | str:
    """Load a persisted JSON ledger from the run workspace."""

    if not paths.run_file.exists():
        return "Coding run is missing."
    raw_text = paths.run_file.read_text(encoding="utf-8")
    try:
        loaded = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        error = f"Coding run ledger is not valid JSON: {exc}"
        return error
    if not isinstance(loaded, dict):
        return "Coding run ledger is not an object."
    ledger = loaded
    return ledger  # type: ignore[return-value]


def append_event(
    *,
    paths: CodingRunPaths,
    run_id: str,
    event_type: str,
    status: str,
    summary: str,
    public_payload: dict[str, object],
    redaction_roots: list[str],
) -> CodingRunEvent:
    """Append one public-safe lifecycle event for a run."""

    existing_events = load_events(paths)
    sequence = len(existing_events) + 1
    event: CodingRunEvent = {
        "event_id": uuid.uuid4().hex,
        "run_id": run_id,
        "sequence": sequence,
        "event_type": event_type,
        "status": status,
        "summary": str(sanitize_public_value(
            summary,
            redaction_roots=redaction_roots,
        )),
        "public_payload": sanitize_public_value(
            public_payload,
            redaction_roots=redaction_roots,
        ),
    }
    serialized = json.dumps(event, ensure_ascii=False, sort_keys=True)
    with paths.event_file.open("a", encoding="utf-8") as file_handle:
        file_handle.write(f"{serialized}\n")
    return event


def load_events(paths: CodingRunPaths) -> list[CodingRunEvent]:
    """Load append-only events from a run workspace."""

    if not paths.event_file.exists():
        return []

    events: list[CodingRunEvent] = []
    for line in paths.event_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)  # type: ignore[arg-type]
    return events


def public_response(
    *,
    ledger: CodingRunLedger,
    events: list[CodingRunEvent],
) -> CodingRunResponse:
    """Project one private ledger into the public run response."""

    redaction_roots = redaction_roots_from_ledger(ledger)
    response: CodingRunResponse = {
        "status": _ledger_text(ledger, "status"),
        "run_id": _ledger_text(ledger, "run_id"),
        "goal": _ledger_text(ledger, "goal"),
        "objective_type": _ledger_text(ledger, "objective_type"),
        "answer_text": _ledger_text(ledger, "answer_text"),
        "repository": ledger.get("repository"),
        "source_scope": ledger.get("source_scope"),
        "evidence": _ledger_list(ledger, "evidence"),
        "patch_artifacts": _ledger_list(ledger, "patch_artifacts"),
        "changed_files": _ledger_list(ledger, "changed_files"),
        "apply_attempts": _ledger_list(ledger, "apply_attempts"),
        "execution_attempts": _ledger_list(ledger, "execution_attempts"),
        "repair_attempts": _ledger_list(ledger, "repair_attempts"),
        "attempts": _ledger_list(ledger, "attempts"),
        "blockers": _ledger_list(ledger, "blockers"),
        "events": events,
        "limitations": _ledger_list(ledger, "limitations"),
        "trace_summary": _ledger_list(ledger, "trace_summary"),
    }
    sanitized = sanitize_public_value(
        response,
        redaction_roots=redaction_roots,
    )
    return sanitized  # type: ignore[return-value]


def empty_response(
    *,
    status: str,
    run_id: str = "",
    objective_type: str = "",
    goal: str = "",
    limitation: str,
    trace_summary: list[str],
) -> CodingRunResponse:
    """Build a public response for failures that do not have a ledger."""

    response: CodingRunResponse = {
        "status": status,
        "run_id": run_id,
        "goal": goal,
        "objective_type": objective_type,
        "answer_text": "",
        "repository": None,
        "source_scope": None,
        "evidence": [],
        "patch_artifacts": [],
        "changed_files": [],
        "apply_attempts": [],
        "execution_attempts": [],
        "repair_attempts": [],
        "attempts": [],
        "blockers": [{
            "code": "request_rejected",
            "message": limitation,
            "details": {},
        }],
        "events": [],
        "limitations": [limitation],
        "trace_summary": trace_summary,
    }
    return response


def redaction_roots_from_ledger(ledger: dict[str, object]) -> list[str]:
    """Collect private local roots that must not enter public projections."""

    roots: list[str] = []
    source_request = ledger.get("source_request")
    if isinstance(source_request, dict):
        for field_name in ("workspace_root", "local_root_hint", "local_path_hint"):
            value = source_request.get(field_name)
            if isinstance(value, str) and value:
                roots.append(value)
    repository = ledger.get("repository")
    if isinstance(repository, dict):
        for field_name in ("local_root", "workspace_root"):
            value = repository.get(field_name)
            if isinstance(value, str) and value:
                roots.append(value)
    return roots


def sanitize_public_value(
    value: object,
    *,
    redaction_roots: list[str],
) -> object:
    """Recursively remove private paths and sensitive tokens from public data."""

    if isinstance(value, str):
        sanitized_text = _sanitize_public_text(
            value,
            redaction_roots=redaction_roots,
        )
        return sanitized_text
    if isinstance(value, list):
        sanitized_list = [
            sanitize_public_value(item, redaction_roots=redaction_roots)
            for item in value
        ]
        return sanitized_list
    if isinstance(value, dict):
        sanitized_dict = {
            _sanitize_public_key(key): sanitize_public_value(
                item,
                redaction_roots=redaction_roots,
            )
            for key, item in value.items()
        }
        return sanitized_dict
    return value


def _sanitize_public_text(
    value: str,
    *,
    redaction_roots: list[str],
) -> str:
    text = value
    for root in redaction_roots:
        if not root:
            continue
        root_path = Path(root).expanduser().resolve(strict=False)
        root_text = str(root_path)
        text = text.replace(root_text, "[local-root]")
        text = text.replace(root_text.replace("\\", "/"), "[local-root]")
        text = text.replace(str(root), "[local-root]")
        text = text.replace(str(root).replace("\\", "/"), "[local-root]")
    text = GIT_INTERNAL_TOKEN_RE.sub("[git-internal]", text)
    text = ENV_FILE_TOKEN_RE.sub("[environment-file]", text)
    text = CACHE_KEY_TOKEN_RE.sub("[cache-key]", text)
    text = SECRET_LIKE_RE.sub("[secret-like-value]", text)
    return text


def _sanitize_public_key(key: object) -> object:
    if not isinstance(key, str):
        return key
    sanitized_key = CACHE_KEY_TOKEN_RE.sub("[cache-key]", key)
    return sanitized_key


def _ledger_text(ledger: dict[str, object], key: str) -> str:
    value = ledger.get(key)
    if isinstance(value, str):
        return value
    return ""


def _ledger_list(ledger: dict[str, object], key: str) -> list:
    value = ledger.get(key)
    if isinstance(value, list):
        return value
    return []
