"""Shared contracts and normalization helpers for the goal resolver POC."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "goal_resolver_poc.v1"
CASEBOOK_ARTIFACT = "goal_resolver_casebook.json"
SUMMARY_ARTIFACT = "goal_resolver_evaluation_summary.json"
REPORT_ARTIFACT = "goal_resolver_report.md"

PLANNER_STAGE = "planner"
TOOL_STAGE = "tool"
VERIFIER_STAGE = "verifier"
FINALIZER_STAGE = "finalizer"
CASE_EVALUATOR_STAGE = "case_evaluator"
CONTRACT_VALIDATION_STAGE = "contract_validation"
REPAIR_FEEDBACK_STAGE = "repair_feedback"

TERMINAL_FINAL = "final"
TERMINAL_NEEDS_HUMAN = "needs_human"
TERMINAL_PENDING_APPROVAL = "pending_approval"
TERMINAL_MAX_ITERATIONS = "max_iterations"

ALLOWED_TOOLS = {
    "rag_research",
    "web_research",
    "workspace_inspect",
    "workspace_command",
    "workspace_patch",
    "local_artifact_inspect",
    "self_goal_generate",
    "prepare_action",
    "ask_human",
    "final_answer",
}

DEFAULT_MAX_ITERATIONS = 8
DEFAULT_REPAIR_PASSES = 1
DEFAULT_REQUIREMENT_ID = "req-001"
REQUIREMENT_STATUS_OPEN = "open"
REQUIREMENT_STATUS_SATISFIED = "satisfied"
REQUIREMENT_STATUS_BLOCKED_HUMAN = "blocked_human"
REQUIREMENT_STATUS_BLOCKED_APPROVAL = "blocked_approval"
REQUIREMENT_STATUS_UNRESOLVED = "unresolved"
REQUIREMENT_STATUSES = {
    REQUIREMENT_STATUS_OPEN,
    REQUIREMENT_STATUS_SATISFIED,
    REQUIREMENT_STATUS_BLOCKED_HUMAN,
    REQUIREMENT_STATUS_BLOCKED_APPROVAL,
    REQUIREMENT_STATUS_UNRESOLVED,
}
REQUIREMENT_TERMINAL_STATUSES = {
    REQUIREMENT_STATUS_SATISFIED,
    REQUIREMENT_STATUS_BLOCKED_HUMAN,
    REQUIREMENT_STATUS_BLOCKED_APPROVAL,
    REQUIREMENT_STATUS_UNRESOLVED,
}
CASE_EVALUATION_ACCEPTED_STATUSES = {"pass", "needs_human_valid"}


def text_field(data: dict[str, Any], field_name: str) -> str:
    """Read a string field as stripped text."""

    value = data.get(field_name)
    if not isinstance(value, str):
        return ""
    return value.strip()


def list_field(data: dict[str, Any], field_name: str) -> list[Any]:
    """Read a list field or return an empty list."""

    value = data.get(field_name)
    if not isinstance(value, list):
        return []
    return value


def dict_field(data: dict[str, Any], field_name: str) -> dict[str, Any]:
    """Read a dict field or return an empty dict."""

    value = data.get(field_name)
    if not isinstance(value, dict):
        return {}
    return value


def bounded_text(value: object, *, limit: int = 4000) -> str:
    """Render an object as bounded text for artifacts and LLM context."""

    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return f"{text[: limit - 13].rstrip()}\n[truncated]"


def normalized_requirement_id(value: object, index: int) -> str:
    """Convert a model-provided id into a stable artifact-safe id.

    Args:
        value: Model-provided requirement id or another external value.
        index: One-based fallback index used when the model omitted an id.

    Returns:
        A short requirement id using only simple artifact-safe characters.
    """

    if isinstance(value, str):
        candidate = value.strip().lower()
    else:
        candidate = ""
    cleaned = "".join(
        char for char in candidate if char.isalnum() or char in {"-", "_"}
    )
    if not cleaned:
        cleaned = f"req-{index:03d}"
    return_value = cleaned[:40]
    return return_value


def new_requirement(
    requirement_id: str,
    description: str,
    required_evidence_type: str,
) -> dict[str, Any]:
    """Create one requirement row in the resolver state.

    Args:
        requirement_id: Stable id referenced by tool actions and verifier rows.
        description: Human-readable requirement the resolver must satisfy.
        required_evidence_type: Short semantic label for the evidence needed.

    Returns:
        A serializable requirement row with open status.
    """

    row = {
        "requirement_id": requirement_id,
        "description": description,
        "required_evidence_type": required_evidence_type,
        "status": REQUIREMENT_STATUS_OPEN,
        "blocking_reason": "",
        "satisfied_by_observation_ids": [],
        "last_verifier_note": "",
    }
    return row


def initial_requirements(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the initial broad requirement before the planner decomposes it."""

    contract = text_field(case, "user_input")
    requirement = new_requirement(
        DEFAULT_REQUIREMENT_ID,
        bounded_text(contract, limit=360),
        "case_resolution",
    )
    requirements = [requirement]
    return requirements


def normalize_requirement_rows(
    rows: list[Any],
    fallback_items: list[str],
) -> list[dict[str, Any]]:
    """Normalize model-proposed requirements without judging semantics.

    Args:
        rows: External LLM rows from `requirements`.
        fallback_items: Legacy list of requirement descriptions when the model
            returns only `open_requirements`.

    Returns:
        Requirement rows with open status and stable ids.
    """

    normalized_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        requirement_id = normalized_requirement_id(
            item.get("requirement_id"),
            index,
        )
        if requirement_id in seen_ids:
            requirement_id = f"req-{index:03d}"
        description = text_field(item, "description")
        if not description:
            continue
        evidence_type = text_field(item, "required_evidence_type")
        if not evidence_type:
            evidence_type = "evidence"
        row = new_requirement(requirement_id, description, evidence_type)
        normalized_rows.append(row)
        seen_ids.add(requirement_id)

    if normalized_rows:
        return normalized_rows

    for index, item in enumerate(fallback_items, start=1):
        row = new_requirement(f"req-{index:03d}", item, "evidence")
        normalized_rows.append(row)

    return normalized_rows


def merge_planner_requirements(
    current_rows: list[dict[str, Any]],
    proposed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge planner requirement decomposition into current resolver state."""

    if not proposed_rows:
        return [dict(row) for row in current_rows]
    if (
        len(current_rows) == 1
        and current_rows[0]["requirement_id"] == DEFAULT_REQUIREMENT_ID
        and current_rows[0]["status"] == REQUIREMENT_STATUS_OPEN
        and not current_rows[0]["satisfied_by_observation_ids"]
    ):
        return [dict(row) for row in proposed_rows]

    current_by_id = {
        row["requirement_id"]: dict(row)
        for row in current_rows
        if isinstance(row.get("requirement_id"), str)
    }
    merged_rows: list[dict[str, Any]] = []
    consumed_ids: set[str] = set()

    for proposed in proposed_rows:
        requirement_id = proposed["requirement_id"]
        current = current_by_id.get(requirement_id)
        if current:
            row = dict(current)
            row["description"] = proposed["description"]
            row["required_evidence_type"] = proposed["required_evidence_type"]
        else:
            row = dict(proposed)
        merged_rows.append(row)
        consumed_ids.add(requirement_id)

    for current in current_rows:
        requirement_id = current["requirement_id"]
        if requirement_id not in consumed_ids:
            merged_rows.append(dict(current))

    return merged_rows


def normalize_requirement_updates(rows: list[Any]) -> list[dict[str, Any]]:
    """Normalize verifier requirement status updates."""

    updates: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        requirement_id = normalized_requirement_id(
            item.get("requirement_id"),
            index,
        )
        status = text_field(item, "status")
        if status not in REQUIREMENT_STATUSES:
            status = REQUIREMENT_STATUS_OPEN
        observation_ids = [
            text.strip()
            for text in list_field(item, "satisfied_by_observation_ids")
            if isinstance(text, str) and text.strip()
        ]
        update = {
            "requirement_id": requirement_id,
            "status": status,
            "blocking_reason": text_field(item, "blocking_reason"),
            "satisfied_by_observation_ids": observation_ids,
            "last_verifier_note": text_field(item, "last_verifier_note"),
        }
        updates.append(update)
    return updates


def apply_requirement_updates(
    current_rows: list[dict[str, Any]],
    updates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply verifier-produced requirement status updates."""

    rows = [dict(row) for row in current_rows]
    row_indexes = {
        row["requirement_id"]: index
        for index, row in enumerate(rows)
        if isinstance(row.get("requirement_id"), str)
    }
    for update in updates:
        requirement_id = update["requirement_id"]
        if requirement_id not in row_indexes:
            row = new_requirement(requirement_id, "", "evidence")
            rows.append(row)
            row_indexes[requirement_id] = len(rows) - 1
        row = rows[row_indexes[requirement_id]]
        row["status"] = update["status"]
        row["blocking_reason"] = update["blocking_reason"]
        row["satisfied_by_observation_ids"] = update[
            "satisfied_by_observation_ids"
        ]
        row["last_verifier_note"] = update["last_verifier_note"]
    return rows


def first_open_requirement_id(requirements: list[dict[str, Any]]) -> str:
    """Return the first open requirement id for targeting a tool action."""

    for requirement in requirements:
        if requirement["status"] == REQUIREMENT_STATUS_OPEN:
            return requirement["requirement_id"]
    return DEFAULT_REQUIREMENT_ID


def all_requirements_terminal(requirements: list[dict[str, Any]]) -> bool:
    """Check that requirement rows are in terminal statuses."""

    for requirement in requirements:
        if requirement["status"] not in REQUIREMENT_TERMINAL_STATUSES:
            return False
    return True


def normalize_planner_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize planner JSON into the POC action contract."""

    goal_frame = text_field(raw, "goal_frame")
    open_requirements = [
        item.strip()
        for item in list_field(raw, "open_requirements")
        if isinstance(item, str) and item.strip()
    ]
    requirements = normalize_requirement_rows(
        list_field(raw, "requirements"),
        open_requirements,
    )
    raw_action = dict_field(raw, "next_action")
    tool = text_field(raw_action, "tool")
    if tool not in ALLOWED_TOOLS:
        raise ValueError(f"planner returned unknown tool: {tool!r}")
    normalized = {
        "goal_frame": goal_frame,
        "open_requirements": open_requirements,
        "requirements": requirements,
        "next_action": {
            "tool": tool,
            "query": text_field(raw_action, "query"),
            "reason": text_field(raw_action, "reason"),
            "target_requirement_id": normalized_requirement_id(
                raw_action.get("target_requirement_id"),
                1,
            ),
        },
    }
    return normalized


def normalize_verifier_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize verifier JSON into stable loop control fields."""

    raw_decision = text_field(raw, "decision")
    if raw_decision not in {
        "continue",
        "final_answer",
        "ask_human",
        "prepare_action",
    }:
        raw_decision = "continue"
    raw_confidence = raw.get("confidence", 0.0)
    if isinstance(raw_confidence, bool) or not isinstance(
        raw_confidence,
        (int, float),
    ):
        confidence = 0.0
    else:
        confidence = max(0.0, min(float(raw_confidence), 1.0))
    raw_resolved = raw.get("resolved", False)
    if isinstance(raw_resolved, bool):
        resolved = raw_resolved
    else:
        resolved = False
    normalized = {
        "resolved": resolved,
        "decision": raw_decision,
        "confidence": confidence,
        "requirement_updates": normalize_requirement_updates(
            list_field(raw, "requirement_updates")
        ),
        "remaining_requirements": [
            item.strip()
            for item in list_field(raw, "remaining_requirements")
            if isinstance(item, str) and item.strip()
        ],
        "feedback": text_field(raw, "feedback"),
        "minimal_human_question": text_field(raw, "minimal_human_question"),
    }
    return normalized


def normalize_patch_output(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a sandbox patch proposal from the LLM."""

    normalized = {
        "file_path": text_field(raw, "file_path"),
        "new_content": text_field(raw, "new_content"),
        "reason": text_field(raw, "reason"),
        "raw_output": text_field(raw, "raw_output"),
    }
    return normalized


def normalize_case_evaluation(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize final LLM case evaluation."""

    status = text_field(raw, "status")
    if status not in {"pass", "fail", "needs_human_valid"}:
        status = "fail"
    raw_score = raw.get("score", 0)
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        score = 0
    else:
        numeric_score = float(raw_score)
        if 0.0 <= numeric_score <= 1.0:
            numeric_score = numeric_score * 100.0
        elif (
            status in CASE_EVALUATION_ACCEPTED_STATUSES
            and numeric_score <= 10.0
        ):
            numeric_score = numeric_score * 10.0
        score = int(max(0.0, min(numeric_score, 100.0)))
    return {
        "status": status,
        "score": score,
        "reason": text_field(raw, "reason"),
        "missing": [
            item.strip()
            for item in list_field(raw, "missing")
            if isinstance(item, str) and item.strip()
        ],
        "loop_quality": text_field(raw, "loop_quality"),
        "tool_use_quality": text_field(raw, "tool_use_quality"),
    }
