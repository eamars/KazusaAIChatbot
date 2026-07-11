"""Strict parser for one controller action per loop turn."""

from __future__ import annotations

from collections.abc import Mapping, Set

from kazusa_ai_chatbot.coding_agent.context_budget import (
    MAX_READ_LINES,
    MAX_REGEX_QUERY_CHARS,
    MAX_SEARCH_EXCERPT_CHARS,
)


_REQUIRED_KEYS = {"schema_version", "action_id", "action", "reason", "args"}
_OPTIONAL_KEYS = {"working_note"}
_MAX_REASON_CHARS = 600
_MAX_WORKING_NOTE_CHARS = 2000
_MAX_ACTION_ID_CHARS = 128
_MAX_PATH_CHARS = 512
_MAX_LIST_ITEMS = 32
_MAX_LIST_ITEM_CHARS = 2000
_MAX_SUMMARY_CHARS = 4000
_MAX_QUESTION_CHARS = 2000
_ACTION_ARGUMENT_KEYS = {
    "read": {"repo_path", "start_line", "symbol", "end_line"},
    "search": {"mode", "query", "path_glob", "cursor"},
    "edit": {
        "operation",
        "repo_path",
        "target_path",
        "expected_sha256",
        "expected_candidate_revision",
        "anchor",
        "replacement",
    },
    "run": {"profile", "targets", "intent"},
    "note": {"completed", "remaining", "assumptions"},
    "finish": {
        "summary",
        "acceptance_criteria",
        "evidence_refs",
        "known_limitations",
    },
    "block": {
        "blocker_type",
        "question",
        "options",
        "blocking_evidence_refs",
    },
}


def parse_action(
    payload: Mapping[str, object],
    *,
    allowed_actions: Set[str],
) -> dict[str, object]:
    """Validate one action before any deterministic executor reads it.

    Args:
        payload: Parsed model JSON object for one requested action.
        allowed_actions: Capability names exposed for this durable loop state.

    Returns:
        Validated action payload or a bounded invalid-action observation.
    """

    payload_keys = set(payload)
    if payload_keys - _REQUIRED_KEYS - _OPTIONAL_KEYS:
        invalid = _invalid_action("unknown action keys")
        return invalid
    if _REQUIRED_KEYS - payload_keys:
        invalid = _invalid_action("required action keys are missing")
        return invalid
    if payload.get("schema_version") != "coding_action.v1":
        invalid = _invalid_action("action schema version is unsupported")
        return invalid
    action = payload.get("action")
    if not isinstance(action, str) or action not in allowed_actions:
        invalid = _invalid_action("action is unavailable")
        return invalid
    action_id = payload.get("action_id")
    if (
        not isinstance(action_id, str)
        or not action_id.strip()
        or len(action_id) > _MAX_ACTION_ID_CHARS
    ):
        invalid = _invalid_action("action id is invalid")
        return invalid
    reason = payload.get("reason")
    if (
        not isinstance(reason, str)
        or not reason.strip()
        or len(reason) > _MAX_REASON_CHARS
    ):
        invalid = _invalid_action("action reason is invalid")
        return invalid
    working_note = payload.get("working_note")
    if working_note is not None and (
        not isinstance(working_note, str)
        or len(working_note) > _MAX_WORKING_NOTE_CHARS
    ):
        invalid = _invalid_action("action working note is invalid")
        return invalid
    arguments = payload.get("args")
    if not isinstance(arguments, Mapping):
        invalid = _invalid_action("action arguments are invalid")
        return invalid
    argument_error = _action_argument_error(action, arguments)
    if argument_error:
        invalid = _invalid_action(argument_error)
        return invalid
    validated_action = dict(payload)
    return {"status": "ok", "action": validated_action}


def invalid_output_blocker(consecutive_invalid_outputs: int) -> dict[str, str] | None:
    """Return the typed blocker after the controller exhausts invalid retries."""

    if consecutive_invalid_outputs < 3:
        return None
    blocker = {
        "blocker_type": "controller_contract_failure",
        "resume_target": "retry_loop",
    }
    return blocker


def _invalid_action(message: str) -> dict[str, object]:
    """Build the bounded observation returned for one invalid parsed action."""

    observation = {"status": "invalid_action", "message": message}
    return observation


def _action_argument_error(action: str, arguments: Mapping[str, object]) -> str:
    """Return the closed-schema error for one semantic action argument map."""

    allowed_keys = _ACTION_ARGUMENT_KEYS[action]
    if set(arguments) - allowed_keys:
        return "action arguments contain unsupported keys"
    if action == "read":
        if not _bounded_text(arguments.get("repo_path"), _MAX_PATH_CHARS):
            return "read action requires a repo path"
        has_start_line = "start_line" in arguments
        has_symbol = "symbol" in arguments
        if has_start_line == has_symbol:
            return "read action requires a line or symbol"
        if has_symbol:
            symbol = arguments.get("symbol")
            if not _bounded_text(symbol, _MAX_PATH_CHARS):
                return "read action symbol is invalid"
            if "end_line" in arguments:
                return "symbol read cannot include an end line"
            return ""
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")
        if not _is_int(start_line) or start_line < 1:
            return "read action start line is invalid"
        if end_line is not None and (
            not _is_int(end_line)
            or end_line < start_line
            or end_line - start_line >= MAX_READ_LINES
        ):
            return "read action end line is invalid"
        return ""
    if action == "search":
        if arguments.get("mode") not in {"literal", "regex", "symbol", "path"}:
            return "search action mode is invalid"
        query = arguments.get("query")
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > MAX_SEARCH_EXCERPT_CHARS
        ):
            return "search action requires a query"
        if arguments.get("mode") == "regex" and len(query) > MAX_REGEX_QUERY_CHARS:
            return "search regex exceeds the configured bound"
        if "path_glob" in arguments and not _bounded_text(
            arguments.get("path_glob"),
            _MAX_PATH_CHARS,
        ):
            return "search path glob is invalid"
        if "cursor" in arguments and not _bounded_text(
            arguments.get("cursor"),
            _MAX_LIST_ITEM_CHARS,
        ):
            return "search cursor is invalid"
        return ""
    if action == "edit":
        operation = arguments.get("operation")
        if operation not in {
            "create_file",
            "replace_anchor",
            "insert_before",
            "insert_after",
            "replace_file_small",
            "delete_file",
            "rename_file",
        }:
            return "edit action requires an operation"
        if not _bounded_text(arguments.get("repo_path"), _MAX_PATH_CHARS):
            return "edit action requires a repo path"
        if not _is_int(arguments.get("expected_candidate_revision")):
            return "edit action requires a candidate revision"
        if int(arguments["expected_candidate_revision"]) < 0:
            return "edit action candidate revision is invalid"
        operation_keys = {
            "create_file": {
                "operation",
                "repo_path",
                "expected_candidate_revision",
                "replacement",
            },
            "replace_anchor": {
                "operation",
                "repo_path",
                "expected_sha256",
                "expected_candidate_revision",
                "anchor",
                "replacement",
            },
            "insert_before": {
                "operation",
                "repo_path",
                "expected_sha256",
                "expected_candidate_revision",
                "anchor",
                "replacement",
            },
            "insert_after": {
                "operation",
                "repo_path",
                "expected_sha256",
                "expected_candidate_revision",
                "anchor",
                "replacement",
            },
            "replace_file_small": {
                "operation",
                "repo_path",
                "expected_sha256",
                "expected_candidate_revision",
                "replacement",
            },
            "delete_file": {
                "operation",
                "repo_path",
                "expected_sha256",
                "expected_candidate_revision",
            },
            "rename_file": {
                "operation",
                "repo_path",
                "target_path",
                "expected_sha256",
                "expected_candidate_revision",
            },
        }
        if set(arguments) != operation_keys[str(operation)]:
            return "edit action arguments do not match the operation"
        expected_hash = arguments.get("expected_sha256")
        if operation == "create_file":
            if expected_hash is not None:
                return "create action cannot include a source hash"
            replacement = arguments.get("replacement")
            if (
                not isinstance(replacement, str)
                or not replacement
                or len(replacement) > MAX_SEARCH_EXCERPT_CHARS
            ):
                return "create action requires replacement content"
            return ""
        if not _is_sha256(expected_hash):
            return "existing-file edit requires a source hash"
        if operation == "rename_file":
            if not _bounded_text(arguments.get("target_path"), _MAX_PATH_CHARS):
                return "rename action requires a target path"
            return ""
        if operation == "delete_file":
            return ""
        if operation in {"replace_anchor", "insert_before", "insert_after"}:
            anchor = arguments.get("anchor")
            if (
                not isinstance(anchor, str)
                or not anchor
                or len(anchor) > MAX_SEARCH_EXCERPT_CHARS
            ):
                return "anchor edit requires an anchor"
        replacement = arguments.get("replacement")
        if (
            not isinstance(replacement, str)
            or len(replacement) > MAX_SEARCH_EXCERPT_CHARS
        ):
            return "edit action requires replacement content"
        return ""
    if action == "run":
        if arguments.get("profile") not in {"derived_base", "focused"}:
            return "run action profile is invalid"
        intent = arguments.get("intent")
        if (
            not isinstance(intent, str)
            or not intent.strip()
            or len(intent) > _MAX_REASON_CHARS
        ):
            return "run action requires an intent"
        targets = arguments.get("targets", [])
        if not _bounded_string_list(
            targets,
            max_items=_MAX_LIST_ITEMS,
            max_chars=_MAX_PATH_CHARS,
        ):
            return "run action targets are invalid"
        return ""
    if action == "note":
        if not all(
            isinstance(arguments.get(key), str)
            and len(str(arguments[key])) <= _MAX_WORKING_NOTE_CHARS
            for key in allowed_keys
        ):
            return "note action requires bounded text fields"
        return ""
    if action == "finish":
        if not _bounded_text(arguments.get("summary"), _MAX_SUMMARY_CHARS):
            return "finish action requires a summary"
        if not all(
            isinstance(arguments.get(key), list)
            for key in ("acceptance_criteria", "evidence_refs", "known_limitations")
        ):
            return "finish action requires list evidence fields"
        if not all(
            _bounded_string_list(
                arguments[key],
                max_items=_MAX_LIST_ITEMS,
                max_chars=_MAX_LIST_ITEM_CHARS,
                allow_empty_items=False,
            )
            for key in ("acceptance_criteria", "evidence_refs", "known_limitations")
        ):
            return "finish action evidence fields are invalid"
        return ""
    if arguments.get("blocker_type") not in {
        "needs_user_input",
        "environment",
        "scope",
        "safety",
    }:
        return "block action blocker type is invalid"
    if not _bounded_text(arguments.get("question"), _MAX_QUESTION_CHARS):
        return "block action requires a question"
    if not _bounded_string_list(
        arguments.get("options"),
        max_items=_MAX_LIST_ITEMS,
        max_chars=_MAX_LIST_ITEM_CHARS,
        allow_empty_items=False,
    ):
        return "block action options are invalid"
    if not _bounded_string_list(
        arguments.get("blocking_evidence_refs"),
        max_items=_MAX_LIST_ITEMS,
        max_chars=_MAX_LIST_ITEM_CHARS,
        allow_empty_items=False,
    ):
        return "block action evidence references are invalid"
    return ""


def _bounded_text(value: object, max_chars: int) -> bool:
    """Return whether a value is nonempty bounded controller text."""

    bounded = (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= max_chars
    )
    return bounded


def _bounded_string_list(
    value: object,
    *,
    max_items: int,
    max_chars: int,
    allow_empty_items: bool = False,
) -> bool:
    """Validate a controller-owned list before prompt data is persisted."""

    if not isinstance(value, list) or len(value) > max_items:
        return False
    for item in value:
        if not isinstance(item, str) or len(item) > max_chars:
            return False
        if not allow_empty_items and not item.strip():
            return False
    return True


def _is_int(value: object) -> bool:
    """Reject booleans while accepting ordinary JSON integers."""

    valid_integer = isinstance(value, int) and not isinstance(value, bool)
    return valid_integer


def _is_sha256(value: object) -> bool:
    """Return whether a value is one lowercase or uppercase SHA-256 hex digest."""

    if not isinstance(value, str) or len(value) != 64:
        return False
    valid_digest = all(
        character in "0123456789abcdefABCDEF" for character in value
    )
    return valid_digest
