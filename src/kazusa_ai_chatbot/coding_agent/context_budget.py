"""Prompt context-budget helpers for coding-agent LLM stages."""

from __future__ import annotations

CHARS_PER_ESTIMATED_TOKEN = 4
PROJECT_CONTEXT_TOKEN_CAP = 50_000
PROJECT_CONTEXT_CHAR_CAP = 50_000
HARD_INPUT_TOKEN_CAP = 42_000
HARD_INPUT_CHAR_CAP = 50_000
PM_TARGET_INPUT_TOKEN_CAP = 32_000
PROGRAMMER_TARGET_INPUT_TOKEN_CAP = 34_000
PATCHER_TARGET_INPUT_TOKEN_CAP = 28_000
SYNTHESIS_TARGET_INPUT_TOKEN_CAP = 34_000
RESERVED_OUTPUT_TOKENS = 8_000
MAX_CONTEXT_IDS = 50
MAX_INDEX_FILE_BYTES = 4 * 1024 * 1024
MAX_INDEX_CHUNK_BYTES = 64 * 1024
MAX_SEARCH_RESULTS_PER_PAGE = 20
MAX_SEARCH_EXCERPT_CHARS = 8_000
MAX_READ_LINES = 500
MAX_REGEX_QUERY_CHARS = 1_000
REGEX_SEARCH_TIMEOUT_MS = 500
MAX_INDEX_STORAGE_BYTES = 512 * 1024 * 1024


def estimate_input_tokens(char_count: int) -> int:
    """Estimate prompt tokens with the plan's ceil(char_count / 4) fallback."""

    if char_count <= 0:
        return 0

    estimated_tokens = (
        char_count + CHARS_PER_ESTIMATED_TOKEN - 1
    ) // CHARS_PER_ESTIMATED_TOKEN
    return estimated_tokens


def prompt_budget_metadata(
    *,
    system_prompt: str,
    payload_text: str,
    target_input_tokens: int,
    selected_evidence_refs: list[str] | None = None,
    pruned_evidence_count: int = 0,
    artifact_ids: list[str] | None = None,
) -> dict[str, object]:
    """Build trace metadata for one prompt input budget check."""

    system_chars = len(system_prompt)
    payload_chars = len(payload_text)
    input_chars = system_chars + payload_chars
    estimated_input_tokens = estimate_input_tokens(input_chars)
    metadata: dict[str, object] = {
        "system_chars": system_chars,
        "payload_chars": payload_chars,
        "input_chars": input_chars,
        "estimated_input_tokens": estimated_input_tokens,
        "target_input_tokens": target_input_tokens,
        "hard_input_tokens": HARD_INPUT_TOKEN_CAP,
        "hard_input_chars": HARD_INPUT_CHAR_CAP,
        "reserved_output_tokens": RESERVED_OUTPUT_TOKENS,
        "project_context_tokens": PROJECT_CONTEXT_TOKEN_CAP,
        "project_context_chars": PROJECT_CONTEXT_CHAR_CAP,
        "over_target": estimated_input_tokens > target_input_tokens,
        "over_hard_token_cap": estimated_input_tokens > HARD_INPUT_TOKEN_CAP,
        "over_hard_char_cap": input_chars > HARD_INPUT_CHAR_CAP,
        "over_hard_cap": (
            estimated_input_tokens > HARD_INPUT_TOKEN_CAP
            or input_chars > HARD_INPUT_CHAR_CAP
        ),
        "selected_evidence_refs": list(
            (selected_evidence_refs or [])[:MAX_CONTEXT_IDS]
        ),
        "pruned_evidence_count": pruned_evidence_count,
        "artifact_ids": list((artifact_ids or [])[:MAX_CONTEXT_IDS]),
    }
    return metadata


def collect_selected_evidence_refs(payload: object) -> list[str]:
    """Collect evidence reference ids from a model-facing payload."""

    refs: list[str] = []
    _collect_values_by_key(
        payload,
        keys={"evidence_ref", "evidence_refs"},
        values=refs,
    )
    return refs


def collect_artifact_ids(payload: object) -> list[str]:
    """Collect patch artifact ids from a model-facing payload."""

    artifact_ids: list[str] = []
    _collect_values_by_key(
        payload,
        keys={"artifact_id", "artifact_ids"},
        values=artifact_ids,
    )
    return artifact_ids


def _collect_values_by_key(
    payload: object,
    *,
    keys: set[str],
    values: list[str],
) -> None:
    if len(values) >= MAX_CONTEXT_IDS:
        return

    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                _append_string_values(value, values)
                if len(values) >= MAX_CONTEXT_IDS:
                    return
                continue
            _collect_values_by_key(value, keys=keys, values=values)
            if len(values) >= MAX_CONTEXT_IDS:
                return
        return

    if isinstance(payload, list):
        for item in payload:
            _collect_values_by_key(item, keys=keys, values=values)
            if len(values) >= MAX_CONTEXT_IDS:
                return


def _append_string_values(value: object, values: list[str]) -> None:
    if isinstance(value, str):
        _append_unique(value, values)
        return

    if isinstance(value, list):
        for item in value:
            if not isinstance(item, str):
                continue
            _append_unique(item, values)
            if len(values) >= MAX_CONTEXT_IDS:
                return


def _append_unique(value: str, values: list[str]) -> None:
    if not value or value in values:
        return
    values.append(value)
