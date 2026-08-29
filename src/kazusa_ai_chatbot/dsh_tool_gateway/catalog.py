"""The fixed model-facing Kazusa semantic capability catalog."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from kazusa_ai_chatbot.dsh_tool_gateway.contracts import canonical_json

SEMANTIC_TOOL_NAMES: tuple[str, ...] = (
    "kazusa_search_conversation_history",
    "kazusa_read_conversation_entries",
    "kazusa_summarize_conversation_participants",
    "kazusa_search_memories",
    "kazusa_read_memories",
    "kazusa_remember_information",
    "kazusa_revise_memory",
    "kazusa_change_memory_lifecycle",
    "kazusa_find_people_by_name",
    "kazusa_read_person_profiles",
    "kazusa_recall_active_context",
    "kazusa_read_calendar_context",
    "kazusa_inspect_attached_media",
)

def _nullable_text() -> dict[str, Any]:
    """Return the shared nullable text schema."""

    return {"type": ["string", "null"]}


def _time_range() -> dict[str, Any]:
    """Return the semantic time-range schema."""

    return {
        "type": "object",
        "properties": {
            "start_at": {"type": "string"},
            "end_at": {"type": "string"},
        },
        "required": [],
        "additionalProperties": False,
    }


def _page_properties() -> dict[str, Any]:
    """Return the shared opaque continuation property."""

    return {"next_page_ref": _nullable_text()}


def _bounded_result() -> dict[str, Any]:
    """Return the shared bounded result property."""

    return {"type": "integer", "minimum": 1, "maximum": 50}


_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "name": "kazusa_search_conversation_history",
        "description": "Find relevant conversation entries by meaning and optional time range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "time_range": _time_range(),
                "max_results": _bounded_result(),
                **_page_properties(),
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_read_conversation_entries",
        "description": "Read complete semantic conversation entries by opaque references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "conversation_entry_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 50,
                },
            },
            "required": ["conversation_entry_refs"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_summarize_conversation_participants",
        "description": "Summarize participants observed in an optional conversation time range.",
        "input_schema": {
            "type": "object",
            "properties": {
                "time_range": _time_range(),
                "max_people": _bounded_result(),
                **_page_properties(),
            },
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_search_memories",
        "description": "Search semantic memories relevant to a query and scope.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "subject_scope": {
                    "type": "string",
                    "enum": ["current_user", "active_character", "shared_world", "all"],
                },
                "memory_kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["profile_fact", "relationship", "commitment", "experience", "world_knowledge"],
                    },
                    "maxItems": 5,
                },
                "max_results": _bounded_result(),
                **_page_properties(),
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_read_memories",
        "description": "Read complete semantic memories by opaque references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 50,
                },
            },
            "required": ["memory_refs"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_remember_information",
        "description": "Retain semantic information with an explicit subject, kind, reason, and provenance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "enum": ["current_user", "active_character", "shared_world"],
                },
                "information": {"type": "string"},
                "memory_kind": {
                    "type": "string",
                    "enum": ["profile_fact", "relationship", "commitment", "experience", "world_knowledge"],
                },
                "reason": {"type": "string"},
                "provenance": {
                    "type": "object",
                    "properties": {
                        "conversation_entry_ref": {"type": "string"},
                        "current_task": {"type": "string"},
                    },
                    "required": [],
                    "additionalProperties": False,
                    "oneOf": [
                        {"required": ["conversation_entry_ref"]},
                        {"required": ["current_task"]},
                    ],
                },
            },
            "required": ["subject", "information", "memory_kind", "reason", "provenance"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_revise_memory",
        "description": "Revise semantic information identified by one opaque memory reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_ref": {"type": "string"},
                "revised_information": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["memory_ref", "revised_information", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_change_memory_lifecycle",
        "description": "Apply one explicit semantic lifecycle transition to an opaque memory reference.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_ref": {"type": "string"},
                "transition": {"type": "string", "enum": ["activate", "complete", "cancel", "archive"]},
                "reason": {"type": "string"},
            },
            "required": ["memory_ref", "transition", "reason"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_find_people_by_name",
        "description": "Find people by display name and semantic relation matching.",
        "input_schema": {
            "type": "object",
            "properties": {
                "display_name": {"type": "string"},
                "match_relation": {
                    "type": "string",
                    "enum": ["exact", "contains", "starts_with", "ends_with"],
                },
                "max_results": _bounded_result(),
                **_page_properties(),
            },
            "required": ["display_name", "match_relation"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_read_person_profiles",
        "description": "Read semantic profiles by opaque person references.",
        "input_schema": {
            "type": "object",
            "properties": {
                "person_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 50,
                },
            },
            "required": ["person_refs"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_recall_active_context",
        "description": "Recall active commitments, progress, history, or calendar context.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kinds": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["commitments", "progress", "history", "calendar"],
                    },
                    "minItems": 1,
                    "maxItems": 4,
                },
                "max_results": _bounded_result(),
            },
            "required": ["kinds"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_read_calendar_context",
        "description": "Read semantic schedule or calendar-run context by view.",
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {"type": "string", "enum": ["schedules", "recent_runs", "pending_runs"]},
                "max_results": _bounded_result(),
                **_page_properties(),
            },
            "required": ["view"],
            "additionalProperties": False,
        },
    },
    {
        "name": "kazusa_inspect_attached_media",
        "description": "Inspect already attached media using an opaque reference and semantic question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "attached_media_ref": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["attached_media_ref", "question"],
            "additionalProperties": False,
        },
    },
)


def semantic_catalog() -> tuple[dict[str, Any], ...]:
    """Return a deep-enough copy of the fixed semantic catalog."""

    return tuple(json.loads(json.dumps(item)) for item in _CATALOG)


def description_stripped_catalog(
    standard_names: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Return description-free Kazusa tools after native names take precedence."""

    native = {name for name in standard_names if isinstance(name, str)}

    def strip(value: object) -> object:
        if isinstance(value, Mapping):
            return {
                key: strip(nested)
                for key, nested in value.items()
                if key != "description"
            }
        if isinstance(value, list):
            return [strip(nested) for nested in value]
        return value

    return tuple(
        strip(item)
        for item in semantic_catalog()
        if item["name"] not in native
    )


def semantic_catalog_projection(
    standard_names: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Return the canonical name/schema projection used by the model route."""

    return tuple(
        {
            "name": item["name"],
            "input_schema": item["input_schema"],
        }
        for item in description_stripped_catalog(standard_names)
    )


def _schema_projection(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Normalize one mounted DSH schema to its description-free projection."""

    name = value.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{field}.name must be a non-empty string")
    schema = value.get("input_schema", value.get("parameters"))
    if not isinstance(schema, Mapping):
        raise ValueError(f"{field}.input_schema must be an object")

    def strip(item: object) -> object:
        if isinstance(item, Mapping):
            return {
                key: strip(nested)
                for key, nested in item.items()
                if key != "description"
            }
        if isinstance(item, list):
            return [strip(nested) for nested in item]
        return item

    return {"name": name, "input_schema": strip(schema)}


def native_catalog_projection(
    schemas: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the canonical projection of the actual mounted native registry."""

    return tuple(
        sorted(
            (_schema_projection(schema, f"native[{index}]") for index, schema in enumerate(schemas)),
            key=lambda item: item["name"],
        )
    )


def published_catalog_projection(
    native_schemas: Sequence[Mapping[str, Any]],
    *,
    submit_schema: Mapping[str, Any],
    standard_names: Iterable[str] = (),
) -> tuple[dict[str, Any], ...]:
    """Return native-precedence plus semantic and terminal tool schemas."""

    native = native_catalog_projection(native_schemas)
    native_names = {item["name"] for item in native}
    semantic = semantic_catalog_projection(
        [*native_names, *standard_names]
    )
    entries = [*native, *semantic]
    submit = _schema_projection(submit_schema, "submit_schema")
    if submit["name"] not in {item["name"] for item in entries}:
        entries.append(submit)
    return tuple(sorted(entries, key=lambda item: item["name"]))


def semantic_catalog_digest(
    catalog: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Return a non-secret digest for the description-free semantic catalog."""

    if catalog is None:
        selected = semantic_catalog_projection()
    else:
        selected = tuple(
            _schema_projection(item, f"semantic[{index}]")
            for index, item in enumerate(catalog)
        )
        selected = tuple(sorted(selected, key=lambda item: item["name"]))
    digest = hashlib.sha256(canonical_json(list(selected))).hexdigest()
    return f"sha256:{digest}"


def native_catalog_digest(
    schemas: Sequence[Mapping[str, Any]],
) -> str:
    """Return the digest of the actual mounted native DSH schema registry."""

    digest = hashlib.sha256(
        canonical_json(list(native_catalog_projection(schemas)))
    ).hexdigest()
    return f"sha256:{digest}"


def published_catalog_digest(
    native_schemas: Sequence[Mapping[str, Any]],
    *,
    submit_schema: Mapping[str, Any],
    standard_names: Iterable[str] = (),
) -> str:
    """Return the digest of the complete model-visible published registry."""

    digest = hashlib.sha256(
        canonical_json(list(published_catalog_projection(
            native_schemas,
            submit_schema=submit_schema,
            standard_names=standard_names,
        )))
    ).hexdigest()
    return f"sha256:{digest}"


def colliding_names(
    standard_names: Iterable[str],
) -> frozenset[str]:
    """Return semantic names occupied by the installed native catalog."""

    native = set(standard_names)
    return frozenset(name for name in SEMANTIC_TOOL_NAMES if name in native)
