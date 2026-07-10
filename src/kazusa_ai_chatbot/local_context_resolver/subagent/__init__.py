"""Resolver-local RAG3 subagent discovery and deterministic dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from kazusa_ai_chatbot.local_context_resolver.contracts import (
    LOCAL_CONTEXT_SUBAGENT_REQUEST_VERSION,
    LocalContextSubagentResultV1,
    validate_local_context_subagent_result,
)

_SUBAGENT_MODULES = (
    "conversation",
    "external",
    "live_context",
    "media",
    "memory",
    "person",
    "recall",
)


@dataclass(frozen=True)
class LocalContextSubagentRegistration:
    """Validated runtime metadata and factory for one RAG3 subagent."""

    subagent: str
    description: str
    supported_actions: tuple[str, ...]
    owned_node_kinds: tuple[str, ...]
    default_action: str
    create: object

    @property
    def OWNED_NODE_KINDS(self) -> tuple[str, ...]:
        """Expose module-style ownership metadata for focused introspection."""

        result = self.owned_node_kinds
        return result


def get_subagent_registry() -> dict[str, LocalContextSubagentRegistration]:
    """Load and validate the complete resolver-local subagent registry."""

    registry: dict[str, LocalContextSubagentRegistration] = {}
    owned_node_kinds: set[str] = set()
    for module_name in _SUBAGENT_MODULES:
        module = import_module(f"{__name__}.{module_name}")
        registration = _registration_from_module(module)
        if registration.subagent in registry:
            raise RuntimeError(f"duplicate RAG3 subagent: {registration.subagent}")
        overlap = owned_node_kinds.intersection(registration.owned_node_kinds)
        if overlap:
            raise RuntimeError(f"duplicate RAG3 node ownership: {sorted(overlap)}")
        registry[registration.subagent] = registration
        owned_node_kinds.update(registration.owned_node_kinds)
    return registry


async def dispatch_subagent_for_node(
    *,
    active_node: dict[str, object],
    context: dict[str, object],
    dependency_context: list[dict[str, object]],
    max_attempts: int,
) -> LocalContextSubagentResultV1:
    """Run the unique registered owner for one source-backed node kind."""

    node_kind = active_node["node_kind"]
    if not isinstance(node_kind, str):
        raise ValueError("active_node.node_kind: expected string")
    registration = _registration_for_node_kind(node_kind)
    if registration is None:
        result = _unavailable_result(node_kind)
        return result
    node_id = active_node["node_id"]
    objective = active_node["objective"]
    if not isinstance(node_id, str) or not isinstance(objective, str):
        raise ValueError("active node: expected id and objective")
    task = {
        "schema_version": LOCAL_CONTEXT_SUBAGENT_REQUEST_VERSION,
        "node_id": node_id,
        "subagent": registration.subagent,
        "action": registration.default_action,
        "objective": objective,
        "payload": {
            "dependency_context": dependency_context,
            "node_kind": node_kind,
        },
        "constraints": {},
    }
    if node_kind in ("current_turn_media", "recent_media"):
        selector_kind = (
            "current"
            if node_kind == "current_turn_media"
            else "recent"
        )
        session_media_refs = context.get("session_media_refs")
        has_current_media = (
            isinstance(session_media_refs, list)
            and any(
                isinstance(media_ref, dict)
                and media_ref.get("turn_relation") == "current"
                for media_ref in session_media_refs
            )
        )
        has_recent_media = (
            isinstance(session_media_refs, list)
            and any(
                isinstance(media_ref, dict)
                and media_ref.get("turn_relation") == "recent"
                for media_ref in session_media_refs
            )
        )
        if (
            selector_kind == "current"
            and not has_current_media
            and has_recent_media
        ):
            selector_kind = "recent"
        task["payload"]["selector"] = {
            "schema_version": "local_context_media_selector.v1",
            "selector_kind": selector_kind,
            "alias": None,
            "ordinal": 1,
            "question": objective,
        }
    subagent = registration.create()
    result = await subagent.run(task, context, max_attempts=max_attempts)
    validated = validate_local_context_subagent_result(result)
    return validated


def _registration_from_module(module: object) -> LocalContextSubagentRegistration:
    """Validate fixed metadata exported by one resolver-local module."""

    subagent = getattr(module, "SUBAGENT")
    description = getattr(module, "DESCRIPTION")
    supported_actions = getattr(module, "SUPPORTED_ACTIONS")
    owned_node_kinds = getattr(module, "OWNED_NODE_KINDS")
    default_action = getattr(module, "DEFAULT_ACTION")
    create = getattr(module, "create")
    if (
        not isinstance(subagent, str)
        or not subagent.strip()
        or not isinstance(description, str)
        or not description.strip()
        or not isinstance(supported_actions, tuple)
        or not all(isinstance(item, str) and item for item in supported_actions)
        or not isinstance(owned_node_kinds, tuple)
        or not all(isinstance(item, str) and item for item in owned_node_kinds)
        or default_action not in supported_actions
        or not callable(create)
    ):
        raise RuntimeError("invalid RAG3 subagent registration")
    result = LocalContextSubagentRegistration(
        subagent=subagent,
        description=description,
        supported_actions=supported_actions,
        owned_node_kinds=owned_node_kinds,
        default_action=default_action,
        create=create,
    )
    return result


def _registration_for_node_kind(
    node_kind: str,
) -> LocalContextSubagentRegistration | None:
    """Return the one registered owner for a source-backed node kind."""

    for registration in get_subagent_registry().values():
        if node_kind in registration.owned_node_kinds:
            return registration
    return None


def _unavailable_result(node_kind: str) -> LocalContextSubagentResultV1:
    """Return an explicit bounded result for non-source graph nodes."""

    result: LocalContextSubagentResultV1 = {
        "schema_version": "local_context_subagent_result.v1",
        "resolved": False,
        "status": "unavailable",
        "result": {
            "source_records": [],
            "artifacts": [],
            "node_update": {
                "evidence_boundary_notes": [
                    f"no resolver-local source owner for {node_kind}",
                ],
            },
        },
        "attempts": 0,
        "cache": {"enabled": False},
        "trace": {},
        "unresolved_items": [],
    }
    return result


__all__ = [
    "LocalContextSubagentRegistration",
    "dispatch_subagent_for_node",
    "get_subagent_registry",
]
