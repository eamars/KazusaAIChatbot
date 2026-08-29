"""Closed dispatcher for the thirteen Kazusa semantic capabilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.dsh_tool_gateway.authority import SignedSemanticCallV1
from kazusa_ai_chatbot.dsh_tool_gateway.catalog import (
    SEMANTIC_TOOL_NAMES,
    semantic_catalog_digest,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
)
from kazusa_ai_chatbot.dsh_tool_gateway.conversation import (
    ConversationSemanticService,
)
from kazusa_ai_chatbot.dsh_tool_gateway.media import MediaSemanticService
from kazusa_ai_chatbot.dsh_tool_gateway.memory import MemorySemanticService
from kazusa_ai_chatbot.dsh_tool_gateway.people import PeopleSemanticService
from kazusa_ai_chatbot.dsh_tool_gateway.recall_calendar import (
    RecallCalendarSemanticService,
)


class SemanticCapabilityDispatcher:
    """Route signed semantic calls to explicitly admitted service methods."""

    def __init__(
        self,
        *,
        conversation: ConversationSemanticService,
        memory: MemorySemanticService,
        people: PeopleSemanticService,
        recall_calendar: RecallCalendarSemanticService,
        media: MediaSemanticService,
        expected_catalog_digest: str | None = None,
    ) -> None:
        self._services = {
            "conversation": conversation,
            "memory": memory,
            "people": people,
            "recall_calendar": recall_calendar,
            "media": media,
        }
        self._expected_catalog_digest = expected_catalog_digest or semantic_catalog_digest()
        self._methods: dict[str, tuple[str, str]] = {
            "kazusa_search_conversation_history": ("conversation", "search_conversation_history"),
            "kazusa_read_conversation_entries": ("conversation", "read_conversation_entries"),
            "kazusa_summarize_conversation_participants": ("conversation", "summarize_conversation_participants"),
            "kazusa_search_memories": ("memory", "search_memories"),
            "kazusa_read_memories": ("memory", "read_memories"),
            "kazusa_remember_information": ("memory", "remember_information"),
            "kazusa_revise_memory": ("memory", "revise_memory"),
            "kazusa_change_memory_lifecycle": ("memory", "change_memory_lifecycle"),
            "kazusa_find_people_by_name": ("people", "find_people_by_name"),
            "kazusa_read_person_profiles": ("people", "read_person_profiles"),
            "kazusa_recall_active_context": ("recall_calendar", "recall_active_context"),
            "kazusa_read_calendar_context": ("recall_calendar", "read_calendar_context"),
            "kazusa_inspect_attached_media": ("media", "inspect_attached_media"),
        }

    @property
    def semantic_tool_names(self) -> tuple[str, ...]:
        """Return the fixed admitted semantic names."""

        return SEMANTIC_TOOL_NAMES

    async def dispatch(
        self,
        call: SignedSemanticCallV1,
    ) -> KazusaSemanticCapabilityResultV1:
        """Dispatch one already authenticated call under catalog authority."""

        if call.authority.catalog_digest != self._expected_catalog_digest:
            return KazusaSemanticCapabilityResultV1.failure(
                "denied", "CATALOG_MISMATCH", "The semantic catalog authority is stale."
            )
        target = self._methods.get(call.operation)
        if target is None:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "SEMANTIC_OPERATION_INVALID", "The semantic operation is unsupported."
            )
        service = self._services[target[0]]
        with_authority = getattr(service, "with_authority", None)
        if callable(with_authority):
            service = with_authority(call.authority)
        method = getattr(service, target[1])
        try:
            arguments = _project_arguments(
                call.operation,
                call.arguments,
                call.authority.service_scope,
            )
        except ValueError:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid",
                "SEMANTIC_ARGUMENTS_INVALID",
                "The semantic arguments are invalid.",
            )
        if call.operation in {
            "kazusa_remember_information",
            "kazusa_revise_memory",
            "kazusa_change_memory_lifecycle",
        }:
            if call.idempotency_key is None:
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid",
                    "MUTATION_LINEAGE_INVALID",
                    "The signed mutation lineage is invalid.",
                )
            arguments["idempotency_key"] = call.idempotency_key
        try:
            result = await method(**arguments)
        except (TypeError, ValueError):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid",
                "SEMANTIC_ARGUMENTS_INVALID",
                "The semantic arguments are invalid.",
            )
        except Exception:  # noqa: BLE001 - service faults map to a safe result
            return KazusaSemanticCapabilityResultV1.failure(
                "unavailable",
                "SEMANTIC_SERVICE_UNAVAILABLE",
                "The semantic service is unavailable.",
            )
        if not isinstance(result, KazusaSemanticCapabilityResultV1):
            raise TypeError("semantic service returned an invalid result")
        return result


_OPERATION_ARGUMENTS: dict[str, frozenset[str]] = {
    "kazusa_search_conversation_history": frozenset({
        "query", "time_range", "max_results", "next_page_ref",
    }),
    "kazusa_read_conversation_entries": frozenset({
        "conversation_entry_refs",
    }),
    "kazusa_summarize_conversation_participants": frozenset({
        "time_range", "max_people", "next_page_ref",
    }),
    "kazusa_search_memories": frozenset({
        "query", "subject_scope", "memory_kinds", "max_results", "next_page_ref",
    }),
    "kazusa_read_memories": frozenset({"memory_refs"}),
    "kazusa_remember_information": frozenset({
        "subject", "information", "memory_kind", "reason", "provenance",
    }),
    "kazusa_revise_memory": frozenset({
        "memory_ref", "revised_information", "reason",
    }),
    "kazusa_change_memory_lifecycle": frozenset({
        "memory_ref", "transition", "reason",
    }),
    "kazusa_find_people_by_name": frozenset({
        "display_name", "match_relation", "max_results", "next_page_ref",
    }),
    "kazusa_read_person_profiles": frozenset({"person_refs"}),
    "kazusa_recall_active_context": frozenset({"kinds", "max_results"}),
    "kazusa_read_calendar_context": frozenset({
        "view", "max_results", "next_page_ref",
    }),
    "kazusa_inspect_attached_media": frozenset({
        "attached_media_ref", "question",
    }),
}


def _project_arguments(
    operation: str,
    arguments: Mapping[str, Any],
    service_scope: Mapping[str, str],
) -> dict[str, Any]:
    """Project model arguments and inject the authenticated service scope."""
    allowed = _OPERATION_ARGUMENTS.get(operation)
    if allowed is None or not isinstance(arguments, Mapping):
        raise ValueError("semantic operation arguments are unsupported")
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError("semantic arguments contain unsupported fields")
    scope_fields = ("platform", "platform_channel_id", "global_user_id")
    if set(service_scope) != set(scope_fields):
        raise ValueError("semantic service scope is invalid")
    if any(
        not isinstance(service_scope[field], str) or not service_scope[field].strip()
        for field in scope_fields
    ):
        raise ValueError("semantic service scope is invalid")
    projected = {key: value for key, value in arguments.items()}
    if operation == "kazusa_search_conversation_history":
        projected.update({
            "platform": service_scope["platform"],
            "platform_channel_id": service_scope["platform_channel_id"],
            "global_user_id": service_scope["global_user_id"],
        })
    elif operation == "kazusa_recall_active_context":
        projected["context"] = {"service_scope": dict(service_scope)}
    elif operation == "kazusa_read_calendar_context":
        projected["source_scope"] = dict(service_scope)
    return projected
