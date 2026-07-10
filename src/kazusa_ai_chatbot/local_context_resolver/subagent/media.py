"""Current and recent session-image inspection for RAG3."""

from __future__ import annotations

from kazusa_ai_chatbot.local_context_resolver.contracts import (
    LOCAL_CONTEXT_ARTIFACT_VERSION,
    LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION,
    LocalContextSubagentRequestV1,
    LocalContextSubagentResultV1,
    validate_local_context_subagent_request,
)
from kazusa_ai_chatbot.media_inspection.service import inspect_media
from kazusa_ai_chatbot.media_inspection.contracts import (
    validate_media_inspection_result,
)
from kazusa_ai_chatbot.media_inspection.session_cache import (
    get_session_media,
    list_session_media_refs,
)

SUBAGENT = "media"
DESCRIPTION = "current or recent local conversation image evidence"
SUPPORTED_ACTIONS = ("inspect_media",)
OWNED_NODE_KINDS = ("current_turn_media", "recent_media")
DEFAULT_ACTION = "inspect_media"


class MediaSubagent:
    """Resolve one prompt-safe local image selector through the shared service."""

    def __init__(self, inspect_func=inspect_media) -> None:
        """Create the media subagent with the production inspector by default."""

        self._inspect_func = inspect_func

    async def run(
        self,
        task: LocalContextSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> LocalContextSubagentResultV1:
        """Inspect one cached image or return a bounded evidence gap."""

        del max_attempts
        request = validate_local_context_subagent_request(task)
        selector = request["payload"].get("selector")
        if not isinstance(selector, dict):
            return _unavailable(request, "invalid_media_selector")
        scope = _scope(context)
        media_ref = _resolve_selector(selector, scope)
        if media_ref is None:
            return _unavailable(request, "cache_miss")
        payload = get_session_media(scope, media_ref["cache_ref"])
        if payload is None:
            return _unavailable(request, "cache_miss")
        question = selector.get("question")
        if not isinstance(question, str) or not question.strip():
            return _unavailable(request, "invalid_media_selector")
        raw_inspection = await self._inspect_func({
            "schema_version": "media_inspection_request.v1",
            "source": "rag3_session_media",
            "media_kind": "image",
            "content_type": payload["content_type"],
            "base64_data": payload["base64_data"],
            "question": question.strip(),
            "existing_descriptor": payload["existing_descriptor"],
        })
        inspection = validate_media_inspection_result(raw_inspection)
        status = inspection["status"]
        answer = inspection["answer"]
        notes = inspection["evidence_boundary_notes"]
        alias = _alias_for_selector(selector)
        artifact = {
            "schema_version": LOCAL_CONTEXT_ARTIFACT_VERSION,
            "artifact_id": f"{request['node_id']}_{alias}",
            "artifact_type": "media_ref",
            "producer_node_id": request["node_id"],
            "summary": answer,
            "projection_payload": {
                "answer": answer,
                "media_evidence": [{
                    "alias": alias,
                    "media_kind": "image",
                    "content_type": payload["content_type"],
                    "description": answer,
                    "recency": _recency(selector),
                    "turn_relation": selector["selector_kind"],
                    "source_summary": payload["source_summary"],
                    "evidence_boundary_notes": notes,
                }],
            },
            "source_policy": "session image cache",
            "prompt_visible": True,
        }
        resolved = status in ("answered", "uncertain")
        result: LocalContextSubagentResultV1 = {
            "schema_version": LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION,
            "resolved": resolved,
            "status": "resolved" if status == "answered" else "partial",
            "result": {
                "source_records": [{
                    "agent": "media",
                    "node_kind": request["subagent"],
                    "resolved": resolved,
                    "source_policy": "session image cache",
                    "evidence": [answer] if answer else [],
                    "missing_context": [],
                    "projection_payload": artifact["projection_payload"],
                }],
                "artifacts": [artifact],
                "node_update": {
                    "status": "resolved" if resolved else "blocked",
                    "investigation_summary": [
                        f"Inspected {alias} for the requested visual detail.",
                    ],
                    "knowledge_we_know_so_far": [answer] if answer else [],
                    "knowledge_still_lacking": (
                        [] if resolved else ["image detail unavailable"]
                    ),
                    "recommended_next_iteration": [],
                    "evidence_boundary_notes": notes,
                    "produces": [artifact["artifact_id"]],
                },
            },
            "attempts": 1,
            "cache": {"enabled": True, "hit": True},
            "trace": {"media_inspection_called": True},
            "unresolved_items": [] if resolved else ["image detail unavailable"],
        }
        return result


def create() -> MediaSubagent:
    """Create the production RAG3 media subagent."""

    return MediaSubagent()


def _scope(context: dict[str, object]) -> tuple[str, str, str]:
    """Extract the exact trusted cache scope from resolver context."""

    result = (
        str(context["platform"]),
        str(context["platform_channel_id"]),
        str(context["global_user_id"]),
    )
    return result


def _resolve_selector(
    selector: dict[str, object],
    scope: tuple[str, str, str],
) -> dict[str, object] | None:
    """Resolve a selector deterministically against scoped trusted refs."""

    selector_kind = selector.get("selector_kind")
    refs = list_session_media_refs(scope)
    if selector_kind == "alias":
        alias = selector.get("alias")
        if not isinstance(alias, str):
            return None
        for relation in ("current", "recent"):
            matching_refs = [
                ref for ref in reversed(refs)
                if ref.get("turn_relation") == relation
            ]
            for index, ref in enumerate(matching_refs, start=1):
                if alias == f"{relation}_media_{index}":
                    return ref
        return None
    ordinal = selector.get("ordinal")
    if not isinstance(ordinal, int) or ordinal < 1:
        return None
    if selector_kind in ("current", "recent"):
        ordered = [
            ref for ref in reversed(refs)
            if ref.get("turn_relation") == selector_kind
        ]
    else:
        return None
    if ordinal > len(ordered):
        return None
    result = ordered[ordinal - 1]
    return result


def _alias_for_selector(selector: dict[str, object]) -> str:
    """Return the stable prompt-safe alias for one selected media item."""

    ordinal = selector.get("ordinal", 1)
    if not isinstance(ordinal, int) or ordinal < 1:
        ordinal = 1
    prefix = (
        "current_media"
        if selector.get("selector_kind") == "current"
        else "recent_media"
    )
    result = f"{prefix}_{ordinal}"
    return result


def _recency(selector: dict[str, object]) -> str:
    """Map a deterministic selector into a short model-facing recency label."""

    if selector.get("selector_kind") == "current":
        return "current turn"
    return "recent conversation"


def _unavailable(
    task: LocalContextSubagentRequestV1,
    reason: str,
) -> LocalContextSubagentResultV1:
    """Return a cache boundary without calling the vision inspector."""

    result: LocalContextSubagentResultV1 = {
        "schema_version": LOCAL_CONTEXT_SUBAGENT_RESULT_VERSION,
        "resolved": False,
        "status": "unavailable",
        "result": {
            "source_records": [],
            "artifacts": [],
            "node_update": {
                "status": "blocked",
                "investigation_summary": [],
                "knowledge_we_know_so_far": [],
                "knowledge_still_lacking": ["requested image is unavailable"],
                "recommended_next_iteration": [],
                "evidence_boundary_notes": [reason],
            },
        },
        "attempts": 0,
        "cache": {"enabled": True, "hit": False},
        "trace": {"media_inspection_called": False},
        "unresolved_items": [reason],
    }
    return result
