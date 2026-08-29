"""Semantic people and profile capabilities."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from typing import Any

from kazusa_ai_chatbot.db.users import (
    get_user_profile,
    list_users_by_display_name,
)
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    KazusaSemanticCapabilityResultV1,
    OpaqueReferenceCodec,
    SemanticPageV1,
    content_digest,
    new_evidence_receipt,
)

_MAX_RESULTS = 50


def _limit(value: object, default: int = 10) -> int:
    """Clamp a requested number of people."""

    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(1, min(value, _MAX_RESULTS))


def _mapping(value: object, field: str) -> dict[str, Any]:
    """Convert one profile result to a mapping."""

    if isinstance(value, Mapping):
        return dict(value)
    try:
        return dict(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an object") from exc


class PeopleSemanticService:
    """Map user/profile leaves to opaque semantic people entities."""

    def __init__(
        self,
        *,
        codec: OpaqueReferenceCodec,
        find: Callable[..., Awaitable[list[dict[str, Any]]]] = list_users_by_display_name,
        read: Callable[..., Awaitable[Mapping[str, Any]]] = get_user_profile,
    ) -> None:
        self._codec = codec
        self._find = find
        self._read = read

    def with_authority(self, authority: Mapping[str, Any] | object) -> "PeopleSemanticService":
        """Return a call-local service bound to the signed authority."""

        bound = copy(self)
        bound._codec = self._codec.with_authority(authority)
        return bound

    async def find_people_by_name(
        self,
        *,
        display_name: str,
        match_relation: str,
        max_results: int = 10,
        next_page_ref: str | None = None,
    ) -> KazusaSemanticCapabilityResultV1:
        """Find people by semantic display-name fragment."""

        if not isinstance(display_name, str) or not display_name.strip():
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "NAME_REQUIRED", "A person name is required."
            )
        if match_relation not in {"exact", "contains", "starts_with", "ends_with"}:
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "MATCH_RELATION_INVALID", "The person match relation is unsupported."
            )
        limit = _limit(max_results)
        offset = 0
        if next_page_ref is not None:
            try:
                payload = self._codec.resolve(next_page_ref, "person-page")
                offset = payload["offset"]
                if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                    raise ValueError
            except (KeyError, TypeError, ValueError):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "PAGE_REFERENCE_INVALID", "The continuation reference is invalid."
                )
        rows = await self._find(
            display_name.strip(),
            operator=match_relation,
            limit=offset + limit + 1,
        )
        selected = list(rows)[offset: offset + limit + 1]
        has_more = len(selected) > limit
        selected = selected[:limit]
        entities: list[dict[str, Any]] = []
        evidence = []
        for index, value in enumerate(selected):
            row = _mapping(value, "person result")
            source_id = row.get("global_user_id")
            if not isinstance(source_id, str) or not source_id:
                continue
            reference = self._codec.issue("person", {"source_id": source_id})
            entity = {
                "person_ref": reference,
                "name": str(row.get("display_name") or ""),
                "platform": str(row.get("platform") or ""),
            }
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=f"receipt-person-{content_digest(reference)}",
                source_kind="person_candidate",
                semantic_ref=reference,
                value=entity,
            ))
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
            page=SemanticPageV1(
                has_more=has_more,
                next_page_ref=(
                    self._codec.issue("person-page", {"offset": offset + limit})
                    if has_more
                    else None
                ),
            ),
        )

    async def read_person_profiles(
        self,
        *,
        person_refs: Sequence[str],
    ) -> KazusaSemanticCapabilityResultV1:
        """Read profile facts for exact opaque person references."""

        if not isinstance(person_refs, Sequence) or isinstance(person_refs, (str, bytes)):
            return KazusaSemanticCapabilityResultV1.failure(
                "invalid", "PERSON_REFS_REQUIRED", "Person references are required."
            )
        entities: list[dict[str, Any]] = []
        evidence = []
        for index, reference in enumerate(person_refs[:_MAX_RESULTS]):
            try:
                payload = self._codec.resolve(str(reference), "person")
            except ValueError:
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "PERSON_REFERENCE_INVALID", "A person reference is invalid."
                )
            source_id = payload.get("source_id")
            if not isinstance(source_id, str):
                return KazusaSemanticCapabilityResultV1.failure(
                    "invalid", "PERSON_REFERENCE_INVALID", "A person reference is invalid."
                )
            profile = await self._read(source_id)
            if profile is None:
                continue
            row = _mapping(profile, "person profile")
            entity = _profile_entity(row, reference=str(reference))
            entities.append(entity)
            evidence.append(new_evidence_receipt(
                receipt_id=f"receipt-profile-{content_digest(str(reference))}",
                source_kind="person_profile",
                semantic_ref=str(reference),
                value=entity,
            ))
        return KazusaSemanticCapabilityResultV1.success(
            entities=entities,
            evidence=evidence,
        )


def _profile_entity(row: Mapping[str, Any], *, reference: str) -> dict[str, Any]:
    """Project only semantic profile fields."""

    entity: dict[str, Any] = {"person_ref": reference}
    accounts = row.get("platform_accounts")
    if isinstance(accounts, list):
        names = [
            item.get("display_name")
            for item in accounts
            if isinstance(item, Mapping) and isinstance(item.get("display_name"), str)
        ]
        if names:
            entity["known_names"] = names[:20]
    for key in ("facts", "suspected_aliases", "cognition_state"):
        value = row.get(key)
        if value not in (None, [], {}):
            entity[key] = value
    return entity
