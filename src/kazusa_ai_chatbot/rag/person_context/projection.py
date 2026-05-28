"""Person worker result projection."""

from __future__ import annotations

from typing import Any

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.rag.person_context.contracts import _clip_text
from kazusa_ai_chatbot.utils import text_or_empty

def _iter_known_refs(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect structured refs from previous RAG2 known facts."""
    refs: list[dict[str, Any]] = []
    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        return refs
    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        raw_result = fact.get("raw_result")
        if not isinstance(raw_result, dict):
            continue
        raw_refs = raw_result.get("resolved_refs")
        if not isinstance(raw_refs, list):
            continue
        for ref in raw_refs:
            if isinstance(ref, dict):
                refs.append(ref)
    return refs

def _first_person_ref(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first structured person reference from previous slots."""
    for ref in _iter_known_refs(context):
        if ref.get("ref_type") == "person":
            return ref
    return_value = None
    return return_value

def _person_ref(
    *,
    role: str,
    global_user_id: object,
    display_name: object = "",
) -> dict[str, str]:
    """Build a normalized person reference for downstream slots."""
    ref = {
        "ref_type": "person",
        "role": role,
        "global_user_id": text_or_empty(global_user_id),
        "display_name": text_or_empty(display_name),
    }
    return ref

def _ref_from_lookup_result(worker_result: dict[str, Any]) -> dict[str, str] | None:
    """Extract a profile-owner ref from a user lookup worker result."""
    raw_result = worker_result.get("result")
    if not isinstance(raw_result, dict):
        return None
    global_user_id = text_or_empty(raw_result.get("global_user_id"))
    display_name = text_or_empty(raw_result.get("display_name"))
    if not global_user_id and not display_name:
        return None
    ref = _person_ref(
        role="profile_owner",
        global_user_id=global_user_id,
        display_name=display_name,
    )
    return ref

def _current_user_ref(context: dict[str, Any]) -> dict[str, str]:
    """Build the current-user profile ref from scoped context."""
    ref = _person_ref(
        role="current_user",
        global_user_id=context.get("global_user_id"),
        display_name=context.get("user_name"),
    )
    return ref

def _active_character_ref(context: dict[str, Any]) -> dict[str, str]:
    """Build the active-character profile ref from scoped context."""
    character_profile = context.get("character_profile")
    display_name = ""
    if isinstance(character_profile, dict):
        display_name = text_or_empty(character_profile.get("name"))
    ref = _person_ref(
        role="active_character",
        global_user_id=CHARACTER_GLOBAL_USER_ID,
        display_name=display_name,
    )
    return ref

def _context_with_ref(context: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    """Append a structured person ref in the shape profile worker can read."""
    worker_context = dict(context)
    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        known_facts = []
    else:
        known_facts = list(known_facts)
    known_facts.append({"raw_result": {"resolved_refs": [ref]}})
    worker_context["known_facts"] = known_facts
    global_user_id = text_or_empty(ref.get("global_user_id"))
    display_name = text_or_empty(ref.get("display_name"))
    if global_user_id:
        worker_context["global_user_id"] = global_user_id
    if display_name:
        worker_context["display_name"] = display_name
    return worker_context

def _summary_from_profile(profile: object) -> str:
    """Build a compact profile summary without parsing hidden internals."""
    if not isinstance(profile, dict):
        return_value = _clip_text(profile)
        return return_value
    display_name = text_or_empty(profile.get("display_name"))
    if not display_name:
        display_name = text_or_empty(profile.get("name"))
    self_image = profile.get("self_image")
    image_summary = ""
    if isinstance(self_image, dict):
        image_summary = text_or_empty(self_image.get("summary"))
    elif isinstance(self_image, str):
        image_summary = self_image.strip()
    parts = [part for part in (display_name, image_summary) if part]
    summary = " | ".join(parts)
    return summary

def _summary_from_people_payload(value: object) -> str:
    """Build summary text for list or relationship worker payloads."""
    if isinstance(value, dict):
        summary = text_or_empty(value.get("summary"))
        if summary:
            return summary
        users = value.get("users")
        if isinstance(users, list):
            names = [
                text_or_empty(user.get("display_name"))
                for user in users
                if isinstance(user, dict)
            ]
            summary = ", ".join(name for name in names if name)
            return summary
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                display_name = text_or_empty(item.get("display_name"))
                relationship = text_or_empty(item.get("relationship_label"))
                if display_name and relationship:
                    parts.append(f"{display_name}: {relationship}")
                elif display_name:
                    parts.append(display_name)
        summary = ", ".join(parts)
        return summary
    return_value = _clip_text(value)
    return return_value

def _owner_id_from_profile(profile: object, fallback_ref: dict[str, Any]) -> str:
    """Return the profile owner id from profile payload or fallback ref."""
    if isinstance(profile, dict):
        global_user_id = text_or_empty(profile.get("global_user_id"))
        if global_user_id:
            return global_user_id
    return_value = text_or_empty(fallback_ref.get("global_user_id"))
    return return_value

def _profile_kind_for_ref(ref: dict[str, Any], requested_target: str) -> str:
    """Classify profile projection kind from target and person ref."""
    if requested_target == "current_user":
        return "current_user"
    if requested_target == "active_character":
        return "active_character"
    role = text_or_empty(ref.get("role"))
    if role == "current_user":
        return "current_user"
    if role == "active_character":
        return "active_character"
    return_value = "third_party"
    return return_value
