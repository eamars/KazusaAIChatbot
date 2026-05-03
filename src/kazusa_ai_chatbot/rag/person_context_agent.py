"""Top-level RAG capability agent for person/profile context."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.relationship_agent import RelationshipAgent
from kazusa_ai_chatbot.rag.user_list_agent import UserListAgent
from kazusa_ai_chatbot.rag.user_lookup_agent import UserLookupAgent
from kazusa_ai_chatbot.rag.user_profile_agent import UserProfileAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "person_context"
_AGENT_NAME = "person_context_agent"
_UNCACHED_REASON = "capability_orchestrator_uncached"
_KNOWN_MODES = {
    "lookup",
    "profile",
    "lookup_profile",
    "user_list",
    "relationship",
    "incompatible",
}


def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact prompt-facing text."""
    text = text_or_empty(value)
    if len(text) <= limit:
        return text
    clipped = text[: limit - 1].rstrip()
    return_value = f"{clipped}…"
    return return_value


def _cache_status() -> dict[str, Any]:
    """Build no-cache metadata for top-level capability orchestration."""
    status = {
        "enabled": False,
        "hit": False,
        "cache_name": "",
        "reason": _UNCACHED_REASON,
    }
    return status


def _result_payload(
    *,
    selected_summary: object = "",
    primary_worker: str = "",
    supporting_workers: list[str] | None = None,
    source_policy: object = "",
    resolved_refs: list[dict[str, Any]] | None = None,
    projection_payload: dict[str, Any] | None = None,
    worker_payloads: dict[str, Any] | None = None,
    evidence: list[str] | None = None,
    missing_context: list[str] | None = None,
    conflicts: list[str] | None = None,
) -> dict[str, Any]:
    """Build the standard top-level person capability payload."""
    payload = {
        "selected_summary": _clip_text(selected_summary),
        "capability": _CAPABILITY_NAME,
        "primary_worker": primary_worker,
        "supporting_workers": list(supporting_workers or []),
        "source_policy": _clip_text(source_policy, limit=400),
        "resolved_refs": list(resolved_refs or []),
        "projection_payload": dict(projection_payload or {}),
        "worker_payloads": dict(worker_payloads or {}),
        "evidence": list(evidence or []),
        "missing_context": list(missing_context or []),
        "conflicts": list(conflicts or []),
    }
    return payload


def _agent_result(*, resolved: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the outer helper result for one capability-agent execution."""
    result = {
        "resolved": resolved,
        "result": payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return result


def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value


def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured person-context slots without selector LLM."""
    task_body = _strip_prefix(task)
    normalized = task_body.lower()

    if "unknown speaker" in normalized and ("said" in normalized or '"' in task_body):
        plan = {
            "mode": "incompatible",
            "target": "",
            "reason": "Conversation-evidence",
        }
        return plan

    if "list users" in normalized or "display names" in normalized:
        plan = {
            "mode": "user_list",
            "target": "",
            "reason": "display-name predicate enumeration",
        }
        return plan

    if "relationship" in normalized or "rank users" in normalized:
        plan = {
            "mode": "relationship",
            "target": "",
            "reason": "relationship ranking",
        }
        return plan

    if "active character" in normalized and "profile" in normalized:
        plan = {
            "mode": "profile",
            "target": "active_character",
            "reason": "active character profile",
        }
        return plan

    if "current user" in normalized and "profile" in normalized:
        plan = {
            "mode": "profile",
            "target": "current_user",
            "reason": "current user profile",
        }
        return plan

    if "resolved in slot" in normalized or "speaker found in slot" in normalized:
        plan = {
            "mode": "profile",
            "target": "known_ref",
            "reason": "profile for structured person ref",
        }
        return plan

    if "display name" in normalized and (
        "profile" in normalized or "impression" in normalized
    ):
        plan = {
            "mode": "lookup_profile",
            "target": "display_name",
            "reason": "display-name lookup followed by profile",
        }
        return plan

    if "display name" in normalized or "resolve" in normalized:
        plan = {
            "mode": "lookup",
            "target": "display_name",
            "reason": "display-name identity lookup",
        }
        return plan

    return None


def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to approved fields."""
    mode = text_or_empty(raw_plan.get("mode"))
    if mode not in _KNOWN_MODES:
        mode = "lookup"
    target = text_or_empty(raw_plan.get("target"))
    reason = text_or_empty(raw_plan.get("reason"))
    plan = {
        "mode": mode,
        "target": target,
        "reason": reason,
    }
    return plan


_SELECTOR_PROMPT = """\
You choose one bounded person-context worker path for a RAG evidence slot.
Do not search conversation history for an unknown speaker. That requires
Conversation-evidence before Person-context.

# Generation Procedure
1. Use mode="incompatible" with reason="Conversation-evidence" when the task
   first needs to find an unknown speaker by quoted message or content.
2. Use mode="user_list" for display-name predicates or enumerating users.
3. Use mode="relationship" for relationship rankings or compatibility bands.
4. Use mode="profile" for current user profile, active character profile, or
   a person already resolved by an earlier slot. Set target to current_user,
   active_character, or known_ref.
5. Use mode="lookup_profile" for display-name -> profile/impression requests.
6. Use mode="lookup" for identity-only display-name resolution.

# Input Format
{
  "task": "Person-context slot text",
  "original_query": "decontextualized user query when available",
  "current_slot": "slot label",
  "known_facts": "ordered facts from previous RAG2 slots"
}

# Output Format
Return valid JSON only:
{
  "mode": "lookup | profile | lookup_profile | user_list | relationship | incompatible",
  "target": "display_name | current_user | active_character | known_ref | empty",
  "reason": "short source selection explanation"
}
"""
_selector_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


async def _select_plan(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Select the bounded person worker path for one slot."""
    deterministic_plan = _deterministic_plan(task)
    if deterministic_plan is not None:
        return deterministic_plan

    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )
    response = await _selector_llm.ainvoke([system_prompt, human_message])
    raw_plan = parse_llm_json_output(response.content)
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    plan = _normalize_selector_plan(raw_plan)
    return plan


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


class PersonContextAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for identity, profile, and relationship context."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached person-context capability agent."""
        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.lookup_agent = UserLookupAgent(cache_runtime=cache_runtime)
        self.profile_agent = UserProfileAgent(cache_runtime=cache_runtime)
        self.user_list_agent = UserListAgent(cache_runtime=cache_runtime)
        self.relationship_agent = RelationshipAgent(cache_runtime=cache_runtime)

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one person-context slot through an approved worker path."""

        del max_attempts

        plan = await _select_plan(task, context)
        mode = plan["mode"]
        if mode == "incompatible":
            reason = text_or_empty(plan["reason"]) or "unsupported"
            result = self._unresolved(
                source_policy=f"incompatible intent; use {reason}",
                missing_context=[f"incompatible_intent:{reason}"],
                primary_worker="",
                supporting_workers=[],
                worker_payloads={},
            )
            return result

        if mode == "user_list":
            result = await self._run_user_list(task, context, plan)
            return result

        if mode == "relationship":
            result = await self._run_relationship(task, context, plan)
            return result

        if mode == "lookup":
            result = await self._run_lookup(task, context, plan)
            return result

        if mode == "lookup_profile":
            result = await self._run_lookup_profile(task, context, plan)
            return result

        result = await self._run_profile(task, context, plan)
        return result

    async def _run_lookup(
        self,
        task: str,
        context: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve identity by display name only."""
        lookup_result = await self.lookup_agent.run(task, context, max_attempts=1)
        ref = _ref_from_lookup_result(lookup_result)
        resolved_refs = [ref] if ref is not None else []
        summary = self._summary_from_ref(ref)
        resolved = bool(lookup_result.get("resolved")) and ref is not None
        missing_context = [] if resolved else ["person_ref"]
        projection_payload = {
            "profile_kind": "third_party",
            "owner_global_user_id": text_or_empty(ref.get("global_user_id")) if ref else "",
            "summary": summary,
        }
        result = self._final_result(
            resolved=resolved,
            selected_summary=summary,
            primary_worker="user_lookup_agent",
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=resolved_refs,
            projection_payload=projection_payload,
            worker_payloads={"user_lookup_agent": lookup_result},
            evidence=[summary] if summary else [],
            missing_context=missing_context,
        )
        return result

    async def _run_lookup_profile(
        self,
        task: str,
        context: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the approved display-name -> profile worker chain."""
        lookup_result = await self.lookup_agent.run(task, context, max_attempts=1)
        ref = _ref_from_lookup_result(lookup_result)
        if ref is None:
            result = self._unresolved(
                source_policy="display-name lookup returned no person ref",
                missing_context=["person_ref"],
                primary_worker="user_lookup_agent",
                supporting_workers=[],
                worker_payloads={"user_lookup_agent": lookup_result},
            )
            return result

        profile_context = _context_with_ref(context, ref)
        profile_result = await self.profile_agent.run(
            task,
            profile_context,
            max_attempts=1,
        )
        worker_payloads = {
            "user_lookup_agent": lookup_result,
            "user_profile_agent": profile_result,
        }
        result = self._profile_result(
            profile_result=profile_result,
            ref=ref,
            requested_target="known_ref",
            primary_worker="user_profile_agent",
            supporting_workers=["user_lookup_agent"],
            source_policy=text_or_empty(plan["reason"]),
            worker_payloads=worker_payloads,
        )
        return result

    async def _run_profile(
        self,
        task: str,
        context: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a direct profile worker request for a known person target."""
        target = text_or_empty(plan["target"])
        if target == "current_user":
            ref = _current_user_ref(context)
        elif target == "active_character":
            ref = _active_character_ref(context)
        else:
            raw_ref = _first_person_ref(context)
            if raw_ref is None:
                result = self._unresolved(
                    source_policy="profile request needs a structured person ref",
                    missing_context=["person_ref"],
                    primary_worker="user_profile_agent",
                    supporting_workers=[],
                    worker_payloads={},
                )
                return result
            ref = raw_ref

        profile_context = _context_with_ref(context, ref)
        profile_result = await self.profile_agent.run(
            task,
            profile_context,
            max_attempts=1,
        )
        result = self._profile_result(
            profile_result=profile_result,
            ref=ref,
            requested_target=target,
            primary_worker="user_profile_agent",
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            worker_payloads={"user_profile_agent": profile_result},
        )
        return result

    async def _run_user_list(
        self,
        task: str,
        context: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Run display-name predicate enumeration."""
        worker_result = await self.user_list_agent.run(task, context, max_attempts=1)
        raw_result = worker_result.get("result")
        summary = _summary_from_people_payload(raw_result)
        resolved = bool(worker_result.get("resolved")) and bool(summary)
        missing_context = [] if resolved else ["person_context"]
        projection_payload = {
            "profile_kind": "user_list",
            "owner_global_user_id": "",
            "summary": summary,
        }
        result = self._final_result(
            resolved=resolved,
            selected_summary=summary,
            primary_worker="user_list_agent",
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=[],
            projection_payload=projection_payload,
            worker_payloads={"user_list_agent": worker_result},
            evidence=[summary] if summary else [],
            missing_context=missing_context,
        )
        return result

    async def _run_relationship(
        self,
        task: str,
        context: dict[str, Any],
        plan: dict[str, Any],
    ) -> dict[str, Any]:
        """Run relationship ranking evidence lookup."""
        worker_result = await self.relationship_agent.run(
            task,
            context,
            max_attempts=1,
        )
        raw_result = worker_result.get("result")
        summary = _summary_from_people_payload(raw_result)
        resolved = bool(worker_result.get("resolved")) and bool(summary)
        missing_context = [] if resolved else ["person_context"]
        projection_payload = {
            "profile_kind": "relationship",
            "owner_global_user_id": "",
            "summary": summary,
        }
        result = self._final_result(
            resolved=resolved,
            selected_summary=summary,
            primary_worker="relationship_agent",
            supporting_workers=[],
            source_policy=text_or_empty(plan["reason"]),
            resolved_refs=[],
            projection_payload=projection_payload,
            worker_payloads={"relationship_agent": worker_result},
            evidence=[summary] if summary else [],
            missing_context=missing_context,
        )
        return result

    def _profile_result(
        self,
        *,
        profile_result: dict[str, Any],
        ref: dict[str, Any],
        requested_target: str,
        primary_worker: str,
        supporting_workers: list[str],
        source_policy: str,
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the result payload for a profile worker response."""
        profile = profile_result.get("result")
        summary = _summary_from_profile(profile)
        owner_global_user_id = _owner_id_from_profile(profile, ref)
        profile_kind = _profile_kind_for_ref(ref, requested_target)
        resolved = bool(profile_result.get("resolved")) and bool(profile)
        missing_context = [] if resolved else ["person_profile"]
        profile_ref = _person_ref(
            role=text_or_empty(ref.get("role")) or "profile_owner",
            global_user_id=owner_global_user_id,
            display_name=ref.get("display_name"),
        )
        projection_payload = {
            "profile_kind": profile_kind,
            "owner_global_user_id": owner_global_user_id,
            "profile": profile,
            "summary": summary,
        }
        result = self._final_result(
            resolved=resolved,
            selected_summary=summary,
            primary_worker=primary_worker,
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=[profile_ref],
            projection_payload=projection_payload,
            worker_payloads=worker_payloads,
            evidence=[summary] if summary else [],
            missing_context=missing_context,
        )
        return result

    def _summary_from_ref(self, ref: dict[str, Any] | None) -> str:
        """Build a display summary from an identity reference."""
        if ref is None:
            return_value = ""
            return return_value
        display_name = text_or_empty(ref.get("display_name"))
        global_user_id = text_or_empty(ref.get("global_user_id"))
        parts = [part for part in (display_name, global_user_id) if part]
        summary = " | ".join(parts)
        return summary

    def _final_result(
        self,
        *,
        resolved: bool,
        selected_summary: str,
        primary_worker: str,
        supporting_workers: list[str],
        source_policy: str,
        resolved_refs: list[dict[str, Any]],
        projection_payload: dict[str, Any],
        worker_payloads: dict[str, Any],
        evidence: list[str],
        missing_context: list[str],
    ) -> dict[str, Any]:
        """Build and log a final person-context result."""
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker=primary_worker,
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=resolved_refs,
            projection_payload=projection_payload,
            worker_payloads=worker_payloads,
            evidence=evidence,
            missing_context=missing_context,
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs={resolved_refs} "
            f"projection_payload={projection_payload} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result

    def _unresolved(
        self,
        *,
        source_policy: str,
        missing_context: list[str],
        primary_worker: str,
        supporting_workers: list[str],
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result without calling another source."""
        payload = _result_payload(
            selected_summary="",
            primary_worker=primary_worker,
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={"profile_kind": "third_party", "summary": ""},
            worker_payloads=worker_payloads,
            evidence=[],
            missing_context=missing_context,
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=False "
            f"primary_worker={primary_worker} "
            f"missing_context={missing_context} selected_summary='' "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] "
            f"projection_payload={payload['projection_payload']} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=False, payload=payload)
        return result
