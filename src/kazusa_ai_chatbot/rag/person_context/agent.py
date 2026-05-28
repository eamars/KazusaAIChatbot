"""Top-level RAG capability agent for person context."""

from __future__ import annotations

import logging
from typing import Any

from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.person_context.contracts import (
    _AGENT_NAME,
    _UNCACHED_REASON,
    _agent_result,
    _result_payload,
)
from kazusa_ai_chatbot.rag.person_context.projection import (
    _active_character_ref,
    _context_with_ref,
    _current_user_ref,
    _first_person_ref,
    _owner_id_from_profile,
    _person_ref,
    _profile_kind_for_ref,
    _ref_from_lookup_result,
    _summary_from_people_payload,
    _summary_from_profile,
)
from kazusa_ai_chatbot.rag.person_context.selector import _select_plan
from kazusa_ai_chatbot.rag.person_context.workers.list import UserListAgent
from kazusa_ai_chatbot.rag.person_context.workers.lookup import UserLookupAgent
from kazusa_ai_chatbot.rag.person_context.workers.profile import UserProfileAgent
from kazusa_ai_chatbot.rag.person_context.workers.relationship import RelationshipAgent
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

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
