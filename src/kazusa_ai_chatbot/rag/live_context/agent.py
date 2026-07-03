"""Top-level RAG capability agent for live context."""

from __future__ import annotations

import logging
from typing import Any

from kazusa_ai_chatbot.rag.conversation_evidence.workers.search import ConversationSearchAgent
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.live_context.runtime_facts import (
    _runtime_selected_summary,
    _validated_runtime_time_context,
)
from kazusa_ai_chatbot.rag.live_context.selector import (
    _deterministic_plan,
    _select_external_live_plan,
)
from kazusa_ai_chatbot.rag.live_context.target_resolution import (
    _extract_worker_text,
    _location_ref,
    _url_from_text,
    _web_task,
)
from kazusa_ai_chatbot.rag.memory_evidence.workers.persistent_search import PersistentMemorySearchAgent
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3
from kazusa_ai_chatbot.utils import text_or_empty

logger = logging.getLogger(__name__)

_CAPABILITY_NAME = "live_context"

_AGENT_NAME = "live_context_agent"

_UNCACHED_REASON = "capability_orchestrator_uncached"

def _clip_text(value: object, *, limit: int = 1000) -> str:
    """Return compact text for prompt-facing evidence fields.

    Args:
        value: Text-like evidence from a worker result.
        limit: Maximum output length.

    Returns:
        Stripped and clipped text.
    """

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
    """Build the standard top-level capability payload.

    Args:
        selected_summary: Short factual summary selected for the evaluator.
        primary_worker: Worker/tool that produced the final live evidence.
        supporting_workers: Optional target/scope resolver workers.
        source_policy: Short authority and freshness explanation.
        resolved_refs: Structured refs available to later ordered slots.
        projection_payload: Public projection adapter payload.
        worker_payloads: Raw worker outputs for debug tracing.
        evidence: Prompt-facing evidence strings.
        missing_context: Missing scope fields for unresolved results.
        conflicts: Source conflicts discovered while resolving the target.

    Returns:
        Result payload accepted by the RAG2 evaluator and projection adapter.
    """

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

class LiveContextAgent(BaseRAGHelperAgent):
    """Top-level RAG helper for runtime and external live facts."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached live-context capability agent.

        Args:
            cache_runtime: Optional cache runtime override for tests.
        """

        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )
        self.web_agent = WebAgent3(cache_runtime=cache_runtime)
        self.memory_search_agent = PersistentMemorySearchAgent(
            cache_runtime=cache_runtime
        )
        self.conversation_search_agent = ConversationSearchAgent(
            cache_runtime=cache_runtime
        )

    def _resolved_runtime_result(
        self,
        *,
        fact_type: str,
        time_context: dict[str, str],
    ) -> dict[str, Any]:
        """Build the final result for a runtime-backed live fact.

        Args:
            fact_type: Runtime-backed live fact type.
            time_context: Validated sanitized time context.

        Returns:
            Standard live-context capability result.
        """

        selected_summary = _runtime_selected_summary(fact_type, time_context)
        projection_payload = {
            "external_text": selected_summary,
            "url": "",
        }
        worker_payloads = {
            "runtime_context_provider": time_context,
        }
        payload = _result_payload(
            selected_summary=selected_summary,
            primary_worker="runtime_context_provider",
            supporting_workers=[],
            source_policy="current-turn runtime state",
            resolved_refs=[],
            projection_payload=projection_payload,
            worker_payloads=worker_payloads,
            evidence=[selected_summary],
            missing_context=[],
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=True "
            f"primary_worker=runtime_context_provider "
            f"missing_context={payload['missing_context']} "
            f"selected_summary={payload['selected_summary']} "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] "
            f"projection_payload={projection_payload} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=True, payload=payload)
        return result

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Resolve one present-tense fact from runtime state or live evidence.

        Args:
            task: ``Live-context:`` slot text from the RAG2 dispatcher.
            context: Existing RAG2 delegate context.
            max_attempts: Accepted for interface compatibility; this agent
                performs one bounded orchestration attempt.

        Returns:
            Standard RAG helper result using top-level capability payload keys.
        """

        del max_attempts

        plan = _deterministic_plan(task)
        if plan is None:
            plan = await _select_external_live_plan(task, context)
        fact_type = plan["fact_type"]
        source_class = plan["source_class"]
        runtime_scope = plan["runtime_scope"]
        target_source = plan["target_source"]
        target = text_or_empty(plan["target"])
        missing_context = list(plan["missing_context"])
        supporting_workers: list[str] = []
        worker_payloads: dict[str, Any] = {}
        resolved_refs: list[dict[str, Any]] = []
        source_policy = ""

        if source_class == "runtime_snapshot":
            if runtime_scope == "current_user":
                user_time_context = _validated_runtime_time_context(
                    context.get("user_time_context")
                )
                if user_time_context is None:
                    result = self._missing_target_result(
                        source_policy=(
                            "current-turn runtime state missing or malformed "
                            "user-local time context"
                        ),
                        missing_context=["user_time_context"],
                        supporting_workers=supporting_workers,
                        worker_payloads=worker_payloads,
                    )
                    return result
                result = self._resolved_runtime_result(
                    fact_type=fact_type,
                    time_context=user_time_context,
                )
                return result

            local_time_context = _validated_runtime_time_context(
                context.get("local_time_context")
            )
            if local_time_context is None:
                result = self._missing_target_result(
                    source_policy=(
                        "current-turn runtime state missing or malformed "
                        "character-local time context"
                    ),
                    missing_context=["local_time_context"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            result = self._resolved_runtime_result(
                fact_type=fact_type,
                time_context=local_time_context,
            )
            return result

        if source_class != "external_live_lookup":
            raise ValueError(
                f"Unsupported live context source_class: {source_class}"
            )

        if target_source == "explicit" and target:
            resolved_refs.append(
                _location_ref(role="target_location", text=target)
            )
            source_policy = "explicit target supplied by Live-context slot"
        elif target_source == "active_character_default":
            target_result = await self._resolve_character_target(task, context)
            worker_payloads["persistent_memory_search_agent"] = target_result
            supporting_workers.append("persistent_memory_search_agent")
            if not bool(target_result.get("resolved")):
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup failed for active character "
                        "default location"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            target = _extract_worker_text(target_result)
            if not target:
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup returned no active character "
                        "location text"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            resolved_refs.append(_location_ref(role="character_default", text=target))
            source_policy = (
                "target_scope_lookup: persistent memory resolved stable "
                "active character target; web resolves the live value"
            )
        elif target_source == "current_user_recent":
            target_result = await self._resolve_user_recent_target(task, context)
            worker_payloads["conversation_search_agent"] = target_result
            supporting_workers.append("conversation_search_agent")
            if not bool(target_result.get("resolved")):
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup failed for user_recent location; "
                        "no character fallback was attempted"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            target = _extract_worker_text(target_result)
            if not target:
                result = self._missing_target_result(
                    source_policy=(
                        "target_scope_lookup returned no user_recent "
                        "location text; no character fallback was attempted"
                    ),
                    missing_context=["location"],
                    supporting_workers=supporting_workers,
                    worker_payloads=worker_payloads,
                )
                return result
            resolved_refs.append(_location_ref(role="user_recent", text=target))
            source_policy = (
                "target_scope_lookup: recent same-user conversation resolved "
                "user_recent target; web resolves the live value"
            )
        else:
            if not missing_context:
                missing_context = ["location"]
            result = self._missing_target_result(
                source_policy="live target/scope unresolved",
                missing_context=missing_context,
                supporting_workers=supporting_workers,
                worker_payloads=worker_payloads,
            )
            return result

        web_task = _web_task(
            fact_type=fact_type,
            target=target,
            original_task=task,
        )
        web_result = await self.web_agent.run(
            web_task,
            context,
            max_attempts=1,
        )
        worker_payloads["web_agent3"] = web_result
        evidence_text = _clip_text(web_result.get("result"))
        resolved = bool(web_result.get("resolved")) and bool(evidence_text)
        projection_payload = {
            "external_text": evidence_text,
            "url": _url_from_text(evidence_text),
        }
        payload = _result_payload(
            selected_summary=evidence_text,
            primary_worker="web_agent3",
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=resolved_refs,
            projection_payload=projection_payload,
            worker_payloads=worker_payloads,
            evidence=[evidence_text] if evidence_text else [],
            missing_context=[] if resolved else ["live_evidence"],
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"primary_worker=web_agent3 "
            f"missing_context={payload['missing_context']} "
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

    async def _resolve_character_target(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the active character's stable live target from memory."""
        lookup_task = (
            "target_scope_lookup: resolve the active character's stable "
            "location or venue for a live external fact; do not retrieve "
            f"live weather, temperature, schedule, price, or status. {task}"
        )
        result = await self.memory_search_agent.run(
            lookup_task,
            context,
            max_attempts=1,
        )
        return result

    async def _resolve_user_recent_target(
        self,
        task: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Resolve the current user's recently stated live target."""
        lookup_task = (
            "target_scope_lookup: find the current user's recently stated "
            "location or venue for a live external fact; same-user scope only; "
            "do not retrieve live weather, temperature, schedule, price, "
            f"or status. {task}"
        )
        result = await self.conversation_search_agent.run(
            lookup_task,
            context,
            max_attempts=1,
        )
        return result

    def _missing_target_result(
        self,
        *,
        source_policy: str,
        missing_context: list[str],
        supporting_workers: list[str],
        worker_payloads: dict[str, Any],
    ) -> dict[str, Any]:
        """Build an unresolved result for missing target/scope."""
        payload = _result_payload(
            selected_summary="",
            primary_worker="",
            supporting_workers=supporting_workers,
            source_policy=source_policy,
            resolved_refs=[],
            projection_payload={},
            worker_payloads=worker_payloads,
            evidence=[],
            missing_context=missing_context,
            conflicts=[],
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved=False primary_worker= "
            f"missing_context={missing_context} selected_summary='' "
            f"cache_reason={_UNCACHED_REASON}"
        )
        logger.debug(
            f"{_AGENT_NAME} debug: resolved_refs=[] projection_payload={{}} "
            f"worker_payloads={worker_payloads}"
        )
        result = _agent_result(resolved=False, payload=payload)
        return result
