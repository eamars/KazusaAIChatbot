"""Resolver-local subagent implementations for complex task resolution."""

from __future__ import annotations

import logging

from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3

from .contracts import (
    COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
    ComplexTaskSubagentRequestV1,
    ComplexTaskSubagentResultV1,
    validate_complex_task_subagent_result,
)

logger = logging.getLogger(__name__)


class ComplexTaskEvidenceSubagent:
    """Evidence subagent boundary for resolver-owned evidence attempts."""

    def __init__(self, web_agent: object | None = None) -> None:
        """Create an evidence subagent backed by the production web helper.

        Args:
            web_agent: Optional WebAgent3-compatible object used by focused
                tests. Production callers leave this unset.
        """

        if web_agent is None:
            web_agent = WebAgent3()
        self._web_agent = web_agent

    async def run(
        self,
        request: ComplexTaskSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> ComplexTaskSubagentResultV1:
        """Collect bounded web evidence through the WebAgent3 public IO."""

        if max_attempts < 0:
            raise ValueError("max_attempts: expected non-negative integer")
        task = _evidence_task(request)
        web_context = _web_context_from_resolver_context(context)
        web_context["original_query"] = task
        try:
            web_result = await self._web_agent.run(
                task,
                web_context,
                max_attempts=max_attempts,
            )
        except Exception as exc:
            logger.exception(f"WebAgent3 evidence call failed: {exc}")
            result = _dependency_failure_result(
                request=request,
                task=task,
                max_attempts=max_attempts,
                reason=str(exc),
                status="failed",
            )
            validated_result = validate_complex_task_subagent_result(result)
            return validated_result

        if not isinstance(web_result, dict):
            raise ValueError("web_agent result: expected object")
        resolved = web_result.get("resolved")
        if not isinstance(resolved, bool):
            raise ValueError("web_agent result.resolved: expected boolean")
        raw_attempts = web_result.get("attempts", 0)
        if not isinstance(raw_attempts, int) or raw_attempts < 0:
            raise ValueError("web_agent result.attempts: expected non-negative int")
        raw_cache = web_result.get("cache", {"enabled": False})
        if not isinstance(raw_cache, dict):
            raise ValueError("web_agent result.cache: expected object")
        summary = str(web_result.get("result", ""))
        status = "resolved" if resolved else _unresolved_status(raw_attempts)
        result: ComplexTaskSubagentResultV1 = {
            "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
            "resolved": resolved,
            "status": status,
            "result": {"summary": summary},
            "attempts": raw_attempts,
            "cache": raw_cache,
            "trace": {
                "node_id": request["node_id"],
                "web_agent3": {
                    "task": task,
                    "max_attempts": max_attempts,
                    "resolved": resolved,
                    "attempts": raw_attempts,
                    "local_time_context": web_context.get(
                        "local_time_context",
                        {},
                    ),
                },
            },
            "unresolved_items": [] if resolved else [summary],
        }
        validated_result = validate_complex_task_subagent_result(result)
        return validated_result


class UnavailableEvidenceSubagent(ComplexTaskEvidenceSubagent):
    """Compatibility name for unavailable review evidence environments."""

    def __init__(self, reason: str) -> None:
        """Create an explicit unavailable evidence provider."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason: expected non-empty string")
        self._reason = reason

    async def run(
        self,
        request: ComplexTaskSubagentRequestV1,
        context: dict[str, object],
        max_attempts: int = 1,
    ) -> ComplexTaskSubagentResultV1:
        """Return a bounded dependency-blocker envelope."""

        del context
        result = _dependency_failure_result(
            request=request,
            task=_evidence_task(request),
            max_attempts=max_attempts,
            reason=self._reason,
            status="unavailable",
        )
        validated_result = validate_complex_task_subagent_result(result)
        return validated_result


def _evidence_task(request: ComplexTaskSubagentRequestV1) -> str:
    """Return the natural-language WebAgent3 task for an evidence request."""

    task_parts = [f"Objective: {request['objective']}"]
    source_hints = _evidence_source_hints(request)
    for source_hint in source_hints:
        task_parts.append(f"Preferred source: {source_hint}")

    payload_query = request["payload"].get("query")
    if isinstance(payload_query, str) and payload_query.strip():
        task_parts.append(f"Search query: {payload_query.strip()}")

    target_date = request["payload"].get("target_date")
    if not isinstance(target_date, str) or not target_date.strip():
        target_date = request["payload"].get("as_of_date")
    if isinstance(target_date, str) and target_date.strip():
        task_parts.append(f"As-of date: {target_date.strip()}")

    task = "\n".join(task_parts)
    return task


def _evidence_source_hints(
    request: ComplexTaskSubagentRequestV1,
) -> list[str]:
    """Return source hints declared by the evidence request payload."""

    source_hints: list[str] = []
    for container in (request["payload"], request["constraints"]):
        for key in (
            "source_hint",
            "source_url",
            "source_urls",
            "source_preference",
            "source_preferences",
            "official_url",
            "official_urls",
            "target_url",
            "target_urls",
            "target_domains",
        ):
            raw_value = container.get(key)
            source_hints.extend(_source_hint_rows(raw_value))
    deduped_hints: list[str] = []
    for source_hint in source_hints:
        if source_hint not in deduped_hints:
            deduped_hints.append(source_hint)
    return_value = deduped_hints
    return return_value


def _source_hint_rows(value: object) -> list[str]:
    """Normalize source hints while preserving their declared meaning."""

    rows: list[str] = []
    if isinstance(value, str) and value.strip():
        rows.append(_normalize_source_hint(value.strip()))
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                rows.append(_normalize_source_hint(item.strip()))
    return rows


def _normalize_source_hint(value: str) -> str:
    """Make URL-like source hints directly readable by WebAgent3."""

    has_scheme = value.startswith(("http://", "https://"))
    if has_scheme or "." not in value or " " in value:
        return_value = value
        return return_value
    return_value = f"https://{value}"
    return return_value


def _web_context_from_resolver_context(
    context: dict[str, object],
) -> dict[str, object]:
    """Adapt resolver context to the WebAgent3 public context contract."""

    web_context = dict(context)
    local_time_context = web_context.get("local_time_context")
    if isinstance(local_time_context, dict) and local_time_context:
        return_value = web_context
        return return_value

    time_context = context.get("time_context")
    if not isinstance(time_context, dict):
        return_value = web_context
        return return_value

    mapped_time_context: dict[str, str] = {}
    current_local_datetime = time_context.get("current_local_datetime")
    if isinstance(current_local_datetime, str) and current_local_datetime.strip():
        mapped_time_context["current_local_datetime"] = (
            current_local_datetime.strip()
        )
    current_date = time_context.get("current_date")
    if (
        "current_local_datetime" not in mapped_time_context
        and isinstance(current_date, str)
        and current_date.strip()
    ):
        mapped_time_context["current_local_datetime"] = current_date.strip()
    current_local_weekday = time_context.get("current_local_weekday")
    if isinstance(current_local_weekday, str) and current_local_weekday.strip():
        mapped_time_context["current_local_weekday"] = (
            current_local_weekday.strip()
        )
    current_weekday = time_context.get("current_weekday")
    if (
        "current_local_weekday" not in mapped_time_context
        and isinstance(current_weekday, str)
        and current_weekday.strip()
    ):
        mapped_time_context["current_local_weekday"] = current_weekday.strip()

    if mapped_time_context:
        web_context["local_time_context"] = mapped_time_context
    return_value = web_context
    return return_value


def _unresolved_status(attempts: int) -> str:
    """Map unresolved WebAgent3 results into resolver subagent status."""

    if attempts == 0:
        return "unavailable"
    return "partial"


def _dependency_failure_result(
    *,
    request: ComplexTaskSubagentRequestV1,
    task: str,
    max_attempts: int,
    reason: str,
    status: str,
) -> ComplexTaskSubagentResultV1:
    """Build a bounded evidence dependency failure envelope."""

    attempts = 0 if max_attempts == 0 else 1
    result: ComplexTaskSubagentResultV1 = {
        "schema_version": COMPLEX_TASK_SUBAGENT_RESULT_VERSION,
        "resolved": False,
        "status": status,
        "result": {},
        "attempts": attempts,
        "cache": {
            "enabled": False,
            "hit": False,
            "cache_name": "web_agent3",
            "reason": "dependency unavailable",
        },
        "trace": {
            "node_id": request["node_id"],
            "web_agent3": {
                "task": task,
                "max_attempts": max_attempts,
                "resolved": False,
                "attempts": attempts,
                "blocker": reason,
            },
        },
        "unresolved_items": [reason],
    }
    return result
