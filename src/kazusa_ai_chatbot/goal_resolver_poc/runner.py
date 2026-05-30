"""Execution loop for the goal resolver POC."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from kazusa_ai_chatbot.goal_resolver_poc.casebook import (
    GOAL_RESOLVER_CASES,
    case_by_id,
)
from kazusa_ai_chatbot.goal_resolver_poc.llm import (
    call_case_evaluator,
    call_finalizer,
    call_planner,
    call_verifier,
)
from kazusa_ai_chatbot.goal_resolver_poc.models import (
    CASE_EVALUATION_ACCEPTED_STATUSES,
    CASE_EVALUATOR_STAGE,
    CONTRACT_VALIDATION_STAGE,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_REPAIR_PASSES,
    FINALIZER_STAGE,
    PLANNER_STAGE,
    REQUIREMENT_STATUS_BLOCKED_HUMAN,
    REQUIREMENT_TERMINAL_STATUSES,
    REPAIR_FEEDBACK_STAGE,
    SCHEMA_VERSION,
    TERMINAL_FINAL,
    TERMINAL_MAX_ITERATIONS,
    TERMINAL_NEEDS_HUMAN,
    TERMINAL_PENDING_APPROVAL,
    TOOL_STAGE,
    VERIFIER_STAGE,
    apply_requirement_updates,
    bounded_text,
    first_open_requirement_id,
    initial_requirements,
    merge_planner_requirements,
)
from kazusa_ai_chatbot.goal_resolver_poc.tools import execute_tool
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso


LOCAL_TIME_ZONE = "Pacific/Auckland"
MAX_CONTEXT_EVENTS = 6
MAX_VISIBLE_FILES = 8
MAX_PUBLIC_SEARCH_ITEMS = 4
MAX_PUBLIC_PAGE_EXCERPTS = 2
MAX_PUBLIC_PAGE_EXCERPT_CHARS = 1200
MAX_PUBLIC_CATALOG_ROWS = 4
MAX_PUBLIC_CATALOG_PRODUCTS = 2
MAX_HARDWARE_EVIDENCE_ROWS = 4
MAX_RESTAURANT_DIRECTORY_ROWS = 4
MAX_GITHUB_RELEASE_ROWS = 4


def _local_time_context() -> dict[str, str]:
    """Return local time context for time-sensitive resolver cases."""

    local_now = datetime.now(ZoneInfo(LOCAL_TIME_ZONE))
    context = {
        "timezone": LOCAL_TIME_ZONE,
        "current_local_datetime": local_now.isoformat(timespec="seconds"),
        "current_local_weekday": local_now.strftime("%A"),
        "current_utc_datetime": storage_utc_now_iso(),
    }
    return context


def _new_state(case: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Create a resolver state object for one case run.

    Args:
        case: Case metadata and natural user input.
        output_dir: Artifact directory used for sandbox state.

    Returns:
        Mutable serializable state for the resolver loop.
    """

    case_id = case["case_id"]
    started_at_utc = storage_utc_now_iso()
    run_token = (
        started_at_utc.replace(":", "")
        .replace(".", "")
        .replace("+", "")
        .replace("-", "")
    )
    sandbox_root = output_dir / "sandboxes" / f"{case_id}_{run_token}"
    state = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "title": case["title"],
        "user_input": case["user_input"],
        "resolver_contract": case["resolver_contract"],
        "valid_terminal_modes": case["valid_terminal_modes"],
        "context_hints": case["context_hints"],
        "started_at_utc": started_at_utc,
        "local_time_context": _local_time_context(),
        "sandbox_root": str(sandbox_root),
        "iteration": 0,
        "goal_frame": "",
        "requirements": initial_requirements(case),
        "known_facts": [],
        "tool_history": [],
        "verifier_history": [],
        "events": [],
        "decision": "",
        "human_request": "",
        "action_candidates": [],
        "final_answer": "",
        "terminal_mode": "",
        "loop_stop_reason": "",
        "terminal_contract_valid": False,
        "repair_feedback": [],
    }
    return state


def _event(
    *,
    iteration: int,
    stage: str,
    summary: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build one serializable loop event."""

    event = {
        "iteration": iteration,
        "stage": stage,
        "summary": bounded_text(summary, limit=800),
        "payload": payload,
    }
    return event


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded payload view for LLM-to-LLM handoff."""

    compact: dict[str, Any] = {}
    for key in [
        "query",
        "answer",
        "result",
        "raw_search_fallback",
        "public_search_fallback_text",
        "stdout",
        "stderr",
        "reason",
        "new_content_preview",
    ]:
        value = payload.get(key)
        if value:
            compact[key] = bounded_text(value, limit=1800)

    for key in [
        "resolved",
        "attempts",
        "loop_count",
        "returncode",
        "command",
        "sandbox_root",
        "artifact_root",
        "action_candidate",
        "question",
    ]:
        if key in payload:
            compact[key] = payload[key]

    files = payload.get("files")
    if isinstance(files, list):
        compact_files = []
        for item in files[:MAX_VISIBLE_FILES]:
            if not isinstance(item, dict):
                continue
            compact_file = {
                "path": bounded_text(item.get("path", ""), limit=240),
                "content": bounded_text(item.get("content", ""), limit=1800),
            }
            compact_files.append(compact_file)
        compact["files"] = compact_files

    known_facts = payload.get("known_facts")
    if isinstance(known_facts, list):
        compact["known_facts"] = [
            bounded_text(item, limit=900)
            for item in known_facts[:MAX_VISIBLE_FILES]
        ]

    unknown_slots = payload.get("unknown_slots")
    if isinstance(unknown_slots, list):
        compact["unknown_slots"] = [
            bounded_text(item, limit=500)
            for item in unknown_slots[:MAX_VISIBLE_FILES]
        ]

    public_catalog_fallback = payload.get("public_catalog_fallback")
    if isinstance(public_catalog_fallback, list):
        compact["public_catalog_fallback"] = _compact_public_catalog(
            public_catalog_fallback
        )

    hardware_catalog_evidence = payload.get("hardware_catalog_evidence")
    if isinstance(hardware_catalog_evidence, dict):
        compact["hardware_catalog_evidence"] = _compact_hardware_catalog(
            hardware_catalog_evidence
        )

    restaurant_directory_fallback = payload.get("restaurant_directory_fallback")
    if isinstance(restaurant_directory_fallback, list):
        compact["restaurant_directory_fallback"] = (
            _compact_restaurant_directory(restaurant_directory_fallback)
        )

    github_release_fallback = payload.get("github_release_fallback")
    if isinstance(github_release_fallback, list):
        compact["github_release_fallback"] = _compact_github_releases(
            github_release_fallback
        )

    public_search_items = payload.get("public_search_items")
    if isinstance(public_search_items, dict):
        compact["public_search_items"] = _compact_public_search_items(
            public_search_items
        )

    public_page_excerpts = payload.get("public_page_excerpts")
    if isinstance(public_page_excerpts, list):
        compact["public_page_excerpts"] = _compact_public_page_excerpts(
            public_page_excerpts
        )

    public_search_fallback = payload.get("public_search_fallback")
    if isinstance(public_search_fallback, dict):
        compact["public_search_fallback"] = {
            "status": bounded_text(
                public_search_fallback.get("status", ""),
                limit=80,
            ),
            "url": bounded_text(
                public_search_fallback.get("url", ""),
                limit=300,
            ),
            "text": bounded_text(
                public_search_fallback.get("text", ""),
                limit=1200,
            ),
            "error": bounded_text(
                public_search_fallback.get("error", ""),
                limit=300,
            ),
        }

    if not compact:
        compact["payload_excerpt"] = bounded_text(payload, limit=2500)
    return compact


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Build a compact tool observation for model context."""

    compact = {
        "observation_id": observation["observation_id"],
        "tool": observation["tool"],
        "target_requirement_id": observation["target_requirement_id"],
        "status": observation["status"],
        "summary": bounded_text(observation["summary"], limit=600),
        "payload": _compact_payload(observation["payload"]),
    }
    return compact


def _compact_public_search_items(value: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded public search item view for model handoff."""

    items = value.get("items")
    compact_items = []
    if isinstance(items, list):
        for item in items[:MAX_PUBLIC_SEARCH_ITEMS]:
            if not isinstance(item, dict):
                continue
            compact_item = {
                "title": bounded_text(item.get("title", ""), limit=180),
                "url": bounded_text(item.get("url", ""), limit=300),
                "description": bounded_text(
                    item.get("description", ""),
                    limit=300,
                ),
                "published_at": bounded_text(
                    item.get("published_at", ""),
                    limit=80,
                ),
            }
            compact_items.append(compact_item)
    compact = {
        "url": bounded_text(value.get("url", ""), limit=300),
        "items": compact_items,
        "error": bounded_text(value.get("error", ""), limit=200),
    }
    return compact


def _compact_public_page_excerpts(value: list[Any]) -> list[dict[str, str]]:
    """Build bounded public page excerpts for model handoff."""

    compact_rows: list[dict[str, str]] = []
    for item in value[:MAX_PUBLIC_PAGE_EXCERPTS]:
        if not isinstance(item, dict):
            continue
        compact_row = {
            "title": bounded_text(item.get("title", ""), limit=180),
            "url": bounded_text(item.get("url", ""), limit=300),
            "error": bounded_text(item.get("error", ""), limit=200),
            "excerpt": bounded_text(
                item.get("excerpt", ""),
                limit=MAX_PUBLIC_PAGE_EXCERPT_CHARS,
            ),
        }
        compact_rows.append(compact_row)
    return compact_rows


def _compact_public_catalog(value: list[Any]) -> list[dict[str, Any]]:
    """Build bounded product-catalog rows for model handoff."""

    compact_rows: list[dict[str, Any]] = []
    for row in value[:MAX_PUBLIC_CATALOG_ROWS]:
        if not isinstance(row, dict):
            continue
        products = row.get("products")
        compact_products = []
        if isinstance(products, list):
            for product in products[:MAX_PUBLIC_CATALOG_PRODUCTS]:
                if not isinstance(product, dict):
                    continue
                compact_product = {
                    "title": bounded_text(product.get("title", ""), limit=180),
                    "available": product.get("available"),
                    "price": bounded_text(product.get("price", ""), limit=80),
                    "vendor": bounded_text(product.get("vendor", ""), limit=120),
                    "type": bounded_text(product.get("type", ""), limit=120),
                    "url": bounded_text(product.get("url", ""), limit=300),
                }
                compact_products.append(compact_product)
        compact_row = {
            "catalog": bounded_text(row.get("catalog", ""), limit=160),
            "query": bounded_text(row.get("query", ""), limit=160),
            "url": bounded_text(row.get("url", ""), limit=300),
            "error": bounded_text(row.get("error", ""), limit=200),
            "products": compact_products,
        }
        compact_rows.append(compact_row)
    return compact_rows


def _compact_hardware_catalog(
    value: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build bounded hardware evidence rows for model handoff."""

    compact: dict[str, list[dict[str, Any]]] = {}
    for key in [
        "available_ready_systems",
        "available_graphics_products",
        "available_component_products",
    ]:
        rows = value.get(key)
        if not isinstance(rows, list):
            continue
        compact_rows: list[dict[str, Any]] = []
        for row in rows[:MAX_HARDWARE_EVIDENCE_ROWS]:
            if not isinstance(row, dict):
                continue
            compact_row = {
                "title": bounded_text(row.get("title", ""), limit=220),
                "available": row.get("available"),
                "price": bounded_text(row.get("price", ""), limit=80),
                "vendor": bounded_text(row.get("vendor", ""), limit=120),
                "type": bounded_text(row.get("type", ""), limit=140),
                "query": bounded_text(row.get("query", ""), limit=120),
                "url": bounded_text(row.get("url", ""), limit=320),
            }
            compact_rows.append(compact_row)
        if compact_rows:
            compact[key] = compact_rows
    return compact


def _compact_restaurant_directory(value: list[Any]) -> list[dict[str, str]]:
    """Build bounded restaurant-directory rows for model handoff."""

    compact_rows: list[dict[str, str]] = []
    for row in value[:MAX_RESTAURANT_DIRECTORY_ROWS]:
        if not isinstance(row, dict):
            continue
        compact_row = {
            "source": bounded_text(row.get("source", ""), limit=100),
            "name": bounded_text(row.get("name", ""), limit=140),
            "rating": bounded_text(row.get("rating", ""), limit=40),
            "reviews": bounded_text(row.get("reviews", ""), limit=40),
            "weekday": bounded_text(row.get("weekday", ""), limit=40),
            "weekday_hours": bounded_text(
                row.get("weekday_hours", ""),
                limit=160,
            ),
            "address": bounded_text(row.get("address", ""), limit=200),
            "url": bounded_text(row.get("url", ""), limit=300),
            "error": bounded_text(row.get("error", ""), limit=180),
        }
        compact_rows.append(compact_row)
    return compact_rows


def _compact_github_releases(value: list[Any]) -> list[dict[str, Any]]:
    """Build bounded GitHub release rows for model handoff."""

    compact_rows: list[dict[str, Any]] = []
    for row in value[:MAX_GITHUB_RELEASE_ROWS]:
        if not isinstance(row, dict):
            continue
        releases = row.get("releases")
        compact_releases = []
        if isinstance(releases, list):
            for release in releases[:MAX_GITHUB_RELEASE_ROWS]:
                if not isinstance(release, dict):
                    continue
                compact_releases.append(
                    {
                        "tag_name": bounded_text(
                            release.get("tag_name", ""),
                            limit=80,
                        ),
                        "name": bounded_text(release.get("name", ""), limit=120),
                        "published_at": bounded_text(
                            release.get("published_at", ""),
                            limit=80,
                        ),
                        "html_url": bounded_text(
                            release.get("html_url", ""),
                            limit=260,
                        ),
                        "draft": bool(release.get("draft")),
                        "prerelease": bool(release.get("prerelease")),
                    }
                )
        compact_row = {
            "source": bounded_text(row.get("source", ""), limit=120),
            "requested_repo": bounded_text(
                row.get("requested_repo", ""),
                limit=140,
            ),
            "url": bounded_text(row.get("url", ""), limit=260),
            "error": bounded_text(row.get("error", ""), limit=180),
            "releases": compact_releases,
        }
        compact_rows.append(compact_row)
    return compact_rows


def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
    """Build a compact loop event for model context."""

    payload = event["payload"]
    if isinstance(payload, dict) and "observation_id" in payload:
        payload_view = _compact_observation(payload)
    elif event["stage"] == PLANNER_STAGE and isinstance(payload, dict):
        payload_view = {
            "goal_frame": bounded_text(payload.get("goal_frame", ""), limit=500),
            "next_action": payload.get("next_action", {}),
            "open_requirements": payload.get("open_requirements", []),
        }
    elif event["stage"] == VERIFIER_STAGE and isinstance(payload, dict):
        payload_view = {
            "resolved": payload.get("resolved", False),
            "decision": payload.get("decision", ""),
            "confidence": payload.get("confidence", 0.0),
            "remaining_requirements": payload.get("remaining_requirements", []),
            "feedback": bounded_text(payload.get("feedback", ""), limit=900),
            "minimal_human_question": payload.get("minimal_human_question", ""),
        }
    elif event["stage"] == FINALIZER_STAGE and isinstance(payload, dict):
        payload_view = {
            "terminal_mode": payload.get("terminal_mode", ""),
            "final_answer": bounded_text(
                payload.get("final_answer", ""),
                limit=1200,
            ),
        }
    elif event["stage"] == CASE_EVALUATOR_STAGE and isinstance(payload, dict):
        payload_view = {
            "status": payload.get("status", ""),
            "score": payload.get("score", 0),
            "reason": bounded_text(payload.get("reason", ""), limit=1000),
            "missing": payload.get("missing", []),
        }
    else:
        payload_view = {
            "excerpt": bounded_text(payload, limit=1600),
        }
    compact = {
        "iteration": event["iteration"],
        "stage": event["stage"],
        "summary": bounded_text(event["summary"], limit=500),
        "payload": payload_view,
    }
    return compact


def _compact_event_for_evaluation(event: dict[str, Any]) -> dict[str, Any]:
    """Build a small trace row for the independent case evaluator."""

    payload = event["payload"]
    payload_view: dict[str, Any]
    if isinstance(payload, dict) and "observation_id" in payload:
        payload_view = {
            "observation_id": payload["observation_id"],
            "tool": payload["tool"],
            "target_requirement_id": payload["target_requirement_id"],
            "status": payload["status"],
        }
    elif event["stage"] == PLANNER_STAGE and isinstance(payload, dict):
        next_action = payload.get("next_action", {})
        payload_view = {
            "next_action": next_action,
            "open_requirements": payload.get("open_requirements", [])[:6],
        }
    elif event["stage"] == VERIFIER_STAGE and isinstance(payload, dict):
        payload_view = {
            "resolved": payload.get("resolved", False),
            "decision": payload.get("decision", ""),
            "remaining_requirements": payload.get("remaining_requirements", [])[:6],
            "requirement_updates": payload.get("requirement_updates", [])[:8],
            "feedback": bounded_text(payload.get("feedback", ""), limit=700),
        }
    elif event["stage"] == FINALIZER_STAGE and isinstance(payload, dict):
        payload_view = {
            "terminal_mode": payload.get("terminal_mode", ""),
            "final_answer": bounded_text(
                payload.get("final_answer", ""),
                limit=900,
            ),
        }
    elif event["stage"] == CONTRACT_VALIDATION_STAGE and isinstance(payload, dict):
        payload_view = {
            "terminal_mode": payload.get("terminal_mode", ""),
            "non_terminal_requirements": payload.get(
                "non_terminal_requirements",
                [],
            ),
            "status": payload.get("status", ""),
        }
    else:
        payload_view = {"excerpt": bounded_text(payload, limit=900)}

    compact = {
        "iteration": event["iteration"],
        "stage": event["stage"],
        "summary": bounded_text(event["summary"], limit=700),
        "payload": payload_view,
    }
    return compact


def _visible_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build bounded state context for the next LLM stage."""

    tool_history = [
        _compact_observation(observation)
        for observation in state["tool_history"][-MAX_CONTEXT_EVENTS:]
    ]
    verifier_history = [
        {
            "resolved": verifier["resolved"],
            "decision": verifier["decision"],
            "confidence": verifier["confidence"],
            "remaining_requirements": verifier["remaining_requirements"],
            "feedback": bounded_text(verifier["feedback"], limit=900),
            "minimal_human_question": verifier["minimal_human_question"],
        }
        for verifier in state["verifier_history"][-MAX_CONTEXT_EVENTS:]
    ]
    view = {
        "user_input": state["user_input"],
        "local_time_context": state["local_time_context"],
        "goal_frame": state["goal_frame"],
        "requirements": state["requirements"],
        "tool_history": tool_history,
        "verifier_history": verifier_history,
        "action_candidates": state["action_candidates"],
        "human_request": state["human_request"],
        "repair_feedback": state["repair_feedback"],
        "final_answer": bounded_text(state["final_answer"], limit=1800),
        "terminal_mode": state["terminal_mode"],
        "loop_stop_reason": state["loop_stop_reason"],
    }
    return view


def _planner_visible_state(state: dict[str, Any]) -> dict[str, Any]:
    """Build planner context without an echoable requirements schema."""

    view = _visible_state(state)
    requirement_state = []
    for requirement in state["requirements"]:
        requirement_state.append(
            {
                "requirement_id": requirement["requirement_id"],
                "description": bounded_text(
                    requirement["description"],
                    limit=300,
                ),
                "required_evidence_type": bounded_text(
                    requirement["required_evidence_type"],
                    limit=120,
                ),
                "status": requirement["status"],
                "blocking_reason": bounded_text(
                    requirement["blocking_reason"],
                    limit=240,
                ),
                "last_verifier_note": bounded_text(
                    requirement["last_verifier_note"],
                    limit=300,
                ),
            }
        )
    view.pop("requirements", None)
    view["requirement_state"] = requirement_state
    return view


def _normalize_action_target(
    action: dict[str, Any],
    requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Force a planner action to target an existing open requirement."""

    open_ids = {
        requirement["requirement_id"]
        for requirement in requirements
        if requirement["status"] == "open"
    }
    normalized = dict(action)
    if normalized["target_requirement_id"] not in open_ids and open_ids:
        normalized["target_requirement_id"] = first_open_requirement_id(
            requirements
        )
    return normalized


def _suppress_repeated_research_action(
    action: dict[str, Any],
    state: dict[str, Any],
) -> dict[str, Any]:
    """Convert duplicate research calls into synthesis attempts."""

    tool = action["tool"]
    if tool not in {"web_research", "rag_research"}:
        return action

    query = action["query"].strip()
    if not query:
        return action

    for observation in state["tool_history"]:
        if observation["tool"] != tool:
            continue
        observed_query = str(observation["payload"].get("query", "")).strip()
        if observed_query != query:
            continue
        normalized = dict(action)
        normalized["tool"] = "final_answer"
        normalized["query"] = (
            "Synthesize the best current answer from existing observations; "
            "do not repeat the same research query."
        )
        normalized["reason"] = (
            f"{action['reason']} Repeated {tool} query suppressed after an "
            "equivalent observation already exists."
        )
        return normalized

    return action


def _terminal_from_action(tool: str) -> str:
    """Map terminal action tools to terminal modes."""

    terminal_mode = ""
    if tool == "ask_human":
        terminal_mode = TERMINAL_NEEDS_HUMAN
    elif tool == "prepare_action":
        terminal_mode = TERMINAL_PENDING_APPROVAL
    elif tool == "final_answer":
        terminal_mode = TERMINAL_FINAL
    return terminal_mode


def _terminal_from_verifier(verifier: dict[str, Any]) -> str:
    """Map verifier decisions to terminal modes when appropriate."""

    decision = verifier["decision"]
    if decision == "ask_human":
        terminal_mode = TERMINAL_NEEDS_HUMAN
    elif decision == "prepare_action":
        terminal_mode = TERMINAL_PENDING_APPROVAL
    elif decision == "final_answer" and verifier["resolved"]:
        terminal_mode = TERMINAL_FINAL
    else:
        terminal_mode = ""
    return terminal_mode


def _record_observation(state: dict[str, Any], observation: dict[str, Any]) -> None:
    """Append a tool observation and event to state."""

    state["tool_history"].append(observation)
    state["events"].append(
        _event(
            iteration=state["iteration"],
            stage=TOOL_STAGE,
            summary=observation["summary"],
            payload=observation,
        )
    )
    if observation["tool"] == "ask_human":
        state["human_request"] = observation["payload"]["question"]


async def _verify_current_state(state: dict[str, Any]) -> dict[str, Any]:
    """Run the verifier and update requirement state from current evidence."""

    verifier = await call_verifier(_visible_state(state))
    state["requirements"] = apply_requirement_updates(
        state["requirements"],
        verifier["requirement_updates"],
    )
    state["verifier_history"].append(verifier)
    state["events"].append(
        _event(
            iteration=state["iteration"],
            stage=VERIFIER_STAGE,
            summary=verifier["feedback"],
            payload=verifier,
        )
    )
    return verifier


def _non_terminal_requirements(
    requirements: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Return requirement ids that still block a terminal state."""

    blocking_rows: list[dict[str, str]] = []
    for requirement in requirements:
        status = requirement["status"]
        if status in REQUIREMENT_TERMINAL_STATUSES:
            continue
        row = {
            "requirement_id": requirement["requirement_id"],
            "status": status,
            "description": bounded_text(
                requirement["description"],
                limit=240,
            ),
        }
        blocking_rows.append(row)
    return blocking_rows


def _terminal_contract_failure(
    state: dict[str, Any],
    terminal_mode: str,
) -> str:
    """Return a contract failure reason, or an empty string when valid."""

    if terminal_mode not in state["valid_terminal_modes"]:
        failure = (
            f"terminal mode {terminal_mode} is not allowed for "
            f"case {state['case_id']}"
        )
        return failure

    if terminal_mode == TERMINAL_NEEDS_HUMAN:
        if not state["human_request"]:
            failure = "terminal mode needs_human requires a recorded human request"
            return failure
        if not any(
            requirement["status"] == REQUIREMENT_STATUS_BLOCKED_HUMAN
            for requirement in state["requirements"]
        ):
            failure = (
                "terminal mode needs_human requires at least one "
                "blocked_human requirement"
            )
            return failure
        return ""

    blocking_rows = _non_terminal_requirements(state["requirements"])
    if blocking_rows:
        blocking_ids = ", ".join(
            f"{row['requirement_id']}({row['status']})"
            for row in blocking_rows
        )
        failure = (
            f"terminal mode {terminal_mode} is blocked by non-terminal "
            f"requirements: {blocking_ids}"
        )
        return failure

    return ""


def _record_contract_validation(
    state: dict[str, Any],
    *,
    terminal_mode: str,
    status: str,
    summary: str,
) -> None:
    """Record deterministic terminal-contract validation in the trace."""

    state["events"].append(
        _event(
            iteration=state["iteration"],
            stage=CONTRACT_VALIDATION_STAGE,
            summary=summary,
            payload={
                "terminal_mode": terminal_mode,
                "valid_terminal_modes": state["valid_terminal_modes"],
                "non_terminal_requirements": _non_terminal_requirements(
                    state["requirements"]
                ),
                "status": status,
            },
        )
    )


async def _finalize(state: dict[str, Any], terminal_mode: str) -> None:
    """Generate the final user-facing answer for a terminal state."""

    state["terminal_mode"] = terminal_mode
    final_answer = await call_finalizer(_visible_state(state))
    state["final_answer"] = final_answer
    state["events"].append(
        _event(
            iteration=state["iteration"],
            stage=FINALIZER_STAGE,
            summary=bounded_text(final_answer, limit=800),
            payload={
                "terminal_mode": terminal_mode,
                "final_answer": final_answer,
            },
        )
    )


async def _try_finalize(state: dict[str, Any], terminal_mode: str) -> bool:
    """Finalize provisionally, then accept only after verifier re-check."""

    await _finalize(state, terminal_mode)
    await _verify_current_state(state)
    failure = _terminal_contract_failure(state, terminal_mode)
    if failure:
        _record_contract_validation(
            state,
            terminal_mode=terminal_mode,
            status="blocked",
            summary=failure,
        )
        state["terminal_mode"] = ""
        state["final_answer"] = ""
        return False

    return True


async def _evaluate_case(state: dict[str, Any]) -> dict[str, Any]:
    """Run the independent LLM case evaluator."""

    evaluation = await call_case_evaluator(
        {
            "case_id": state["case_id"],
            "user_input": state["user_input"],
            "resolver_contract": state["resolver_contract"],
            "valid_terminal_modes": state["valid_terminal_modes"],
            "terminal_mode": state["terminal_mode"],
            "requirements": state["requirements"],
            "events": [
                _compact_event_for_evaluation(event)
                for event in state["events"]
            ],
            "final_answer": state["final_answer"],
        }
    )
    state["events"].append(
        _event(
            iteration=state["iteration"],
            stage=CASE_EVALUATOR_STAGE,
            summary=evaluation["reason"],
            payload=evaluation,
        )
    )
    return evaluation


def _add_repair_feedback(
    state: dict[str, Any],
    evaluation: dict[str, Any],
    repair_pass: int,
) -> None:
    """Convert evaluator failure feedback into another open requirement."""

    missing_items = evaluation["missing"]
    if not missing_items:
        missing_items = [evaluation["reason"]]
    feedback = {
        "repair_pass": repair_pass,
        "evaluation_status": evaluation["status"],
        "reason": evaluation["reason"],
        "missing": missing_items,
    }
    state["repair_feedback"].append(feedback)
    next_index = len(state["requirements"]) + 1
    for item in missing_items:
        requirement = {
            "requirement_id": f"repair-{next_index:03d}",
            "description": item,
            "required_evidence_type": "evaluator_missing_evidence",
            "status": "open",
            "blocking_reason": "",
            "satisfied_by_observation_ids": [],
            "last_verifier_note": "",
        }
        state["requirements"].append(requirement)
        next_index += 1
    state["terminal_mode"] = ""
    state["final_answer"] = ""
    state["loop_stop_reason"] = ""
    state["terminal_contract_valid"] = False
    state["events"].append(
        _event(
            iteration=state["iteration"],
            stage=REPAIR_FEEDBACK_STAGE,
            summary=evaluation["reason"],
            payload=feedback,
        )
    )


async def _run_loop(
    state: dict[str, Any],
    case: dict[str, Any],
    max_iterations: int,
) -> None:
    """Run planner, tool, and verifier iterations until a terminal state."""

    iteration_limit = state["iteration"] + max_iterations
    while state["iteration"] < iteration_limit and not state["terminal_mode"]:
        state["iteration"] += 1
        planner = await call_planner(_planner_visible_state(state))
        state["goal_frame"] = planner["goal_frame"]
        state["requirements"] = merge_planner_requirements(
            state["requirements"],
            planner["requirements"],
        )
        action = _normalize_action_target(
            planner["next_action"],
            state["requirements"],
        )
        action = _suppress_repeated_research_action(action, state)
        planner["next_action"] = action
        state["events"].append(
            _event(
                iteration=state["iteration"],
                stage=PLANNER_STAGE,
                summary=action["reason"],
                payload=planner,
            )
        )

        action_terminal_mode = _terminal_from_action(action["tool"])
        if action["tool"] == "final_answer":
            finalized = await _try_finalize(state, action_terminal_mode)
            if finalized:
                break
            continue

        observation_id = f"obs-{len(state['tool_history']) + 1:03d}"
        observation = await execute_tool(action, state, case, observation_id)
        _record_observation(state, observation)
        verifier = await _verify_current_state(state)

        if action_terminal_mode:
            finalized = await _try_finalize(state, action_terminal_mode)
            if finalized:
                break
            continue

        verifier_terminal_mode = _terminal_from_verifier(verifier)
        if verifier_terminal_mode:
            finalized = await _try_finalize(state, verifier_terminal_mode)
            if finalized:
                break

    if not state["terminal_mode"]:
        state["loop_stop_reason"] = TERMINAL_MAX_ITERATIONS
        verifier = await _verify_current_state(state)
        verifier_terminal_mode = _terminal_from_verifier(verifier)
        if verifier_terminal_mode:
            finalized = await _try_finalize(state, verifier_terminal_mode)
            if finalized:
                return
        state["terminal_mode"] = TERMINAL_MAX_ITERATIONS
        _record_contract_validation(
            state,
            terminal_mode=TERMINAL_MAX_ITERATIONS,
            status="fail",
            summary="iteration limit reached before a valid terminal state",
        )


async def run_goal_resolver_case_async(
    case: dict[str, Any],
    output_dir: Path,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    repair_passes: int = DEFAULT_REPAIR_PASSES,
) -> dict[str, Any]:
    """Run one resolver case through the POC loop.

    Args:
        case: Case metadata from the casebook.
        output_dir: Artifact directory for run data and sandboxes.
        max_iterations: Maximum planner/tool/verifier iterations per pass.
        repair_passes: Number of evaluator-driven repair passes.

    Returns:
        Full run state including LLM evaluation.
    """

    state = _new_state(case, output_dir)
    evaluation: dict[str, Any] = {}
    for repair_pass in range(repair_passes + 1):
        await _run_loop(state, case, max_iterations)
        evaluation = await _evaluate_case(state)
        terminal_failure = _terminal_contract_failure(
            state,
            state["terminal_mode"],
        )
        state["terminal_contract_valid"] = not terminal_failure
        if not state["terminal_contract_valid"]:
            evaluation = dict(evaluation)
            evaluation["status"] = "fail"
            evaluation["reason"] = terminal_failure
            evaluation["missing"] = [terminal_failure]
            state["events"].append(
                _event(
                    iteration=state["iteration"],
                    stage=CONTRACT_VALIDATION_STAGE,
                    summary=evaluation["reason"],
                    payload={
                        "terminal_mode": state["terminal_mode"],
                        "valid_terminal_modes": state["valid_terminal_modes"],
                        "non_terminal_requirements": (
                            _non_terminal_requirements(state["requirements"])
                        ),
                        "status": "fail",
                    },
                )
            )
        accepted = evaluation["status"] in CASE_EVALUATION_ACCEPTED_STATUSES
        if accepted and state["terminal_contract_valid"]:
            break
        if repair_pass < repair_passes:
            _add_repair_feedback(state, evaluation, repair_pass + 1)

    state["case_evaluation"] = evaluation
    state["iterations"] = state["iteration"]
    state["ended_at_utc"] = storage_utc_now_iso()
    return state


async def run_goal_resolver_cases_async(
    cases: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    repair_passes: int = DEFAULT_REPAIR_PASSES,
) -> list[dict[str, Any]]:
    """Run several resolver cases sequentially."""

    runs: list[dict[str, Any]] = []
    for case in cases:
        run = await run_goal_resolver_case_async(
            case,
            output_dir,
            max_iterations=max_iterations,
            repair_passes=repair_passes,
        )
        runs.append(run)
    return runs


def select_cases(case_ids: list[str]) -> list[dict[str, Any]]:
    """Select casebook rows by id, or all rows when no ids are provided."""

    if not case_ids:
        cases = [dict(case) for case in GOAL_RESOLVER_CASES]
        return cases
    cases = [case_by_id(case_id) for case_id in case_ids]
    return cases
