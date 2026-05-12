"""Dispatcher, agent registry, and executor for the RAG supervisor."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_PLANNER_LLM_API_KEY,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.conversation_aggregate_agent import ConversationAggregateAgent
from kazusa_ai_chatbot.rag.conversation_evidence_agent import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.conversation_filter_agent import ConversationFilterAgent
from kazusa_ai_chatbot.rag.conversation_keyword_agent import ConversationKeywordAgent
from kazusa_ai_chatbot.rag.conversation_search_agent import ConversationSearchAgent
from kazusa_ai_chatbot.rag.live_context_agent import LiveContextAgent
from kazusa_ai_chatbot.rag.memory_evidence_agent import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context_agent import PersonContextAgent
from kazusa_ai_chatbot.rag.persistent_memory_keyword_agent import PersistentMemoryKeywordAgent
from kazusa_ai_chatbot.rag.persistent_memory_search_agent import PersistentMemorySearchAgent
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.rag.recall_agent import RecallAgent
from kazusa_ai_chatbot.rag.relationship_agent import RelationshipAgent
from kazusa_ai_chatbot.rag.user_list_agent import UserListAgent
from kazusa_ai_chatbot.rag.user_lookup_agent import UserLookupAgent
from kazusa_ai_chatbot.rag.user_profile_agent import UserProfileAgent
from kazusa_ai_chatbot.rag.web_search_agent import WebSearchAgent
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_prompt_views import (
    _known_facts_llm_view,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_types import (
    ProgressiveRAGState,
    RAGAgentRegistryEntry,
)
from kazusa_ai_chatbot.utils import (
    get_llm,
    log_preview,
    parse_llm_json_output,
)

logger = logging.getLogger(__name__)


# ── Dispatcher ─────────────────────────────────────────────────────
_RAG_SUPERVISOR_AGENT_REGISTRY: dict[str, RAGAgentRegistryEntry] = {
    "live_context_agent": {
        "agent": LiveContextAgent().run,
        "fact_source": {
            "source_kind": "external",
            "source_system": "live_context",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_evidence_agent": {
        "agent": ConversationEvidenceAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "memory_evidence_agent": {
        "agent": MemoryEvidenceAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "person_context_agent": {
        "agent": PersonContextAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_lookup_agent": {
        "agent": UserLookupAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_list_agent": {
        "agent": UserListAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles_or_conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_profile_agent": {
        "agent": UserProfileAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "relationship_agent": {
        "agent": RelationshipAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "web_search_agent2": {
        "agent": WebSearchAgent().run,
        "fact_source": {
            "source_kind": "external",
            "source_system": "web",
            "consolidation_policy": "eligible_external_knowledge",
            "can_consolidate_as_new_knowledge": True,
        },
    },
    "conversation_aggregate_agent": {
        "agent": ConversationAggregateAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_filter_agent": {
        "agent": ConversationFilterAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_keyword_agent": {
        "agent": ConversationKeywordAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_search_agent": {
        "agent": ConversationSearchAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "persistent_memory_keyword_agent": {
        "agent": PersistentMemoryKeywordAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "persistent_memory_search_agent": {
        "agent": PersistentMemorySearchAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "recall_agent": {
        "agent": RecallAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "recall",
            "consolidation_policy": "operational_recall_evidence",
            "can_consolidate_as_new_knowledge": False,
        },
    },
}

_PREFIX_DISPATCH_TABLE: tuple[tuple[str, str, int], ...] = (
    ("Live-context:", "live_context_agent", 1),
    ("Conversation-evidence:", "conversation_evidence_agent", 1),
    ("Memory-evidence:", "memory_evidence_agent", 1),
    ("Person-context:", "person_context_agent", 1),
    ("Web-evidence:", "web_search_agent2", 3),
    ("Identity:", "user_lookup_agent", 3),
    ("User-list:", "user_list_agent", 3),
    ("Relationship:", "relationship_agent", 3),
    ("Profile:", "user_profile_agent", 3),
    ("Conversation-aggregate:", "conversation_evidence_agent", 1),
    ("Conversation-filter:", "conversation_evidence_agent", 1),
    ("Conversation-keyword:", "conversation_evidence_agent", 1),
    ("Conversation-semantic:", "conversation_evidence_agent", 1),
    ("Memory-keyword:", "memory_evidence_agent", 1),
    ("Memory-search:", "memory_evidence_agent", 1),
    ("Recall:", "recall_agent", 1),
    ("Web-search:", "web_search_agent2", 3),
)

_DISPATCH_AGENT_ALIASES = {
    "conversation_aggregate_agent": "conversation_evidence_agent",
    "conversation_filter_agent": "conversation_evidence_agent",
    "conversation_keyword_agent": "conversation_evidence_agent",
    "conversation_search_agent": "conversation_evidence_agent",
    "persistent_memory_keyword_agent": "memory_evidence_agent",
    "persistent_memory_search_agent": "memory_evidence_agent",
}

_DISPATCHER_PROMPT = '''\
You are a RAG Dispatcher. For each slot, select exactly one inner-loop retrieval agent and produce a concise task description for it.

## Agent Roster

- `live_context_agent`: Top-level present-tense live context capability.
  Answers runtime-backed current local time/date/weekday directly. For weather,
  temperature, opening status, schedules, prices, exchange rates, or current
  public status, resolves target/scope and then delegates to web.
  Use for `Live-context:` slots.

- `conversation_evidence_agent`: Top-level conversation-history evidence capability.
  Chooses hybrid exact/fuzzy search, structured filter, or aggregate
  conversation worker internally. Use for `Conversation-evidence:` slots and
  legacy `Conversation-keyword:`, `Conversation-semantic:`,
  `Conversation-filter:`, and `Conversation-aggregate:` slots.

- `memory_evidence_agent`: Top-level durable memory evidence capability.
  Handles durable memory evidence relevant to answering the slot.
  Chooses hybrid exact/fuzzy persistent-memory workers internally.
  Use for `Memory-evidence:`, legacy `Memory-search:`, and legacy
  `Memory-keyword:` slots.

- `person_context_agent`: Top-level person/profile/relationship capability.
  Chooses identity, profile, user-list, or relationship worker internally.
  Use for `Person-context:` slots.

- `user_lookup_agent`: Direct user-profile lookup by display name.
  Queries the user_profiles collection (NOT conversation history). Returns global_user_id.
  Use as the ONLY choice when the slot is about resolving who a named person is.

- `user_list_agent`: User enumeration by display-name predicates or participant metadata.
  Handles listing users whose display names equal, contain, start with, or end with a literal value.
  Use for "all users whose names..." and similar user-list questions.

- `user_profile_agent`: Reads a user's full profile from the user-profile store.
  Use ONLY when global_user_id is already present in known_facts. Never for unknown identities.

- `relationship_agent`: Ranks profiled users by the character's relationship data.
  Use for `Relationship:` slots. The agent extracts its own ranking parameters.

- `conversation_filter_agent`: Internal structured filter over conversation history.
  Handles: fetching messages from a known user (by global_user_id); filtering by channel, time range, or message count.
  Top-level dispatch should use `conversation_evidence_agent`, which invokes this worker internally when appropriate.

- `conversation_aggregate_agent`: Internal factual aggregate over conversation history.
  Handles: counts and rankings grouped by user, optionally filtered by literal keyword, known user, channel, and time window.
  Top-level dispatch should use `conversation_evidence_agent` for "who spoke most", "how many messages", and "who mentioned X most" questions.
  It returns evidence only, not opinions or persona interpretation.

- `conversation_keyword_agent`: Internal exact-string worker used by hybrid conversation search.
  Top-level dispatch should use `conversation_evidence_agent` for URLs, filenames, exact phrases, and proper nouns.

- `conversation_search_agent`: Internal hybrid semantic plus literal-anchor search over message content.
  Top-level dispatch should use `conversation_evidence_agent` for fuzzy topic recall and exact/literal recall.

- `persistent_memory_keyword_agent`: Internal exact-keyword worker used by hybrid persistent-memory search.
  Top-level dispatch should use `memory_evidence_agent` for tags, event names, and exact memory identifiers.

- `persistent_memory_search_agent`: Internal hybrid search over persistent memories.
  Top-level dispatch should use `memory_evidence_agent` for durable memory evidence.

- `recall_agent`: Reconciles active agreements, ongoing promises, current plans,
  open loops, and current-episode state from scoped progress, active commitments,
  pending scheduled events, and gated history proof. Use for `Recall:` slots.

- `web_search_agent2`: Public internet search.
  Use ONLY when information cannot exist in local conversation history or persistent memory.

## Slot prefix → agent mapping (check this FIRST — overrides everything below)

Slots produced by the initializer start with a prefix that maps directly to an agent.
Match the prefix literally and use the mapped agent without further deliberation:

| Slot prefix                  | Agent                            |
|------------------------------|----------------------------------|
| "Live-context: ..."          | `live_context_agent`             |
| "Conversation-evidence: ..." | `conversation_evidence_agent`    |
| "Memory-evidence: ..."       | `memory_evidence_agent`          |
| "Person-context: ..."        | `person_context_agent`           |
| "Web-evidence: ..."          | `web_search_agent2`              |
| "Identity: ..."              | `user_lookup_agent`              |
| "User-list: ..."             | `user_list_agent`                |
| "Relationship: ..."          | `relationship_agent`             |
| "Profile: ..."               | `user_profile_agent`             |
| "Conversation-aggregate: ..."| `conversation_evidence_agent`    |
| "Conversation-filter: ..."   | `conversation_evidence_agent`    |
| "Conversation-keyword: ..."  | `conversation_evidence_agent`    |
| "Conversation-semantic: ..." | `conversation_evidence_agent`    |
| "Memory-keyword: ..."        | `memory_evidence_agent`          |
| "Memory-search: ..."         | `memory_evidence_agent`          |
| "Recall: ..."                | `recall_agent`                   |
| "Web-search: ..."            | `web_search_agent2`              |

## Fallback decision sequence — use only when slot has no recognised prefix

Evaluate top to bottom, pick the first match:

1. Slot resolves a named person's identity (display_name → global_user_id)?
   → `user_lookup_agent`.

2. Slot enumerates users by display-name pattern or participant metadata?
   → `user_list_agent`.

3. Slot needs a user's full profile AND global_user_id is already in known_facts?
   → `user_profile_agent`.

4. Slot targets conversation history, including literal strings, URLs,
   filenames, exact phrases, fuzzy topics, structured filters, counts, or
   rankings?
   → `conversation_evidence_agent`.

5. Slot targets durable persistent memory by exact keyword, tag, event name,
   proper noun, memory identifier, or semantic memory meaning?
   → `memory_evidence_agent`.

6. Slot asks what was agreed, promised, planned, left unresolved, or where the
    current episode left off?
   → `recall_agent`.

7. Slot requires public internet data?
   → `web_search_agent2`.

## Input

{{
    "current_slot": "the slot to resolve this turn",
    "known_facts": [{{"slot": "...", "agent": "...", "resolved": true/false, "summary": "concise fact summary", "attempts": 1}}, ...],
    "context": runtime info (platform, channel, timestamp, etc.)
}}

## Generation Procedure
1. Read `current_slot` and match its prefix against the prefix-to-agent table first.
2. If there is no recognized prefix, use the fallback decision sequence.
3. Build a concise task for the chosen inner-loop agent, preserving dependency references such as "slot N".
4. Pass through only relevant context needed by the chosen agent.

## Output

Return valid JSON only:
{{
    "agent_name": "{agent_name_union}",
    "task": "task description for the chosen agent",
    "context": {{}},
    "max_attempts": 3
}}
'''
_dispatcher_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_PLANNER_LLM_MODEL,
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
)


def build_rag_fact_source_map() -> dict[str, dict[str, Any]]:
    """Build the deterministic fact-source map for RAG known_facts.

    The consolidator can combine this map with each known fact's ``agent``
    field to decide whether a fact came from internal memory/history/profile
    stores or from an external source.

    Returns:
        Dict keyed by agent name with source and consolidation policy metadata.
    """
    return_value = {
        agent_name: dict(entry["fact_source"])
        for agent_name, entry in _RAG_SUPERVISOR_AGENT_REGISTRY.items()
    }
    return return_value


def _dispatch_from_known_prefix(current_slot: str) -> dict[str, Any] | None:
    """Build a dispatch directly from a recognized slot prefix.

    Args:
        current_slot: The ordered slot selected for this loop iteration.

    Returns:
        Normalized dispatch payload, or ``None`` when no prefix matches.
    """

    if not isinstance(current_slot, str):
        return None

    for prefix, agent_name, max_attempts in _PREFIX_DISPATCH_TABLE:
        if current_slot.startswith(prefix):
            dispatch = {
                "agent_name": agent_name,
                "task": current_slot.strip(),
                "context": {},
                "max_attempts": max_attempts,
                "route_source": "deterministic_prefix",
            }
            return dispatch

    return_value = None
    return return_value


async def rag_dispatcher(state: ProgressiveRAGState) -> dict:
    """Generate one plain JSON dispatch targeting the first unknown slot.

    Args:
        state: Current state with unknown_slots and known_facts.

    Returns:
        Partial state update with dispatcher JSON recorded in state,
        current_slot set, and loop_count incremented.
    """
    current_slot = state["unknown_slots"][0]
    prefix_dispatch = _dispatch_from_known_prefix(current_slot)
    if prefix_dispatch is not None:
        logger.info(
            f'RAG2 dispatch output: agent={prefix_dispatch["agent_name"]} '
            f"task={log_preview(prefix_dispatch['task'])} "
            f'route_source={prefix_dispatch["route_source"]}'
        )
        logger.debug(
            f'RAG2 dispatch metadata: loop={state.get("loop_count", 0) + 1} '
            f"slot={log_preview(current_slot)} "
            f'max_attempts={prefix_dispatch["max_attempts"]} '
            f"dispatch_context={log_preview(prefix_dispatch['context'])}"
        )
        return_value = {
            "messages": [
                AIMessage(
                    content=json.dumps(prefix_dispatch, ensure_ascii=False)
                )
            ],
            "current_slot": current_slot,
            "current_dispatch": prefix_dispatch,
            "loop_count": state.get("loop_count", 0) + 1,
        }
        return return_value

    system_prompt = SystemMessage(
        content=_DISPATCHER_PROMPT.format(
            agent_name_union=_build_agent_name_union(),
        )
    )
    llm_context = project_runtime_context_for_llm(state.get("context", {}))
    user_input = {
        "current_slot": current_slot,
        "known_facts": _known_facts_llm_view(state.get("known_facts", [])),
        "context": llm_context,
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False, default=str))

    # Only the last two messages — enough to preserve immediate loop history
    # without polluting the current slot with distant attempts.
    recent_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]

    response = await _dispatcher_llm.ainvoke([system_prompt, human_message] + recent_messages)
    dispatch = parse_llm_json_output(response.content)
    if not isinstance(dispatch, dict):
        dispatch = {}
    normalized_dispatch = _normalize_dispatch(dispatch, current_slot)
    logger.info(
        f'RAG2 dispatch output: agent={normalized_dispatch["agent_name"] or "<invalid>"} '
        f"task={log_preview(normalized_dispatch['task'])} "
        f'route_source={normalized_dispatch["route_source"]}'
    )
    logger.debug(
        f'RAG2 dispatch metadata: loop={state.get("loop_count", 0) + 1} '
        f"slot={log_preview(current_slot)} "
        f'max_attempts={normalized_dispatch["max_attempts"]} '
        f'route_source={normalized_dispatch["route_source"]} '
        f"dispatch_context={log_preview(normalized_dispatch['context'])} "
        f"raw={log_preview(dispatch)}"
    )

    return_value = {
        "messages": [AIMessage(content=response.content)],
        "current_slot": current_slot,
        "current_dispatch": normalized_dispatch,
        "loop_count": state.get("loop_count", 0) + 1,
    }
    return return_value


# ── Executor ───────────────────────────────────────────────────────


def _build_delegate_context(state: ProgressiveRAGState, dispatch: dict) -> dict:
    """Merge runtime context, dispatcher hints, and known facts for one agent call."""
    delegate_context = dict(state.get("context", {}))
    raw_dispatch_context = dispatch.get("context", {})
    if isinstance(raw_dispatch_context, dict):
        delegate_context.update(raw_dispatch_context)
    delegate_context["known_facts"] = _known_facts_llm_view(
        state.get("known_facts", []),
    )
    delegate_context["original_query"] = state.get("original_query", "")
    delegate_context["current_slot"] = state.get("current_slot", "")
    return delegate_context



def _build_agent_name_union() -> str:
    """Render the dispatcher's allowed ``agent_name`` values as a union string."""
    agent_names = [
        agent_name
        for agent_name in _RAG_SUPERVISOR_AGENT_REGISTRY
        if agent_name not in _DISPATCH_AGENT_ALIASES
    ]
    return_value = " | ".join(agent_names)
    return return_value


def _normalize_dispatch(raw_dispatch: dict, current_slot: str) -> dict:
    """Normalize dispatcher JSON into a safe executable dispatch payload."""
    raw_agent_name = raw_dispatch.get("agent_name", "")
    agent_name = raw_agent_name.strip() if isinstance(raw_agent_name, str) else ""
    agent_name = _DISPATCH_AGENT_ALIASES.get(agent_name, agent_name)
    if agent_name not in _RAG_SUPERVISOR_AGENT_REGISTRY:
        agent_name = ""

    raw_task = raw_dispatch.get("task", "")
    fallback_task = current_slot if isinstance(current_slot, str) else ""
    task = raw_task.strip() if isinstance(raw_task, str) else ""
    task = task or fallback_task
    context = raw_dispatch.get("context", {})
    if not isinstance(context, dict):
        context = {}

    raw_max_attempts = raw_dispatch.get("max_attempts", 3)
    if isinstance(raw_max_attempts, int) and not isinstance(raw_max_attempts, bool) and raw_max_attempts > 0:
        max_attempts = raw_max_attempts
    else:
        max_attempts = 3

    raw_route_source = raw_dispatch.get("route_source", "dispatcher_llm")
    if isinstance(raw_route_source, str) and raw_route_source.strip():
        route_source = raw_route_source.strip()
    else:
        route_source = "dispatcher_llm"

    return_value = {
        "agent_name": agent_name,
        "task": task,
        "context": context,
        "max_attempts": max_attempts,
        "route_source": route_source,
    }
    return return_value


def _serialize_agent_result(result: dict) -> str:
    """Serialize one agent result for dispatcher history."""
    return_value = json.dumps(result, ensure_ascii=False, default=str)
    return return_value


def _agent_result_info_view(result: dict) -> dict:
    """Build an INFO-safe view of an agent result.

    Args:
        result: Raw helper-agent result from the executor.

    Returns:
        Compact operational fields suitable for INFO logging. Heavy payloads
        such as resolved refs, projection payloads, and worker payloads remain
        available in DEBUG logs and in graph state.
    """

    raw_result = result.get("result")
    if isinstance(raw_result, dict) and "capability" in raw_result:
        compact_result = {
            "capability": raw_result.get("capability", ""),
            "primary_worker": raw_result.get("primary_worker", ""),
            "supporting_workers": raw_result.get("supporting_workers", []),
            "missing_context": raw_result.get("missing_context", []),
            "selected_summary": raw_result.get("selected_summary", ""),
        }
    else:
        compact_result = raw_result

    raw_cache = result.get("cache", {})
    cache = raw_cache if isinstance(raw_cache, dict) else {}
    info_view = {
        "agent": result.get("agent", ""),
        "resolved": bool(result.get("resolved", False)),
        "attempts": result.get("attempts", 0),
        "cache": {
            "enabled": cache.get("enabled", False),
            "hit": cache.get("hit", False),
            "reason": cache.get("reason", ""),
        },
        "result": compact_result,
    }
    return info_view


async def rag_executor(state: ProgressiveRAGState) -> dict:
    """Execute the delegate-agent call produced by the dispatcher.

    Args:
        state: Current state whose dispatch payload names the agent to run.

    Returns:
        Partial state update with the normalized agent result.
    """
    dispatch = _normalize_dispatch(
        state.get("current_dispatch", {}),
        state.get("current_slot", ""),
    )
    agent_name = dispatch["agent_name"]

    if not agent_name:
        result = {
            "agent": "",
            "resolved": False,
            "result": "Dispatcher failed to choose a valid agent.",
            "attempts": 0,
        }
        logger.info(
            f"RAG2 agent output: result={log_preview(result)}"
        )
        logger.debug(
            f'RAG2 agent metadata: slot={log_preview(state.get("current_slot", ""))} '
            f"agent=<invalid> resolved=False attempts=0"
        )
        return_value = {
            "last_agent_result": result,
            "messages": [
                HumanMessage(
                    content=_serialize_agent_result(_agent_result_info_view(result)),
                    name="agent_result",
                )
            ],
        }
        return return_value

    agent = _RAG_SUPERVISOR_AGENT_REGISTRY[agent_name]["agent"]
    try:
        result = await agent(
            task=dispatch["task"],
            context=_build_delegate_context(state, dispatch),
            max_attempts=dispatch["max_attempts"],
        )
        if not isinstance(result, dict):
            result = {
                "resolved": False,
                "result": str(result),
                "attempts": dispatch["max_attempts"],
            }
        result.setdefault("agent", agent_name)
    except Exception as exc:
        logger.exception(f'Error executing agent {agent_name}: {exc}')
        result = {
            "agent": agent_name,
            "resolved": False,
            "result": f"{type(exc).__name__}: {exc}",
            "attempts": 0,
        }

    info_view = _agent_result_info_view(result)
    logger.info(f"RAG2 agent output: result={log_preview(info_view)}")
    logger.debug(
        f'RAG2 agent metadata: slot={log_preview(state.get("current_slot", ""))} '
        f'agent={agent_name} resolved={bool(result.get("resolved", False))} '
        f'attempts={result.get("attempts", 0)} raw={log_preview(result)}'
    )
    return_value = {
        "last_agent_result": result,
        "messages": [
            HumanMessage(
                content=_serialize_agent_result(info_view),
                name="agent_result",
            )
        ],
    }
    return return_value
