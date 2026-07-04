"""Production-owned LLM stages for the local-context resolver."""

from __future__ import annotations

import copy
import json

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_PLANNER_LLM_API_KEY,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS,
    RAG_PLANNER_LLM_MODEL,
    RAG_PLANNER_LLM_THINKING_ENABLED,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)

from .constants import STAGE_LLM_TEMPERATURE, STAGE_LLM_TOP_P
from .contracts import LocalContextValidationError

_STAGE_TRACE_RECORDS: list[dict[str, object]] = []


def drain_stage_trace_records() -> list[dict[str, object]]:
    """Return and clear model-facing stage traces from the current process."""

    records = copy.deepcopy(_STAGE_TRACE_RECORDS)
    _STAGE_TRACE_RECORDS.clear()
    return records


def _parse_stage_json_output(
    content: object,
    stage_name: str,
) -> dict[str, object]:
    """Parse one stage JSON object without invoking JSON repair LLMs."""

    if not isinstance(content, str):
        raise LocalContextValidationError(
            f"{stage_name}: raw output must be a string"
        )
    raw_text = content.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        object_text = _outer_json_object_text(raw_text)
        try:
            parsed = json.loads(object_text)
        except json.JSONDecodeError as object_exc:
            repaired_text = _escape_string_control_characters(object_text)
            try:
                parsed = json.loads(repaired_text)
            except json.JSONDecodeError as repaired_exc:
                if object_text == raw_text:
                    raise LocalContextValidationError(
                        f"{stage_name}: invalid JSON: {exc}"
                    ) from repaired_exc
                raise LocalContextValidationError(
                    f"{stage_name}: invalid JSON object: {object_exc}"
                ) from repaired_exc
    if not isinstance(parsed, dict) or not parsed:
        raise LocalContextValidationError(f"{stage_name}: empty parsed output")
    result = parsed
    return result


def _outer_json_object_text(raw_text: str) -> str:
    """Return the outermost object substring when a response wraps JSON."""

    try:
        start_index = raw_text.index("{")
        end_index = raw_text.rindex("}")
    except ValueError:
        return_value = raw_text
        return return_value
    if end_index <= start_index:
        return_value = raw_text
        return return_value
    return_value = raw_text[start_index:end_index + 1]
    return return_value


def _escape_string_control_characters(raw_text: str) -> str:
    """Escape raw control characters that appear inside JSON strings."""

    chars: list[str] = []
    in_string = False
    escaped = False
    for char in raw_text:
        if escaped:
            chars.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            chars.append(char)
            escaped = True
            continue
        if char == '"':
            chars.append(char)
            in_string = not in_string
            continue
        if in_string and char == "\n":
            chars.append("\\n")
            continue
        if in_string and char == "\r":
            chars.append("\\r")
            continue
        if in_string and char == "\t":
            chars.append("\\t")
            continue
        if in_string and ord(char) < 32:
            chars.append(f"\\u{ord(char):04x}")
            continue
        chars.append(char)
    return_value = "".join(chars)
    return return_value


def _record_stage_trace(
    *,
    stage_name: str,
    route_name: str,
    model: str,
    payload: dict[str, object],
    raw_output: str,
    parsed_output: dict[str, object],
) -> None:
    """Keep raw stage evidence for live LLM review artifacts."""

    record = {
        "stage_name": stage_name,
        "prompt_id": stage_name,
        "route_name": route_name,
        "model": model,
        "input_payload": copy.deepcopy(payload),
        "raw_model_output": raw_output,
        "parsed_output": copy.deepcopy(parsed_output),
    }
    _STAGE_TRACE_RECORDS.append(record)


def _record_failed_stage_trace(
    *,
    stage_name: str,
    route_name: str,
    model: str,
    payload: dict[str, object],
    raw_output: str,
    error: LocalContextValidationError,
) -> None:
    """Keep raw stage evidence when deterministic parsing fails."""

    _record_stage_trace(
        stage_name=stage_name,
        route_name=route_name,
        model=model,
        payload=payload,
        raw_output=raw_output,
        parsed_output={"parse_error": str(error)},
    )

_PLANNER_PROMPT = '''\
You split one local-context recall objective into a small semantic task list.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Task
Use the objective and compact context to identify the smallest local evidence
nodes that should be checked. Planning only decomposes; it does not answer.

# Output Format
{
  "tasks": [
    {
      "objective": "one concrete local evidence task",
      "node_kind": "memory_evidence|conversation_evidence|person_context|recall_evidence|live_context|external_evidence|scoped_memory|subtask"
    }
  ]
}

# Rules
- Return 1 to 5 tasks.
- Prefer one task when one source domain can satisfy the objective.
- Do not split extraction and verification into separate tasks when the same
  supplied row can provide the speaker, quote, URL, or adjacent context.
- Use memory_evidence for durable shared memory or command/lore anchors.
- Use scoped_memory for current-user private continuity.
- Use conversation_evidence for recent or historical chat rows, exact phrases,
  speakers, URLs, reply context, or neighboring dialog.
- Use person_context for profile, identity, relationship, or impression facts.
- Use recall_evidence for active commitments, agreements, plans, or episode
  progress.
- When the objective asks what was agreed, promised, scheduled, committed, or
  planned and a recall source row is supplied, use one recall_evidence task.
  Do not add scoped_memory merely to double-check active agreements.
- Use live_context for supplied local time/date/runtime context.
- Use external_evidence only when local context points at public URL or web
  content that must be read.
- Do not add recall_evidence for recent chat events, command responses,
  direct-address behavior, tags, URLs, exact phrases, or neighboring dialog.
- Do not add person_context for command behavior, tags, direct address, or a
  speaker name unless the objective explicitly asks for that person's profile,
  identity, relationship, or impression.
- For current time/date/weekday questions, use one live_context task unless
  the objective explicitly asks for a character-specific timezone or profile.
- For supplied web_content rows, use one external_evidence task unless the
  objective separately asks for chat provenance.
- Keep direct address, tags, and mentions as social context; preserve the
  semantic anchor in the message, such as a command or quoted phrase.
- Do not return synthesis tasks. The service performs final synthesis after
  evidence-node traversal.
- Do not include graph ids, storage ids, adapter ids, database filters,
  embedding settings, cache keys, final dialog wording, or behavior controls.
'''

_planner_llm = LLInterface()
_planner_llm_config = LLMCallConfig(
    stage_name="local_context_resolver.graph_planner",
    route_name="RAG_PLANNER_LLM",
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
    model=RAG_PLANNER_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=RAG_PLANNER_LLM_THINKING_ENABLED),
)


async def plan_local_context_graph(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return a semantic local-context task decomposition."""

    response = await _planner_llm.ainvoke(
        [
            SystemMessage(content=_PLANNER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_planner_llm_config,
    )
    try:
        parsed = _parse_stage_json_output(response.content, "planner stage")
    except LocalContextValidationError as exc:
        _record_failed_stage_trace(
            stage_name="local_context_resolver.graph_planner",
            route_name="RAG_PLANNER_LLM",
            model=RAG_PLANNER_LLM_MODEL,
            payload=payload,
            raw_output=response.content,
            error=exc,
        )
        raise
    _record_stage_trace(
        stage_name="local_context_resolver.graph_planner",
        route_name="RAG_PLANNER_LLM",
        model=RAG_PLANNER_LLM_MODEL,
        payload=payload,
        raw_output=response.content,
        parsed_output=parsed,
    )
    result = parsed
    return result


_NODE_PROMPT = '''\
You resolve exactly one local-context evidence node.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Task
Use only the active node, compact context, and dependency context. Record
prompt-safe local evidence. Do not write final character dialog.

# Output Format
{
  "node_update": {
    "status": "resolved|blocked|cannot_answer",
    "investigation_summary": ["what this node checked"],
    "knowledge_we_know_so_far": ["evidence-backed local fact"],
    "knowledge_still_lacking": ["specific missing local fact"],
    "recommended_next_iteration": ["narrow next evidence direction"],
    "evidence_boundary_notes": ["source or confidence boundary"],
    "produces": ["semantic artifact name"]
  },
  "artifacts": [
    {
      "schema_version": "local_context_artifact.v1",
      "artifact_id": "short semantic artifact id",
      "artifact_type": "memory_ref|conversation_ref|person_ref|recall_ref|live_context_ref|external_ref|semantic_packet",
      "producer_node_id": "active_node",
      "summary": "prompt-safe evidence summary",
      "projection_payload": {
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "third_party_profiles": [],
        "user_memory_unit_candidates": []
      },
      "source_policy": "short source policy",
      "prompt_visible": true
    }
  ]
}

# Rules
- Use provided context rows as source material. Do not invent storage results.
- A resolved node means the local evidence step completed, not that the whole
  user goal is fully answered.
- Put graph ids, trace details, raw message ids, adapter ids, database ids,
  cache keys, embeddings, raw timestamps, and raw wire syntax outside
  projection_payload.
- Return valid JSON. If source text contains double quotes, escape them or
  paraphrase them in JSON string values.
- Treat chat row local_time values as message timestamps only. Do not infer
  the current time from message timestamps. Only local_time_context supplies
  current date/time; if a current time value is absent, do not judge whether a
  scheduled time has passed.
- Match artifact_type and projection_payload field ownership:
  memory_ref is for durable/shared/scoped memory evidence and writes
  memory_evidence or user_memory_unit_candidates.
  Current-user scoped memory or user_memory_units source rows write
  user_memory_unit_candidates, not memory_evidence.
  conversation_ref is for chat messages, speakers, quotes, URL provenance, and
  nearby/reply context, and writes conversation_evidence.
  person_ref is for named-person profile, identity, relationship, or
  impression evidence, and writes third_party_profiles, user_image, or
  character_image as appropriate.
  recall_ref is for active agreements, commitments, plans, open loops, and
  episode state, and writes recall_evidence.
  external_ref is only for supplied public URL or web-content evidence, and
  writes external_evidence.
  live_context_ref is for supplied current date, current time, weather,
  opening, or runtime context. Put prompt-facing live context in
  conversation_evidence because the retained rag_result surface has no
  live_context_evidence list.
- Do not put person profile evidence, URL provenance, or active agreements
  into memory_evidence unless the source row is explicitly durable memory.
- Do not use recall_ref for exact quoted phrases, command definitions,
  direct-address events, URLs, or ordinary recent chat. Use conversation_ref
  for chat/provenance/direct-address anchors and memory_ref for durable command
  rules.
- If context is insufficient, set status to blocked and explain the missing
  evidence in knowledge_still_lacking.
- For confirmation, provenance, quote, URL, speaker, or command-definition
  objectives, leave knowledge_still_lacking empty once the requested anchor is
  found. Do not list optional background such as causes, future dates,
  biographies, or unrelated details.
- For named-person profile or impression objectives, if supplied
  profile/impression evidence answers the requested impression, do not list
  missing recent interactions unless the objective explicitly asks for recent
  interactions.
- Use artifacts only for prompt-visible evidence that belongs in rag_result.
- Keep source text, URLs, command names, and quoted literals exact when they
  are supplied by context.
'''

_node_llm = LLInterface()
_node_llm_config = LLMCallConfig(
    stage_name="local_context_resolver.active_node_resolver",
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED),
)


async def resolve_local_context_node(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return one active-node local evidence update."""

    response = await _node_llm.ainvoke(
        [
            SystemMessage(content=_NODE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_node_llm_config,
    )
    try:
        parsed = _parse_stage_json_output(response.content, "node resolver stage")
    except LocalContextValidationError as exc:
        _record_failed_stage_trace(
            stage_name="local_context_resolver.active_node_resolver",
            route_name="RAG_SUBAGENT_LLM",
            model=RAG_SUBAGENT_LLM_MODEL,
            payload=payload,
            raw_output=response.content,
            error=exc,
        )
        raise
    _record_stage_trace(
        stage_name="local_context_resolver.active_node_resolver",
        route_name="RAG_SUBAGENT_LLM",
        model=RAG_SUBAGENT_LLM_MODEL,
        payload=payload,
        raw_output=response.content,
        parsed_output=parsed,
    )
    result = parsed
    return result


_COLLAPSE_PROMPT = '''\
You review whether one local-context node duplicates a resolved candidate.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Output Format
{
  "collapse_decision": {
    "should_collapse": true,
    "target_candidate_ref": "",
    "reason": "short observable reason"
  }
}

When no collapse is justified, return should_collapse false and an empty
target_candidate_ref.

# Rules
- Collapse only clear semantic duplicates.
- Use only the candidate_ref supplied in the candidates list.
- Leave graph bookkeeping and traversal decisions to deterministic code.
'''

_collapse_llm = LLInterface()
_collapse_llm_config = LLMCallConfig(
    stage_name="local_context_resolver.collapse_review",
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED),
)


async def review_local_context_collapse(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return one bounded collapse decision."""

    response = await _collapse_llm.ainvoke(
        [
            SystemMessage(content=_COLLAPSE_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_collapse_llm_config,
    )
    try:
        parsed = _parse_stage_json_output(response.content, "collapse stage")
    except LocalContextValidationError as exc:
        _record_failed_stage_trace(
            stage_name="local_context_resolver.collapse_review",
            route_name="RAG_SUBAGENT_LLM",
            model=RAG_SUBAGENT_LLM_MODEL,
            payload=payload,
            raw_output=response.content,
            error=exc,
        )
        raise
    _record_stage_trace(
        stage_name="local_context_resolver.collapse_review",
        route_name="RAG_SUBAGENT_LLM",
        model=RAG_SUBAGENT_LLM_MODEL,
        payload=payload,
        raw_output=response.content,
        parsed_output=parsed,
    )
    result = parsed
    return result


_SYNTHESIZER_PROMPT = '''\
You synthesize a local-context evidence packet, not final character dialog.
Return one JSON object only.
Do not include hidden chain-of-thought.

# Task
Synthesize bottom-up from resolved and unresolved local-context node summaries.
Preserve uncertainty and missing source coverage.

# Output Format
{
  "investigation_summary": ["what local context was investigated"],
  "knowledge_we_know_so_far": ["evidence-backed local fact"],
  "knowledge_still_lacking": ["specific missing local fact"],
  "recommended_next_iteration": ["narrow next evidence direction"],
  "evidence_boundary_notes": ["source, freshness, or confidence boundary"]
}

# Rules
- Do not judge whether the character should speak.
- Do not write visible reply text.
- Do not expose graph ids, trace counters, prompt text, storage internals,
  cache keys, adapter ids, or raw wire syntax.
- Keep supplied command names, URLs, quoted text, and source literals exact.
- Treat chat row local_time values as message timestamps only. Do not infer
  current time from them. Only local_time_context supplies current date/time.
- Report missing knowledge only when it is needed to satisfy the current
  local-context objective. Do not ask for extra profile background, future
  timeline details, or unrelated source coverage merely because it could be
  useful.
- For confirmation, provenance, quote, URL, speaker, or command-definition
  objectives, leave knowledge_still_lacking empty once the requested anchor is
  found.
- For named-person profile or impression objectives, supplied
  profile/impression evidence is enough unless the objective explicitly asks
  for recent interactions.
'''

_synthesizer_llm = LLInterface()
_synthesizer_llm_config = LLMCallConfig(
    stage_name="local_context_resolver.bottom_up_synthesis",
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=STAGE_LLM_TEMPERATURE,
    top_p=STAGE_LLM_TOP_P,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED),
)


async def synthesize_local_context_packet(
    payload: dict[str, object],
) -> dict[str, object]:
    """Return final semantic packet fields for one resolver run."""

    response = await _synthesizer_llm.ainvoke(
        [
            SystemMessage(content=_SYNTHESIZER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ],
        config=_synthesizer_llm_config,
    )
    try:
        parsed = _parse_stage_json_output(response.content, "synthesizer stage")
    except LocalContextValidationError as exc:
        _record_failed_stage_trace(
            stage_name="local_context_resolver.bottom_up_synthesis",
            route_name="RAG_SUBAGENT_LLM",
            model=RAG_SUBAGENT_LLM_MODEL,
            payload=payload,
            raw_output=response.content,
            error=exc,
        )
        raise
    _record_stage_trace(
        stage_name="local_context_resolver.bottom_up_synthesis",
        route_name="RAG_SUBAGENT_LLM",
        model=RAG_SUBAGENT_LLM_MODEL,
        payload=payload,
        raw_output=response.content,
        parsed_output=parsed,
    )
    result = parsed
    return result
