"""Production-owned LLM stages for the local-context resolver."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from collections.abc import Callable

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
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
from kazusa_ai_chatbot.utils import parse_llm_json_output

from .constants import STAGE_LLM_TEMPERATURE, STAGE_LLM_TOP_P
from .contracts import LocalContextValidationError

_STAGE_TRACE_RECORDS: list[dict[str, object]] = []


def drain_stage_trace_records() -> list[dict[str, object]]:
    """Return and clear model-facing stage traces from the current process."""

    records = copy.deepcopy(_STAGE_TRACE_RECORDS)
    _STAGE_TRACE_RECORDS.clear()
    return records


_LOCAL_CONTEXT_STAGE_ATTEMPT_LIMIT = V2_MODEL_TOTAL_ATTEMPTS
_LOCAL_CONTEXT_ERROR_CAP = 500
_LOCAL_CONTEXT_REPAIR_OUTPUT_CAP = 8000
_LOCAL_CONTEXT_REPAIR_INSTRUCTION = (
    "请保留原始语义判断，并按当前阶段已有字段、值类型、列表基数和边界返回完整对象。"
)


class LocalContextStageContractError(LocalContextValidationError):
    """Raised when a local-context stage exhausts its model contract attempts."""


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


async def _run_local_context_stage(
    *,
    payload: dict[str, object],
    system_prompt: str,
    llm: LLInterface,
    config: LLMCallConfig,
    stage_name: str,
    output_state_fields: tuple[str, ...],
    candidate_validator: Callable[
        [dict[str, object]],
        None,
    ],
) -> dict[str, object]:
    """Run one local-context stage with bounded content-contract recovery."""

    base_payload = dict(payload)
    request_payload = base_payload
    system_message = SystemMessage(content=system_prompt)
    last_error = ""
    for attempt_index in range(1, _LOCAL_CONTEXT_STAGE_ATTEMPT_LIMIT + 1):
        human_message = HumanMessage(
            content=json.dumps(request_payload, ensure_ascii=False)
        )
        messages = [system_message, human_message]
        started_at = time.perf_counter()
        try:
            response = await llm.ainvoke(messages, config=config)
        except (
            OpenAIError,
            httpx.HTTPError,
            ConnectionError,
            OSError,
            RuntimeError,
            TimeoutError,
        ) as exc:
            last_error = str(exc)
            await _record_local_context_trace(
                stage_name=stage_name,
                config=config,
                messages=messages,
                response_text="",
                parsed_output={},
                parse_status="provider_error",
                status="failed",
                attempt_index=attempt_index,
                validation_error=last_error,
                started_at=started_at,
                output_state_fields=output_state_fields,
            )
            _record_stage_trace(
                stage_name=stage_name,
                route_name=config.route_name,
                model=config.model,
                payload=request_payload,
                raw_output="",
                parsed_output={"provider_error": last_error},
            )
            if attempt_index >= _LOCAL_CONTEXT_STAGE_ATTEMPT_LIMIT:
                raise LocalContextStageContractError(
                    f"{stage_name}: provider attempts exhausted: {last_error}"
                ) from exc
            request_payload = _local_context_repair_payload(
                base_payload,
                reason=(
                    "上一轮模型调用未返回可用候选，请在相同语境下重新生成完整对象。"
                ),
                contract_error="",
                invalid_candidate="",
            )
            continue

        response_content = getattr(response, "content", "")
        response_text = str(response_content)
        parsed: object = {}
        try:
            if not isinstance(response_content, str):
                raise LocalContextStageContractError(
                    f"{stage_name}: raw output must be text"
                )
            try:
                parsed = parse_llm_json_output(response_content)
            except LocalContextValidationError as exc:
                raise LocalContextStageContractError(str(exc)) from exc
            if not isinstance(parsed, dict) or not parsed:
                raise LocalContextStageContractError(
                    f"{stage_name}: output must be a non-empty object"
                )
            try:
                candidate_validator(parsed)
            except LocalContextStageContractError:
                raise
            except LocalContextValidationError as exc:
                raise LocalContextStageContractError(str(exc)) from exc
        except LocalContextStageContractError as exc:
            last_error = str(exc)
            _record_failed_stage_trace(
                stage_name=stage_name,
                route_name=config.route_name,
                model=config.model,
                payload=request_payload,
                raw_output=response_text,
                error=exc,
            )
            await _record_local_context_trace(
                stage_name=stage_name,
                config=config,
                messages=messages,
                response_text=response_text,
                parsed_output=parsed,
                parse_status="contract_error",
                status="failed",
                attempt_index=attempt_index,
                validation_error=last_error,
                started_at=started_at,
                output_state_fields=output_state_fields,
            )
            if attempt_index >= _LOCAL_CONTEXT_STAGE_ATTEMPT_LIMIT:
                raise LocalContextStageContractError(
                    f"{stage_name}: contract attempts exhausted: {last_error}"
                ) from exc
            request_payload = _local_context_repair_payload(
                base_payload,
                reason=(
                    "上一份候选未通过当前阶段的字段、值类型或列表边界校验。"
                ),
                contract_error=last_error,
                invalid_candidate=response_text,
            )
            continue

        _record_stage_trace(
            stage_name=stage_name,
            route_name=config.route_name,
            model=config.model,
            payload=request_payload,
            raw_output=response_text,
            parsed_output=parsed,
        )
        await _record_local_context_trace(
            stage_name=stage_name,
            config=config,
            messages=messages,
            response_text=response_text,
            parsed_output=parsed,
            parse_status="succeeded",
            status="succeeded",
            attempt_index=attempt_index,
            validation_error="",
            started_at=started_at,
            output_state_fields=output_state_fields,
        )
        return parsed

    raise LocalContextStageContractError(
        f"{stage_name}: contract attempts exhausted: {last_error}"
    )


def _local_context_repair_payload(
    payload: dict[str, object],
    *,
    reason: str,
    contract_error: str,
    invalid_candidate: str,
) -> dict[str, object]:
    """Add exactly one bounded local-context contract-repair block."""

    repair_payload = dict(payload)
    repair_payload["contract_repair"] = {
        "repair_instruction": _LOCAL_CONTEXT_REPAIR_INSTRUCTION,
        "reason": reason,
        "contract_error": contract_error[:_LOCAL_CONTEXT_ERROR_CAP],
        "invalid_candidate": invalid_candidate[:_LOCAL_CONTEXT_REPAIR_OUTPUT_CAP],
    }
    return repair_payload


async def _record_local_context_trace(
    *,
    stage_name: str,
    config: LLMCallConfig,
    messages: list[SystemMessage | HumanMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    attempt_index: int,
    validation_error: str,
    started_at: float,
    output_state_fields: tuple[str, ...],
) -> None:
    """Record one local-context model attempt through protected tracing."""

    await llm_tracing.record_llm_trace_step(
        trace_id=llm_tracing.current_trace_id(),
        stage_name=stage_name,
        route_name=config.route_name,
        model_name=config.model,
        messages=messages,
        response_text=response_text,
        parsed_output=parsed_output,
        parse_status=parse_status,
        status=status,
        duration_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
        output_state_fields=output_state_fields,
        sequence=attempt_index - 1,
        call_config=config,
        attempt_index=attempt_index,
        validation_error=validation_error[:_LOCAL_CONTEXT_ERROR_CAP],
        attempt_started_at=started_at,
    )


def _prompt_digest(prompt: str) -> str:
    """Return a stable digest for one runtime prompt contract."""

    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return digest


def _stage_cache_identity(
    *,
    prompt: str,
    route_name: str,
    model: str,
    max_completion_tokens: int,
    thinking_enabled: bool,
) -> dict[str, object]:
    """Return stage identity fields that make cached LLM output reusable."""

    identity = {
        "prompt_digest": _prompt_digest(prompt),
        "route_name": route_name,
        "model": model,
        "temperature": STAGE_LLM_TEMPERATURE,
        "top_p": STAGE_LLM_TOP_P,
        "max_completion_tokens": max_completion_tokens,
        "thinking_enabled": thinking_enabled,
    }
    return identity


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
      "node_kind": "memory_evidence|conversation_evidence|person_context|recall_evidence|live_context|external_evidence|scoped_memory|current_turn_media|recent_media|subtask"
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
- Use current_turn_media or recent_media only when a prompt-safe conversation
  image alias is present and the objective requires a visual detail.
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


def planner_stage_cache_identity() -> dict[str, object]:
    """Return the graph-planner prompt and model cache identity."""

    identity = _stage_cache_identity(
        prompt=_PLANNER_PROMPT,
        route_name="RAG_PLANNER_LLM",
        model=RAG_PLANNER_LLM_MODEL,
        max_completion_tokens=RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS,
        thinking_enabled=RAG_PLANNER_LLM_THINKING_ENABLED,
    )
    return identity


async def plan_local_context_graph(
    payload: dict[str, object],
    *,
    candidate_validator: Callable[
        [dict[str, object]],
        None,
    ],
) -> dict[str, object]:
    """Return a semantic local-context task decomposition."""

    return await _run_local_context_stage(
        payload=payload,
        system_prompt=_PLANNER_PROMPT,
        llm=_planner_llm,
        config=_planner_llm_config,
        stage_name="local_context_resolver.graph_planner",
        output_state_fields=("planner_response",),
        candidate_validator=candidate_validator,
    )


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
      "artifact_id": "short semantic artifact id",
      "artifact_type": "memory_ref|conversation_ref|person_ref|recall_ref|live_context_ref|external_ref|media_ref|semantic_packet",
      "summary": "prompt-safe evidence summary",
      "projection_payload": {
        "memory_evidence": [],
        "recall_evidence": [],
        "conversation_evidence": [],
        "external_evidence": [],
        "media_evidence": [],
        "third_party_profiles": [],
        "user_memory_unit_candidates": []
      },
      "source_policy": "short source policy"
    }
  ]
}

# Rules
- Use provided context rows as source material. Do not invent storage results.
- If context.source_context is present, treat those rows as source-agent
  retrieval evidence for this node and prefer them over inference from
  supplied chat/history rows.
- A resolved node means the local evidence step completed, not that the whole
  user goal is fully answered.
- Put graph ids, trace details, raw message ids, adapter ids, database ids,
  cache keys, embeddings, raw timestamps, and raw wire syntax outside
  projection_payload.
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
  media_ref is only for a supplied prompt-safe conversation image alias and
  writes media_evidence. Never write image payloads, hashes, cache refs, or
  platform message identifiers.
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


def active_node_stage_cache_identity() -> dict[str, object]:
    """Return the active-node prompt and model cache identity."""

    identity = _stage_cache_identity(
        prompt=_NODE_PROMPT,
        route_name="RAG_SUBAGENT_LLM",
        model=RAG_SUBAGENT_LLM_MODEL,
        max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
        thinking_enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED,
    )
    return identity


async def resolve_local_context_node(
    payload: dict[str, object],
    *,
    candidate_validator: Callable[
        [dict[str, object]],
        None,
    ],
) -> dict[str, object]:
    """Return one active-node local evidence update."""

    return await _run_local_context_stage(
        payload=payload,
        system_prompt=_NODE_PROMPT,
        llm=_node_llm,
        config=_node_llm_config,
        stage_name="local_context_resolver.active_node_resolver",
        output_state_fields=("node_update", "artifacts"),
        candidate_validator=candidate_validator,
    )


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
    *,
    candidate_validator: Callable[
        [dict[str, object]],
        None,
    ],
) -> dict[str, object]:
    """Return one bounded collapse decision."""

    return await _run_local_context_stage(
        payload=payload,
        system_prompt=_COLLAPSE_PROMPT,
        llm=_collapse_llm,
        config=_collapse_llm_config,
        stage_name="local_context_resolver.collapse_review",
        output_state_fields=("collapse_decision",),
        candidate_validator=candidate_validator,
    )


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
    *,
    candidate_validator: Callable[
        [dict[str, object]],
        None,
    ],
) -> dict[str, object]:
    """Return final semantic packet fields for one resolver run."""

    return await _run_local_context_stage(
        payload=payload,
        system_prompt=_SYNTHESIZER_PROMPT,
        llm=_synthesizer_llm,
        config=_synthesizer_llm_config,
        stage_name="local_context_resolver.bottom_up_synthesis",
        output_state_fields=(
            "investigation_summary",
            "knowledge_we_know_so_far",
            "knowledge_still_lacking",
            "recommended_next_iteration",
            "evidence_boundary_notes",
        ),
        candidate_validator=candidate_validator,
    )
