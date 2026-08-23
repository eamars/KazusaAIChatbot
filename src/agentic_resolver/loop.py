"""Serialized native-tool stream loop and terminal handling."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import cast

from agentic_resolver.context_budget import ContextBudget
from agentic_resolver.contracts import (
    AgenticResolverContractError,
    AgenticResolverLimitsV1,
    AgenticResolverRequestV1,
    AgenticResolverResultV1,
    AgenticResolverSubagentTaskV1,
    AgenticResolverUsageV1,
    ResolverStatus,
    SubmitResultV1,
)
from agentic_resolver.json_protocol import (
    compacted_observation_message,
    contract_error_message,
    parse_json_object,
    skill_content_message,
    subagent_result_message,
    tool_observation_message,
)
from agentic_resolver.model import AgenticModelClient
from agentic_resolver.session import (
    ResolverObservation,
    ResolverSession,
)
from agentic_resolver.skills import SkillCatalog
from agentic_resolver.streaming import (
    AssembledAssistantTurn,
    ModelStreamAssembler,
)
from agentic_resolver.subagents import SubagentRunner
from agentic_resolver.tools import (
    ToolExecutionResult,
    ToolRegistry,
    sanitized_exception_text,
    validate_json_schema_arguments,
)


class AgentLoop:
    """Run one bounded root or child session to submit_result or hard stop."""

    def __init__(
        self,
        *,
        model: AgenticModelClient,
        registry: ToolRegistry,
        skills: SkillCatalog,
        limits: AgenticResolverLimitsV1,
        permission_scope: Mapping[str, object],
        subagent_runner: SubagentRunner,
    ) -> None:
        self._model = model
        self._registry = registry
        self._skills = skills
        self._limits = limits
        self._permission_scope = permission_scope
        self._subagent_runner = subagent_runner
        self._context_budget = ContextBudget(limits)

    async def run(
        self,
        *,
        session: ResolverSession,
        request: AgenticResolverRequestV1,
        deadline: float,
    ) -> AgenticResolverResultV1:
        """Run serialized model steps while enforcing every deterministic cap."""

        model_tools = self._registry.model_definitions()
        while True:
            if time.monotonic() >= deadline:
                result = _hard_result(
                    session,
                    request=request,
                    status="budget_exhausted",
                    summary="The resolver session reached its wall-clock limit.",
                    reason="session_deadline",
                )
                return result
            if session.usage.model_steps >= self._limits.max_model_steps:
                result = _hard_result(
                    session,
                    request=request,
                    status="budget_exhausted",
                    summary="The resolver reached its model-step limit.",
                    reason="model_step_cap",
                )
                return result

            admission = self._context_budget.prepare(session, model_tools)
            if admission is None:
                result = _hard_result(
                    session,
                    request=request,
                    status="budget_exhausted",
                    summary="The resolver request cannot fit the context ceiling.",
                    reason="context_budget",
                )
                return result
            session.record_model_step_started()
            remaining_seconds = max(0.0, deadline - time.monotonic())
            try:
                turn = await asyncio.wait_for(
                    self._stream_turn(
                        session=session,
                        messages=admission.messages,
                        tools=model_tools,
                    ),
                    timeout=remaining_seconds,
                )
            except TimeoutError as exc:
                result = _hard_result(
                    session,
                    request=request,
                    status="budget_exhausted",
                    summary=(
                        "The model stream exceeded the remaining session time: "
                        f"{sanitized_exception_text(exc)}"
                    ),
                    reason="model_stream_timeout",
                )
                return result
            except AgenticResolverContractError as exc:
                status: ResolverStatus = (
                    "budget_exhausted"
                    if exc.code == "stream_budget_exhausted"
                    else "failed"
                )
                result = _hard_result(
                    session,
                    request=request,
                    status=status,
                    summary=f"The model stream contract failed: {exc}",
                    reason=exc.code,
                )
                return result
            # Injected model transports may raise SDK-specific exception types.
            except Exception as exc:  # noqa: BLE001
                result = _hard_result(
                    session,
                    request=request,
                    status="failed",
                    summary=(
                        "The model stream failed: "
                        f"{sanitized_exception_text(exc)}"
                    ),
                    reason="provider_failure",
                )
                return result

            session.record_assembled_turn(turn)
            if turn.finish.reason == "max_tokens":
                result = _hard_result(
                    session,
                    request=request,
                    status="budget_exhausted",
                    summary="The model exhausted its completion budget.",
                    reason="model_max_tokens",
                )
                return result
            if turn.finish.reason in {"aborted", "error"}:
                result = _hard_result(
                    session,
                    request=request,
                    status="failed",
                    summary="The model stream ended without a usable turn.",
                    reason=f"model_{turn.finish.reason}",
                )
                return result
            content_error = _assistant_content_error(turn.content)
            if content_error is not None:
                can_continue = self._append_unpaired_contract_error(
                    session,
                    turn=turn,
                    code="invalid_assistant_content",
                    message=content_error,
                )
                if can_continue:
                    continue
                result = _hard_result(
                    session,
                    request=request,
                    status="failed",
                    summary="The model exhausted structural replacements.",
                    reason="contract_replacement_cap",
                )
                return result
            if turn.invalid_tool_calls:
                can_continue = self._append_unpaired_contract_error(
                    session,
                    turn=turn,
                    code="invalid_tool_call",
                    message="Return one complete native tool call with JSON arguments.",
                )
                if can_continue:
                    continue
                result = _hard_result(
                    session,
                    request=request,
                    status="failed",
                    summary="The model exhausted structural replacements.",
                    reason="contract_replacement_cap",
                )
                return result
            if len(turn.tool_calls) != 1:
                code = (
                    "no_tool_call"
                    if not turn.tool_calls
                    else "multiple_tool_calls"
                )
                can_continue = self._append_unpaired_contract_error(
                    session,
                    turn=turn,
                    code=code,
                    message="Return exactly one registered native tool call.",
                )
                if can_continue:
                    continue
                result = _hard_result(
                    session,
                    request=request,
                    status="failed",
                    summary="The model exhausted structural replacements.",
                    reason="contract_replacement_cap",
                )
                return result

            tool_call = turn.tool_calls[0]
            try:
                definition = self._registry.get(tool_call.name)
            except AgenticResolverContractError:
                can_continue = self._append_paired_contract_error(
                    session,
                    turn=turn,
                    tool_call_id=tool_call.call_id,
                    code="unknown_tool",
                    message=f"Use one registered tool; {tool_call.name!r} is unknown.",
                )
                if can_continue:
                    continue
                result = _hard_result(
                    session,
                    request=request,
                    status="failed",
                    summary="The model exhausted structural replacements.",
                    reason="contract_replacement_cap",
                )
                return result

            if definition.name == "submit_result":
                terminal = self._submit_result(
                    session,
                    turn=turn,
                    request=request,
                    arguments=tool_call.arguments,
                )
                if terminal is not None:
                    return terminal
                continue
            if session.usage.tool_calls >= self._limits.max_tool_calls:
                result = _hard_result(
                    session,
                    request=request,
                    status="budget_exhausted",
                    summary="The resolver reached its non-terminal tool-call limit.",
                    reason="tool_call_cap",
                )
                return result
            session.record_tool_call(tool_name=definition.name)
            if definition.name == "skill":
                self._load_skill(
                    session,
                    turn=turn,
                    tool_call_id=tool_call.call_id,
                    arguments=tool_call.arguments,
                    definition_schema=definition.input_schema,
                )
                continue
            if definition.name == "run_subagent":
                await self._run_subagent(
                    session,
                    turn=turn,
                    tool_call_id=tool_call.call_id,
                    arguments=tool_call.arguments,
                    definition_schema=definition.input_schema,
                    deadline=deadline,
                )
                continue
            await self._run_ordinary_tool(
                session,
                turn=turn,
                tool_name=definition.name,
                tool_call_id=tool_call.call_id,
                arguments=tool_call.arguments,
                deadline=deadline,
            )

    async def _stream_turn(
        self,
        *,
        session: ResolverSession,
        messages,
        tools,
    ) -> AssembledAssistantTurn:
        """Consume every chunk into one bounded assembler before dispatch."""

        assembler = ModelStreamAssembler(
            max_output_characters=(
                self._limits.completion_reserve_tokens * 4
            )
        )
        async for chunk in self._model.astream(messages, tools=tools):
            session.record_stream_chunk(chunk)
            assembler.consume(chunk)
        turn = assembler.finalize()
        return turn

    def _submit_result(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        request: AgenticResolverRequestV1,
        arguments: Mapping[str, object],
    ) -> AgenticResolverResultV1 | None:
        """Validate model-owned terminal semantics and known evidence refs."""

        tool_call = turn.tool_calls[0]
        definition = self._registry.get("submit_result")
        try:
            normalized = validate_json_schema_arguments(
                arguments,
                definition.input_schema,
                label="submit_result arguments",
            )
            submitted = SubmitResultV1.from_mapping(normalized)
            _validate_terminal_evidence(submitted, session)
            _validate_observation_handle_placement(submitted, session)
        except AgenticResolverContractError as exc:
            can_continue = self._append_paired_contract_error(
                session,
                turn=turn,
                tool_call_id=tool_call.call_id,
                code="invalid_submit_result",
                message=f"Replace submit_result with valid terminal arguments: {exc}",
            )
            if can_continue:
                return None
            result = _hard_result(
                session,
                request=request,
                status="failed",
                summary="The model exhausted structural replacements.",
                reason="contract_replacement_cap",
            )
            return result

        session.append_terminal_turn(turn)
        session.record_terminal(status=submitted.status, reason="submit_result")
        result = AgenticResolverResultV1(
            session_id=session.session_id,
            status=submitted.status,
            summary=submitted.summary,
            evidence=submitted.evidence,
            completed_tasks=submitted.completed_tasks,
            remaining_needs=submitted.remaining_needs,
            usage=_usage_copy(session.usage),
        )
        return result

    def _load_skill(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        tool_call_id: str,
        arguments: Mapping[str, object],
        definition_schema: Mapping[str, object],
    ) -> None:
        """Validate and append one lazily loaded trusted skill body."""

        try:
            normalized = validate_json_schema_arguments(
                arguments,
                definition_schema,
                label="skill arguments",
            )
            name = cast(str, normalized["name"])
            definition = self._skills.load(name)
        except AgenticResolverContractError as exc:
            self._append_tool_error(
                session,
                turn=turn,
                tool_call_id=tool_call_id,
                tool_name="skill",
                error=str(exc),
            )
            return
        content = skill_content_message(
            name=definition.name,
            description=definition.description,
            catalog_digest=self._skills.catalog_digest,
            content=definition.body,
        )
        session.append_exchange(
            turn,
            tool_content=content,
            tool_call_id=tool_call_id,
        )
        session.mark_skill_loaded(definition.name)

    async def _run_subagent(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        tool_call_id: str,
        arguments: Mapping[str, object],
        definition_schema: Mapping[str, object],
        deadline: float,
    ) -> None:
        """Run one same-runtime child and append only its bounded projection."""

        try:
            normalized = validate_json_schema_arguments(
                arguments,
                definition_schema,
                label="run_subagent arguments",
            )
            task = AgenticResolverSubagentTaskV1.from_mapping(normalized)
            child_result = await self._subagent_runner.run(
                parent_session=session,
                task=task,
                deadline=deadline,
            )
        except AgenticResolverContractError as exc:
            self._append_tool_error(
                session,
                turn=turn,
                tool_call_id=tool_call_id,
                tool_name="run_subagent",
                error=str(exc),
            )
            return
        # The child boundary converts nested transport failures into one result.
        except Exception as exc:  # noqa: BLE001
            self._append_tool_error(
                session,
                turn=turn,
                tool_call_id=tool_call_id,
                tool_name="run_subagent",
                error=sanitized_exception_text(exc),
            )
            return
        content = subagent_result_message(child_result)
        observation_id = child_result.observation_id
        evidence_refs = tuple(
            reference
            for evidence in child_result.evidence
            for reference in evidence.provenance_refs
        )
        observation = ResolverObservation(
            observation_id=observation_id,
            tool_name="run_subagent",
            status=child_result.status,
            summary=child_result.summary,
            evidence_refs=evidence_refs,
        )
        compacted_content = compacted_observation_message(
            observation_id=observation_id,
            tool_name="run_subagent",
            status=child_result.status,
            summary=child_result.summary,
            evidence_refs=evidence_refs,
        )
        session.append_exchange(
            turn,
            tool_content=content,
            tool_call_id=tool_call_id,
            compacted_content=compacted_content,
            observation=observation,
        )

    async def _run_ordinary_tool(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        tool_name: str,
        tool_call_id: str,
        arguments: Mapping[str, object],
        deadline: float,
    ) -> None:
        """Execute one registered ordinary capability and append its observation."""

        remaining_seconds = max(0.0, deadline - time.monotonic())
        timeout_seconds = min(
            self._limits.tool_timeout_seconds,
            remaining_seconds,
        )
        if timeout_seconds <= 0:
            execution_result = ToolExecutionResult(
                status="error",
                output={},
                error="session deadline elapsed before tool execution",
            )
        else:
            try:
                execution_result = await self._registry.execute_tool(
                    tool_name,
                    arguments,
                    permission_scope=self._permission_scope,
                    timeout_seconds=timeout_seconds,
                    maximum_result_characters=(
                        self._limits.max_tool_result_characters
                    ),
                )
            except AgenticResolverContractError as exc:
                execution_result = ToolExecutionResult(
                    status="error",
                    output={},
                    error=str(exc),
                )
        self._append_execution_result(
            session,
            turn=turn,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            execution_result=execution_result,
        )

    def _append_execution_result(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        tool_call_id: str,
        tool_name: str,
        execution_result: ToolExecutionResult,
    ) -> None:
        """Project one bounded ordinary result into full and compact history."""

        observation_id = session.next_observation_id()
        summary = _result_summary(
            tool_name,
            execution_result.output,
            execution_result.error,
        )
        evidence_refs = _result_text_tuple(
            execution_result.output,
            keys=("provenance_refs", "evidence_refs"),
        )
        content = tool_observation_message(
            tool_call_id=tool_call_id,
            observation_id=observation_id,
            tool_name=tool_name,
            status=execution_result.status,
            output=execution_result.output,
            error=execution_result.error,
        )
        observation = ResolverObservation(
            observation_id=observation_id,
            tool_name=tool_name,
            status=execution_result.status,
            summary=summary,
            evidence_refs=evidence_refs,
        )
        compacted_content = compacted_observation_message(
            observation_id=observation_id,
            tool_name=tool_name,
            status=execution_result.status,
            summary=summary,
            evidence_refs=evidence_refs,
        )
        session.append_exchange(
            turn,
            tool_content=content,
            tool_call_id=tool_call_id,
            compacted_content=compacted_content,
            observation=observation,
        )

    def _append_tool_error(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        tool_call_id: str,
        tool_name: str,
        error: str,
    ) -> None:
        """Append one schema or core-tool failure as a normal tool observation."""

        execution_result = ToolExecutionResult(
            status="error",
            output={},
            error=error[:500],
        )
        self._append_execution_result(
            session,
            turn=turn,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            execution_result=execution_result,
        )

    def _append_unpaired_contract_error(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        code: str,
        message: str,
    ) -> bool:
        """Append JSON replacement feedback when no valid call can be paired."""

        if session.usage.contract_errors >= self._limits.max_contract_replacements:
            return False
        remaining = (
            self._limits.max_contract_replacements
            - session.usage.contract_errors
            - 1
        )
        content = contract_error_message(
            code=code,
            message=message,
            remaining_replacements=remaining,
        )
        session.record_rejected_turn(turn, code=code)
        session.append_protocol_feedback(content, code=code)
        return True

    def _append_paired_contract_error(
        self,
        session: ResolverSession,
        *,
        turn: AssembledAssistantTurn,
        tool_call_id: str,
        code: str,
        message: str,
    ) -> bool:
        """Pair invalid selected calls with JSON feedback for replay fidelity."""

        if session.usage.contract_errors >= self._limits.max_contract_replacements:
            return False
        remaining = (
            self._limits.max_contract_replacements
            - session.usage.contract_errors
            - 1
        )
        content = contract_error_message(
            code=code,
            message=message,
            remaining_replacements=remaining,
        )
        session.append_exchange(
            turn,
            tool_content=content,
            tool_call_id=tool_call_id,
        )
        session.record_contract_error(code=code)
        return True


def _assistant_content_error(content: str) -> str | None:
    """Return structural feedback for non-empty non-object assistant text."""

    if not content:
        return None
    try:
        parse_json_object(content)
    except AgenticResolverContractError:
        return "Assistant text must be empty or exactly one JSON object."
    return None


def _validate_terminal_evidence(
    submitted: SubmitResultV1,
    session: ResolverSession,
) -> None:
    """Require every terminal evidence row to bind known accepted provenance."""

    seen_ids: set[str] = set()
    for evidence in submitted.evidence:
        if evidence.observation_id in seen_ids:
            raise AgenticResolverContractError(
                "submit_result.evidence contains a duplicate observation_id"
            )
        seen_ids.add(evidence.observation_id)
        observation = session.observations.get(evidence.observation_id)
        if observation is None:
            raise AgenticResolverContractError(
                "submit_result.evidence references an unknown observation_id"
            )
        if observation.status not in {"success", "resolved", "partial"}:
            raise AgenticResolverContractError(
                "submit_result.evidence references an unsuccessful observation"
            )
        unknown_refs = set(evidence.provenance_refs) - set(
            observation.evidence_refs
        )
        if unknown_refs:
            raise AgenticResolverContractError(
                "submit_result.evidence contains unknown provenance_refs"
            )


def _validate_observation_handle_placement(
    submitted: SubmitResultV1,
    session: ResolverSession,
) -> None:
    """Keep accepted observation handles in the structured evidence field."""

    semantic_texts: list[tuple[str, str]] = [
        ("summary", submitted.summary),
    ]
    for index, evidence in enumerate(submitted.evidence):
        semantic_texts.append((f"evidence[{index}].summary", evidence.summary))
        semantic_texts.extend(
            (
                f"evidence[{index}].limitations[{limitation_index}]",
                limitation,
            )
            for limitation_index, limitation in enumerate(evidence.limitations)
        )
    semantic_texts.extend(
        (f"completed_tasks[{index}]", task)
        for index, task in enumerate(submitted.completed_tasks)
    )
    semantic_texts.extend(
        (f"remaining_needs[{index}]", need)
        for index, need in enumerate(submitted.remaining_needs)
    )

    observation_ids = tuple(session.observations)
    for field_name, text in semantic_texts:
        if any(observation_id in text for observation_id in observation_ids):
            raise AgenticResolverContractError(
                f"{field_name} must not repeat an observation handle; keep "
                "handles only in evidence[].observation_id"
            )


def _result_summary(
    tool_name: str,
    output: Mapping[str, object],
    error: str | None,
) -> str:
    """Return one compact observation summary from an explicit projection."""

    summary = output.get("summary")
    if isinstance(summary, str) and summary.strip():
        normalized_summary = summary.strip()[:2000]
        return normalized_summary
    if error:
        return error[:500]
    default_summary = (
        f"Tool {tool_name} completed with a bounded object result."
    )
    return default_summary


def _result_text_tuple(
    output: Mapping[str, object],
    *,
    keys: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the first bounded string-list field from a tool projection."""

    for key in keys:
        value = output.get(key)
        if not isinstance(value, list):
            continue
        items = tuple(
            item[:2000]
            for item in value[:16]
            if isinstance(item, str) and item.strip()
        )
        return items
    empty_items: tuple[str, ...] = ()
    return empty_items


def _usage_copy(usage: AgenticResolverUsageV1) -> AgenticResolverUsageV1:
    """Freeze the session's current counters into a result-owned value."""

    copied = AgenticResolverUsageV1(
        model_steps=usage.model_steps,
        tool_calls=usage.tool_calls,
        subagent_runs=usage.subagent_runs,
        contract_errors=usage.contract_errors,
        compactions=usage.compactions,
        estimated_context_tokens_peak=usage.estimated_context_tokens_peak,
        provider_usage=dict(usage.provider_usage),
    )
    return copied


def _hard_result(
    session: ResolverSession,
    *,
    request: AgenticResolverRequestV1,
    status: ResolverStatus,
    summary: str,
    reason: str,
) -> AgenticResolverResultV1:
    """Build a deterministic fail-closed result for one hard runtime cap."""

    remaining_needs: tuple[str, ...] = ()
    if status != "resolved":
        remaining_needs = (request.objective,)
    session.record_terminal(status=status, reason=reason)
    result = AgenticResolverResultV1(
        session_id=session.session_id,
        status=status,
        summary=summary[:4000],
        evidence=(),
        completed_tasks=(),
        remaining_needs=remaining_needs,
        usage=_usage_copy(session.usage),
    )
    return result
