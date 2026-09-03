"""Shared Brain/DSH task-resolution lifecycle and durable handoff helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Mapping
from inspect import isawaitable

from kazusa_ai_chatbot.cognition_episode import validate_goal_continuation_ref
from kazusa_ai_chatbot.db.errors import DatabaseOperationError
from kazusa_ai_chatbot.dsh_tool_gateway.contracts import (
    content_digest,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    DshResolutionRefV1,
    TaskResolutionAdmissionV1,
    TaskResolutionContractError,
    TaskResolutionExecutionContextV2,
    TaskResolutionResultV1,
    validate_accepted_task_control,
    validate_dsh_resolution_ref,
    validate_dsh_task_start_spec,
    validate_task_resolution_admission,
    validate_task_resolution_execution_context,
)
from kazusa_ai_chatbot.task_resolution.projection import (
    build_model_facts,
    project_dsh_exhaust,
)

MINIMUM_INLINE_BUDGET_SECONDS = 0.001
MAXIMUM_INLINE_BUDGET_SECONDS = 120.0

_TASK_RESOLUTION_RUNTIME: object | None = None
_TASK_RESOLUTION_BINDING_STORE: object | None = None
_TASK_RESOLUTION_ACCEPTED_TASK_STORE: object | None = None
_TASK_RESOLUTION_BACKGROUND_QUEUE: object | None = None


def configure_task_resolution_runtime(
    runtime: object | None,
    *,
    binding_store: object | None = None,
    accepted_task_store: object | None = None,
    background_queue: object | None = None,
) -> None:
    """Bind or clear the shared runtime and durable collaborators."""

    global _TASK_RESOLUTION_RUNTIME
    global _TASK_RESOLUTION_BINDING_STORE
    global _TASK_RESOLUTION_ACCEPTED_TASK_STORE
    global _TASK_RESOLUTION_BACKGROUND_QUEUE
    configured = (
        runtime,
        binding_store,
        accepted_task_store,
        background_queue,
    )
    if runtime is None and any(value is not None for value in configured[1:]):
        raise TaskResolutionContractError(
            "task-resolution runtime collaborators cannot be partially cleared",
        )
    if runtime is not None and any(value is None for value in configured[1:]):
        raise TaskResolutionContractError(
            "task-resolution runtime requires every durable collaborator",
        )
    _TASK_RESOLUTION_RUNTIME = runtime
    _TASK_RESOLUTION_BINDING_STORE = binding_store
    _TASK_RESOLUTION_ACCEPTED_TASK_STORE = accepted_task_store
    _TASK_RESOLUTION_BACKGROUND_QUEUE = background_queue


async def resolve_task_inline(
    request: Mapping[str, object],
    execution_context: Mapping[str, object],
    *,
    inline_budget_seconds: float,
    runtime: object | None = None,
    binding_store: object | None = None,
) -> TaskResolutionResultV1:
    """Run one foreground operation and terminally close failed admission."""

    context = _context_for_service(execution_context)
    task_session_id = _task_session_id(request, context)
    configured_store = (
        binding_store
        if binding_store is not None
        else _TASK_RESOLUTION_BINDING_STORE
    )
    try:
        return await _resolve_task_inline_operation(
            request,
            context,
            inline_budget_seconds=inline_budget_seconds,
            runtime=runtime,
            binding_store=configured_store,
        )
    except Exception:
        if configured_store is not None:
            try:
                await fault_task_resolution_binding(
                    binding_store=configured_store,
                    task_session_id=task_session_id,
                    updated_at=str(context.get("current_timestamp_utc", "")),
                )
            except Exception as cleanup_exc:
                raise TaskResolutionContractError(
                    "failed DSH admission could not be terminally recorded",
                ) from cleanup_exc
        raise


async def _resolve_task_inline_operation(
    request: Mapping[str, object],
    execution_context: Mapping[str, object],
    *,
    inline_budget_seconds: float,
    runtime: object | None = None,
    binding_store: object | None = None,
) -> TaskResolutionResultV1:
    """Run one DSH operation under a shielded caller budget."""

    _validate_inline_budget(inline_budget_seconds)
    context = _context_for_service(execution_context)
    semantic_objective = _request_objective(request)
    task_session_id = _task_session_id(request, context)
    start_spec = _build_start_spec(request, context)
    binding_store = (
        binding_store
        if binding_store is not None
        else _TASK_RESOLUTION_BINDING_STORE
    )
    runtime = runtime if runtime is not None else _TASK_RESOLUTION_RUNTIME
    _require_runtime(runtime)
    _require_store(binding_store, "create_task_binding")
    binding: Mapping[str, object] = {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": task_session_id,
        "semantic_objective": semantic_objective,
        "goal_continuation_ref": context.get("goal_continuation_ref", {}),
        "source_scope": _source_scope(context),
        "state": "queued",
        "start_spec": start_spec,
        "resolution_thread_id": None,
        "segment_id": None,
        "resolution_ref": None,
        "operation_generation": 0,
        "current_accepted_task_id": None,
        "current_background_work_job_id": None,
        "latest_task_resolution_result": None,
        "revision": 0,
        "created_at": str(context.get("current_timestamp_utc", "")),
        "updated_at": str(context.get("current_timestamp_utc", "")),
    }
    stored_binding = await _store_call(
        binding_store,
        "create_task_binding",
        binding=dict(binding),
    )
    if not isinstance(stored_binding, Mapping):
        raise TaskResolutionContractError(
            "DSH task-binding creation returned no durable binding",
        )
    binding = dict(stored_binding)
    if binding.get("task_session_id") != task_session_id:
        raise TaskResolutionContractError(
            "DSH task-binding creation returned a different session",
        )

    open_method = getattr(runtime, "open", None)
    if not callable(open_method):
        raise TaskResolutionContractError("DSH runtime does not expose open")

    async def before_resolve(reference: Mapping[str, object]) -> object:
        """Persist the admitted runtime identity before sidecar execution."""

        nonlocal binding
        binding = await _open_binding_before_activation(
            binding_store=binding_store,
            binding=binding,
            task_session_id=task_session_id,
            updated_at=str(context.get("current_timestamp_utc", "")),
        )
        attached = await _store_call(
            binding_store,
            "attach_resolution_ref",
            task_session_id=task_session_id,
            expected_revision=int(binding.get("revision", 0)),
            resolution_ref=dict(reference),
            updated_at=str(context.get("current_timestamp_utc", "")),
        )
        if isinstance(attached, Mapping):
            binding = dict(attached)
        return binding

    opening = open_method(
        task_session_id=task_session_id,
        operation_generation=0,
        request=dict(request),
        execution_context=dict(context),
        start_spec=start_spec,
        before_resolve=before_resolve,
    )
    deadline = asyncio.get_running_loop().time() + inline_budget_seconds
    if isawaitable(opening):
        opening_task = asyncio.ensure_future(opening)
        try:
            opened = await asyncio.wait_for(
                asyncio.shield(opening_task),
                timeout=inline_budget_seconds,
            )
        except asyncio.TimeoutError:
            opening_task.add_done_callback(_consume_task_outcome)
            checkpoint = await _request_checkpoint(
                runtime,
                task_session_id=task_session_id,
                operation_generation=0,
            )
            if _runtime_result_is_terminal(opening_task):
                opened = await _await_value(opening_task)
                if _is_exhaust(opened):
                    terminal_result = project_dsh_exhaust(opened, start_spec)
                    await _record_inline_binding_outcome(
                        binding_store=binding_store,
                        binding=binding,
                        task_session_id=task_session_id,
                        exhaust=opened,
                        result=terminal_result,
                    )
                    return terminal_result
            if _is_terminal_mapping(checkpoint):
                terminal_result = project_dsh_exhaust(checkpoint, start_spec)
                await _record_inline_binding_outcome(
                    binding_store=binding_store,
                    binding=binding,
                    task_session_id=task_session_id,
                    exhaust=checkpoint,
                    result=terminal_result,
                )
                return terminal_result
            await _record_inline_binding_outcome(
                binding_store=binding_store,
                binding=binding,
                task_session_id=task_session_id,
                exhaust=checkpoint,
                result=None,
            )
            return _deferred_result(
                semantic_objective,
                context,
                _inline_resolution_ref(
                    checkpoint,
                    task_session_id=task_session_id,
                    binding=binding,
                ),
                start_spec,
            )
    else:
        opened = opening
    if (
        isinstance(opened, Mapping)
        and (
            opened.get("kind") == "checkpointed"
            or (
                isinstance(opened.get("exhaust"), Mapping)
                and opened["exhaust"].get("kind") == "checkpointed"
            )
        )
    ):
        result = await _checkpoint_after_open(
            runtime,
            opened,
            semantic_objective,
            context,
            start_spec,
            task_session_id,
            binding,
        )
        await _record_inline_binding_outcome(
            binding_store=binding_store,
            binding=binding,
            task_session_id=task_session_id,
            exhaust=opened,
            result=None,
        )
        return result
    if _is_exhaust(opened):
        result = project_dsh_exhaust(opened, start_spec)
        await _record_inline_binding_outcome(
            binding_store=binding_store,
            binding=binding,
            task_session_id=task_session_id,
            exhaust=opened,
            result=result,
        )
        return result
    if isinstance(opened, Mapping):
        result = await _checkpoint_after_open(
            runtime,
            opened,
            semantic_objective,
            context,
            start_spec,
            task_session_id,
            binding,
        )
        await _record_inline_binding_outcome(
            binding_store=binding_store,
            binding=binding,
            task_session_id=task_session_id,
            exhaust=opened,
            result=None,
        )
        return result
    if not isinstance(opened, Awaitable):
        raise TaskResolutionContractError("DSH runtime open returned an invalid handle")

    reasoning = asyncio.ensure_future(opened)
    try:
        exhaust = await asyncio.wait_for(
            asyncio.shield(reasoning),
            timeout=max(0.0, deadline - asyncio.get_running_loop().time()),
        )
    except asyncio.TimeoutError:
        checkpoint = await _request_checkpoint(
            runtime,
            task_session_id=task_session_id,
            operation_generation=0,
        )
        if _runtime_result_is_terminal(reasoning):
            exhaust = await _await_value(reasoning)
            if _is_exhaust(exhaust):
                terminal_result = project_dsh_exhaust(exhaust, start_spec)
                await _record_inline_binding_outcome(
                    binding_store=binding_store,
                    binding=binding,
                    task_session_id=task_session_id,
                    exhaust=exhaust,
                    result=terminal_result,
                )
                return terminal_result
        if _is_terminal_mapping(checkpoint):
            terminal_result = project_dsh_exhaust(checkpoint, start_spec)
            await _record_inline_binding_outcome(
                binding_store=binding_store,
                binding=binding,
                task_session_id=task_session_id,
                exhaust=checkpoint,
                result=terminal_result,
            )
            return terminal_result
        await _record_inline_binding_outcome(
            binding_store=binding_store,
            binding=binding,
            task_session_id=task_session_id,
            exhaust=checkpoint,
            result=None,
        )
        return _deferred_result(
            semantic_objective,
            context,
            _inline_resolution_ref(
                checkpoint,
                task_session_id=task_session_id,
                binding=binding,
            ),
            start_spec,
        )
    result = project_dsh_exhaust(exhaust, start_spec)
    await _record_inline_binding_outcome(
        binding_store=binding_store,
        binding=binding,
        task_session_id=task_session_id,
        exhaust=exhaust,
        result=result,
    )
    return result


def _is_exhaust(value: object) -> bool:
    """Identify a completed DSH exhaust before treating mappings as metadata."""

    if isinstance(value, Mapping):
        nested = value.get("exhaust")
        candidate = nested if isinstance(nested, Mapping) else value
        return candidate.get("kind") in {
            "terminal",
            "checkpointed",
            "runtime_fault",
            "canceled",
        }
    return hasattr(value, "kind") and hasattr(value, "to_dict")


def _is_terminal_mapping(value: object) -> bool:
    """Identify a terminal exhaust returned by a checkpoint race."""

    if isinstance(value, Mapping):
        nested = value.get("exhaust")
        candidate = nested if isinstance(nested, Mapping) else value
        return candidate.get("kind") == "terminal"
    return getattr(value, "kind", None) == "terminal"


def _runtime_result_is_terminal(task: asyncio.Future[object]) -> bool:
    """Check a completed shielded operation without consuming exceptions."""

    if not task.done() or task.cancelled():
        return False
    value = task.result()
    return _is_terminal_mapping(value) or (
        _is_exhaust(value) and getattr(value, "kind", None) == "terminal"
    )


async def _open_binding_before_activation(
    *,
    binding_store: object,
    binding: Mapping[str, object],
    task_session_id: str,
    updated_at: str,
) -> Mapping[str, object]:
    """Move a foreground binding to opening before sidecar RPC execution."""

    current = dict(binding)
    state = current.get("state")
    if state == "opening":
        return current
    if state != "queued":
        raise TaskResolutionContractError(
            "foreground DSH binding is not openable in its current state",
        )
    revision = current.get("revision", 0)
    generation = current.get("operation_generation", 0)
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(generation, int)
        or isinstance(generation, bool)
    ):
        raise TaskResolutionContractError("foreground DSH binding CAS fields are invalid")
    transitioned = await _store_call(
        binding_store,
        "transition_task_binding",
        task_session_id=task_session_id,
        expected_revision=revision,
        expected_state="queued",
        next_state="opening",
        operation_generation=generation,
        updated_at=updated_at,
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "foreground DSH binding opening transition was not durable",
        )
    return dict(transitioned)


async def fault_task_resolution_binding(
    *,
    task_session_id: str,
    updated_at: str,
    binding_store: object | None = None,
) -> None:
    """Close one failed task-resolution operation through its durable graph."""

    selected_store = (
        binding_store
        if binding_store is not None
        else _TASK_RESOLUTION_BINDING_STORE
    )
    if selected_store is None:
        raise TaskResolutionContractError(
            "failed DSH operation has no durable binding store",
        )

    stored = await _store_call(
        selected_store,
        "find_binding_by_session",
        task_session_id=task_session_id,
    )
    if stored is None:
        return
    if not isinstance(stored, Mapping):
        raise TaskResolutionContractError(
            "failed DSH admission returned an invalid durable binding",
        )
    current = dict(stored)
    state = current.get("state")
    if state in {"faulted", "canceled", "terminal", "consumed_inline"}:
        return
    if state not in {"queued", "opening", "checkpointed", "active"}:
        raise TaskResolutionContractError(
            "failed DSH admission has an unsupported binding state",
        )
    revision = current.get("revision")
    generation = current.get("operation_generation")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(generation, int)
        or isinstance(generation, bool)
    ):
        raise TaskResolutionContractError(
            "failed DSH admission has invalid binding fences",
        )
    transitioned = await _store_call(
        selected_store,
        "transition_task_binding",
        task_session_id=task_session_id,
        expected_revision=revision,
        expected_state=state,
        next_state="faulted",
        operation_generation=generation,
        updated_at=updated_at,
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "failed DSH admission fault transition was not durable",
        )


async def _record_inline_binding_outcome(
    *,
    binding_store: object,
    binding: Mapping[str, object],
    task_session_id: str,
    exhaust: object,
    result: Mapping[str, object] | None,
) -> None:
    """Persist foreground identity/result and close the inline generation."""

    current = dict(binding)
    if current.get("state") == "queued":
        current = dict(await _open_binding_before_activation(
            binding_store=binding_store,
            binding=current,
            task_session_id=task_session_id,
            updated_at=_binding_updated_at(current),
        ))
    reference = _inline_resolution_ref(
        exhaust,
        task_session_id=task_session_id,
        binding=current,
    )
    stored_reference = current.get("resolution_ref")
    if isinstance(stored_reference, Mapping):
        stored_reference = validate_dsh_resolution_ref(stored_reference)
        if stored_reference != reference:
            raise TaskResolutionContractError(
                "foreground DSH exhaust reference conflicts with its binding",
            )
    else:
        revision = current.get("revision", 0)
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise TaskResolutionContractError(
                "foreground DSH binding revision is invalid",
            )
        attached = await _store_call(
            binding_store,
            "attach_resolution_ref",
            task_session_id=task_session_id,
            expected_revision=revision,
            resolution_ref=dict(reference),
            updated_at=_binding_updated_at(current),
        )
        if not isinstance(attached, Mapping):
            raise TaskResolutionContractError(
                "foreground DSH resolution reference was not durable",
            )
        current = dict(attached)
    kind = _inline_exhaust_kind(exhaust)
    next_state = {
        "terminal": "consumed_inline",
        "checkpointed": "checkpointed",
        "runtime_fault": "faulted",
        "canceled": "canceled",
    }.get(kind)
    if next_state is None:
        raise TaskResolutionContractError(
            "foreground DSH exhaust kind is unsupported",
        )
    if result is not None and kind == "terminal":
        latest = current.get("latest_task_resolution_result")
        revision = current.get("revision", 0)
        if latest != dict(result):
            if not isinstance(revision, int) or isinstance(revision, bool):
                raise TaskResolutionContractError(
                    "foreground DSH binding revision is invalid",
                )
            reconciled = await _store_call(
                binding_store,
                "reconcile_task_resolution_result",
                task_session_id=task_session_id,
                expected_revision=revision,
                operation_generation=0,
                task_resolution_result=dict(result),
                updated_at=_binding_updated_at(current),
            )
            if not isinstance(reconciled, Mapping):
                raise TaskResolutionContractError(
                    "foreground DSH result reconciliation was not durable",
                )
            current = dict(reconciled)
    state = current.get("state")
    if not isinstance(state, str) or state == next_state:
        return
    if not _inline_transition_allowed(state, next_state):
        raise TaskResolutionContractError(
            f"foreground DSH binding cannot transition {state} to {next_state}",
        )
    revision = current.get("revision", 0)
    generation = current.get("operation_generation", 0)
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(generation, int)
        or isinstance(generation, bool)
    ):
        raise TaskResolutionContractError("foreground DSH binding CAS fields are invalid")
    transitioned = await _store_call(
        binding_store,
        "transition_task_binding",
        task_session_id=task_session_id,
        expected_revision=revision,
        expected_state=state,
        next_state=next_state,
        operation_generation=generation,
        updated_at=_binding_updated_at(current),
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "foreground DSH terminal transition was not durable",
        )


def _inline_resolution_ref(
    exhaust: object,
    *,
    task_session_id: str,
    binding: Mapping[str, object],
) -> DshResolutionRefV1:
    """Extract the exact durable reference from one inline exhaust or binding."""

    def same_session(reference: DshResolutionRefV1) -> DshResolutionRefV1:
        if reference["dsh_session_id"] != task_session_id:
            raise TaskResolutionContractError(
                "foreground DSH exhaust reference has a different session",
            )
        return reference

    value: object = exhaust.to_dict() if hasattr(exhaust, "to_dict") else exhaust
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(
            "DSH inline exhaust does not contain a durable reference",
        )
    nested = value.get("exhaust")
    candidate = nested if isinstance(nested, Mapping) else value
    checkpoint = candidate.get("dsh_resolution_ref")
    if not isinstance(checkpoint, Mapping):
        checkpoint = candidate.get("checkpoint")
    if isinstance(checkpoint, Mapping):
        try:
            reference = validate_dsh_resolution_ref(checkpoint)
        except TaskResolutionContractError as exc:
            raise TaskResolutionContractError(
                "DSH inline exhaust contains an invalid durable reference",
            ) from exc
        else:
            return same_session(reference)
    identity = candidate.get("identity")
    if isinstance(identity, Mapping):
        stored = binding.get("resolution_ref")
        if isinstance(stored, Mapping):
            reference = validate_dsh_resolution_ref(stored)
            for field in (
                "resolution_thread_id",
                "segment_id",
                "dsh_session_id",
                "activation_id",
                "lease_epoch",
                "document_revision",
                "last_committed_seq",
            ):
                if field in identity and identity[field] != reference[field]:
                    raise TaskResolutionContractError(
                        "DSH inline exhaust identity conflicts with its binding",
                    )
            return same_session(reference)
    try:
        reference = validate_dsh_resolution_ref({
            field: candidate.get(field)
            for field in (
                "schema_version",
                "resolution_thread_id",
                "segment_id",
                "dsh_session_id",
                "activation_id",
                "lease_epoch",
                "document_revision",
                "last_committed_seq",
            )
        })
    except TaskResolutionContractError:
        stored = binding.get("resolution_ref")
        if isinstance(stored, Mapping):
            reference = validate_dsh_resolution_ref(stored)
            if reference["dsh_session_id"] != task_session_id:
                raise TaskResolutionContractError(
                    "foreground DSH binding reference has a different session",
                )
            return reference
        raise TaskResolutionContractError(
            "DSH inline exhaust does not contain a complete durable reference",
        )
    return same_session(reference)


def _inline_exhaust_kind(exhaust: object) -> str | None:
    """Read the closed kind discriminator from an inline exhaust."""

    if hasattr(exhaust, "kind"):
        return str(exhaust.kind)
    if isinstance(exhaust, Mapping):
        candidate = exhaust.get("exhaust")
        if not isinstance(candidate, Mapping):
            candidate = exhaust
        kind = candidate.get("kind", candidate.get("disposition"))
        return kind if isinstance(kind, str) else None
    return None


def _inline_transition_allowed(current: str, next_state: str) -> bool:
    """Keep foreground binding state inside the DSH transition graph."""

    allowed = {
        "queued": {"opening", "canceled", "faulted"},
        "opening": {
            "checkpointed",
            "terminal",
            "canceled",
            "faulted",
            "consumed_inline",
        },
        "checkpointed": {"active", "canceled", "faulted"},
        "active": {
            "checkpointed",
            "terminal",
            "canceled",
            "faulted",
        },
        "terminal": {"active", "canceled", "terminal"},
        "consumed_inline": {"consumed_inline"},
    }
    return next_state in allowed.get(current, set())


def _consume_task_outcome(task: asyncio.Future[object]) -> None:
    """Consume a timed-out runtime task outcome to avoid unobserved errors."""

    try:
        task.exception()
    except asyncio.CancelledError:
        return


async def _checkpoint_after_open(
    runtime: object,
    opened: Mapping[str, object],
    semantic_objective: str,
    context: Mapping[str, object],
    start_spec: Mapping[str, object],
    task_session_id: str,
    binding: Mapping[str, object],
) -> TaskResolutionResultV1:
    """Use the cooperative checkpoint RPC for a synchronous open response."""

    checkpoint = await _request_checkpoint(
        runtime,
        task_session_id=task_session_id,
        operation_generation=0,
        opened=opened,
    )
    selected = checkpoint if checkpoint else opened
    return _deferred_result(
        semantic_objective,
        context,
        _inline_resolution_ref(
            selected,
            task_session_id=task_session_id,
            binding=binding,
        ),
        start_spec,
    )


async def _request_checkpoint(runtime: object, **kwargs: object) -> Mapping[str, object]:
    """Request a cooperative checkpoint without canceling the open operation."""

    method = getattr(runtime, "request_checkpoint", None)
    if not callable(method):
        raise TaskResolutionContractError("DSH runtime does not expose checkpoint")
    value = await _await_value(method(**kwargs))
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError("DSH checkpoint returned an invalid result")
    return value


async def start_task_resolution_in_background(
    request: Mapping[str, object],
    execution_context: Mapping[str, object],
    *,
    source_trigger_source: str = "user_message",
    source_platform_bot_id: str = "",
    requester_display_name: str = "",
    source_llm_trace_id: str = "",
    binding_store: object | None = None,
    accepted_task_store: object | None = None,
    background_queue: object | None = None,
    authority_broker: object | None = None,
) -> TaskResolutionAdmissionV1:
    """Persist a generation-zero DSH job with no queued authority token."""

    del authority_broker
    _require_runtime(_TASK_RESOLUTION_RUNTIME)
    context = _context_for_service(execution_context)
    semantic_objective = _request_objective(request)
    task_session_id = _task_session_id(request, context)
    start_spec = _build_start_spec(request, context)
    binding_store = (
        binding_store
        if binding_store is not None
        else _TASK_RESOLUTION_BINDING_STORE
    )
    accepted_task_store = (
        accepted_task_store
        if accepted_task_store is not None
        else _TASK_RESOLUTION_ACCEPTED_TASK_STORE
    )
    background_queue = (
        background_queue
        if background_queue is not None
        else _TASK_RESOLUTION_BACKGROUND_QUEUE
    )
    _require_store(binding_store, "create_task_binding")
    _require_store(binding_store, "find_binding_by_session")
    _require_store(accepted_task_store, "create_or_return_active_accepted_task")
    _require_queue(background_queue)
    binding = {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": task_session_id,
        "semantic_objective": semantic_objective,
        "goal_continuation_ref": context.get("goal_continuation_ref", {}),
        "source_scope": _source_scope(context),
        "state": "queued",
        "start_spec": start_spec,
        "resolution_thread_id": None,
        "segment_id": None,
        "resolution_ref": None,
        "operation_generation": 0,
        "current_accepted_task_id": None,
        "current_background_work_job_id": None,
        "latest_task_resolution_result": None,
        "revision": 0,
        "created_at": str(context.get("current_timestamp_utc", "")),
        "updated_at": str(context.get("current_timestamp_utc", "")),
    }
    accepted_task: Mapping[str, object] = {}
    accepted_request = _accepted_task_request(
        semantic_objective=semantic_objective,
        context=context,
        source_trigger_source=source_trigger_source,
        source_platform_bot_id=source_platform_bot_id,
        requester_display_name=requester_display_name,
    )
    value = await _store_call(
        accepted_task_store,
        "create_or_return_active_accepted_task",
        request=accepted_request,
        dsh_task_session_id=task_session_id,
        dsh_operation_generation=0,
        dsh_followup_open=False,
    )
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(
            "accepted-task creation returned no durable task",
        )
    nested = value.get("task")
    accepted_task = nested if isinstance(nested, Mapping) else value
    accepted_task_id = _required_mapping_text(
        accepted_task,
        "accepted_task_id",
    )
    task_identity_key = _required_mapping_text(
        accepted_task,
        "task_identity_key",
    )
    creation_status = value.get("status")
    if creation_status == "already_active":
        if accepted_task.get("dsh_task_session_id") != task_session_id:
            raise TaskResolutionContractError(
                "active accepted task belongs to a different DSH session",
            )
        existing_binding = await _store_call(
            binding_store,
            "find_binding_by_session",
            task_session_id=task_session_id,
        )
        if not isinstance(existing_binding, Mapping):
            raise TaskResolutionContractError(
                "active accepted task has no durable DSH binding",
            )
        existing_binding = dict(existing_binding)
        if existing_binding.get("task_session_id") != task_session_id:
            raise TaskResolutionContractError(
                "active accepted task binding has a different session",
            )
        if existing_binding.get("operation_generation") != 0:
            raise TaskResolutionContractError(
                "active accepted task binding is not generation zero",
            )
        if existing_binding.get("current_accepted_task_id") != accepted_task_id:
            raise TaskResolutionContractError(
                "active accepted task binding has a different task identity",
            )
        existing_job_id = _required_mapping_text(
            existing_binding,
            "current_background_work_job_id",
        )
        admission = _task_resolution_admission(
            accepted_task_id=accepted_task_id,
            background_work_job_id=existing_job_id,
            task_session_id=task_session_id,
        )
        return admission
    try:
        stored_binding = await _store_call(
            binding_store,
            "create_task_binding",
            binding=binding,
        )
    except (DatabaseOperationError, TypeError, ValueError):
        await _store_call(
            accepted_task_store,
            "mark_accepted_task_enqueue_failed",
            accepted_task_id=accepted_task_id,
            failure_summary="DSH task binding creation failed.",
            updated_at=str(context.get("current_timestamp_utc", "")),
        )
        raise
    if not isinstance(stored_binding, Mapping):
        raise TaskResolutionContractError(
            "DSH task-binding creation returned no durable binding",
        )
    binding = dict(stored_binding)
    if binding.get("task_session_id") != task_session_id:
        raise TaskResolutionContractError(
            "DSH task-binding creation returned a different session",
        )
    binding_snapshot = await _store_call(
        binding_store,
        "attach_accepted_task",
        task_session_id=task_session_id,
        expected_revision=_required_revision(binding),
        operation_generation=0,
        accepted_task_id=accepted_task_id,
        updated_at=str(context.get("current_timestamp_utc", "")),
    )
    if not isinstance(binding_snapshot, Mapping):
        raise TaskResolutionContractError(
            "DSH task-binding accepted-task attachment was not durable",
        )
    binding = dict(binding_snapshot)
    job_id = f"dsh-task:{accepted_task_id}:generation:0"
    start_action_attempt_id = f"dsh-start:{task_session_id}"
    payload = {
        "schema_version": "task_orchestrator_worker_payload.v2",
        "operation": "open_dsh_resolution",
        "task_session_id": task_session_id,
        "operation_generation": 0,
        "control": None,
    }
    pending_task = await _store_call(
        accepted_task_store,
        "mark_accepted_task_pending",
        accepted_task_id=accepted_task_id,
        executor_ref=job_id,
        updated_at=str(context.get("current_timestamp_utc", "")),
    )
    if not isinstance(pending_task, Mapping):
        await _store_call(
            accepted_task_store,
            "mark_accepted_task_enqueue_failed",
            accepted_task_id=accepted_task_id,
            failure_summary="Accepted task could not become pending.",
            updated_at=str(context.get("current_timestamp_utc", "")),
        )
        raise TaskResolutionContractError(
            "accepted task pending transition failed",
        )
    try:
        queued = await _queue_payload(
            background_queue,
            payload,
            job_id=job_id,
            source_action_attempt_id=start_action_attempt_id,
            idempotency_key=f"dsh-start:{task_session_id}:0",
            accepted_task_id=accepted_task_id,
            task_identity_key=task_identity_key,
            semantic_objective=semantic_objective,
            goal_continuation_ref=context.get("goal_continuation_ref"),
            source_llm_trace_id=source_llm_trace_id,
            requested_worker="task_orchestrator",
            task_execution_context=dict(context),
            source_platform=str(context.get("platform", "")),
            source_channel_id=str(context.get("channel_id", "")),
            source_channel_type=str(context.get("channel_type", "")),
            source_message_id=str(context.get("source_message_id", "")),
            source_platform_bot_id=(
                source_platform_bot_id.strip()
                or str(context.get("source_platform_bot_id", ""))
            ),
            source_character_name=str(context.get("character_name", "")),
            requester_global_user_id=str(
                context.get("requester_global_user_id", ""),
            ),
            requester_platform_user_id=str(
                context.get("requester_platform_user_id", ""),
            ),
            requester_display_name=(
                requester_display_name.strip()
                or str(context.get("requester_display_name", ""))
            ),
            requested_delivery="send_result_when_done",
            max_output_chars=int(context.get("max_output_chars", 0)),
            storage_timestamp_utc=str(
                context.get("current_timestamp_utc", ""),
            ),
        )
    except (DatabaseOperationError, TypeError, ValueError):
        await _store_call(
            accepted_task_store,
            "mark_accepted_task_enqueue_failed",
            accepted_task_id=accepted_task_id,
            failure_summary="Background DSH task enqueue failed.",
            updated_at=str(context.get("current_timestamp_utc", "")),
        )
        await _store_call(
            binding_store,
            "transition_task_binding",
            task_session_id=task_session_id,
            expected_state="queued",
            next_state="faulted",
            operation_generation=0,
            expected_revision=_required_revision(binding),
        )
        raise
    if not isinstance(queued, Mapping):
        raise TaskResolutionContractError(
            "background queue returned no durable job",
        )
    queued_job_id = _required_mapping_text(queued, "job_id")
    if queued_job_id != job_id:
        raise TaskResolutionContractError(
            "background queue returned an unexpected job id",
        )
    attached_job = await _store_call(
        binding_store,
        "attach_background_job",
        task_session_id=task_session_id,
        expected_revision=_required_revision(binding),
        operation_generation=0,
        background_work_job_id=queued_job_id,
        updated_at=str(context.get("current_timestamp_utc", "")),
    )
    if not isinstance(attached_job, Mapping):
        raise TaskResolutionContractError(
            "DSH task-binding background-job attachment was not durable",
        )
    binding = dict(attached_job)
    if binding.get("task_session_id") != task_session_id:
        raise TaskResolutionContractError(
            "DSH task binding returned a different session",
        )
    if binding.get("operation_generation") != 0:
        raise TaskResolutionContractError(
            "DSH task binding returned a different generation",
        )
    bound_task_id = _required_mapping_text(
        binding,
        "current_accepted_task_id",
    )
    if bound_task_id != accepted_task_id:
        raise TaskResolutionContractError(
            "DSH task binding returned a different accepted-task identity",
        )
    bound_job_id = _required_mapping_text(
        binding,
        "current_background_work_job_id",
    )
    if bound_job_id != queued_job_id:
        raise TaskResolutionContractError(
            "DSH task binding returned a different background-job identity",
        )
    admission = _task_resolution_admission(
        accepted_task_id=bound_task_id,
        background_work_job_id=bound_job_id,
        task_session_id=task_session_id,
    )
    return admission


async def resume_task_resolution(
    checkpoint: Mapping[str, object],
    execution_context: Mapping[str, object],
    *,
    runtime: object | None = None,
) -> TaskResolutionResultV1:
    """Resume one generation-bound DSH checkpoint through the shared runtime."""

    context = _context_for_service(execution_context)
    ref_value = checkpoint.get("dsh_resolution_ref")
    if not isinstance(ref_value, Mapping):
        raise TaskResolutionContractError(
            "resume_task_resolution requires a durable DSH reference",
        )
    ref = validate_dsh_resolution_ref(ref_value)
    objective_value = checkpoint.get("semantic_objective")
    if not isinstance(objective_value, str) or not objective_value.strip():
        raise TaskResolutionContractError(
            "resume_task_resolution requires a semantic objective",
        )
    objective = objective_value.strip()
    runtime = runtime if runtime is not None else _TASK_RESOLUTION_RUNTIME
    _require_runtime(runtime)
    binding_store = _TASK_RESOLUTION_BINDING_STORE
    _require_store(binding_store, "find_binding_by_session")
    binding = await _store_call(
        binding_store,
        "find_binding_by_session",
        task_session_id=ref["dsh_session_id"],
    )
    if not isinstance(binding, Mapping):
        raise TaskResolutionContractError(
            "resume_task_resolution requires a durable task binding",
        )
    bound_ref = binding.get("resolution_ref")
    if not isinstance(bound_ref, Mapping):
        raise TaskResolutionContractError(
            "resume_task_resolution binding has no durable DSH reference",
        )
    if validate_dsh_resolution_ref(bound_ref) != ref:
        raise TaskResolutionContractError(
            "resume_task_resolution reference conflicts with its binding",
        )
    start_spec_value = binding.get("start_spec")
    if not isinstance(start_spec_value, Mapping):
        raise TaskResolutionContractError(
            "resume_task_resolution binding has no persisted start spec",
        )
    start_spec = validate_dsh_task_start_spec(start_spec_value)
    if start_spec["resolver_request"]["semantic_goal"] != objective:
        raise TaskResolutionContractError(
            "resume_task_resolution objective conflicts with its start spec",
        )
    method = getattr(runtime, "continue_after_terminal", None)
    if not callable(method):
        raise TaskResolutionContractError(
            "DSH runtime does not expose terminal continuation",
        )
    value = await _await_value(method(
        resolution_thread_id=ref["resolution_thread_id"],
        segment_id=ref["segment_id"],
        activation_id=ref["activation_id"],
        lease_epoch=ref["lease_epoch"],
        continuation_delta={"checkpoint": dict(checkpoint)},
        execution_context=dict(context),
        start_spec=start_spec,
    ))
    if isinstance(value, Mapping) and value.get("disposition") == "terminal":
        return project_dsh_exhaust(
            value,
            start_spec,
        )
    return _deferred_result(objective, context, ref, start_spec)


async def promote_deferred_task_resolution(
    result: Mapping[str, object],
    execution_context: Mapping[str, object],
    **kwargs: object,
) -> dict[str, object]:
    """Materialize a deferred result on its existing DSH binding."""

    context = _context_for_service(execution_context)
    if result.get("status") != "deferred":
        raise TaskResolutionContractError(
            "only deferred task results can be promoted",
        )
    checkpoint = result.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise TaskResolutionContractError(
            "deferred task result is missing its DSH reference",
        )
    reference = validate_dsh_resolution_ref(checkpoint)
    binding_store = kwargs.get("binding_store")
    if binding_store is None:
        binding_store = _TASK_RESOLUTION_BINDING_STORE
    accepted_task_store = kwargs.get("accepted_task_store")
    if accepted_task_store is None:
        accepted_task_store = _TASK_RESOLUTION_ACCEPTED_TASK_STORE
    background_queue = kwargs.get("background_queue")
    if background_queue is None:
        background_queue = _TASK_RESOLUTION_BACKGROUND_QUEUE
    _require_store(binding_store, "find_binding_by_session")
    _require_store(
        accepted_task_store,
        "create_or_return_active_accepted_task",
    )
    _require_queue(background_queue)

    session_id = reference["dsh_session_id"]
    binding = await _store_call(
        binding_store,
        "find_binding_by_session",
        task_session_id=session_id,
    )
    if not isinstance(binding, Mapping):
        raise TaskResolutionContractError(
            "deferred DSH result has no durable task binding",
        )
    binding_snapshot = dict(binding)
    bound_reference = binding_snapshot.get("resolution_ref")
    if not isinstance(bound_reference, Mapping):
        raise TaskResolutionContractError(
            "deferred DSH binding has no durable resolution reference",
        )
    if validate_dsh_resolution_ref(bound_reference) != reference:
        raise TaskResolutionContractError(
            "deferred DSH reference conflicts with its binding",
        )

    semantic_objective = result.get("semantic_objective")
    if not isinstance(semantic_objective, str) or not semantic_objective.strip():
        raise TaskResolutionContractError(
            "deferred task semantic objective is required",
        )
    accepted_request = _accepted_task_request(
        semantic_objective=semantic_objective,
        context=context,
        source_trigger_source=str(
            kwargs.get("source_trigger_source", "user_message"),
        ),
        source_platform_bot_id=str(kwargs.get("source_platform_bot_id", "")),
        requester_display_name=str(kwargs.get("requester_display_name", "")),
    )
    value = await _store_call(
        accepted_task_store,
        "create_or_return_active_accepted_task",
        request=accepted_request,
        dsh_task_session_id=session_id,
        dsh_operation_generation=0,
        dsh_followup_open=False,
    )
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(
            "accepted-task promotion returned no durable task",
        )
    nested = value.get("task")
    accepted_task = nested if isinstance(nested, Mapping) else value
    accepted_task_id = _required_mapping_text(
        accepted_task,
        "accepted_task_id",
    )
    state = binding_snapshot.get("state")
    revision = binding_snapshot.get("revision")
    generation = binding_snapshot.get("operation_generation", 0)
    if (
        not isinstance(state, str)
        or not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation != 0
    ):
        raise TaskResolutionContractError(
            "deferred DSH binding generation or revision is invalid",
        )
    if state != "checkpointed":
        raise TaskResolutionContractError(
            "deferred DSH binding is not promotable",
        )
    transitioned = await _store_call(
        binding_store,
        "transition_task_binding",
        task_session_id=session_id,
        expected_revision=revision,
        expected_state=state,
        next_state="active",
        operation_generation=0,
        updated_at=_binding_updated_at(binding_snapshot),
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "promoted DSH binding activation was not durable",
        )
    binding_snapshot = dict(transitioned)
    revision = _required_revision(binding_snapshot)
    pending = await _store_call(
        accepted_task_store,
        "mark_accepted_task_pending",
        accepted_task_id=accepted_task_id,
        executor_ref=f"dsh-task:{session_id}:generation-0",
        updated_at=_binding_updated_at(binding_snapshot),
    )
    if not isinstance(pending, Mapping):
        raise TaskResolutionContractError(
            "promoted accepted task could not become pending",
        )

    payload = {
        "schema_version": "task_orchestrator_worker_payload.v2",
        "operation": "continue_dsh_resolution",
        "task_session_id": session_id,
        "operation_generation": 0,
        "control": None,
    }
    queue_kwargs = _followup_queue_kwargs(
        binding_snapshot,
        accepted_task=accepted_task,
        accepted_task_id=accepted_task_id,
        action_attempt_id=f"dsh-promote:{session_id}",
        task_session_id=session_id,
        operation_generation=0,
        payload=payload,
    )
    job_id = _required_mapping_text(queue_kwargs, "job_id")
    queued = await _queue_payload(background_queue, payload, **queue_kwargs)
    if not isinstance(queued, Mapping):
        raise TaskResolutionContractError(
            "promotion queue returned no durable job",
        )
    if _required_mapping_text(queued, "job_id") != job_id:
        raise TaskResolutionContractError(
            "promotion queue returned an unexpected job id",
        )
    attached = await _store_call(
        binding_store,
        "attach_accepted_task",
        task_session_id=session_id,
        expected_revision=revision,
        operation_generation=0,
        accepted_task_id=accepted_task_id,
        updated_at=_binding_updated_at(binding_snapshot),
    )
    if not isinstance(attached, Mapping):
        raise TaskResolutionContractError(
            "promotion accepted-task attachment was not durable",
        )
    binding_snapshot = dict(attached)
    revision = _required_revision(binding_snapshot)
    attached_job = await _store_call(
        binding_store,
        "attach_background_job",
        task_session_id=session_id,
        expected_revision=revision,
        operation_generation=0,
        background_work_job_id=job_id,
        updated_at=_binding_updated_at(binding_snapshot),
    )
    if not isinstance(attached_job, Mapping):
        raise TaskResolutionContractError(
            "promotion background-job attachment was not durable",
        )
    return dict(_deferred_result(
        semantic_objective,
        context,
        reference,
        binding_snapshot.get("start_spec", {}),
        status="deferred",
        job_id=job_id,
        accepted_task_id=accepted_task_id,
    ))


async def reconcile_task_resolution_result(
    *,
    task_session_id: str | None = None,
    operation_generation: int,
    task_resolution_result: Mapping[str, object] | None = None,
    result: Mapping[str, object] | None = None,
    binding_store: object | None = None,
    accepted_task_store: object | None = None,
    promoted: bool = False,
    binding_id: str | None = None,
    **_: object,
) -> dict[str, object]:
    """Reconcile an interaction result exactly once before/after promotion."""

    del accepted_task_store
    selected = task_resolution_result or result
    if not isinstance(selected, Mapping):
        raise TaskResolutionContractError("task-resolution result is required")
    if not isinstance(task_session_id, str) or not task_session_id.strip():
        raise TaskResolutionContractError(
            "task-resolution binding session is required",
        )
    _require_store(binding_store, "find_binding_by_session")
    loaded = await _store_call(
        binding_store,
        "find_binding_by_session",
        task_session_id=task_session_id,
    )
    if not isinstance(loaded, Mapping):
        raise TaskResolutionContractError(
            "task-resolution result has no durable task binding",
        )
    current = dict(loaded)
    generation = current.get("operation_generation")
    if generation != operation_generation:
        raise TaskResolutionContractError(
            "task-resolution result generation does not match its binding",
        )
    existing = current.get("latest_task_resolution_result")
    if existing is not None:
        if not isinstance(existing, Mapping) or dict(existing) != dict(selected):
            raise TaskResolutionContractError(
                "task-resolution result conflicts with its durable binding",
            )
        return {
            "binding_id": binding_id,
            "task_session_id": task_session_id,
            "operation_generation": operation_generation,
            "disposition": "reconciled" if promoted else "already_reconciled",
            "result": dict(existing),
        }
    expected_revision = _.get("expected_revision")
    if not isinstance(expected_revision, int) or isinstance(expected_revision, bool):
        expected_revision = current.get("revision")
    if not isinstance(expected_revision, int) or isinstance(expected_revision, bool):
        raise TaskResolutionContractError(
            "task-resolution binding revision is required",
        )
    reconciled = await _store_call(
        binding_store,
        "reconcile_task_resolution_result",
        task_session_id=task_session_id,
        expected_revision=expected_revision,
        operation_generation=operation_generation,
        task_resolution_result=dict(selected),
    )
    if not isinstance(reconciled, Mapping):
        raise TaskResolutionContractError(
            "task-resolution result reconciliation was not durable",
        )
    return {
        "binding_id": binding_id,
        "task_session_id": task_session_id,
        "operation_generation": operation_generation,
        "disposition": "reconciled" if promoted else "stored_before_promotion",
        "result": dict(selected),
    }


async def continue_delivered_task(
    control: Mapping[str, object],
    *,
    action_attempt_id: str,
    binding: Mapping[str, object],
    binding_store: object | None = None,
    accepted_task_store: object | None = None,
    background_queue: object | None = None,
    authority_broker: object | None = None,
) -> dict[str, object]:
    """Queue one typed next-generation follow-up on the same DSH thread."""

    del authority_broker
    validated = validate_accepted_task_control(control)
    if not isinstance(action_attempt_id, str) or not action_attempt_id.strip():
        raise TaskResolutionContractError("action_attempt_id is required")
    accepted_ref = validated["accepted_task_ref"]
    accepted_task_id = accepted_ref.removeprefix("accepted_task:")
    _require_store(binding_store, "find_binding_by_accepted_task")
    _require_store(accepted_task_store, "find_accepted_task_by_id")
    _require_store(accepted_task_store, "find_dsh_followup_by_action_attempt")
    _require_store(accepted_task_store, "create_followup")
    _require_queue(background_queue)
    accepted_task = await _store_call(
        accepted_task_store,
        "find_accepted_task_by_id",
        accepted_task_id=accepted_task_id,
    )
    if not isinstance(accepted_task, Mapping):
        raise TaskResolutionContractError(
            "accepted-task control references no durable task",
        )
    session_id = _required_mapping_text(
        accepted_task,
        "dsh_task_session_id",
    )
    binding_value = await _store_call(
        binding_store,
        "find_binding_by_accepted_task",
        accepted_task_id=accepted_task_id,
    )
    if not isinstance(binding_value, Mapping):
        binding_value = await _store_call(
            binding_store,
            "find_binding_by_session",
            task_session_id=session_id,
        )
    if not isinstance(binding_value, Mapping):
        raise TaskResolutionContractError(
            "accepted-task control references no durable DSH binding",
        )
    binding_snapshot = dict(binding_value)
    if binding_snapshot.get("task_session_id") != session_id:
        raise TaskResolutionContractError(
            "accepted-task control binding session is invalid",
        )
    if isinstance(binding, Mapping) and dict(binding):
        supplied_session = binding.get("task_session_id")
        if supplied_session is not None and supplied_session != session_id:
            raise TaskResolutionContractError(
                "accepted-task control binding conflicts with durable binding",
            )
    thread_id = _required_binding_text(binding_snapshot, "resolution_thread_id")
    segment_id = _required_binding_text(binding_snapshot, "segment_id")
    binding_ref = _binding_resolution_ref(binding_snapshot, session_id)
    if (
        binding_ref["resolution_thread_id"] != thread_id
        or binding_ref["segment_id"] != segment_id
    ):
        raise TaskResolutionContractError(
            "accepted-task control binding identity is inconsistent",
        )
    current_generation = binding_snapshot.get("operation_generation")
    if (
        not isinstance(current_generation, int)
        or isinstance(current_generation, bool)
        or current_generation < 0
    ):
        raise TaskResolutionContractError("binding operation_generation is invalid")
    generation = current_generation + 1
    expected_state = binding_snapshot.get("state")
    if expected_state != "terminal":
        replay = await _store_call(
            accepted_task_store,
            "find_dsh_followup_by_action_attempt",
            task_session_id=session_id,
            action_attempt_id=action_attempt_id,
            operation_generation=current_generation,
        )
        if isinstance(replay, Mapping):
            replay_task_id = _required_mapping_text(
                replay,
                "accepted_task_id",
            )
            replay_generation = replay.get("dsh_operation_generation")
            if validated["operation"] == "cancel":
                replay_generation = current_generation
                replay_status = "canceled"
            else:
                replay_status = "queued"
            if replay_generation != current_generation:
                raise TaskResolutionContractError(
                    "durable follow-up replay generation is invalid",
                )
            return {
                "accepted_task_id": replay_task_id,
                "task_session_id": session_id,
                "resolution_thread_id": thread_id,
                "segment_id": segment_id,
                "operation_generation": current_generation,
                "status": replay_status,
            }
    if expected_state != "terminal":
        raise TaskResolutionContractError(
            "delivered follow-up requires a terminal DSH binding",
        )
    expected_revision = binding_snapshot.get("revision")
    if (
        not isinstance(expected_revision, int)
        or isinstance(expected_revision, bool)
        or expected_revision < 0
    ):
        raise TaskResolutionContractError("binding revision is invalid")
    accepted_revision = accepted_task.get("revision")
    if (
        not isinstance(accepted_revision, int)
        or isinstance(accepted_revision, bool)
        or accepted_revision < 0
    ):
        raise TaskResolutionContractError("accepted-task revision is invalid")
    replay_generation = current_generation if validated["operation"] == "cancel" else generation
    replay = await _store_call(
        accepted_task_store,
        "find_dsh_followup_by_action_attempt",
        task_session_id=session_id,
        action_attempt_id=action_attempt_id,
        operation_generation=replay_generation,
    )
    if isinstance(replay, Mapping):
        replay_task_id = _required_mapping_text(replay, "accepted_task_id")
        if validated["operation"] == "cancel":
            return {
                "accepted_task_id": replay_task_id,
                "task_session_id": session_id,
                "resolution_thread_id": thread_id,
                "segment_id": segment_id,
                "operation_generation": current_generation,
                "status": "canceled",
            }
        replay_generation_value = replay.get("dsh_operation_generation")
        if replay_generation_value != generation:
            raise TaskResolutionContractError(
                "durable follow-up replay generation is invalid",
            )
        return {
            "accepted_task_id": replay_task_id,
            "task_session_id": session_id,
            "resolution_thread_id": thread_id,
            "segment_id": segment_id,
            "operation_generation": generation,
            "status": "queued",
        }
    followup_value = await _store_call(
        accepted_task_store,
        "create_followup",
        accepted_task_id=accepted_task_id,
        action_attempt_id=action_attempt_id,
        operation=validated["operation"],
        instruction=validated["instruction"],
        task_session_id=session_id,
        operation_generation=generation,
        binding=dict(binding_snapshot),
        expected_revision=accepted_revision,
    )
    if not isinstance(followup_value, Mapping):
        raise TaskResolutionContractError(
            "accepted-task follow-up was not durable",
        )
    nested_task = followup_value.get("task")
    next_task = nested_task if isinstance(nested_task, Mapping) else followup_value
    next_task_id = _required_mapping_text(next_task, "accepted_task_id")
    if validated["operation"] == "cancel":
        transitioned = await _store_call(
            binding_store,
            "transition_task_binding",
            task_session_id=session_id,
        expected_state=expected_state,
        next_state="canceled",
        operation_generation=current_generation,
        expected_operation_generation=current_generation,
        expected_revision=expected_revision,
        )
        if not isinstance(transitioned, Mapping):
            raise TaskResolutionContractError(
                "DSH cancellation transition was not durable",
            )
        output = {
            "accepted_task_id": next_task_id,
            "task_session_id": session_id,
            "resolution_thread_id": thread_id,
            "segment_id": segment_id,
            "operation_generation": current_generation,
            "status": "canceled",
        }
        return output
    transitioned = await _store_call(
        binding_store,
        "transition_task_binding",
        task_session_id=session_id,
        expected_state=expected_state,
        next_state="active",
        operation_generation=generation,
        expected_operation_generation=current_generation,
        expected_revision=expected_revision,
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up activation was not durable",
        )
    binding_snapshot = dict(transitioned)
    expected_revision = _required_revision(binding_snapshot)
    attached = await _store_call(
        binding_store,
        "attach_accepted_task",
        task_session_id=session_id,
        expected_revision=expected_revision,
        operation_generation=generation,
        accepted_task_id=next_task_id,
        updated_at=_binding_updated_at(binding_snapshot),
    )
    if not isinstance(attached, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up accepted-task attachment was not durable",
        )
    binding_snapshot = dict(attached)
    expected_revision = _required_revision(binding_snapshot)
    payload = {
        "schema_version": "task_orchestrator_worker_payload.v2",
        "operation": "continue_dsh_resolution",
        "task_session_id": session_id,
        "operation_generation": generation,
        "control": dict(validated),
    }
    queue_kwargs = _followup_queue_kwargs(
        binding_snapshot,
        accepted_task=next_task,
        accepted_task_id=next_task_id,
        action_attempt_id=action_attempt_id,
        task_session_id=session_id,
        operation_generation=generation,
        payload=payload,
    )
    queued = await _queue_payload(background_queue, payload, **queue_kwargs)
    if not isinstance(queued, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up queue result was not durable",
        )
    job_id = _required_mapping_text(queued, "job_id")
    if job_id != _required_mapping_text(queue_kwargs, "job_id"):
        raise TaskResolutionContractError(
            "DSH follow-up queue returned an unexpected job id",
        )
    attached_job = await _store_call(
        binding_store,
        "attach_background_job",
        task_session_id=session_id,
        expected_revision=expected_revision,
        operation_generation=generation,
        background_work_job_id=job_id,
        updated_at=_binding_updated_at(binding_snapshot),
    )
    if not isinstance(attached_job, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up background-job attachment was not durable",
        )
    output = {
        "accepted_task_id": next_task_id,
        "task_session_id": session_id,
        "resolution_thread_id": thread_id,
        "segment_id": segment_id,
        "operation_generation": generation,
        "status": "queued",
    }
    return output


def _binding_updated_at(binding: Mapping[str, object]) -> str:
    """Read the durable update timestamp used by binding CAS helpers."""

    value = binding.get("updated_at")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _followup_queue_kwargs(
    binding: Mapping[str, object],
    *,
    accepted_task: Mapping[str, object] | None,
    accepted_task_id: str,
    action_attempt_id: str,
    task_session_id: str,
    operation_generation: int,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Build the complete trusted queue request for a DSH follow-up."""

    start_spec = binding.get("start_spec")
    if not isinstance(start_spec, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up binding has no persisted start specification",
        )
    validated_spec = validate_dsh_task_start_spec(start_spec)
    raw_context = validated_spec.get("execution_context")
    if not isinstance(raw_context, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up start specification has no execution context",
        )
    execution_context = validate_task_resolution_execution_context(raw_context)
    semantic_objective = binding.get("semantic_objective")
    raw_request = validated_spec.get("resolver_request")
    if not isinstance(semantic_objective, str) or not semantic_objective.strip():
        raise TaskResolutionContractError(
            "DSH follow-up semantic objective is required",
        )
    if not isinstance(raw_request, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up start specification request is required",
        )
    if semantic_objective.strip() != raw_request["semantic_goal"]:
        raise TaskResolutionContractError(
            "DSH follow-up semantic objective conflicts with its start spec",
        )
    goal_continuation_ref = binding.get("goal_continuation_ref")
    if not isinstance(goal_continuation_ref, Mapping):
        raise TaskResolutionContractError(
            "DSH follow-up goal continuation reference is required",
        )
    if dict(goal_continuation_ref) != dict(execution_context["goal_continuation_ref"]):
        raise TaskResolutionContractError(
            "DSH follow-up goal continuation conflicts with its start spec",
        )

    def text(field: str) -> str:
        value = execution_context.get(field)
        if isinstance(value, str) and (value.strip() or field == "source_llm_trace_id"):
            return value.strip()
        raise TaskResolutionContractError(
            f"DSH follow-up queue context field {field} is required",
        )

    task_identity_key = ""
    if isinstance(accepted_task, Mapping):
        candidate_identity = accepted_task.get("task_identity_key")
        if isinstance(candidate_identity, str):
            task_identity_key = candidate_identity.strip()
    if not task_identity_key:
        raise TaskResolutionContractError(
            "DSH follow-up accepted task identity is required",
        )
    timestamp = text("current_timestamp_utc")
    max_output_chars = execution_context.get("max_output_chars")
    if not isinstance(max_output_chars, int) or isinstance(max_output_chars, bool):
        raise TaskResolutionContractError(
            "DSH follow-up max_output_chars is invalid",
        )
    return {
        "job_id": (
            f"dsh-task:{accepted_task_id}:generation:{operation_generation}"
        ),
        "source_action_attempt_id": action_attempt_id,
        "source_llm_trace_id": text("source_llm_trace_id"),
        "idempotency_key": (
            f"dsh-followup:{task_session_id}:"
            f"{operation_generation}:{action_attempt_id}"
        ),
        "accepted_task_id": accepted_task_id,
        "task_identity_key": task_identity_key,
        "semantic_objective": semantic_objective.strip(),
        "goal_continuation_ref": dict(goal_continuation_ref),
        "requested_worker": "task_orchestrator",
        "worker_payload": dict(payload),
        "task_execution_context": dict(execution_context),
        "source_platform": text("platform"),
        "source_channel_id": text("channel_id"),
        "source_channel_type": text("channel_type"),
        "source_message_id": text("source_message_id"),
        "source_platform_bot_id": text("source_platform_bot_id"),
        "source_character_name": text("character_name"),
        "requester_global_user_id": text("requester_global_user_id"),
        "requester_platform_user_id": text("requester_platform_user_id"),
        "requester_display_name": text("requester_display_name"),
        "requested_delivery": "send_result_when_done",
        "max_output_chars": max_output_chars,
        "storage_timestamp_utc": timestamp,
    }


def _context_for_service(
    value: Mapping[str, object],
) -> TaskResolutionExecutionContextV2:
    """Validate the complete trusted context at the service boundary."""

    return validate_task_resolution_execution_context(value)


def _build_start_spec(
    request: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object]:
    """Build the model-hidden DSH start carrier with ten ordered facts."""

    facts = build_model_facts(context)
    start_spec = {
        "schema_version": "dsh_task_start_spec.v1",
        "resolver_request": {
            "capability": "task_resolution_request",
            "semantic_goal": _request_objective(request),
            "reason": str(request.get("reason", "")),
            "evidence_handles": list(request.get("evidence_handles", []))
            if isinstance(request.get("evidence_handles", []), list)
            else [],
            "start_in_background": bool(request.get("start_in_background", False)),
            "goal_continuation_ref": context.get("goal_continuation_ref"),
        },
        "execution_context": dict(context),
        "model_facts": facts,
        "model_facts_digest": content_digest(facts),
        "objective_ref": content_digest(context.get("goal_continuation_ref", {})),
    }
    return dict(validate_dsh_task_start_spec(start_spec))


def _accepted_task_request(
    *,
    semantic_objective: str,
    context: Mapping[str, object],
    source_trigger_source: str,
    source_platform_bot_id: str,
    requester_display_name: str,
) -> dict[str, object]:
    """Build the trusted accepted-task request for one DSH generation."""

    def text(field: str, fallback: str = "") -> str:
        value = context.get(field, fallback)
        return value.strip() if isinstance(value, str) else fallback

    max_output_chars = context.get("max_output_chars")
    if not isinstance(max_output_chars, int) or max_output_chars < 1:
        raise TaskResolutionContractError(
            "task-resolution context max_output_chars is invalid",
        )
    return {
        "task_kind": "task_resolution",
        "semantic_objective": semantic_objective,
        "accepted_task_summary": semantic_objective,
        "goal_continuation_ref": context.get("goal_continuation_ref"),
        "requested_delivery": "send_result_when_done",
        "max_output_chars": max_output_chars,
        "source_trigger_source": source_trigger_source.strip() or text(
            "source_trigger_source",
        ),
        "source_platform": text("platform"),
        "source_channel_id": text("channel_id"),
        "source_channel_type": text("channel_type"),
        "source_message_id": text("source_message_id"),
        "source_platform_bot_id": source_platform_bot_id.strip() or text(
            "source_platform_bot_id",
        ),
        "source_character_name": text("character_name"),
        "requester_global_user_id": text("requester_global_user_id"),
        "requester_platform_user_id": text("requester_platform_user_id"),
        "requester_display_name": requester_display_name.strip() or text(
            "requester_display_name",
        ),
        "storage_timestamp_utc": text("current_timestamp_utc"),
    }


def _deferred_result(
    semantic_objective: str,
    context: Mapping[str, object],
    resolution_ref: Mapping[str, object],
    start_spec: Mapping[str, object],
    *,
    status: str = "deferred",
    job_id: str = "",
    accepted_task_id: str = "",
) -> TaskResolutionResultV1:
    """Build one prompt-safe deferred/pending result with the DSH ref."""

    ref = validate_dsh_resolution_ref(resolution_ref)
    result: dict[str, object] = {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": semantic_objective,
        "status": status,
        "scene_context": dict(context.get("scene_context", {})),
        "goal_continuation_ref": context.get("goal_continuation_ref", {}),
        "evidence_state": "pending",
        "evidence_excerpts": [],
        "evidence_handles": [],
        "prompt_safe_summary": "The task needs durable continuation.",
        "evidence": [],
        "completed_subgoals": [],
        "remaining_needs": ["DSH resolution continuation"],
        "checkpoint": dict(ref),
        "coding_run_context": {},
    }
    return result  # type: ignore[return-value]


def _task_resolution_admission(
    *,
    accepted_task_id: str,
    background_work_job_id: str,
    task_session_id: str,
) -> TaskResolutionAdmissionV1:
    """Build the transient identity returned before DSH claims a thread."""

    admission = validate_task_resolution_admission({
        "schema_version": "task_resolution_admission.v1",
        "accepted_task_id": accepted_task_id,
        "background_work_job_id": background_work_job_id,
        "task_session_id": task_session_id,
    })
    return admission


def _request_objective(request: Mapping[str, object]) -> str:
    """Read the semantic objective from the canonical or focused carrier."""

    value = request.get("semantic_goal", request.get("objective"))
    if not isinstance(value, str) or not value.strip():
        raise TaskResolutionContractError(
            "task-resolution semantic objective is required",
        )
    return value.strip()


def _task_session_id(
    _request: Mapping[str, object],
    context: Mapping[str, object],
) -> str:
    """Derive one stable session identity from trusted scope and continuation."""

    try:
        continuation_ref = validate_goal_continuation_ref(
            context.get("goal_continuation_ref")
        )
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            "task-resolution session requires a validated goal continuation"
        ) from exc
    material = {
        "task_kind": "task_resolution",
        "source_scope": _durable_task_identity_scope(context),
        "goal_continuation_ref": continuation_ref,
    }
    return "session-" + content_digest(material).removeprefix("sha256:")[:32]


def _durable_task_identity_scope(context: Mapping[str, object]) -> dict[str, object]:
    """Project only trusted scope fields used by durable task identity."""

    source_scope = _source_scope(context)
    return {
        key: source_scope[key]
        for key in (
            "platform",
            "channel_id",
            "channel_type",
            "requester_global_user_id",
            "requester_platform_user_id",
        )
    }


def _source_scope(context: Mapping[str, object]) -> dict[str, object]:
    """Project trusted source scope without model authority fields."""

    return {
        "schema_version": "dsh_task_source_scope.v1",
        **{
            key: context.get(key, "")
            for key in (
                "platform",
                "channel_id",
                "channel_type",
                "requester_global_user_id",
                "requester_platform_user_id",
                "source_message_id",
                "source_platform_bot_id",
            )
        },
    }


def _validate_inline_budget(value: float) -> None:
    """Validate the bounded caller budget."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TaskResolutionContractError("inline_budget_seconds must be numeric")
    if value < MINIMUM_INLINE_BUDGET_SECONDS or value > MAXIMUM_INLINE_BUDGET_SECONDS:
        raise TaskResolutionContractError("inline_budget_seconds is outside its bound")


async def _store_call(store: object, name: str, **kwargs: object) -> object:
    """Invoke one required durable collaborator method."""

    method = getattr(store, name, None)
    if not callable(method):
        raise TaskResolutionContractError(
            f"task-resolution collaborator lacks required method {name}",
        )
    return await _await_value(method(**kwargs))


async def _queue_payload(
    queue: object,
    payload: Mapping[str, object],
    **kwargs: object,
) -> object:
    """Send one complete V2 queue request through the configured queue owner."""

    request = {"worker_payload": dict(payload), **kwargs}
    method = getattr(queue, "enqueue_background_work_request", None)
    request_style = True
    if not callable(method):
        method = getattr(queue, "enqueue", None)
        request_style = False
    if not callable(method):
        raise TaskResolutionContractError(
            "task-resolution background queue lacks its enqueue method",
        )
    value = method(request if request_style else dict(payload))
    return await _await_value(value)


async def _await_value(value: object) -> object:
    """Await one collaborator return value when needed."""

    return await value if isawaitable(value) else value


def _required_binding_text(binding: Mapping[str, object], field: str) -> str:
    """Read one required binding identity field."""

    value = binding.get(field)
    if not isinstance(value, str) or not value:
        raise TaskResolutionContractError(f"binding {field} is required")
    return value


def _require_runtime(runtime: object | None) -> object:
    """Require the Brain-composed DSH runtime at a service boundary."""

    if runtime is None:
        raise TaskResolutionContractError(
            "task-resolution service requires the Brain-composed DSH runtime",
        )
    return runtime


def _require_store(store: object | None, method_name: str) -> object:
    """Require one configured durable repository and its named operation."""

    if store is None:
        raise TaskResolutionContractError(
            f"task-resolution service requires durable store {method_name}",
        )
    if not callable(getattr(store, method_name, None)):
        raise TaskResolutionContractError(
            f"task-resolution durable store lacks {method_name}",
        )
    return store


def _require_queue(queue: object | None) -> object:
    """Require the configured background-work queue owner."""

    if queue is None:
        raise TaskResolutionContractError(
            "task-resolution service requires the background-work queue",
        )
    if not callable(
        getattr(queue, "enqueue_background_work_request", None),
    ) and not callable(getattr(queue, "enqueue", None)):
        raise TaskResolutionContractError(
            "task-resolution background queue lacks enqueue",
        )
    return queue


def _required_mapping_text(mapping: Mapping[str, object], field: str) -> str:
    """Read one non-empty durable identity returned by a collaborator."""

    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise TaskResolutionContractError(
            f"durable collaborator response lacks {field}",
        )
    return value.strip()


def _required_revision(binding: Mapping[str, object]) -> int:
    """Read one non-negative binding CAS revision."""

    value = binding.get("revision")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TaskResolutionContractError(
            "task-resolution binding revision is required",
        )
    return value


def _binding_resolution_ref(
    binding: Mapping[str, object],
    task_session_id: str,
) -> DshResolutionRefV1:
    """Require the exact reference already persisted on a binding."""

    value = binding.get("resolution_ref")
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError(
            "task-resolution binding has no durable DSH reference",
        )
    reference = validate_dsh_resolution_ref(value)
    if reference["dsh_session_id"] != task_session_id:
        raise TaskResolutionContractError(
            "task-resolution binding reference belongs to another session",
        )
    return reference
