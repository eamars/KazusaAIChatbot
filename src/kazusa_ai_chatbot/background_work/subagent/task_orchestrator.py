"""Leased worker entrypoint for generation-bound DSH task resolution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from kazusa_ai_chatbot.background_work.jobs import (
    validate_task_orchestrator_worker_payload,
)
from kazusa_ai_chatbot.task_resolution.contracts import (
    DshResolutionRefV1,
    TaskResolutionContractError,
    TaskResolutionResultV1,
    validate_dsh_resolution_ref,
    validate_dsh_task_start_spec,
    validate_task_resolution_execution_context,
    validate_task_resolution_result,
)
from kazusa_ai_chatbot.task_resolution.projection import (
    project_dsh_exhaust,
)

_TASK_RESOLUTION_RUNTIME: object | None = None
_TASK_RESOLUTION_BINDING_STORE: object | None = None


def set_task_resolution_runtime(runtime: object | None) -> None:
    """Bind the Brain-composed DSH runtime used by claimed task jobs."""

    global _TASK_RESOLUTION_RUNTIME
    _TASK_RESOLUTION_RUNTIME = runtime


def set_task_resolution_binding_store(store: object | None) -> None:
    """Bind the durable DSH task-binding owner used by claimed jobs."""

    global _TASK_RESOLUTION_BINDING_STORE
    _TASK_RESOLUTION_BINDING_STORE = store


async def execute_task_orchestrator_job(
    job: Mapping[str, object],
    *,
    lease_owner: str,
) -> TaskResolutionResultV1:
    """Execute one open or next-generation DSH operation from a leased row."""

    del lease_owner
    if not isinstance(job, Mapping):
        raise TaskResolutionContractError("background job must be an object")
    payload_value = job.get("worker_payload")
    payload = validate_task_orchestrator_worker_payload(payload_value)
    context_value = job.get("task_execution_context")
    try:
        context = validate_task_resolution_execution_context(context_value)
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            f"task_execution_context is invalid: {exc}",
        ) from exc
    objective = _required_text(job, "semantic_objective")
    session_id = str(payload["task_session_id"])
    generation = int(payload["operation_generation"])
    operation = payload["operation"]
    runtime = _require_runtime(_TASK_RESOLUTION_RUNTIME)
    binding = await _load_binding(
        task_session_id=session_id,
        operation_generation=generation,
        job=job,
    )
    persisted_start_spec = binding.get("start_spec")
    if not isinstance(persisted_start_spec, Mapping):
        raise TaskResolutionContractError(
            "claimed DSH binding has no persisted start specification",
        )
    start_spec = validate_dsh_task_start_spec(persisted_start_spec)
    resolver_request = start_spec["resolver_request"]
    spec_context = start_spec["execution_context"]
    if resolver_request["semantic_goal"] != objective:
        raise TaskResolutionContractError(
            "claimed DSH job objective conflicts with its persisted start spec",
        )
    if dict(spec_context) != dict(context):
        raise TaskResolutionContractError(
            "claimed DSH job context conflicts with its persisted start spec",
        )
    if operation == "open_dsh_resolution":
        await _prepare_open_binding(binding)

    if operation == "open_dsh_resolution":
        if generation != 0:
            raise TaskResolutionContractError(
                "open_dsh_resolution must use generation zero",
            )
        method = getattr(runtime, "open", None)
        if not callable(method):
            raise TaskResolutionContractError("DSH runtime does not expose open")
        before_resolve = _binding_before_resolve(
            task_session_id=session_id,
            operation_generation=generation,
        )
        exhaust = await _await_result(method(
            task_session_id=session_id,
            operation_generation=generation,
            request=dict(resolver_request),
            execution_context=dict(context),
            start_spec=start_spec,
            before_resolve=before_resolve,
        ))
    else:
        control = payload.get("control")
        if control is not None and not isinstance(control, Mapping):
            raise TaskResolutionContractError(
                "continue_dsh_resolution control is invalid",
            )
        await _prepare_continue_binding(
            binding,
            operation_generation=generation,
        )
        reference = _job_resolution_ref(job, context, binding)
        if control is None:
            method = getattr(runtime, "continue_after_checkpoint", None)
            continuation_delta = {
                "operation_generation": generation,
            }
        else:
            method = getattr(runtime, "continue_after_terminal", None)
            continuation_delta = {
                "operation_generation": generation,
                "control": dict(control),
            }
        if not callable(method):
            raise TaskResolutionContractError(
                "DSH runtime does not expose the required continuation",
            )
        exhaust = await _await_result(method(
            resolution_thread_id=reference["resolution_thread_id"],
            segment_id=reference["segment_id"],
            activation_id=reference["activation_id"],
            lease_epoch=reference["lease_epoch"],
            continuation_delta=continuation_delta,
            execution_context=dict(context),
            start_spec=start_spec,
        ))

    if isinstance(exhaust, Mapping) and isinstance(exhaust.get("exhaust"), Mapping):
        exhaust = exhaust["exhaust"]
    result = project_dsh_exhaust(exhaust, start_spec)
    result = validate_task_resolution_result(result)
    await _record_binding_outcome(
        task_session_id=session_id,
        operation_generation=generation,
        exhaust=exhaust,
        result=result,
        allow_reference_advance=(operation == "continue_dsh_resolution"),
    )
    return result


def _job_resolution_ref(
    job: Mapping[str, object],
    context: Mapping[str, object],
    binding: Mapping[str, object] | None,
) -> DshResolutionRefV1:
    """Require the durable DSH reference for a next-generation operation."""

    if not isinstance(binding, Mapping):
        raise TaskResolutionContractError(
            "continue_dsh_resolution requires a durable DSH binding",
        )
    candidate = binding.get("resolution_ref")
    if not isinstance(candidate, Mapping):
        raise TaskResolutionContractError(
            "continue_dsh_resolution requires a durable DSH reference",
        )
    try:
        reference = validate_dsh_resolution_ref(candidate)
    except (TypeError, ValueError) as exc:
        raise TaskResolutionContractError(
            f"dsh_resolution_ref is invalid: {exc}",
        ) from exc
    if reference["dsh_session_id"] != _required_text(
        binding,
        "task_session_id",
    ):
        raise TaskResolutionContractError(
            "dsh_resolution_ref session does not match its binding",
        )
    for carrier in (job, context):
        candidate = carrier.get("dsh_resolution_ref")
        if isinstance(candidate, Mapping):
            try:
                if validate_dsh_resolution_ref(candidate) != reference:
                    raise TaskResolutionContractError(
                        "dsh_resolution_ref conflicts with its binding",
                    )
            except (TypeError, ValueError) as exc:
                raise TaskResolutionContractError(
                    f"dsh_resolution_ref is invalid: {exc}",
                ) from exc
    return reference


def _required_text(value: Mapping[str, object], field: str) -> str:
    """Read one required non-empty job text field."""

    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise TaskResolutionContractError(f"background job {field} is required")
    return raw.strip()


async def _await_result(value: object) -> object:
    """Await a runtime result while accepting deterministic sync fakes."""

    if hasattr(value, "__await__"):
        return await value  # type: ignore[misc]
    return value


async def _load_binding(
    *,
    task_session_id: str,
    operation_generation: int,
    job: Mapping[str, object],
) -> Mapping[str, object]:
    """Load and fence the sole binding for one claimed DSH operation."""

    store = _TASK_RESOLUTION_BINDING_STORE
    if store is None:
        raise TaskResolutionContractError(
            "claimed DSH execution requires the durable task-binding store",
        )
    method = getattr(store, "find_binding_by_session", None)
    if not callable(method):
        raise TaskResolutionContractError(
            "DSH task-binding store does not expose session lookup",
        )
    value = await _await_result(method(task_session_id=task_session_id))
    if value is None:
        raise TaskResolutionContractError(
            "claimed DSH job has no durable task binding",
        )
    if not isinstance(value, Mapping):
        raise TaskResolutionContractError("DSH task binding is invalid")
    if value.get("task_session_id") != task_session_id:
        raise TaskResolutionContractError("DSH task binding session is invalid")
    if value.get("operation_generation") != operation_generation:
        raise TaskResolutionContractError(
            "DSH task binding generation does not match the claimed job",
        )
    job_id = job.get("job_id")
    bound_job_id = value.get("current_background_work_job_id")
    if job_id is not None:
        if not isinstance(job_id, str) or not job_id.strip():
            raise TaskResolutionContractError(
                "background job job_id is required",
            )
        if not isinstance(bound_job_id, str) or not bound_job_id.strip():
            raise TaskResolutionContractError(
                "DSH task binding has no attached background job",
            )
        if bound_job_id != job_id:
            raise TaskResolutionContractError(
                "DSH task binding background job does not match the claimed job",
            )
    return dict(value)


async def _prepare_open_binding(
    binding: Mapping[str, object],
) -> None:
    """Move a newly claimed generation into opening under revision CAS."""

    if _TASK_RESOLUTION_BINDING_STORE is None:
        raise TaskResolutionContractError(
            "opening a DSH job requires the durable task-binding store",
        )
    state = binding.get("state")
    if state == "opening":
        return
    if state != "queued":
        raise TaskResolutionContractError(
            "DSH task binding is not openable in its current state",
        )
    revision = binding.get("revision")
    generation = binding.get("operation_generation")
    session_id = binding.get("task_session_id")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(generation, int)
        or not isinstance(session_id, str)
    ):
        raise TaskResolutionContractError("DSH task binding CAS fields are invalid")
    transitioned = await _store_call(
        "transition_task_binding",
        task_session_id=session_id,
        expected_revision=revision,
        expected_state="queued",
        next_state="opening",
        operation_generation=generation,
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "DSH task binding opening transition was not durable",
        )


async def _prepare_continue_binding(
    binding: Mapping[str, object],
    *,
    operation_generation: int,
) -> None:
    """Activate a checkpointed generation before a continuation call."""

    if _TASK_RESOLUTION_BINDING_STORE is None:
        raise TaskResolutionContractError(
            "continuing a DSH job requires the durable task-binding store",
        )
    state = binding.get("state")
    if state == "active":
        return
    if state != "checkpointed":
        raise TaskResolutionContractError(
            "DSH task binding is not continuable in its current state",
        )
    revision = binding.get("revision")
    session_id = binding.get("task_session_id")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(session_id, str)
        or not session_id
    ):
        raise TaskResolutionContractError("DSH task binding CAS fields are invalid")
    transitioned = await _store_call(
        "transition_task_binding",
        task_session_id=session_id,
        expected_revision=revision,
        expected_state=state,
        next_state="active",
        operation_generation=operation_generation,
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "DSH task binding activation was not durable",
        )


def _binding_before_resolve(
    *,
    task_session_id: str,
    operation_generation: int,
) -> Callable[[Mapping[str, object]], Awaitable[object]]:
    """Return the pre-RPC callback that persists the admitted DSH identity."""

    if _TASK_RESOLUTION_BINDING_STORE is None:
        raise TaskResolutionContractError(
            "DSH runtime admission requires the durable task-binding store",
        )

    async def persist(reference: Mapping[str, object]) -> object:
        return await _attach_binding_reference(
            task_session_id=task_session_id,
            operation_generation=operation_generation,
            reference=reference,
        )

    return persist


async def _attach_binding_reference(
    *,
    task_session_id: str,
    operation_generation: int,
    reference: Mapping[str, object],
) -> Mapping[str, object]:
    """Attach one admitted DSH identity through the binding revision fence."""

    store = _TASK_RESOLUTION_BINDING_STORE
    if store is None:
        raise TaskResolutionContractError(
            "DSH identity attachment requires the durable task-binding store",
        )
    validated = validate_dsh_resolution_ref(reference)
    binding = await _load_binding(
        task_session_id=task_session_id,
        operation_generation=operation_generation,
        job={},
    )
    existing = binding.get("resolution_ref")
    if isinstance(existing, Mapping):
        current = validate_dsh_resolution_ref(existing)
        if current != validated:
            raise TaskResolutionContractError(
                "DSH task binding identity changed during execution",
            )
        else:
            return binding
    revision = binding.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool):
        raise TaskResolutionContractError("DSH task binding revision is invalid")
    attached = await _store_call(
        "attach_resolution_ref",
        task_session_id=task_session_id,
        expected_revision=revision,
        resolution_ref=dict(validated),
    )
    if not isinstance(attached, Mapping):
        raise TaskResolutionContractError(
            "DSH task binding identity attachment was not durable",
        )
    return dict(attached)


async def _record_binding_outcome(
    *,
    task_session_id: str,
    operation_generation: int,
    exhaust: object,
    result: Mapping[str, object],
    allow_reference_advance: bool = False,
) -> None:
    """Persist the result, identity, and state transition for one generation."""

    if _TASK_RESOLUTION_BINDING_STORE is None:
        raise TaskResolutionContractError(
            "DSH result settlement requires the durable task-binding store",
        )
    binding = await _load_binding(
        task_session_id=task_session_id,
        operation_generation=operation_generation,
        job={},
    )
    reference = _reference_from_exhaust(
        exhaust,
        task_session_id=task_session_id,
        binding=binding,
    )
    existing = binding.get("resolution_ref")
    if not isinstance(existing, Mapping):
        binding = await _attach_binding_reference(
            task_session_id=task_session_id,
            operation_generation=operation_generation,
            reference=reference,
        )
    else:
        current_reference = validate_dsh_resolution_ref(existing)
        if current_reference != reference:
            if not _reference_progression_allowed(
                current_reference,
                reference,
                allow_fresh_fence=allow_reference_advance,
            ):
                raise TaskResolutionContractError(
                    "DSH result identity conflicts with its durable binding",
                )
            revision = binding.get("revision")
            if not isinstance(revision, int) or isinstance(revision, bool):
                raise TaskResolutionContractError(
                    "DSH task binding revision is invalid",
                )
            advanced = await _store_call(
                "attach_resolution_ref",
                task_session_id=task_session_id,
                expected_revision=revision,
                resolution_ref=dict(reference),
            )
            if not isinstance(advanced, Mapping):
                raise TaskResolutionContractError(
                    "DSH continuation fence advance was not durable",
                )
            binding = dict(advanced)
    state = binding.get("state")
    next_state = _binding_state_for_exhaust(exhaust)
    revision = binding.get("revision")
    if next_state is None:
        raise TaskResolutionContractError(
            "DSH exhaust does not map to a binding state",
        )
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or not isinstance(state, str)
    ):
        raise TaskResolutionContractError(
            "DSH task binding CAS fields are invalid",
        )
    if next_state == "terminal":
        stored = await _store_call(
            "reconcile_task_resolution_result",
            task_session_id=task_session_id,
            expected_revision=revision,
            operation_generation=operation_generation,
            task_resolution_result=dict(result),
        )
        if not isinstance(stored, Mapping):
            raise TaskResolutionContractError(
                "DSH result reconciliation was not durable",
            )
        binding = stored
        revision = stored.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool):
            raise TaskResolutionContractError(
                "DSH result reconciliation returned an invalid revision",
            )
    if state == next_state:
        return
    if not _binding_transition_allowed(state, next_state):
        raise TaskResolutionContractError(
            f"DSH task binding cannot transition {state} to {next_state}",
        )
    transitioned = await _store_call(
        "transition_task_binding",
        task_session_id=task_session_id,
        expected_revision=revision,
        expected_state=state,
        next_state=next_state,
        operation_generation=operation_generation,
    )
    if not isinstance(transitioned, Mapping):
        raise TaskResolutionContractError(
            "DSH task binding outcome transition was not durable",
        )


def _reference_from_exhaust(
    exhaust: object,
    *,
    task_session_id: str,
    binding: Mapping[str, object],
) -> DshResolutionRefV1:
    """Project hidden runtime identity into the durable reference carrier."""

    if hasattr(exhaust, "to_dict"):
        exhaust = exhaust.to_dict()
    if not isinstance(exhaust, Mapping):
        raise TaskResolutionContractError(
            "DSH exhaust does not contain a durable identity",
        )
    nested = exhaust.get("exhaust")
    candidate = nested if isinstance(nested, Mapping) else exhaust
    direct = candidate.get("dsh_resolution_ref")
    if isinstance(direct, Mapping):
        try:
            reference = validate_dsh_resolution_ref(direct)
        except (TypeError, ValueError) as exc:
            raise TaskResolutionContractError(
                f"dsh_resolution_ref is invalid: {exc}",
            ) from exc
        if reference["dsh_session_id"] != task_session_id:
            raise TaskResolutionContractError(
                "DSH result reference belongs to another task session",
            )
        return reference
    bound = binding.get("resolution_ref")
    if not isinstance(bound, Mapping):
        raise TaskResolutionContractError(
            "DSH exhaust identity is missing from its durable binding",
        )
    reference = validate_dsh_resolution_ref(bound)
    identity = candidate.get("identity")
    if isinstance(identity, Mapping):
        projected = dict(reference)
        for field in (
            "resolution_thread_id",
            "segment_id",
            "dsh_session_id",
            "activation_id",
            "lease_epoch",
            "document_revision",
            "last_committed_seq",
        ):
            if field in identity:
                projected[field] = identity[field]
        try:
            reference = validate_dsh_resolution_ref(projected)
        except (TypeError, ValueError) as exc:
            raise TaskResolutionContractError(
                f"DSH exhaust identity is invalid: {exc}",
            ) from exc
    if reference["dsh_session_id"] != task_session_id:
        raise TaskResolutionContractError(
            "DSH result reference belongs to another task session",
        )
    return reference


def _reference_progression_allowed(
    current: DshResolutionRefV1,
    candidate: DshResolutionRefV1,
    *,
    allow_fresh_fence: bool,
) -> bool:
    """Allow only monotonic progress on the same durable DSH lineage."""

    for field in (
        "resolution_thread_id",
        "segment_id",
        "dsh_session_id",
    ):
        if candidate[field] != current[field]:
            return False
    if (
        candidate["document_revision"] < current["document_revision"]
        or candidate["last_committed_seq"] < current["last_committed_seq"]
    ):
        return False
    same_fence = (
        candidate["activation_id"] == current["activation_id"]
        and candidate["lease_epoch"] == current["lease_epoch"]
    )
    if same_fence:
        return True
    return (
        allow_fresh_fence
        and candidate["activation_id"] != current["activation_id"]
        and candidate["lease_epoch"] == current["lease_epoch"] + 1
    )


def _binding_state_for_exhaust(exhaust: object) -> str | None:
    """Map a DSH exhaust kind to the closed binding state graph."""

    if hasattr(exhaust, "kind"):
        kind = exhaust.kind
    elif isinstance(exhaust, Mapping):
        nested = exhaust.get("exhaust")
        candidate = nested if isinstance(nested, Mapping) else exhaust
        kind = candidate.get("kind")
    else:
        kind = None
    return {
        "terminal": "terminal",
        "checkpointed": "checkpointed",
        "runtime_fault": "faulted",
        "canceled": "canceled",
    }.get(kind)


def _binding_transition_allowed(current: str, next_state: str) -> bool:
    """Keep the worker's state updates inside the durable closed graph."""

    allowed = {
        "opening": {
            "checkpointed",
            "terminal",
            "canceled",
            "faulted",
        },
        "active": {
            "checkpointed",
            "terminal",
            "canceled",
            "faulted",
        },
        "checkpointed": {"active", "canceled", "faulted"},
        "terminal": {"terminal"},
    }
    return next_state in allowed.get(current, set())


async def _store_call(name: str, **kwargs: object) -> object:
    """Call one bound task-binding repository method."""

    store = _TASK_RESOLUTION_BINDING_STORE
    if store is None:
        raise TaskResolutionContractError(
            "DSH task-binding store is not configured",
        )
    method = getattr(store, name, None)
    if not callable(method):
        raise TaskResolutionContractError(
            f"DSH task-binding store does not expose {name}",
        )
    return await _await_result(method(**kwargs))


def _require_runtime(runtime: object | None) -> object:
    """Require the shared Brain-composed runtime for a claimed job."""

    if runtime is None:
        raise TaskResolutionContractError(
            "claimed DSH job requires the Brain-composed DSH runtime",
        )
    return runtime
