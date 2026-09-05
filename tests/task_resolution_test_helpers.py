"""Retained deterministic fixtures shared by task-resolution tests."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref


def _scene_context() -> dict[str, object]:
    """Build the canonical prompt-safe scene shared by task fixtures."""

    return {
        "channel_scope": "private",
        "character_role": "Test Character",
        "current_user_role": "Test User",
        "semantic_scene": "A bounded task-resolution test scene.",
        "public_group_scene": "",
        "conversation_continuity": "The current turn continues one test goal.",
        "semantic_temporal_context": "The current test turn is active.",
    }


def _goal_continuation_ref() -> dict[str, object]:
    """Build the deterministic continuation identity used by task fixtures."""

    return build_goal_continuation_ref(
        source_episode_id="episode-task-001",
        source_message_id="message-1",
        branch_id="b1",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal-001",
        },
    )


def resolver_task_observation() -> dict[str, object]:
    """Build one canonical succeeded task-resolution observation."""

    return {
        "schema_version": "resolver_observation.v1",
        "observation_id": "raw-tool-run-123",
        "capability_kind": "task_resolution_request",
        "request_objective": "Retrieve relationship evidence.",
        "request_reason": "The current cycle lacks enough evidence.",
        "status": "succeeded",
        "prompt_safe_summary": "Found two relevant relationship evidence rows.",
        "evidence_refs": [
            {
                "schema_version": "evidence_ref.v1",
                "evidence_kind": "tool_result",
                "evidence_id": "raw-evidence-row-456",
                "owner": "cognition_resolver",
                "excerpt": "bounded summary only",
                "observed_at": "2026-05-30T00:00:00+00:00",
            }
        ],
        "task_resolution_evidence_state": {
            "schema_version": "resolver_evidence_state.v1",
            "state": "complete",
            "remaining_needs": [],
        },
        "goal_continuation_ref": build_goal_continuation_ref(
            source_episode_id="resolver-test-episode",
            source_message_id="resolver-test-message",
            branch_id="task_resolution",
            goal_ref={
                "scope": "user",
                "kind": "goal",
                "entity_id": "resolver-test-goal",
            },
        ),
        "created_at_utc": "2026-05-30T00:00:00+00:00",
    }


def _context() -> dict[str, object]:
    """Build the canonical V2 context shared by retained task tests."""

    return {
        "schema_version": "task_resolution_execution_context.v2",
        "character_name": "Test Character",
        "platform": "debug",
        "channel_id": "debug:user:test-user",
        "channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "source_message_id": "message-1",
        "source_platform_bot_id": "debug-bot",
        "source_trigger_source": "user_message",
        "source_llm_trace_id": "",
        "brain_conversation_ref": "chat:debug:test-user",
        "scene_context": _scene_context(),
        "goal_continuation_ref": _goal_continuation_ref(),
        "local_time_context": {"local_time": "2026-08-01 10:00"},
        "prompt_message_context": {"text": "Resolve this public source."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "conversation_progress": {},
        "persona_summary": "A bounded test persona.",
        "conversation_summary": "A bounded test conversation.",
        "current_timestamp_utc": "2026-07-31T22:00:00+00:00",
        "active_turn_platform_message_ids": ["message-1"],
        "active_turn_conversation_row_ids": [],
        "session_media_refs": [],
        "max_output_chars": 3000,
    }


def _resolution_ref(
    *,
    session_id: str = "session-task-001",
    thread_id: str = "thread-task-001",
    segment_id: str = "segment-task-001",
    activation_id: str = "activation-task-001",
) -> dict[str, object]:
    """Build one complete model-hidden DSH reference for deterministic fakes."""

    return {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": thread_id,
        "segment_id": segment_id,
        "dsh_session_id": session_id,
        "activation_id": activation_id,
        "lease_epoch": 1,
        "document_revision": 0,
        "last_committed_seq": 0,
    }


def recorded_task_checkpoint(
    *,
    status: str = "resolved",
    initial_checkpoint: dict[str, object] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build a typed DSH result and its durable reference fixture."""

    checkpoint = initial_checkpoint or _resolution_ref()
    summary = "A public source resolved the requested fact."
    semantic_ref = "https://example.com/source"
    is_resolved = status == "resolved"
    result = {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve one bounded public question.",
        "status": status,
        "scene_context": _context()["scene_context"],
        "goal_continuation_ref": _context()["goal_continuation_ref"],
        "evidence_state": "complete" if is_resolved else "pending",
        "evidence_excerpts": [summary] if is_resolved else [],
        "evidence_handles": [semantic_ref] if is_resolved else [],
        "prompt_safe_summary": summary if is_resolved else "Continuation is pending.",
        "evidence": [{
            "schema_version": "task_resolution_evidence.v1",
            "evidence_id": "public-evidence-1",
            "task_node_id": "dsh",
            "specialist": "dsh",
            "summary": semantic_ref,
            "provenance_refs": [semantic_ref],
            "limitations": [],
        }] if is_resolved else [],
        "completed_subgoals": [
            "Resolve one bounded public question."
        ] if is_resolved else [],
        "remaining_needs": [] if is_resolved else ["Continue the DSH task."],
        "checkpoint": {} if is_resolved else checkpoint,
        "coding_run_context": {},
    }
    return checkpoint, result


def resume_queue_request() -> dict[str, object]:
    """Build one generation-zero DSH queue request without authority."""

    return {
        "job_id": "job-001",
        "source_action_attempt_id": "action_attempt:task-resolution-001",
        "source_llm_trace_id": "llmtrace_source-1",
        "idempotency_key": "background_work:task-resolution-001",
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v2:abc",
        "semantic_objective": "Resolve one bounded public question.",
        "goal_continuation_ref": _goal_continuation_ref(),
        "requested_worker": "task_orchestrator",
        "worker_payload": {
            "schema_version": "task_orchestrator_worker_payload.v2",
            "operation": "open_dsh_resolution",
            "task_session_id": "session-task-001",
            "operation_generation": 0,
            "control": None,
        },
        "task_execution_context": _context(),
        "source_platform": "debug",
        "source_channel_id": "debug:user:test-user",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "debug-bot",
        "source_character_name": "Test Character",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "requester_display_name": "Test User",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "storage_timestamp_utc": "2026-06-06T00:00:00+00:00",
    }


def accepted_task_completed_job() -> dict[str, object]:
    """Build one completed accepted-task job with a canonical DSH result."""

    _checkpoint, task_result = recorded_task_checkpoint(status="resolved")
    return {
        "job_id": "job-001",
        "task_brief": "Generate a Fibonacci function snippet.",
        "worker": "text_artifact",
        "status": "completed",
        "artifact_text": "def fib(n): return n",
        "failure_summary": "",
        "result_summary": "Generated a compact Fibonacci snippet.",
        "worker_metadata": {"task_type": "coding_snippet"},
        "source_platform": "debug",
        "source_channel_id": "debug-private-1",
        "source_channel_type": "private",
        "source_message_id": "message-1",
        "source_platform_bot_id": "bot-1",
        "source_character_name": "Test Character",
        "requester_global_user_id": "global-user-1",
        "requester_platform_user_id": "platform-user-1",
        "requester_display_name": "Test User",
        "created_at": "2026-06-06T00:00:00+00:00",
        "completed_at": "2026-06-06T00:01:00+00:00",
        "goal_continuation_ref": _goal_continuation_ref(),
        "accepted_task_id": "task-001",
        "task_identity_key": "accepted_task:v1:abc",
        "source_llm_trace_id": "llmtrace-parent-1",
        "task_resolution_result": task_result,
    }


class InMemoryDshBindingStore:
    """Revision-fenced binding owner used by executable service tests."""

    def __init__(
        self,
        *,
        preassigned_ref: Mapping[str, object] | None = None,
    ) -> None:
        self.bindings: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.preassigned_ref = (
            dict(preassigned_ref) if preassigned_ref is not None else None
        )

    async def create_task_binding(
        self,
        *,
        binding: Mapping[str, object],
    ) -> dict[str, object]:
        row = deepcopy(dict(binding))
        session_id = str(row["task_session_id"])
        if self.preassigned_ref is not None:
            reference = deepcopy(self.preassigned_ref)
            row["resolution_ref"] = reference
            row["resolution_thread_id"] = reference["resolution_thread_id"]
            row["segment_id"] = reference["segment_id"]
        existing = self.bindings.get(session_id)
        if existing is not None and existing != row:
            raise ValueError("task session id was reused with a different binding")
        self.bindings[session_id] = row if existing is None else existing
        self.calls.append(("create_task_binding", {"binding": row}))
        return deepcopy(self.bindings[session_id])

    async def find_binding_by_session(
        self,
        *,
        task_session_id: str,
    ) -> dict[str, object] | None:
        self.calls.append((
            "find_binding_by_session",
            {"task_session_id": task_session_id},
        ))
        row = self.bindings.get(task_session_id)
        return deepcopy(row) if row is not None else None

    async def find_binding_by_accepted_task(
        self,
        *,
        accepted_task_id: str,
    ) -> dict[str, object] | None:
        self.calls.append((
            "find_binding_by_accepted_task",
            {"accepted_task_id": accepted_task_id},
        ))
        for row in self.bindings.values():
            if row.get("current_accepted_task_id") == accepted_task_id:
                return deepcopy(row)
        return None

    async def transition_task_binding(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("transition_task_binding", dict(kwargs)))
        session_id = str(kwargs["task_session_id"])
        row = self.bindings[session_id]
        expected_revision = kwargs["expected_revision"]
        expected_state = kwargs["expected_state"]
        generation = kwargs["operation_generation"]
        expected_generation = kwargs.get(
            "expected_operation_generation",
            generation,
        )
        if (
            row["revision"] != expected_revision
            or row["state"] != expected_state
            or row["operation_generation"] != expected_generation
        ):
            raise ValueError("binding CAS rejected")
        row["state"] = kwargs["next_state"]
        row["operation_generation"] = generation
        row["revision"] = int(row["revision"]) + 1
        if isinstance(kwargs.get("updated_at"), str):
            row["updated_at"] = kwargs["updated_at"]
        return deepcopy(row)

    async def attach_resolution_ref(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("attach_resolution_ref", dict(kwargs)))
        session_id = str(kwargs["task_session_id"])
        row = self.bindings[session_id]
        reference = deepcopy(dict(kwargs["resolution_ref"]))
        if row.get("resolution_ref") == reference:
            return deepcopy(row)
        if row["revision"] != kwargs["expected_revision"]:
            raise ValueError("binding CAS rejected")
        row["resolution_ref"] = reference
        row["resolution_thread_id"] = reference["resolution_thread_id"]
        row["segment_id"] = reference["segment_id"]
        row["revision"] = int(row["revision"]) + 1
        return deepcopy(row)

    async def attach_accepted_task(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("attach_accepted_task", dict(kwargs)))
        return await self._attach_generation_field(
            kwargs,
            "current_accepted_task_id",
            str(kwargs["accepted_task_id"]),
        )

    async def attach_background_job(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("attach_background_job", dict(kwargs)))
        return await self._attach_generation_field(
            kwargs,
            "current_background_work_job_id",
            str(kwargs["background_work_job_id"]),
        )

    async def _attach_generation_field(
        self,
        kwargs: Mapping[str, object],
        field: str,
        value: str,
    ) -> dict[str, object]:
        session_id = str(kwargs["task_session_id"])
        row = self.bindings[session_id]
        if row.get(field) == value:
            return deepcopy(row)
        if (
            row["revision"] != kwargs["expected_revision"]
            or row["operation_generation"] != kwargs["operation_generation"]
        ):
            raise ValueError("binding CAS rejected")
        row[field] = value
        row["revision"] = int(row["revision"]) + 1
        return deepcopy(row)

    async def reconcile_task_resolution_result(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        self.calls.append(("reconcile_task_resolution_result", dict(kwargs)))
        session_id = str(kwargs["task_session_id"])
        row = self.bindings[session_id]
        result = deepcopy(dict(kwargs["task_resolution_result"]))
        existing = row.get("latest_task_resolution_result")
        if existing is not None:
            if existing != result:
                raise ValueError("result replay conflicts")
            return deepcopy(row)
        if (
            row["revision"] != kwargs["expected_revision"]
            or row["operation_generation"] != kwargs["operation_generation"]
        ):
            raise ValueError("binding CAS rejected")
        row["latest_task_resolution_result"] = result
        row["revision"] = int(row["revision"]) + 1
        return deepcopy(row)


class InMemoryAcceptedTaskStore:
    """Durable accepted-task/follow-up fake with replay-by-attempt behavior."""

    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, object]] = {}
        self.calls: list[tuple[str, dict[str, object]]] = []
        self._next_id = 1

    async def create_or_return_active_accepted_task(
        self,
        *,
        request: Mapping[str, object],
        dsh_task_session_id: str,
        dsh_operation_generation: int,
        dsh_followup_open: bool,
    ) -> dict[str, object]:
        self.calls.append((
            "create_or_return_active_accepted_task",
            {
                "request": dict(request),
                "dsh_task_session_id": dsh_task_session_id,
                "dsh_operation_generation": dsh_operation_generation,
                "dsh_followup_open": dsh_followup_open,
            },
        ))
        identity = str(request["semantic_objective"])
        for task in self.tasks.values():
            if (
                task.get("task_identity_key") == identity
                and task.get("state") in {"enqueueing", "pending", "running"}
            ):
                return {"status": "already_active", "task": deepcopy(task)}
        task_id = f"task-{self._next_id}"
        self._next_id += 1
        task = {
            "schema_version": "accepted_task.v2",
            "accepted_task_id": task_id,
            "task_kind": "task_resolution",
            "task_identity_key": identity,
            "active_identity_key": identity,
            "accepted_task_summary": request["accepted_task_summary"],
            "semantic_objective": request["semantic_objective"],
            "goal_continuation_ref": deepcopy(
                request["goal_continuation_ref"],
            ),
            "requested_delivery": request["requested_delivery"],
            "max_output_chars": request["max_output_chars"],
            "source_trigger_source": request["source_trigger_source"],
            "source_platform": request["source_platform"],
            "source_channel_id": request["source_channel_id"],
            "source_channel_type": request["source_channel_type"],
            "source_message_id": request["source_message_id"],
            "source_platform_bot_id": request["source_platform_bot_id"],
            "source_character_name": request["source_character_name"],
            "requester_global_user_id": request["requester_global_user_id"],
            "requester_platform_user_id": request["requester_platform_user_id"],
            "requester_display_name": request["requester_display_name"],
            "storage_timestamp_utc": request["storage_timestamp_utc"],
            "state": "enqueueing",
            "executor_ref": "",
            "revision": 0,
            "dsh_task_session_id": dsh_task_session_id,
            "dsh_operation_generation": dsh_operation_generation,
            "dsh_followup_open": dsh_followup_open,
            "dsh_followup_claim_action_attempt_id": None,
        }
        self.tasks[task_id] = task
        return {"status": "created", "task": deepcopy(task)}

    async def find_accepted_task_by_id(
        self,
        *,
        accepted_task_id: str,
    ) -> dict[str, object] | None:
        self.calls.append((
            "find_accepted_task_by_id",
            {"accepted_task_id": accepted_task_id},
        ))
        task = self.tasks.get(accepted_task_id)
        return deepcopy(task) if task is not None else None

    async def find_dsh_followup_by_action_attempt(
        self,
        *,
        task_session_id: str,
        action_attempt_id: str,
        operation_generation: int,
    ) -> dict[str, object] | None:
        self.calls.append((
            "find_dsh_followup_by_action_attempt",
            {
                "task_session_id": task_session_id,
                "action_attempt_id": action_attempt_id,
                "operation_generation": operation_generation,
            },
        ))
        for task in self.tasks.values():
            if (
                task.get("dsh_task_session_id") == task_session_id
                and task.get("dsh_operation_generation")
                == operation_generation
                and task.get("dsh_followup_claim_action_attempt_id")
                == action_attempt_id
            ):
                return deepcopy(task)
        return None

    async def create_followup(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("create_followup", dict(kwargs)))
        session_id = str(kwargs["task_session_id"])
        generation = int(kwargs["operation_generation"])
        action_attempt_id = str(kwargs["action_attempt_id"])
        operation = str(kwargs["operation"])
        replay = await self.find_dsh_followup_by_action_attempt(
            task_session_id=session_id,
            action_attempt_id=action_attempt_id,
            operation_generation=(generation if operation != "cancel" else generation - 1),
        )
        if replay is not None:
            return replay
        source_id = str(kwargs["accepted_task_id"])
        source = self.tasks[source_id]
        if source.get("dsh_followup_open") is not True:
            raise ValueError("follow-up is not open")
        source["dsh_followup_open"] = False
        source["dsh_followup_claim_action_attempt_id"] = action_attempt_id
        source["revision"] = int(source.get("revision", 0)) + 1
        if operation == "cancel":
            source["state"] = "cancelled"
            return deepcopy(source)
        next_id = f"task-{self._next_id}"
        self._next_id += 1
        next_task = deepcopy(source)
        next_task.update({
            "accepted_task_id": next_id,
            "task_identity_key": f"{source_id}:dsh:{generation}",
            "active_identity_key": f"{source_id}:dsh:{generation}",
            "state": "pending",
            "executor_ref": "",
            "revision": 0,
            "dsh_operation_generation": generation,
            "dsh_followup_open": False,
        })
        self.tasks[next_id] = next_task
        return deepcopy(next_task)

    async def mark_accepted_task_pending(
        self,
        *,
        accepted_task_id: str,
        executor_ref: str,
        updated_at: str,
    ) -> dict[str, object] | None:
        self.calls.append((
            "mark_accepted_task_pending",
            {
                "accepted_task_id": accepted_task_id,
                "executor_ref": executor_ref,
                "updated_at": updated_at,
            },
        ))
        task = self.tasks.get(accepted_task_id)
        if task is None:
            return None
        task["state"] = "pending"
        task["executor_ref"] = executor_ref
        task["updated_at"] = updated_at
        return deepcopy(task)

    async def mark_accepted_task_enqueue_failed(self, **kwargs: object) -> dict[str, object] | None:
        self.calls.append(("mark_accepted_task_enqueue_failed", dict(kwargs)))
        task = self.tasks.get(str(kwargs["accepted_task_id"]))
        if task is None:
            return None
        task["state"] = "enqueue_failed"
        return deepcopy(task)


class InMemoryBackgroundQueue:
    """Canonical queue fake that returns the caller's durable job identity."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def enqueue_background_work_request(
        self,
        request: Mapping[str, object],
    ) -> dict[str, object]:
        value = deepcopy(dict(request))
        self.requests.append(value)
        return {
            "job_id": value["job_id"],
            "accepted_task_id": value["accepted_task_id"],
        }
