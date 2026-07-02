"""Process-local scoped pipeline admission and cooperative cancellation."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from uuid import uuid4

from kazusa_ai_chatbot.runtime_coordination.models import (
    PipelineCancellation,
    PipelineCancelled,
    PipelinePrecedence,
    PipelineScope,
)


@dataclass(frozen=True)
class PipelineRunAdmission:
    """Result of attempting to admit a scoped pipeline run.

    Args:
        admitted: True when the caller may continue the run.
        handle: Active run handle when admitted; otherwise `None`.
        defer_reason: Stable reason string when admission is denied.
    """

    admitted: bool
    handle: "PipelineRunHandle | None"
    defer_reason: str | None


class PipelineRunHandle:
    """Active pipeline run handle with cooperative cancellation checkpoints."""

    def __init__(
        self,
        *,
        coordinator: "PipelineCoordinator",
        run_id: str,
        scope: PipelineScope,
        owner: str,
        precedence: PipelinePrecedence,
        run_kind: str,
    ) -> None:
        """Create one active run handle owned by a coordinator."""

        self._coordinator = coordinator
        self.run_id = run_id
        self.scope = scope
        self.owner = owner
        self.precedence = precedence
        self.run_kind = run_kind
        self._released = False
        self._cancellation: PipelineCancellation | None = None

    async def __aenter__(self) -> "PipelineRunHandle":
        """Return this active handle for async-context-manager callers."""

        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        """Release this run when the caller leaves the pipeline context."""

        del exc_type, exc, tb
        await self.release()

    async def release(self) -> None:
        """Release this active run, allowing repeated calls safely."""

        if self._released:
            return
        self._released = True
        self._coordinator._release_handle(self)

    def cancelled(self) -> bool:
        """Return whether this run has been marked for cancellation."""

        return_value = self._cancellation is not None
        return return_value

    def raise_if_cancelled(self, checkpoint: str) -> None:
        """Raise `PipelineCancelled` when this run has been cancelled.

        Args:
            checkpoint: Stable checkpoint name where the caller observed the
                cancellation.
        """

        if self._cancellation is None:
            return

        cancellation = PipelineCancellation(
            run_id=self.run_id,
            scope=self.scope,
            requested_by=self._cancellation.requested_by,
            reason=self._cancellation.reason,
            checkpoint=checkpoint,
        )
        raise PipelineCancelled(cancellation)

    def _mark_cancelled(
        self,
        *,
        requested_by: str,
        reason: str,
    ) -> None:
        """Store cancellation metadata for later cooperative checkpoints."""

        if self._cancellation is not None:
            return

        self._cancellation = PipelineCancellation(
            run_id=self.run_id,
            scope=self.scope,
            requested_by=requested_by,
            reason=reason,
            checkpoint="",
        )


class PipelineCoordinator:
    """Coordinate foreground/background pipeline runs within one process."""

    def __init__(self) -> None:
        """Initialize an empty process-local active-run registry."""

        self._active_runs: dict[
            PipelineScope,
            dict[str, PipelineRunHandle],
        ] = {}

    def request_cancellation(
        self,
        *,
        scope: PipelineScope,
        requested_by: str,
        reason: str,
        target_precedence: Collection[str] = ("background",),
    ) -> list[str]:
        """Request cooperative cancellation for matching active runs.

        Args:
            scope: Channel scope whose active runs should be inspected.
            requested_by: Runtime component requesting cancellation.
            reason: Stable machine-readable cancellation reason.
            target_precedence: Run precedence values eligible for cancellation.

        Returns:
            Run ids that were marked cancelled by this request.
        """

        active_for_scope = self._active_runs.get(scope)
        if not active_for_scope:
            cancelled_run_ids: list[str] = []
            return cancelled_run_ids

        cancelled_run_ids = []
        for handle in active_for_scope.values():
            if handle.precedence not in target_precedence:
                continue
            if handle.cancelled():
                continue
            handle._mark_cancelled(
                requested_by=requested_by,
                reason=reason,
            )
            cancelled_run_ids.append(handle.run_id)

        return cancelled_run_ids

    async def start_run(
        self,
        *,
        scope: PipelineScope,
        owner: str,
        precedence: PipelinePrecedence,
        run_kind: str,
    ) -> PipelineRunAdmission:
        """Admit a scoped pipeline run or return a deterministic deferral.

        Args:
            scope: Channel scope for this pipeline run.
            owner: Runtime component that owns the run.
            precedence: Foreground or background precedence.
            run_kind: Stable runtime kind for diagnostics.

        Returns:
            Admission result with an active handle when admitted.
        """

        if precedence == "background" and self._foreground_active(scope):
            admission = PipelineRunAdmission(
                admitted=False,
                handle=None,
                defer_reason="same_scope_foreground_active",
            )
            return admission

        run_id = uuid4().hex
        handle = PipelineRunHandle(
            coordinator=self,
            run_id=run_id,
            scope=scope,
            owner=owner,
            precedence=precedence,
            run_kind=run_kind,
        )
        self._active_runs.setdefault(scope, {})[run_id] = handle
        admission = PipelineRunAdmission(
            admitted=True,
            handle=handle,
            defer_reason=None,
        )
        return admission

    def _foreground_active(self, scope: PipelineScope) -> bool:
        """Return whether a foreground run is active for the scope."""

        active_for_scope = self._active_runs.get(scope)
        if not active_for_scope:
            return_value = False
            return return_value

        return_value = any(
            handle.precedence == "foreground"
            for handle in active_for_scope.values()
        )
        return return_value

    def _release_handle(self, handle: PipelineRunHandle) -> None:
        """Remove a handle from the active-run registry if still present."""

        active_for_scope = self._active_runs.get(handle.scope)
        if not active_for_scope:
            return

        active_for_scope.pop(handle.run_id, None)
        if not active_for_scope:
            self._active_runs.pop(handle.scope, None)
