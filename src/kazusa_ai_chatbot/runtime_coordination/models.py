"""Public data contracts for scoped runtime pipeline coordination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


PipelinePrecedence = Literal["foreground", "background"]


@dataclass(frozen=True)
class PipelineScope:
    """Canonical process-local pipeline scope for one channel.

    Args:
        platform: Adapter platform key, such as `qq`, `discord`, or `debug`.
        platform_channel_id: Platform channel or private conversation id.
        channel_type: Channel class, such as `group` or `private`.
    """

    platform: str
    platform_channel_id: str
    channel_type: str


@dataclass(frozen=True)
class PipelineCancellation:
    """Cancellation metadata reported at deterministic checkpoints.

    Args:
        run_id: Coordinator run id for the cancelled pipeline.
        scope: Channel scope that owns the cancelled run.
        requested_by: Runtime component that requested cancellation.
        reason: Stable machine-readable cancellation reason.
        checkpoint: Runtime checkpoint where the cancellation was observed.
    """

    run_id: str
    scope: PipelineScope
    requested_by: str
    reason: str
    checkpoint: str


class PipelineCancelled(Exception):
    """Raised when a cooperative pipeline checkpoint observes cancellation."""

    def __init__(self, cancellation: PipelineCancellation) -> None:
        """Initialize the exception with its structured cancellation record."""

        self.cancellation = cancellation
        message = (
            f"pipeline run {cancellation.run_id} cancelled at "
            f"{cancellation.checkpoint}: {cancellation.reason}"
        )
        super().__init__(message)
