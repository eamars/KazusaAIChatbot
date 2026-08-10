"""Runtime adapter registration helpers for the brain service."""

from __future__ import annotations

from collections.abc import Callable

from .contracts import (
    RuntimeAdapterRegistrationRequest,
    RuntimeAdapterRegistrationResponse,
)


def register_runtime_adapter_payload(
    req: RuntimeAdapterRegistrationRequest,
    *,
    status: str,
    character_name: str,
    register_remote_runtime_adapter_func: Callable[..., None],
) -> RuntimeAdapterRegistrationResponse:
    """Register one remote adapter payload and return a normalized response.

    Args:
        req: Remote adapter registration or heartbeat payload.
        status: Response status string to return to the caller.
        character_name: Active brain-owned character display name.
        register_remote_runtime_adapter_func: Service-level registration hook.

    Returns:
        Structured confirmation for the adapter process.
    """

    registration_kwargs = {
        "platform": req.platform,
        "callback_url": req.callback_url,
        "platform_bot_id": req.platform_bot_id,
        "shared_secret": req.shared_secret,
        "timeout_seconds": req.timeout_seconds,
    }

    register_remote_runtime_adapter_func(**registration_kwargs)
    response = RuntimeAdapterRegistrationResponse(
        status=status,
        platform=req.platform,
        callback_url=req.callback_url,
        character_name=character_name,
    )
    return response
