"""FastAPI callback surface and active binding for the NapCat adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from kazusa_ai_chatbot.dispatcher import SendResult


class RuntimeNapCatAdapter(Protocol):
    """Narrow send interface required by the runtime callback API."""

    runtime_shared_secret: str

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Return whether one target can receive a runtime send."""
        ...

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: Sequence[dict] | None = None,
    ) -> SendResult:
        """Deliver one runtime message through the live adapter."""
        ...


runtime_app = FastAPI(title="Kazusa NapCat Runtime Adapter")
_runtime_adapter: RuntimeNapCatAdapter | None = None


def bind_runtime_adapter(adapter: RuntimeNapCatAdapter | None) -> None:
    """Set or clear the process-local adapter used by callback endpoints."""

    global _runtime_adapter

    _runtime_adapter = adapter


def current_runtime_adapter() -> RuntimeNapCatAdapter | None:
    """Return the adapter currently bound to callback endpoints, if any."""

    return _runtime_adapter


class RuntimeSendMessageRequest(BaseModel):
    channel_id: str
    channel_type: str
    text: str
    reply_to_msg_id: str | None = None
    delivery_mentions: list[dict] | None = None


class RuntimeSendMessageResponse(BaseModel):
    platform: str
    channel_id: str
    message_id: str
    sent_at: str


class RuntimeSendMessageCapabilityRequest(BaseModel):
    channel_id: str
    channel_type: str


class RuntimeSendMessageCapabilityResponse(BaseModel):
    available: bool
    reason: str = ""


@runtime_app.post(
    "/send_message/capability",
    response_model=RuntimeSendMessageCapabilityResponse,
)
async def send_message_capability_endpoint(
    req: RuntimeSendMessageCapabilityRequest,
    authorization: str = Header(default=""),
):
    """Report whether the live NapCat adapter can send to one target."""

    adapter = current_runtime_adapter()
    if adapter is None:
        raise HTTPException(status_code=503, detail="Runtime adapter is not ready")
    if adapter.runtime_shared_secret:
        expected = f"Bearer {adapter.runtime_shared_secret}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    available = await adapter.can_send_message(
        channel_id=req.channel_id,
        channel_type=req.channel_type,
    )
    response = RuntimeSendMessageCapabilityResponse(available=available)
    return response


@runtime_app.post("/send_message", response_model=RuntimeSendMessageResponse)
async def send_message_endpoint(
    req: RuntimeSendMessageRequest,
    authorization: str = Header(default=""),
):
    """Deliver one scheduled outbound message through the live NapCat adapter."""

    adapter = current_runtime_adapter()
    if adapter is None:
        raise HTTPException(status_code=503, detail="Runtime adapter is not ready")
    if adapter.runtime_shared_secret:
        expected = f"Bearer {adapter.runtime_shared_secret}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = await adapter.send_message(
            channel_id=req.channel_id,
            text=req.text,
            reply_to_msg_id=req.reply_to_msg_id,
            channel_type=req.channel_type,
            delivery_mentions=req.delivery_mentions,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return_value = RuntimeSendMessageResponse(
        platform=result.platform,
        channel_id=result.channel_id,
        message_id=result.message_id,
        sent_at=result.sent_at.isoformat(),
    )
    return return_value
