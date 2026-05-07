"""Pydantic contracts for the Kazusa brain service API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from kazusa_ai_chatbot.message_envelope import MentionEntityKind


class AttachmentIn(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""
    size_bytes: int | None = None


class DebugModesIn(BaseModel):
    listen_only: bool = False
    think_only: bool = False
    no_remember: bool = False


class MentionIn(BaseModel):
    platform_user_id: str = ""
    global_user_id: str = ""
    display_name: str = ""
    entity_kind: MentionEntityKind = "unknown"
    raw_text: str = ""


class ReplyTargetIn(BaseModel):
    platform_message_id: str = ""
    platform_user_id: str = ""
    global_user_id: str = ""
    display_name: str = ""
    excerpt: str = ""
    derivation: str = ""


class AttachmentRefIn(AttachmentIn):
    storage_shape: str = ""


class MessageEnvelopeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    body_text: str
    raw_wire_text: str
    mentions: list[MentionIn]
    reply: ReplyTargetIn | None = None
    attachments: list[AttachmentRefIn]
    addressed_to_global_user_ids: list[str]
    broadcast: bool


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str
    platform_channel_id: str = ""
    channel_type: str = "group"
    platform_message_id: str = ""
    platform_user_id: str
    platform_bot_id: str = ""
    display_name: str = ""
    channel_name: str = ""
    content_type: str = "text"
    message_envelope: MessageEnvelopeIn
    timestamp: str = ""
    debug_modes: DebugModesIn = Field(default_factory=DebugModesIn)


class AttachmentOut(BaseModel):
    media_type: str = ""
    url: str = ""
    base64_data: str = ""
    description: str = ""
    size_bytes: int | None = None


class ChatResponse(BaseModel):
    messages: list[str] = Field(default_factory=list)
    content_type: str = "text"
    attachments: list[AttachmentOut] = Field(default_factory=list)
    use_reply_feature: bool = False
    scheduled_followups: int = 0


class EventRequest(BaseModel):
    platform: str
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Cache2AgentStatsResponse(BaseModel):
    agent_name: str
    hit_count: int
    miss_count: int
    hit_rate: float


class Cache2HealthResponse(BaseModel):
    agents: list[Cache2AgentStatsResponse] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    db: bool
    scheduler: bool
    cache2: Cache2HealthResponse = Field(default_factory=Cache2HealthResponse)


class RuntimeAdapterRegistrationRequest(BaseModel):
    platform: str
    callback_url: str

    shared_secret: str = ""
    timeout_seconds: float = 10.0


class RuntimeAdapterRegistrationResponse(BaseModel):
    status: str
    platform: str
    callback_url: str
