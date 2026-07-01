"""Thin Discord adapter — translates Discord events into brain service HTTP calls.

Usage:
    python -m adapters.discord_adapter \
        --brain-url http://localhost:8000 \
        --channels 12345 67890 \
        --no-listen-all
"""

from __future__ import annotations

import argparse
import asyncio
import base64
from collections.abc import Sequence
from datetime import datetime, timezone
import logging
import os
import re
import sys
from types import SimpleNamespace

import discord
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from adapters.delivery_receipts import post_delivery_receipt
from adapters.envelope_common import (
    addressed_to_from_envelope_parts,
    attachment_refs,
    normalize_mention_display_map,
    normalize_body_spacing,
    normalize_numeric_platform_user_id,
    readable_mention_token,
    resolve_mentions,
)
from adapters.inline_mentions import InlineMention, inline_mention_parts
from adapters.outbound_sequence import followup_delay_seconds
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.dispatcher import SendResult
from kazusa_ai_chatbot.logging_config import configure_adapter_logging
from kazusa_ai_chatbot.message_envelope import (
    AttachmentHandlerRegistryProtocol,
    MentionResolver,
    PassthroughMentionResolver,
    build_default_attachment_handler_registry,
)
from kazusa_ai_chatbot.message_envelope.types import (
    MessageEnvelope,
    RawMention,
    ReplyTarget,
)
from kazusa_ai_chatbot.utils import log_preview

configure_adapter_logging()

logger = logging.getLogger(__name__)

BRAIN_URL: str = "http://localhost:8000"
runtime_app = FastAPI(title="Kazusa Discord Runtime Adapter")
_runtime_adapter: "DiscordAdapter | None" = None

_USER_MENTION_PATTERN = re.compile(r"<@!?(\d+)>")
_ROLE_MENTION_PATTERN = re.compile(r"<@&(\d+)>")
_CHANNEL_MENTION_PATTERN = re.compile(r"<#(\d+)>")
_CUSTOM_EMOJI_PATTERN = re.compile(r"<a?:[A-Za-z0-9_]+:\d+>")
_EVERYONE_PATTERN = re.compile(r"@(everyone|here)\b")


class DiscordEnvelopeNormalizer:
    """Adapter-local Discord normalizer for brain-safe envelopes."""

    def normalize(
        self,
        request: object,
        mention_resolver: MentionResolver,
        attachment_handlers: AttachmentHandlerRegistryProtocol,
    ) -> MessageEnvelope:
        """Normalize one Discord request.

        Args:
            request: Request-like object with Discord wire content and metadata.
            mention_resolver: Resolver used to project raw mentions.
            attachment_handlers: Registry used to normalize attachments.

        Returns:
            Message envelope with Discord mention tags represented as readable
            tokens in `body_text`.
        """

        raw_wire_text = str(request.content or "")
        platform_bot_id = str(request.platform_bot_id)
        channel_type = str(request.channel_type)
        reply_context = dict(request.reply_context)
        user_display_names = normalize_mention_display_map(
            getattr(request, "user_mention_display_names", {}),
        )
        role_display_names = normalize_mention_display_map(
            getattr(request, "role_mention_display_names", {}),
        )
        channel_display_names = normalize_mention_display_map(
            getattr(request, "channel_mention_display_names", {}),
        )

        raw_mentions = self._raw_mentions(
            raw_wire_text,
            platform_bot_id,
            user_display_names,
            role_display_names,
            channel_display_names,
        )
        reply = self._reply_target(reply_context, platform_bot_id)
        body_text = self._body_text(
            raw_wire_text,
            platform_bot_id,
            user_display_names,
            role_display_names,
            channel_display_names,
        )

        mentions = resolve_mentions(raw_mentions, mention_resolver)
        addressed_to = addressed_to_from_envelope_parts(
            mentions=mentions,
            reply=reply,
            channel_type=channel_type,
        )
        envelope: MessageEnvelope = {
            "body_text": body_text,
            "raw_wire_text": raw_wire_text,
            "mentions": mentions,
            "attachments": attachment_refs(request.attachments, attachment_handlers),
            "addressed_to_global_user_ids": addressed_to,
            "broadcast": False,
        }
        if reply:
            envelope["reply"] = reply

        return envelope

    def _body_text(
        self,
        raw_wire_text: str,
        platform_bot_id: str,
        user_display_names: dict[str, str],
        role_display_names: dict[str, str],
        channel_display_names: dict[str, str],
    ) -> str:
        """Replace Discord mention wire tags with readable body tokens."""

        occurrence_counts: dict[str, int] = {}

        def next_occurrence(kind: str) -> int:
            occurrence_counts[kind] = occurrence_counts.get(kind, 0) + 1
            return occurrence_counts[kind]

        def replace_user(match: re.Match[str]) -> str:
            platform_user_id = match.group(1)
            entity_kind = "bot" if platform_user_id == platform_bot_id else "user"
            occurrence_index = next_occurrence("user")
            token = readable_mention_token(
                entity_kind=entity_kind,
                display_name=user_display_names.get(platform_user_id, ""),
                occurrence_index=occurrence_index,
            )
            return_value = f" {token} "
            return return_value

        def replace_role(match: re.Match[str]) -> str:
            platform_role_id = match.group(1)
            occurrence_index = next_occurrence("platform_role")
            token = readable_mention_token(
                entity_kind="platform_role",
                display_name=role_display_names.get(platform_role_id, ""),
                occurrence_index=occurrence_index,
            )
            return_value = f" {token} "
            return return_value

        def replace_channel(match: re.Match[str]) -> str:
            platform_channel_id = match.group(1)
            occurrence_index = next_occurrence("channel")
            token = readable_mention_token(
                entity_kind="channel",
                display_name=channel_display_names.get(platform_channel_id, ""),
                occurrence_index=occurrence_index,
            )
            return_value = f" {token} "
            return return_value

        def replace_everyone(match: re.Match[str]) -> str:
            raw_label = match.group(1)
            occurrence_index = next_occurrence("everyone")
            token = readable_mention_token(
                entity_kind="everyone",
                display_name="",
                occurrence_index=occurrence_index,
                raw_label=raw_label,
            )
            return_value = f" {token} "
            return return_value

        body_text = _USER_MENTION_PATTERN.sub(replace_user, raw_wire_text)
        body_text = _ROLE_MENTION_PATTERN.sub(replace_role, body_text)
        body_text = _CHANNEL_MENTION_PATTERN.sub(replace_channel, body_text)
        body_text = _CUSTOM_EMOJI_PATTERN.sub(" ", body_text)
        body_text = _EVERYONE_PATTERN.sub(replace_everyone, body_text)
        body_text = normalize_body_spacing(body_text)
        return body_text

    def _raw_mentions(
        self,
        raw_wire_text: str,
        platform_bot_id: str,
        user_display_names: dict[str, str],
        role_display_names: dict[str, str],
        channel_display_names: dict[str, str],
    ) -> list[RawMention]:
        """Extract typed Discord mention fragments from wire text."""

        raw_mentions: list[RawMention] = []
        for match in _USER_MENTION_PATTERN.finditer(raw_wire_text):
            platform_user_id = match.group(1)
            entity_kind = "bot" if platform_user_id == platform_bot_id else "user"
            raw_mentions.append({
                "platform": "discord",
                "platform_user_id": platform_user_id,
                "entity_kind": entity_kind,
                "raw_text": match.group(0),
                "display_name": user_display_names.get(platform_user_id, ""),
            })

        for match in _ROLE_MENTION_PATTERN.finditer(raw_wire_text):
            platform_role_id = match.group(1)
            raw_mentions.append({
                "platform": "discord",
                "platform_user_id": platform_role_id,
                "entity_kind": "platform_role",
                "raw_text": match.group(0),
                "display_name": role_display_names.get(platform_role_id, ""),
            })

        for match in _CHANNEL_MENTION_PATTERN.finditer(raw_wire_text):
            platform_channel_id = match.group(1)
            raw_mentions.append({
                "platform": "discord",
                "platform_user_id": platform_channel_id,
                "entity_kind": "channel",
                "raw_text": match.group(0),
                "display_name": channel_display_names.get(platform_channel_id, ""),
            })

        for match in _EVERYONE_PATTERN.finditer(raw_wire_text):
            raw_mentions.append({
                "platform": "discord",
                "platform_user_id": match.group(1),
                "entity_kind": "everyone",
                "raw_text": match.group(0),
                "display_name": match.group(1),
            })

        return raw_mentions

    def _reply_target(
        self,
        reply_context: dict,
        platform_bot_id: str,
    ) -> ReplyTarget:
        """Project Discord reply metadata into a typed reply target."""

        reply: ReplyTarget = {}
        if reply_context.get("reply_to_message_id"):
            reply["platform_message_id"] = str(reply_context["reply_to_message_id"])
            reply["derivation"] = "platform_native"
        if reply_context.get("reply_to_platform_user_id"):
            platform_user_id = str(reply_context["reply_to_platform_user_id"])
            reply["platform_user_id"] = platform_user_id
            if platform_user_id == platform_bot_id:
                reply["global_user_id"] = CHARACTER_GLOBAL_USER_ID
        if reply_context.get("reply_to_display_name"):
            reply["display_name"] = str(reply_context["reply_to_display_name"])
        if reply_context.get("reply_excerpt"):
            reply["excerpt"] = str(reply_context["reply_excerpt"])
        return reply


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
    """Report whether the live Discord adapter can send to one target."""

    if _runtime_adapter is None:
        raise HTTPException(status_code=503, detail="Runtime adapter is not ready")
    if _runtime_adapter.runtime_shared_secret:
        expected = f"Bearer {_runtime_adapter.runtime_shared_secret}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    available = await _runtime_adapter.can_send_message(
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
    """Deliver one scheduled outbound message through the live Discord adapter."""

    if _runtime_adapter is None:
        raise HTTPException(status_code=503, detail="Runtime adapter is not ready")
    if _runtime_adapter.runtime_shared_secret:
        expected = f"Bearer {_runtime_adapter.runtime_shared_secret}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        result = await _runtime_adapter.send_message(
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


class DiscordAdapter(discord.Client):
    """Discord client that forwards messages to the Kazusa brain service."""

    platform = "discord"

    def __init__(
        self,
        brain_url: str,
        runtime_host: str,
        runtime_port: int,
        runtime_public_url: str,
        runtime_shared_secret: str = "",
        heartbeat_seconds: float = 30.0,
        channel_ids: list[str] | None = None,
        debug_modes: dict | None = None,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents, **kwargs)

        self.brain_url = brain_url.rstrip("/")
        self.runtime_host = runtime_host
        self.runtime_port = runtime_port
        self.runtime_public_url = runtime_public_url.rstrip("/")
        self.runtime_shared_secret = runtime_shared_secret
        self.heartbeat_seconds = heartbeat_seconds
        self.channel_ids = set(channel_ids) if channel_ids is not None else None
        self.debug_modes = debug_modes or {}
        self._envelope_normalizer = DiscordEnvelopeNormalizer()
        self._attachment_handlers = build_default_attachment_handler_registry()
        self._mention_resolver = PassthroughMentionResolver()
        self._http_client = httpx.AsyncClient(timeout=120.0)
        self._runtime_server: uvicorn.Server | None = None
        self._runtime_server_task: asyncio.Task | None = None
        self._brain_registration_done = False
        self._heartbeat_task: asyncio.Task | None = None
        self._normal_chat_delivery_tasks: set[asyncio.Task] = set()

    def _outbound_channel_allowed(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Return whether this adapter may visibly send to the target."""

        if channel_type == "private":
            return_value = True
            return return_value
        if channel_type != "group":
            return_value = False
            return return_value
        if self.channel_ids is None:
            return_value = False
            return return_value
        return_value = str(channel_id) in self.channel_ids
        return return_value

    async def setup_hook(self) -> None:
        """Start the runtime callback server before Discord events begin."""

        await self._ensure_runtime_server_started()

    async def on_ready(self):
        logger.info(f"Discord adapter logged in as {self.user} (id={self.user.id})")
        if not self._brain_registration_done:
            await self._register_with_brain()
            self._brain_registration_done = True
            self._ensure_heartbeat_started()
        if self.channel_ids is not None:
            logger.info(f"Active in channels: {self.channel_ids}. Other channels are listen-only.")
        else:
            logger.info("No active channels configured — all channels are listen-only (DMs are active).")

    async def _ensure_runtime_server_started(self) -> None:
        """Start the adapter callback server exactly once for brain-side sends."""

        global _runtime_adapter

        _runtime_adapter = self
        if self._runtime_server_task is not None:
            return

        config = uvicorn.Config(
            runtime_app,
            host=self.runtime_host,
            port=self.runtime_port,
            log_level="warning",
        )
        self._runtime_server = uvicorn.Server(config)
        self._runtime_server_task = asyncio.create_task(self._runtime_server.serve())
        logger.info(f"Discord runtime callback listening on {self.runtime_host}:{self.runtime_port}")

    async def _register_with_brain(self) -> None:
        """Register this adapter's outbound callback URL with the brain service."""

        payload = self._runtime_registration_payload()
        response = await self._http_client.post(
            f"{self.brain_url}/runtime/adapters/register",
            json=payload,
        )
        response.raise_for_status()
        logger.info(f"Registered Discord runtime adapter with brain: callback_url={self.runtime_public_url}")

    def _runtime_registration_payload(self) -> dict:
        """Return the shared registration payload for startup and heartbeat."""

        return_value = {
            "platform": self.platform,
            "callback_url": self.runtime_public_url,
            "shared_secret": self.runtime_shared_secret,
            "timeout_seconds": 10.0,
        }
        user = getattr(self, "user", None)
        if user is not None:
            return_value["platform_bot_id"] = str(user.id)
            return_value["display_name"] = str(user.display_name)
        return return_value

    async def _send_heartbeat_once(self) -> None:
        """Refresh the adapter registration in the brain service."""

        response = await self._http_client.post(
            f"{self.brain_url}/runtime/adapters/heartbeat",
            json=self._runtime_registration_payload(),
        )
        response.raise_for_status()

    def _ensure_heartbeat_started(self) -> None:
        """Start the re-registration heartbeat exactly once."""

        if self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(self._run_brain_heartbeat())

    async def _run_brain_heartbeat(self) -> None:
        """Keep refreshing adapter registration so brain restarts self-heal."""

        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await self._send_heartbeat_once()
            except httpx.HTTPError as exc:
                logger.warning(f"Discord runtime heartbeat failed: {exc}")

    async def _post_delivery_receipt(
        self,
        *,
        channel_id: str,
        delivery_tracking_id: str,
        platform_message_id: str,
    ) -> None:
        """Best-effort report of a delivered normal chat response."""

        await post_delivery_receipt(
            http_client=self._http_client,
            brain_url=self.brain_url,
            platform="discord",
            platform_channel_id=channel_id,
            delivery_tracking_id=delivery_tracking_id,
            platform_message_id=platform_message_id,
            adapter="discord",
            logger=logger,
            log_label="Discord",
        )

    def _track_normal_chat_delivery_task(self, task: asyncio.Task) -> None:
        """Track adapter-owned follow-up sends for cancellation and tests."""

        self._normal_chat_delivery_tasks.add(task)
        task.add_done_callback(self._finalize_normal_chat_delivery_task)

    def _finalize_normal_chat_delivery_task(self, task: asyncio.Task) -> None:
        """Discard completed follow-up tasks and log unexpected failures."""

        self._normal_chat_delivery_tasks.discard(task)
        if task.cancelled():
            return
        task_exception = task.exception()
        if task_exception is None:
            return
        logger.warning(
            f"Discord follow-up delivery task failed unexpectedly: "
            f"{task_exception}",
            exc_info=(
                type(task_exception),
                task_exception,
                task_exception.__traceback__,
            ),
        )

    async def _send_normal_chat_followups(
        self,
        *,
        messages: Sequence[str],
        channel: object,
        channel_id: str,
        channel_type: str,
        delivery_mentions: Sequence[dict] | None,
    ) -> None:
        """Send delayed follow-up messages from one brain cognition."""

        mention_candidates = (
            delivery_mentions
            if channel_type == "group"
            else None
        )
        for message_text in messages:
            await asyncio.sleep(followup_delay_seconds(message_text))
            outbound_text = _outbound_text_with_delivery_mentions(
                message_text,
                mention_candidates,
            )
            for chunk in _split_message(outbound_text):
                try:
                    await channel.send(chunk)
                except discord.HTTPException as exc:
                    logger.warning(
                        f"Discord follow-up send failed: "
                        f"channel_id={channel_id} error={exc}"
                    )
                    return

    async def on_message(self, message: discord.Message):
        """Normalize one Discord message event and forward it to the brain.

        Args:
            message: Discord message emitted by the client event stream.

        Returns:
            None.
        """

        if message.author == self.user:
            return

        is_dm = message.guild is None
        channel_id_str = str(message.channel.id)
        reply_context: dict[str, str] = {}

        if message.reference is not None and message.reference.message_id is not None:
            reply_context["reply_to_message_id"] = str(message.reference.message_id)

        referenced_message = message.reference.resolved if message.reference is not None else None
        if isinstance(referenced_message, discord.Message):
            reply_context["reply_to_platform_user_id"] = str(referenced_message.author.id)
            reply_context["reply_to_display_name"] = referenced_message.author.display_name
            reply_context["reply_excerpt"] = referenced_message.content
        message_debug_modes = dict(self.debug_modes)
        is_active = is_dm or (self.channel_ids is not None and channel_id_str in self.channel_ids)
        
        if not is_active:
            message_debug_modes["listen_only"] = True
            
        mode_label = "LISTEN-ONLY" if message_debug_modes.get("listen_only") else "ACTIVE"

        logger.info(
            f"[{mode_label}] Incoming Discord message: "
            f"channel_id={channel_id_str} is_dm={is_dm} "
            f"author={message.author.display_name}"
        )
        logger.debug(
            f"Incoming Discord message detail: channel_id={channel_id_str} "
            f"content={log_preview(message.content)}"
        )

        # Build attachments
        attachments = []
        for att in message.attachments:
            if att.size and att.size > (5 * 1024 * 1024):
                logger.warning(f"Attachment {att.filename} exceeds 5MB — skipping")
                continue

            if att.content_type and att.content_type.startswith("image/"):
                try:
                    async with httpx.AsyncClient() as client:
                        resp = await client.get(att.url)
                        resp.raise_for_status()
                        b64 = base64.b64encode(resp.content).decode("utf-8")
                        attachments.append({
                            "media_type": att.content_type,
                            "base64_data": b64,
                            "description": "",
                        })
                except httpx.HTTPError as exc:
                    logger.exception(f"Failed to fetch attachment {att.url}: {exc}")

        # Determine channel name
        if message.guild is None:
            channel_name = f"Private chat with {message.author.display_name}"
        else:
            channel_name = str(message.channel.name)

        user_mention_display_names: dict[str, str] = {}
        for mentioned_user in getattr(message, "mentions", []):
            platform_user_id = getattr(mentioned_user, "id", None)
            if platform_user_id is None:
                continue
            display_name = getattr(mentioned_user, "display_name", "")
            if not isinstance(display_name, str):
                display_name = ""
            user_mention_display_names[str(platform_user_id)] = display_name

        role_mention_display_names: dict[str, str] = {}
        for mentioned_role in getattr(message, "role_mentions", []):
            platform_role_id = getattr(mentioned_role, "id", None)
            if platform_role_id is None:
                continue
            display_name = getattr(mentioned_role, "name", "")
            if not isinstance(display_name, str):
                display_name = ""
            role_mention_display_names[str(platform_role_id)] = display_name

        channel_mention_display_names: dict[str, str] = {}
        for mentioned_channel in getattr(message, "channel_mentions", []):
            platform_channel_id = getattr(mentioned_channel, "id", None)
            if platform_channel_id is None:
                continue
            display_name = getattr(mentioned_channel, "name", "")
            if not isinstance(display_name, str):
                display_name = ""
            channel_mention_display_names[str(platform_channel_id)] = display_name

        envelope_request = SimpleNamespace(
            platform="discord",
            channel_type="private" if is_dm else "group",
            content=message.content,
            platform_bot_id=str(self.user.id) if self.user else "",
            reply_context=reply_context,
            user_mention_display_names=user_mention_display_names,
            role_mention_display_names=role_mention_display_names,
            channel_mention_display_names=channel_mention_display_names,
            attachments=attachments,
        )
        envelope = self._envelope_normalizer.normalize(
            envelope_request,
            self._mention_resolver,
            self._attachment_handlers,
        )

        # Build the request payload
        payload = {
            "platform": "discord",
            "platform_channel_id": str(message.channel.id),
            "channel_type": "private" if is_dm else "group",
            "platform_message_id": str(message.id),
            "platform_user_id": str(message.author.id),
            "platform_bot_id": str(self.user.id) if self.user else "",
            "display_name": message.author.display_name,
            "channel_name": channel_name,
            "content_type": "text",
            "message_envelope": envelope,
            "debug_modes": message_debug_modes,
        }

        try:
            # Only show typing indicator if the bot is actually going to think and reply
            if not message_debug_modes.get("listen_only"):
                async with message.channel.typing():
                    resp = await self._http_client.post(
                        f"{self.brain_url}/chat",
                        json=payload,
                    )
            else:
                resp = await self._http_client.post(
                    f"{self.brain_url}/chat",
                    json=payload,
                )
            
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.exception(f"Brain service request failed: {exc}")
            if not message_debug_modes.get("listen_only"):
                await message.reply("I'm having trouble thinking right now, please try again later.")
            return

        # Send response messages
        messages = data.get("messages", [])
        if not messages:
            return
        if not self._outbound_channel_allowed(
            channel_id_str,
            channel_type="private" if is_dm else "group",
        ):
            logger.warning(
                "Suppressing Discord response for disallowed target: "
                f"channel_type={'private' if is_dm else 'group'} "
                f"channel_id={channel_id_str}"
            )
            return

        use_reply = data.get("use_reply_feature", False)
        raw_delivery_mentions = data.get("delivery_mentions")
        delivery_mentions = (
            raw_delivery_mentions
            if isinstance(raw_delivery_mentions, list)
            else None
        )
        channel_type = "private" if is_dm else "group"
        mention_candidates = delivery_mentions if channel_type == "group" else None
        first_outbound_text = _outbound_text_with_delivery_mentions(
            messages[0],
            mention_candidates,
        )
        first_sent_message_id = ""
        for chunk in _split_message(first_outbound_text):
            try:
                if use_reply:
                    sent_message = await message.reply(chunk)
                    use_reply = False
                else:
                    sent_message = await message.channel.send(chunk)
            except discord.HTTPException as exc:
                logger.warning(
                    f"Discord normal chat send failed: "
                    f"channel_id={channel_id_str} error={exc}"
                )
                if not first_sent_message_id:
                    return
                break
            if not first_sent_message_id:
                first_sent_message_id = str(sent_message.id)

        delivery_tracking_id = str(data.get("delivery_tracking_id") or "")
        await self._post_delivery_receipt(
            channel_id=channel_id_str,
            delivery_tracking_id=delivery_tracking_id,
            platform_message_id=first_sent_message_id,
        )

        followup_messages = messages[1:]
        if followup_messages:
            followup_task = asyncio.create_task(
                self._send_normal_chat_followups(
                    messages=followup_messages,
                    channel=message.channel,
                    channel_id=channel_id_str,
                    channel_type=channel_type,
                    delivery_mentions=delivery_mentions,
                )
            )
            self._track_normal_chat_delivery_task(followup_task)

        # Handle multimedia attachments from the response
        for att in data.get("attachments", []):
            # TODO: send images/stickers when the brain supports it
            pass

    async def close(self):
        """Close adapter-owned HTTP clients and callback server."""

        normal_chat_delivery_tasks = list(self._normal_chat_delivery_tasks)
        for task in normal_chat_delivery_tasks:
            task.cancel()
        if normal_chat_delivery_tasks:
            await asyncio.gather(
                *normal_chat_delivery_tasks,
                return_exceptions=True,
            )
            self._normal_chat_delivery_tasks.clear()

        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError as exc:
                logger.debug(f"Heartbeat task cancelled during close: {exc}")
                pass
        await self._http_client.aclose()
        if self._runtime_server is not None:
            self._runtime_server.should_exit = True
        if self._runtime_server_task is not None:
            await self._runtime_server_task
        await super().close()

    async def can_send_message(
        self,
        channel_id: str,
        *,
        channel_type: str,
    ) -> bool:
        """Return whether Discord can accept one outbound target."""

        if not self._outbound_channel_allowed(
            channel_id,
            channel_type=channel_type,
        ):
            return_value = False
            return return_value
        try:
            channel_key = int(channel_id)
        except ValueError:
            return_value = False
            return return_value
        channel = self.get_channel(channel_key)
        if channel is not None:
            return_value = True
            return return_value
        try:
            await self.fetch_channel(channel_key)
        except (
            discord.Forbidden,
            discord.NotFound,
            discord.HTTPException,
        ):
            return_value = False
            return return_value
        return_value = True
        return return_value

    async def send_message(
        self,
        channel_id: str,
        text: str,
        *,
        channel_type: str,
        reply_to_msg_id: str | None = None,
        delivery_mentions: Sequence[dict] | None = None,
    ) -> SendResult:
        """Send one outbound message through the Discord client.

        Args:
            channel_id: Discord channel identifier.
            text: Message body to send.
            channel_type: Scheduler target scope. Discord sends by channel id
                for both guild channels and DM channels.
            reply_to_msg_id: Optional message id to reply to.
            delivery_mentions: Optional adapter-owned mention requests.

        Returns:
            Structured send result for the dispatcher.
        """

        if channel_type not in {"group", "private"}:
            raise RuntimeError(f"Unsupported Discord channel_type: {channel_type}")
        if not self._outbound_channel_allowed(
            channel_id,
            channel_type=channel_type,
        ):
            raise RuntimeError(
                "Discord target channel is not allowed: "
                f"channel_type={channel_type} channel_id={channel_id}"
            )

        channel = self.get_channel(int(channel_id))
        if channel is None:
            channel = await self.fetch_channel(int(channel_id))

        send_kwargs = {}
        if reply_to_msg_id:
            send_kwargs["reference"] = discord.PartialMessage(
                channel=channel,
                id=int(reply_to_msg_id),
            )

        outbound_text = _outbound_text_with_delivery_mentions(
            text,
            delivery_mentions if channel_type == "group" else None,
        )
        message = await channel.send(outbound_text, **send_kwargs)
        return_value = SendResult(
            platform=self.platform,
            channel_id=channel_id,
            message_id=str(message.id),
            sent_at=datetime.now(timezone.utc),
        )
        return return_value


def _outbound_text_with_delivery_mentions(
    text: str,
    delivery_mentions: Sequence[dict] | None,
) -> str:
    """Return Discord text with exact inline user tags rendered natively."""

    fragments: list[str] = []
    for part in inline_mention_parts(text, delivery_mentions):
        if isinstance(part, InlineMention):
            normalized_user_id = normalize_numeric_platform_user_id(
                part.platform_user_id
            )
            if normalized_user_id:
                fragments.append(f"<@{normalized_user_id}>")
            else:
                fragments.append(part.token)
            continue
        fragments.append(part)

    return_value = "".join(fragments)
    return return_value


def _split_message(text: str, limit: int = 2000) -> list[str]:
    """Split a message into chunks that fit within Discord's character limit."""
    if len(text) <= limit:
        return_value = [text]
        return return_value
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, limit)
        if split_at == -1:
            split_at = text.rfind(" ", 0, limit)
        if split_at == -1:
            split_at = limit
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Discord adapter for Kazusa Brain Service")
    parser.add_argument("--channels", type=str, nargs="*", default=None, help="Discord channel IDs to actively participate in. Other channels will be listen-only.")
    parser.add_argument("--runtime-host", type=str, default=os.getenv("ADAPTER_RUNTIME_HOST", "127.0.0.1"))
    parser.add_argument("--runtime-port", type=int, default=int(os.getenv("DISCORD_RUNTIME_PORT", "8012")))
    parser.add_argument("--runtime-public-url", type=str, default=os.getenv("ADAPTER_RUNTIME_PUBLIC_URL", ""))
    parser.add_argument("--heartbeat-seconds", type=float, default=float(os.getenv("ADAPTER_HEARTBEAT_SECONDS", "30")))
    args = parser.parse_args()

    token = os.getenv("DISCORD_TOKEN", "")
    if not token:
        logger.error("DISCORD_TOKEN environment variable is required")
        sys.exit(1)

    brain_url = os.getenv("BRAIN_URL")
    if not brain_url:
        logger.error("BRAIN_URL environment variable is required")
        sys.exit(1)
    runtime_public_url = args.runtime_public_url or f"http://127.0.0.1:{args.runtime_port}"
    runtime_shared_secret = os.getenv("ADAPTER_RUNTIME_SHARED_SECRET", "")

    adapter = DiscordAdapter(
        brain_url=brain_url,
        runtime_host=args.runtime_host,
        runtime_port=args.runtime_port,
        runtime_public_url=runtime_public_url,
        runtime_shared_secret=runtime_shared_secret,
        heartbeat_seconds=args.heartbeat_seconds,
        channel_ids=args.channels,
        debug_modes={},
    )
    adapter.run(token, log_handler=None)


if __name__ == "__main__":
    main()
