import os
import argparse
import asyncio
import sys
from dotenv import load_dotenv
import json
import base64
import logging
import httpx
import websockets
from typing import Optional


logger = logging.getLogger(__name__)


class NapCatWSAdapter:
    def __init__(
        self,
        ws_url: str, 
        ws_token: str,
        brain_url: str, 
        brain_response_timeout: int,
    ):
        # User arguments
        self.ws_url: str = ws_url   
        self.ws_token = ws_token
        self.brain_url = brain_url.rstrip("/")

        # The following will be populated on connect
        self.bot_id: Optional[str] = None
        self.bot_name: Optional[str] = None

        # Initialize connection
        self.brain_client = httpx.AsyncClient(base_url=self.brain_url, timeout=brain_response_timeout)

    async def connect(self):
        """Connects to NapCat and synchronizes bot identity."""
        headers = {"Authorization": f"Bearer {self.ws_token}"} if self.ws_token else {}
        
        while True:
            try:
                logger.info(f"Connecting to NapCat at {self.ws_url}...")
                async with websockets.connect(self.ws_url, additional_headers=headers) as ws:
                    # 1. Sync Bot Info immediately after connecting
                    await self._fetch_bot_info(ws)
                    logger.info(f"Logged in as {self.bot_name} (ID: {self.bot_id})")
                    
                    # 2. Main Event Loop
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)

                        # Handle Events (messages, etc.)
                        asyncio.create_task(self.handle_event(data, ws))
            except Exception as e:
                logger.error(f"Connection lost: {e}. Retrying in 5s...")
                await asyncio.sleep(5)

    async def _fetch_bot_info(self, ws):
        """Calls get_login_info to retrieve the bot's QQ ID and Nickname."""
        response = await self._call_api(ws, "get_login_info")
        if response.get("status") == "ok":
            data = response.get("data", {})
            self.bot_id = str(data.get("user_id"))
            self.bot_name = data.get("nickname")
        else:
            logger.warning("Could not retrieve bot info, using defaults.")
            self.bot_id = "unknown"
            self.bot_name = "NapCat Bot"

    async def _call_api(self, ws, action: str, params: dict = None):
        """Helper to call OneBot APIs and wait for the specific response."""
        echo_id = f"sync_{action}_{asyncio.get_event_loop().time()}"
        payload = {
            "action": action,
            "params": params or {},
            "echo": echo_id
        }

        await ws.send(json.dumps(payload))

        while True:
            message = await asyncio.wait_for(ws.recv(), timeout=10.0)
            response = json.loads(message)
            if response.get("echo") == echo_id:
                return response

    async def handle_event(self, data: dict, ws):
        """Processes incoming messages (only if we are identified)."""
        if data.get("post_type") != "message" or not self.bot_id:
            return

        user_id = str(data.get("user_id"))
        group_id = data.get("group_id")
        raw_content = data.get("raw_message", "")
        sender_name = data.get("sender", {}).get("nickname", f"User {user_id}")
        
        is_group = data.get("message_type") == "group"
        channel_id = str(group_id) if is_group else user_id

        logger.info(
            "Incoming QQ message: user_id=%s channel_id=%s is_group=%s sender=%s content=%s",
            user_id,
            channel_id,
            is_group,
            sender_name,
            raw_content,
        )

        # Attachments processing (same as before)
        attachments = []
        for seg in data.get("message", []):
            if isinstance(seg, dict) and seg.get("type") == "image":
                url = seg["data"].get("url")
                if url:
                    try:
                        resp = await self.brain_client.get(url, timeout=10.0)
                        attachments.append({
                            "media_type": "image/jpeg",
                            "base64_data": base64.b64encode(resp.content).decode("utf-8"),
                            "description": ""
                        })
                    except Exception as e:
                        logger.error(f"Image fetch error: {e}")

        # Brain Forwarding
        payload = {
            "platform": "qq",
            "platform_channel_id": channel_id,
            "platform_user_id": user_id,
            "platform_bot_id": self.bot_id,
            "display_name": sender_name,
            "channel_name": f"Group {group_id}" if is_group else "Private",
            "content": raw_content,
            "content_type": "text",
            "attachments": attachments,
        }

        logger.info(
            "Forwarding to brain: channel_id=%s user_id=%s content=%s attachments=%s",
            channel_id,
            user_id,
            raw_content,
            len(attachments),
        )

        try:
            resp = await self.brain_client.post(f"{self.brain_url}/chat", json=payload)
            resp.raise_for_status()
            brain_data = resp.json()
        except Exception:
            logger.exception("Brain service failed")
            return

        # Response handling
        replies = brain_data.get("messages", [])
        logger.info(
            "Brain output: should_reply=%s message_count=%s messages=%s",
            brain_data.get("should_reply"),
            len(replies),
            replies,
        )
        if replies:
            combined = "\n".join(replies)
            msg_params = {
                "message_type": "group" if is_group else "private",
                "group_id" if is_group else "user_id": int(channel_id),
                "message": combined
            }
            if brain_data.get("should_reply"):
                msg_params["message"] = f"[CQ:reply,id={data['message_id']}]" + combined

            logger.info(
                "Sending QQ message: channel_id=%s message=%s",
                channel_id,
                msg_params["message"],
            )
            
            # We don't need to wait for 'send_msg' response usually
            await ws.send(json.dumps({"action": "send_msg", "params": msg_params}))

    async def close(self):
        await self.brain_client.aclose()



def main():
    parser = argparse.ArgumentParser(description="Discord adapter for Kazusa Brain Service")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Todo: Add channel filter

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load dotenv
    load_dotenv()

    ws_url = os.getenv("NAPCAT_WS_URL")
    if not ws_url:
        logger.error("NAPCAT_WS_URL environment variable is required")
        sys.exit(1)

    ws_token = os.getenv("NAPCAT_WS_TOKEN")
    if not ws_token:
        logger.error("NAPCAT_WS_TOKEN environment variable is required")
        sys.exit(1)

    brain_url = os.getenv("BRAIN_URL")
    if not brain_url:
        logger.error("BRAIN_URL environment variable is required")
        sys.exit(1)

    brain_response_timeout = os.getenv("BRAIN_RESPONSE_TIMEOUT", "120")
    brain_response_timeout = int(brain_response_timeout)

    adapter = NapCatWSAdapter(
        ws_url=ws_url,
        ws_token=ws_token,
        brain_url=brain_url,
        brain_response_timeout=brain_response_timeout,
    )

    try:
        asyncio.run(adapter.connect())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(adapter.close())


if __name__ == "__main__":
    main()