"""Debug adapter — a browser-based chat UI that talks directly to the brain service.

Usage:
    python -m adapters.debug_adapter --brain-url http://localhost:8000 --port 8080

Opens a simple web chat at http://localhost:8080
"""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

import httpx

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.logging_config import configure_adapter_logging

configure_adapter_logging()

logger = logging.getLogger(__name__)

BRAIN_URL = "http://localhost:8000"

debug_app = FastAPI(title="Kazusa Debug Adapter")


_HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Kazusa Debug Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0f0f0f; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  header { background: #1a1a2e; padding: 14px 20px; border-bottom: 1px solid #333;
           display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 1.1rem; color: #c084fc; }
  .config { display: flex; gap: 8px; margin-left: auto; font-size: 0.8rem; }
  .config input { background: #222; border: 1px solid #444; color: #ccc; padding: 4px 8px;
                  border-radius: 4px; font-size: 0.8rem; }
  .config label { color: #888; line-height: 28px; }
  #chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 10px; }
  .msg { max-width: 75%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
  .msg.user { align-self: flex-end; background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
  .msg.bot  { align-self: flex-start; background: #1e1e2e; color: #e0e0e0; border: 1px solid #333; border-bottom-left-radius: 4px; }
  .msg.system { align-self: center; background: transparent; color: #666; font-size: 0.8rem; font-style: italic; }
  .meta { font-size: 0.7rem; color: #888; margin-top: 4px; }
  #input-area { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #333; background: #1a1a1a; }
  #input-area textarea { flex: 1; resize: none; background: #222; border: 1px solid #444; color: #e0e0e0;
                         padding: 10px; border-radius: 8px; font-size: 0.95rem; font-family: inherit; }
  #input-area button { background: #7c3aed; color: #fff; border: none; padding: 10px 20px;
                       border-radius: 8px; cursor: pointer; font-size: 0.95rem; }
  #input-area button:hover { background: #6d28d9; }
  #input-area button:disabled { background: #444; cursor: not-allowed; }
  .spinner { display: inline-block; width: 16px; height: 16px; border: 2px solid #666;
             border-top-color: #c084fc; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Kazusa Debug Chat</h1>
  <div class="config">
    <label>Platform:</label>
    <input id="cfg-platform" value="debug" style="width:80px"/>
    <label>User ID:</label>
    <input id="cfg-uid" value="debug-user-001" style="width:130px"/>
    <label>Name:</label>
    <input id="cfg-name" value="DebugUser" style="width:100px"/>
    <label>Channel:</label>
    <input id="cfg-channel" value="debug-channel" style="width:120px"/>
    <span style="border-left:1px solid #444;margin:0 6px"></span>
    <label><input type="checkbox" id="dbg-listen"/> Listen Only</label>
    <label><input type="checkbox" id="dbg-think"/> Think Only</label>
    <label><input type="checkbox" id="dbg-noremember"/> No Remember</label>
  </div>
</header>
<div id="chat"></div>
<div id="input-area">
  <textarea id="msg-input" rows="2" placeholder="Type a message... (Enter to send, Shift+Enter for newline)"></textarea>
  <button id="send-btn" onclick="sendMessage()">Send</button>
</div>
<script>
const chatEl = document.getElementById('chat');
const inputEl = document.getElementById('msg-input');
const sendBtn = document.getElementById('send-btn');

inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

function addMsg(text, cls, meta) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  if (meta) { const m = document.createElement('div'); m.className = 'meta'; m.textContent = meta; d.appendChild(m); }
  chatEl.appendChild(d);
  chatEl.scrollTop = chatEl.scrollHeight;
  return d;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;
  inputEl.value = '';
  addMsg(text, 'user');
  sendBtn.disabled = true;
  const spinner = addMsg('', 'system');
  spinner.innerHTML = '<span class="spinner"></span> Thinking...';

  const payload = {
    platform: document.getElementById('cfg-platform').value,
    platform_channel_id: document.getElementById('cfg-channel').value,
    channel_type: 'private',
    platform_user_id: document.getElementById('cfg-uid').value,
    platform_bot_id: 'debug-bot-001',
    display_name: document.getElementById('cfg-name').value,
    channel_name: document.getElementById('cfg-channel').value,
    content_type: 'text',
    message_envelope: {
      body_text: text,
      raw_wire_text: text,
      mentions: [],
      attachments: [],
      addressed_to_global_user_ids: ['__CHARACTER_GLOBAL_USER_ID__'],
      broadcast: false,
    },
    debug_modes: {
      listen_only: document.getElementById('dbg-listen').checked,
      think_only: document.getElementById('dbg-think').checked,
      no_remember: document.getElementById('dbg-noremember').checked,
    },
  };

  try {
    const resp = await fetch('/api/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
    const data = await resp.json();
    spinner.remove();
    const msgs = data.messages || [];
    if (msgs.length === 0) { addMsg('(no response)', 'system'); }
    else { msgs.forEach(m => addMsg(m, 'bot')); }
    if (data.scheduled_followups > 0) { addMsg(`${data.scheduled_followups} follow-up(s) scheduled`, 'system'); }
  } catch (err) {
    spinner.remove();
    addMsg('Error: ' + err.message, 'system');
  }
  sendBtn.disabled = false;
  inputEl.focus();
}
</script>
</body>
</html>"""


class DebugModesIn(BaseModel):
    listen_only: bool = False
    think_only: bool = False
    no_remember: bool = False


class DebugChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    platform: str = "debug"
    platform_channel_id: str = ""
    channel_type: str = "private"
    platform_user_id: str = ""
    platform_bot_id: str = ""
    display_name: str = ""
    channel_name: str = ""
    content_type: str = "text"
    message_envelope: dict
    debug_modes: DebugModesIn = Field(default_factory=DebugModesIn)


@debug_app.get("/", response_class=HTMLResponse)
async def index():
    rendered_page = _HTML_PAGE.replace(
        "__CHARACTER_GLOBAL_USER_ID__",
        CHARACTER_GLOBAL_USER_ID,
    )
    return rendered_page


@debug_app.post("/api/chat")
async def proxy_chat(req: DebugChatRequest):
    """Forward chat request to the brain service."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{BRAIN_URL}/chat", json=req.model_dump())
        resp.raise_for_status()
        return_value = resp.json()
        return return_value


@debug_app.get("/api/health")
async def proxy_health():
    """Forward health check to the brain service."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{BRAIN_URL}/health")
        resp.raise_for_status()
        return_value = resp.json()
        return return_value


def main():
    global BRAIN_URL

    parser = argparse.ArgumentParser(description="Debug web adapter for Kazusa Brain Service")
    parser.add_argument("--brain-url", type=str, default="http://localhost:8000", help="Brain service URL")
    parser.add_argument("--port", type=int, default=8080, help="Port for the debug web UI")
    args = parser.parse_args()

    BRAIN_URL = args.brain_url.rstrip("/")
    logger.info(f"Debug adapter proxying to brain at {BRAIN_URL}")
    logger.info(f"Open http://localhost:{args.port} in your browser")

    uvicorn.run(debug_app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
