"""Attached-media semantic capability tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest


@pytest.mark.asyncio
async def test_attached_media_inspection_accepts_only_brain_issued_semantic_refs() -> None:
    from kazusa_ai_chatbot.dsh_tool_gateway.contracts import OpaqueReferenceCodec
    from kazusa_ai_chatbot.dsh_tool_gateway.media import (
        MediaSemanticService,
        issue_attached_media_reference,
    )

    scope = ("debug", "channel-1", "user-1")
    codec = OpaqueReferenceCodec(b"secret")
    reference = issue_attached_media_reference(codec=codec, scope=scope, cache_ref="cache-1")

    async def inspect(request):
        return {"status": "answered", "answer": "a cup", "evidence_boundary_notes": ["visible"]}

    def get_media(received_scope, cache_ref):
        assert received_scope == scope
        return {"content_type": "image/png", "base64_data": "aGVsbG8=", "existing_descriptor": ""}

    service = MediaSemanticService(scope=scope, codec=codec, get_media=get_media, inspect=inspect)
    result = await service.inspect_attached_media(attached_media_ref=reference, question="What is visible?")
    assert result.entities[0]["answer"] == "a cup"
    foreign = issue_attached_media_reference(codec=codec, scope=("debug", "other", "user-1"), cache_ref="cache-1")
    denied = await service.inspect_attached_media(attached_media_ref=foreign, question="What is visible?")
    assert denied.status == "denied"


def test_attached_media_cache_is_available_to_a_separate_worker_process(
    tmp_path,
    monkeypatch,
) -> None:
    """The Brain cache mirror crosses the sidecar worker process boundary."""

    from kazusa_ai_chatbot.dsh_tool_gateway.media import persist_attached_media
    from kazusa_ai_chatbot.media_inspection.session_cache import (
        clear_session_media,
        put_session_media,
    )

    scope = ("debug", "channel-1", "user-1")
    monkeypatch.setenv("KAZUSA_DSH_DATA_ROOT", str(tmp_path.resolve()))
    references = put_session_media(scope, [{
        "media_kind": "image",
        "content_type": "image/png",
        "base64_data": "AA==",
        "source_summary": "one pixel",
    }])
    persist_attached_media(scope, references)
    clear_session_media(scope)
    cache_ref = str(references[0]["cache_ref"])
    script = (
        "import json,sys; "
        "from kazusa_ai_chatbot.dsh_tool_gateway.media import get_attached_media; "
        "value=get_attached_media(('debug','channel-1','user-1'),sys.argv[1]); "
        "print(json.dumps(value))"
    )
    environment = os.environ.copy()
    completed = subprocess.run(
        [sys.executable, "-c", script, cache_ref],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["content_type"] == "image/png"
    assert payload["base64_data"] == "AA=="
