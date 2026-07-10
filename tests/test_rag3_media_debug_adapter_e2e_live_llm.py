"""Opt-in real debug-adapter test for precise recent-image follow-ups."""

from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

import httpx
from PIL import Image, ImageDraw
import pytest

from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from tests.llm_trace import write_llm_trace

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVICE_STARTUP_TIMEOUT_SECONDS = 90
CHAT_REQUEST_TIMEOUT_SECONDS = 130
HEALTH_POLL_INTERVAL_SECONDS = 1
SCENE_WIDTH = 960
SCENE_HEIGHT = 640

pytestmark = [pytest.mark.asyncio, pytest.mark.live_db, pytest.mark.live_llm]


@pytest.mark.skipif(
    os.environ.get('KAZUSA_RUN_REAL_DEBUG_MEDIA_E2E') != '1',
    reason=(
        'real debug-media E2E starts local brain and debug-adapter processes; '
        'set KAZUSA_RUN_REAL_DEBUG_MEDIA_E2E=1 to run it'
    ),
)
async def test_debug_adapter_answers_precise_recent_image_followups(
    tmp_path: Path,
    unused_tcp_port_factory,
) -> None:
    """Observe recent-image grounding through the real debug adapter.

    The image has simple, unambiguous ground truth while the follow-ups require
    relative position and exact counting that a generic visual summary may omit.
    The assertions establish transport, telemetry, and privacy contract gates;
    the recorded trace supports qualitative review of visual grounding.
    """

    brain_port = unused_tcp_port_factory()
    adapter_port = unused_tcp_port_factory()
    brain_url = f'http://127.0.0.1:{brain_port}'
    adapter_url = f'http://127.0.0.1:{adapter_port}'
    test_scope = uuid4().hex
    image_base64 = _precision_scene_png_base64()
    brain_log_path = tmp_path / 'brain.log'
    adapter_log_path = tmp_path / 'debug_adapter.log'
    process_env = os.environ.copy()
    source_path = str(REPOSITORY_ROOT / 'src')
    existing_python_path = process_env.get('PYTHONPATH', '')
    process_env['PYTHONPATH'] = os.pathsep.join(
        path for path in (source_path, existing_python_path) if path
    )

    with (
        brain_log_path.open('w', encoding='utf-8') as brain_log,
        adapter_log_path.open('w', encoding='utf-8') as adapter_log,
    ):
        brain_process = _start_service_process(
            command=[
                sys.executable,
                '-m',
                'uvicorn',
                'kazusa_ai_chatbot.service:app',
                '--host',
                '127.0.0.1',
                '--port',
                str(brain_port),
            ],
            environment=process_env,
            log_handle=brain_log,
        )
        adapter_process: subprocess.Popen[str] | None = None
        try:
            await _wait_for_health(
                url=f'{brain_url}/health',
                process=brain_process,
            )
            adapter_process = _start_service_process(
                command=[
                    sys.executable,
                    '-m',
                    'adapters.debug_adapter',
                    '--brain-url',
                    brain_url,
                    '--port',
                    str(adapter_port),
                ],
                environment=process_env,
                log_handle=adapter_log,
            )
            await _wait_for_health(
                url=f'{adapter_url}/api/health',
                process=adapter_process,
            )

            async with httpx.AsyncClient(
                timeout=CHAT_REQUEST_TIMEOUT_SECONDS,
            ) as client:
                upload_response = await client.post(
                    f'{adapter_url}/api/chat',
                    json=_debug_chat_payload(
                        channel_id=f'debug-rag3-media-{test_scope}',
                        text=(
                            'Please keep this image available for three precise '
                            'follow-up questions.'
                        ),
                        attachments=[{
                            'media_type': 'image/png',
                            'base64_data': image_base64,
                            'description': 'A geometric scene with several shapes.',
                        }],
                    ),
                )
                upload_response.raise_for_status()
                upload_payload = upload_response.json()
                upload_graph = await _latest_cognition_graph(client, brain_url)

                left_response = await client.post(
                    f'{adapter_url}/api/chat',
                    json=_debug_chat_payload(
                        channel_id=f'debug-rag3-media-{test_scope}',
                        text=(
                            'In the image from my immediately previous turn, '
                            'which shape is immediately left of the white circle?'
                        ),
                        attachments=[],
                    ),
                )
                left_response.raise_for_status()
                left_payload = left_response.json()
                left_graph = await _latest_cognition_graph(client, brain_url)

                below_response = await client.post(
                    f'{adapter_url}/api/chat',
                    json=_debug_chat_payload(
                        channel_id=f'debug-rag3-media-{test_scope}',
                        text=(
                            'Still using that same image, which colored shape sits '
                            'directly below the white circle?'
                        ),
                        attachments=[],
                    ),
                )
                below_response.raise_for_status()
                below_payload = below_response.json()
                below_graph = await _latest_cognition_graph(client, brain_url)

                count_response = await client.post(
                    f'{adapter_url}/api/chat',
                    json=_debug_chat_payload(
                        channel_id=f'debug-rag3-media-{test_scope}',
                        text=(
                            'One final check on that image: how many red dots are '
                            'inside the outlined square in the upper-left?'
                        ),
                        attachments=[],
                    ),
                )
                count_response.raise_for_status()
                count_payload = count_response.json()
                count_graph = await _latest_cognition_graph(client, brain_url)

            trace_path = write_llm_trace(
                'rag3_media_debug_adapter_e2e_live_llm',
                'recent_image_precision_followups',
                {
                    'transport': {
                        'debug_adapter_endpoint': '/api/chat',
                        'brain_endpoint': '/chat',
                        'debug_mode': {'no_remember': True},
                    },
                    'image_fixture': {
                        'kind': 'synthetic_geometric_scene',
                        'ground_truth': {
                            'left_of_white_circle': 'orange triangle',
                            'below_white_circle': 'teal square',
                            'upper_left_red_dot_count': 3,
                        },
                    },
                    'turns': [
                        {
                            'turn': 1,
                            'purpose': 'image upload',
                            'response': upload_payload,
                            'cognition_graph': upload_graph,
                        },
                        {
                            'turn': 2,
                            'purpose': 'recent-image relative-position check',
                            'response': left_payload,
                            'cognition_graph': left_graph,
                        },
                        {
                            'turn': 3,
                            'purpose': 'recent-image vertical-relation check',
                            'response': below_payload,
                            'cognition_graph': below_graph,
                        },
                        {
                            'turn': 4,
                            'purpose': 'recent-image exact-count check',
                            'response': count_payload,
                            'cognition_graph': count_graph,
                        },
                    ],
                    'expected_review_points': [
                        'The follow-up turns use the cached prior image, not a new attachment.',
                        'The final wording is grounded in the three fixture facts.',
                        'The operator telemetry contains no raw base64 payload.',
                    ],
                    'process_logs': {
                        'brain': str(brain_log_path),
                        'debug_adapter': str(adapter_log_path),
                    },
                },
            )
        finally:
            if adapter_process is not None:
                _stop_service_process(adapter_process)
            _stop_service_process(brain_process)

    assert _response_messages(left_payload)
    assert _response_messages(below_payload)
    assert _response_messages(count_payload)
    assert upload_graph is not None
    assert left_graph is not None
    assert below_graph is not None
    assert count_graph is not None
    serialized_graphs = json.dumps(
        [upload_graph, left_graph, below_graph, count_graph],
        ensure_ascii=False,
    )
    assert image_base64 not in serialized_graphs
    assert trace_path.exists()


def _precision_scene_png_base64() -> str:
    """Build the exact visual fixture used by the recent-image questions.

    Returns:
        Base64-encoded PNG bytes for a scene with known shape relations and an
        exact red-dot count.
    """

    image = Image.new('RGB', (SCENE_WIDTH, SCENE_HEIGHT), '#172033')
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 250, 250), outline='#d9e2f2', width=6)
    for center in ((90, 90), (150, 90), (120, 155)):
        x, y = center
        draw.ellipse((x - 18, y - 18, x + 18, y + 18), fill='#e84855')
    draw.polygon([(290, 240), (380, 145), (380, 335)], fill='#f28e2b')
    draw.ellipse((425, 185, 535, 295), fill='#f5f7fa')
    draw.rectangle((425, 405, 535, 515), fill='#1bb3a9')
    draw.polygon([(700, 420), (765, 485), (700, 550), (635, 485)], fill='#f6c945')
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    encoded_image = base64.b64encode(image_bytes.getvalue()).decode('ascii')
    return encoded_image


def _debug_chat_payload(
    *,
    channel_id: str,
    text: str,
    attachments: list[dict[str, str]],
) -> dict[str, object]:
    """Build one real debug-adapter request with a typed message envelope.

    Args:
        channel_id: Stable channel scope for the complete test conversation.
        text: Prompt-facing user message.
        attachments: Normalized image attachment rows for this turn.

    Returns:
        Debug adapter request body that the adapter forwards unchanged to the
        brain service.
    """

    payload = {
        'platform': 'debug',
        'platform_channel_id': channel_id,
        'channel_type': 'private',
        'platform_user_id': 'debug-rag3-media-e2e-user',
        'platform_bot_id': 'debug-rag3-media-e2e-bot',
        'display_name': 'Debug Media E2E User',
        'channel_name': 'RAG3 media E2E',
        'content_type': 'mixed' if attachments else 'text',
        'message_envelope': {
            'body_text': text,
            'raw_wire_text': text,
            'mentions': [],
            'attachments': attachments,
            'addressed_to_global_user_ids': [CHARACTER_GLOBAL_USER_ID],
            'broadcast': False,
        },
        'debug_modes': {
            'listen_only': False,
            'think_only': False,
            'no_remember': True,
        },
    }
    return payload


async def _latest_cognition_graph(
    client: httpx.AsyncClient,
    brain_url: str,
) -> dict[str, object] | None:
    """Read the bounded operator snapshot after one completed debug turn.

    Args:
        client: HTTP client shared by the E2E conversation.
        brain_url: Running brain service base URL.

    Returns:
        Latest bounded cognition graph supplied by the brain service.
    """

    response = await client.get(f'{brain_url}/ops/latest-cognition-graph')
    response.raise_for_status()
    payload = response.json()
    graph = payload['cognition_graph']
    return graph


async def _wait_for_health(
    *,
    url: str,
    process: subprocess.Popen[str],
) -> None:
    """Wait for one local service health endpoint or surface its startup error.

    Args:
        url: Health endpoint belonging to the local process.
        process: Process expected to serve the endpoint.

    Returns:
        None after a successful health response.
    """

    deadline = time.monotonic() + SERVICE_STARTUP_TIMEOUT_SECONDS
    last_error = ''
    async with httpx.AsyncClient(timeout=HEALTH_POLL_INTERVAL_SECONDS) as client:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f'local service exited before health check: return_code={return_code}'
                )
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                last_error = str(exc)
                await asyncio.sleep(HEALTH_POLL_INTERVAL_SECONDS)
                continue
            return
    raise TimeoutError(f'health endpoint did not become ready: {url}; last_error={last_error}')


def _start_service_process(
    *,
    command: list[str],
    environment: dict[str, str],
    log_handle,
) -> subprocess.Popen[str]:
    """Start one local E2E service with captured stdout and stderr.

    Args:
        command: Python command line for the service process.
        environment: Inherited environment with the source package path added.
        log_handle: UTF-8 file handle receiving process output.

    Returns:
        Running subprocess handle for later lifecycle cleanup.
    """

    process = subprocess.Popen(
        command,
        cwd=REPOSITORY_ROOT,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return process


def _stop_service_process(process: subprocess.Popen[str]) -> None:
    """Stop one service process started by this test without leaving it running."""

    if process.poll() is None:
        process.terminate()
        process.wait(timeout=SERVICE_STARTUP_TIMEOUT_SECONDS)


def _response_messages(payload: dict[str, object]) -> list[str]:
    """Read non-empty debug response messages for structural E2E assertions."""

    raw_messages = payload['messages']
    assert isinstance(raw_messages, list)
    messages = [message for message in raw_messages if isinstance(message, str)]
    assert all(message.strip() for message in messages)
    return messages
