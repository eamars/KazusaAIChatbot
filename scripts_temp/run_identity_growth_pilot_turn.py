"""Send one normal debug-chat turn and retain its raw HTTP evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from uuid import uuid4

import httpx


DEFAULT_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "test_artifacts"
    / "character_identity_growth"
    / "longitudinal_pilot"
)


def _parser() -> argparse.ArgumentParser:
    """Build the one-turn pilot client command contract."""

    parser = argparse.ArgumentParser(
        description=(
            "Send one persistent turn through the normal public chat endpoint"
        ),
    )
    parser.add_argument("--message", required=True)
    parser.add_argument("--observation-target", required=True)
    parser.add_argument("--channel-id", required=True)
    parser.add_argument("--platform-user-id", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--character-id", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument(
        "--local-timestamp",
        help=(
            "Explicit configured-local wall-clock timestamp for accelerated "
            "test turns"
        ),
    )
    parser.add_argument(
        "--channel-type",
        choices=("private", "group"),
        default="private",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_ARTIFACT_DIRECTORY,
    )
    return parser


def _local_timestamp(explicit_timestamp: str | None) -> str:
    """Return the current local wall-clock value expected by debug chat."""

    if explicit_timestamp is not None:
        parsed = datetime.fromisoformat(explicit_timestamp)
        if parsed.tzinfo is not None:
            raise ValueError("--local-timestamp must be timezone-naive")
        return parsed.isoformat(sep=" ", timespec="microseconds")
    local_timestamp = datetime.now().astimezone().replace(
        tzinfo=None,
    ).isoformat(sep=" ", timespec="microseconds")
    return local_timestamp


def _request_payload(args: argparse.Namespace) -> dict[str, object]:
    """Build one typed public chat request without growth-state access."""

    message_token = uuid4().hex
    platform_message_id = f"identity-growth-pilot-{message_token}"
    message_envelope = {
        "body_text": args.message,
        "raw_wire_text": args.message,
        "mentions": [],
        "reply": None,
        "attachments": [],
        "addressed_to_global_user_ids": [args.character_id],
        "broadcast": False,
    }
    request_payload = {
        "platform": "debug",
        "platform_channel_id": args.channel_id,
        "channel_type": args.channel_type,
        "platform_message_id": platform_message_id,
        "platform_user_id": args.platform_user_id,
        "platform_bot_id": args.character_id,
        "display_name": args.display_name,
        "channel_name": "Identity Growth Longitudinal Pilot",
        "content_type": "text",
        "message_envelope": message_envelope,
        "local_timestamp": _local_timestamp(args.local_timestamp),
        "debug_modes": {
            "listen_only": False,
            "think_only": False,
            "no_remember": False,
        },
    }
    return request_payload


def _artifact_path(
    *,
    output_directory: Path,
    platform_message_id: str,
) -> Path:
    """Return the durable raw artifact path for one pilot turn."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_name = f"{timestamp}_{platform_message_id}.json"
    artifact_path = output_directory / artifact_name
    return artifact_path


def main() -> None:
    """Send one turn, fail on HTTP errors, and retain request/response data."""

    args = _parser().parse_args()
    request_payload = _request_payload(args)
    base_url = str(args.base_url).rstrip("/")
    with httpx.Client(base_url=base_url, timeout=None) as client:
        health_response = client.get("/health")
        health_response.raise_for_status()
        chat_response = client.post("/chat", json=request_payload)
        chat_response.raise_for_status()

    response_payload = chat_response.json()
    platform_message_id = str(request_payload["platform_message_id"])
    artifact = {
        "schema_version": "character_identity_longitudinal_pilot_turn.v1",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "observation_target": args.observation_target,
        "request": request_payload,
        "response": response_payload,
        "health_status_code": health_response.status_code,
        "chat_status_code": chat_response.status_code,
    }
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = _artifact_path(
        output_directory=output_directory,
        platform_message_id=platform_message_id,
    )
    artifact_path.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    response_messages = response_payload.get("messages", [])
    print(f"artifact={artifact_path}")
    print(
        json.dumps(
            {
                "platform_message_id": platform_message_id,
                "messages": response_messages,
                "operational_error": response_payload.get(
                    "operational_error"
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    main()
