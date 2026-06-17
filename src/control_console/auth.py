"""Local operator authentication and CSRF helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import binascii
import hashlib
import hmac
import secrets

from fastapi import HTTPException, Request, Response, status

from control_console.contracts import ControlConsoleOperator
from control_console.settings import ControlConsoleSettings


HASH_ITERATIONS = 260_000
SESSION_TTL_SECONDS = 12 * 60 * 60


@dataclass(slots=True)
class SessionRecord:
    """In-memory session metadata for one browser session."""

    operator_id: str
    csrf_token: str
    authenticated_at: datetime


class SessionStore:
    """Process-local session store for the local console."""

    def __init__(self) -> None:
        """Create an empty in-memory session store."""

        self._sessions: dict[str, SessionRecord] = {}

    def create_session(self, operator_id: str) -> tuple[str, SessionRecord]:
        """Create a session id and CSRF token for one operator."""

        session_id = secrets.token_urlsafe(32)
        record = SessionRecord(
            operator_id=operator_id,
            csrf_token=secrets.token_urlsafe(32),
            authenticated_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = record
        return session_id, record

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Return a live session record when one exists."""

        record = self._sessions.get(session_id)
        return record


def hash_operator_token(token: str) -> str:
    """Hash an operator token for `KAZUSA_CONTROL_OPERATOR_TOKEN_HASH`."""

    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        token.encode("utf-8"),
        salt,
        HASH_ITERATIONS,
    )
    salt_text = base64.urlsafe_b64encode(salt).decode("ascii")
    digest_text = base64.urlsafe_b64encode(digest).decode("ascii")
    token_hash = f"pbkdf2_sha256${HASH_ITERATIONS}${salt_text}${digest_text}"
    return token_hash


def verify_operator_token(token: str, token_hash: str) -> bool:
    """Return whether a plaintext token matches a stored hash."""

    if not token_hash:
        return False

    parts = token_hash.split("$")
    if len(parts) != 4 or parts[0] != "pbkdf2_sha256":
        return False

    try:
        iterations = int(parts[1])
        salt = base64.urlsafe_b64decode(parts[2].encode("ascii"))
        expected = base64.urlsafe_b64decode(parts[3].encode("ascii"))
    except (ValueError, TypeError, binascii.Error):
        return False
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        token.encode("utf-8"),
        salt,
        iterations,
    )
    is_match = hmac.compare_digest(digest, expected)
    return is_match


def issue_login_response(
    *,
    response: Response,
    settings: ControlConsoleSettings,
    sessions: SessionStore,
    operator_id: str,
) -> dict[str, object]:
    """Create a browser session and set the session cookie."""

    session_id, record = sessions.create_session(operator_id)
    response.set_cookie(
        settings.session_cookie_name,
        session_id,
        httponly=True,
        samesite="strict",
        secure=False,
        max_age=SESSION_TTL_SECONDS,
    )
    payload: dict[str, object] = {
        "operator": {
            "operator_id": record.operator_id,
            "authenticated_at": record.authenticated_at.isoformat(),
        },
        "csrf_token": record.csrf_token,
        "csrf_header_name": settings.csrf_header_name,
    }
    return payload


def require_operator(
    request: Request,
    *,
    settings: ControlConsoleSettings,
    sessions: SessionStore,
) -> ControlConsoleOperator:
    """Validate the current session cookie and return the operator."""

    if not settings.require_auth:
        operator = ControlConsoleOperator(
            operator_id="local_operator",
            authenticated_at=datetime.now(timezone.utc),
        )
        return operator

    record = _require_session_record(
        request=request,
        settings=settings,
        sessions=sessions,
    )
    operator = ControlConsoleOperator(
        operator_id=record.operator_id,
        authenticated_at=record.authenticated_at,
    )
    return operator


def require_csrf(
    request: Request,
    *,
    settings: ControlConsoleSettings,
    sessions: SessionStore,
) -> None:
    """Validate the browser CSRF header for state-changing API routes."""

    if not settings.require_auth:
        return

    record = _require_session_record(
        request=request,
        settings=settings,
        sessions=sessions,
    )
    supplied_token = request.headers.get(settings.csrf_header_name)
    if not supplied_token or not hmac.compare_digest(
        supplied_token,
        record.csrf_token,
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)


def session_csrf_token(
    request: Request,
    *,
    settings: ControlConsoleSettings,
    sessions: SessionStore,
) -> str:
    """Return the CSRF token for the authenticated browser session."""

    if not settings.require_auth:
        return ""

    record = _require_session_record(
        request=request,
        settings=settings,
        sessions=sessions,
    )
    return record.csrf_token


def get_session_record(
    request: Request,
    *,
    settings: ControlConsoleSettings,
    sessions: SessionStore,
) -> SessionRecord | None:
    """Return the current session record without raising on locked browsers."""

    if not settings.require_auth:
        record = SessionRecord(
            operator_id="local_operator",
            csrf_token="",
            authenticated_at=datetime.now(timezone.utc),
        )
        return record

    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        return None
    record = sessions.get_session(session_id)
    return record


def _require_session_record(
    *,
    request: Request,
    settings: ControlConsoleSettings,
    sessions: SessionStore,
) -> SessionRecord:
    """Return a live session record or raise an auth failure."""

    session_id = request.cookies.get(settings.session_cookie_name)
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    record = sessions.get_session(session_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return record
