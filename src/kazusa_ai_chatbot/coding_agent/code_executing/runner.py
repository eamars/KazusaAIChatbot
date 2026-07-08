"""Bounded argv subprocess runner for managed code execution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import time
from collections.abc import Sequence


@dataclass(frozen=True)
class CommandRunResult:
    """Normalized result for a completed or timed-out command.

    Args:
        stdout: Sanitized and capped standard output excerpt.
        stderr: Sanitized and capped standard error excerpt.
        returncode: Process exit code when the command completed.
        timed_out: Whether the process exceeded its runtime limit.
        duration_ms: Measured wall-clock runtime in milliseconds.
        output_truncated: Whether stdout or stderr exceeded its cap.
    """

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    duration_ms: int
    output_truncated: bool


def run_argv(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    max_stdout_chars: int,
    max_stderr_chars: int,
    scrub_roots: Sequence[Path],
) -> CommandRunResult:
    """Run one allowed argv command with timeout and output caps.

    Args:
        command: Already-validated argv list.
        cwd: Managed apply source root.
        timeout_seconds: Maximum runtime before termination.
        max_stdout_chars: Maximum public stdout excerpt characters.
        max_stderr_chars: Maximum public stderr excerpt characters.
        scrub_roots: Absolute roots to redact from public output.

    Returns:
        Public-safe normalized command result.
    """

    started_at = time.monotonic()
    execution_env = _execution_env(cwd)
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            env=execution_env,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = _elapsed_ms(started_at)
        stdout_text = _timeout_text(exc.stdout)
        stderr_text = _timeout_text(exc.stderr)
        stdout_excerpt, stdout_truncated = _public_excerpt(
            stdout_text,
            max_chars=max_stdout_chars,
            scrub_roots=scrub_roots,
            scrub_values=execution_env.values(),
        )
        stderr_excerpt, stderr_truncated = _public_excerpt(
            stderr_text,
            max_chars=max_stderr_chars,
            scrub_roots=scrub_roots,
            scrub_values=execution_env.values(),
        )
        result = CommandRunResult(
            stdout=stdout_excerpt,
            stderr=stderr_excerpt,
            returncode=None,
            timed_out=True,
            duration_ms=duration_ms,
            output_truncated=stdout_truncated or stderr_truncated,
        )
        return result
    except FileNotFoundError as exc:
        duration_ms = _elapsed_ms(started_at)
        stderr_excerpt, stderr_truncated = _public_excerpt(
            str(exc),
            max_chars=max_stderr_chars,
            scrub_roots=scrub_roots,
            scrub_values=execution_env.values(),
        )
        result = CommandRunResult(
            stdout="",
            stderr=stderr_excerpt,
            returncode=None,
            timed_out=False,
            duration_ms=duration_ms,
            output_truncated=stderr_truncated,
        )
        return result

    duration_ms = _elapsed_ms(started_at)
    stdout_excerpt, stdout_truncated = _public_excerpt(
        completed.stdout,
        max_chars=max_stdout_chars,
        scrub_roots=scrub_roots,
        scrub_values=execution_env.values(),
    )
    stderr_excerpt, stderr_truncated = _public_excerpt(
        completed.stderr,
        max_chars=max_stderr_chars,
        scrub_roots=scrub_roots,
        scrub_values=execution_env.values(),
    )
    result = CommandRunResult(
        stdout=stdout_excerpt,
        stderr=stderr_excerpt,
        returncode=completed.returncode,
        timed_out=False,
        duration_ms=duration_ms,
        output_truncated=stdout_truncated or stderr_truncated,
    )
    return result


def _execution_env(cwd: Path) -> dict[str, str]:
    """Build a minimal environment for Python verification commands."""

    env: dict[str, str] = {
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(cwd),
    }
    for key in ("SYSTEMROOT", "TEMP", "TMP"):
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    return env


def _elapsed_ms(started_at: float) -> int:
    elapsed = int((time.monotonic() - started_at) * 1000)
    return elapsed


def _timeout_text(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
        return text
    return value


def _public_excerpt(
    text: str,
    *,
    max_chars: int,
    scrub_roots: Sequence[Path],
    scrub_values: Sequence[str],
) -> tuple[str, bool]:
    sanitized_text = _redact_roots(text, scrub_roots)
    sanitized_text = _redact_values(sanitized_text, scrub_values)
    output_truncated = len(sanitized_text) > max_chars
    excerpt = sanitized_text[:max_chars]
    return excerpt, output_truncated


def _redact_roots(text: str, scrub_roots: Sequence[Path]) -> str:
    redacted_text = text
    for root in scrub_roots:
        resolved_root = root.resolve(strict=False)
        variants = {
            str(resolved_root),
            str(resolved_root).replace("\\", "/"),
        }
        for variant in variants:
            redacted_text = redacted_text.replace(variant, "[managed-workspace]")
    return redacted_text


def _redact_values(text: str, values: Sequence[str]) -> str:
    redacted_text = text
    for value in values:
        if len(value) < 4:
            continue
        redacted_text = redacted_text.replace(value, "[execution-env]")
    return redacted_text
