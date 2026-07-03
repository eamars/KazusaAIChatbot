"""Git command facade used by coding-agent modules."""

from dataclasses import dataclass
import subprocess
from collections.abc import Sequence

from kazusa_ai_chatbot.coding_agent.models import GIT_COMMAND_TIMEOUT_SECONDS


@dataclass(frozen=True)
class GitCommandResult:
    """Normalized result for a completed git command.

    Args:
        stdout: Captured standard output with replacement decoding.
        stderr: Captured standard error with replacement decoding.
        returncode: Process exit code.

    Returns:
        Immutable command result used by callers that need stdout or stderr.
    """

    stdout: str
    stderr: str
    returncode: int


class GitCommandError(RuntimeError):
    """Raised when a git command cannot complete successfully."""


def run_git_command(
    args: Sequence[str],
    *,
    cwd: str | None = None,
    timeout_seconds: int = GIT_COMMAND_TIMEOUT_SECONDS,
) -> GitCommandResult:
    """Run one git command through list arguments and bounded output capture.

    Args:
        args: Git arguments excluding the `git` executable.
        cwd: Optional working directory for the command.
        timeout_seconds: Maximum runtime before the command is aborted.

    Returns:
        Captured command result for successful exit code zero.
    """

    command = ["git", *args]

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        message = f"git executable is unavailable: {exc}"
        raise GitCommandError(message) from exc
    except subprocess.TimeoutExpired as exc:
        message = f"git command timed out after {timeout_seconds} seconds: {exc}"
        raise GitCommandError(message) from exc

    result = GitCommandResult(
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
        returncode=completed.returncode,
    )
    if result.returncode != 0:
        message = (
            f"git command failed with exit code {result.returncode}: "
            f"{result.stderr or result.stdout}"
        )
        raise GitCommandError(message)

    return result
