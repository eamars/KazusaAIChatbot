"""Command registry for the fixture CLI."""

from __future__ import annotations


def hello_command(argv: list[str]) -> str:
    """Return the greeting command output."""

    del argv
    return "hello"


def version_command(argv: list[str]) -> str:
    """Return the version command output."""

    del argv
    return "tooling 1.0"


COMMANDS = {
    "hello": hello_command,
    "version": version_command,
}


def discover_commands() -> dict[str, object]:
    """Return the available command handlers keyed by command name."""

    commands = dict(COMMANDS)
    return commands
