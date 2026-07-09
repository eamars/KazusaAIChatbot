"""Command-line entrypoint for the fixture CLI."""

from __future__ import annotations

import argparse

from tooling.commands import discover_commands


def build_parser() -> argparse.ArgumentParser:
    """Build the parser from the command registry."""

    parser = argparse.ArgumentParser(prog="tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in sorted(discover_commands()):
        subparsers.add_parser(command_name)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a registered fixture command."""

    parser = build_parser()
    args, command_args = parser.parse_known_args(argv)
    commands = discover_commands()
    output = commands[args.command](command_args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
