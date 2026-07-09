# Command Discovery CLI

This tiny package exposes a command registry and a CLI entrypoint. The command
discovery owner is `src/tooling/commands.py`; `src/tooling/cli.py` consumes the
registry when building subcommands.
