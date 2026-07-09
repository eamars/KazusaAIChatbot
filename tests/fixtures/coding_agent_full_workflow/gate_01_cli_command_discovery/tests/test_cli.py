from tooling.commands import discover_commands


def test_discovers_hello_and_version_commands() -> None:
    commands = discover_commands()

    assert sorted(commands) == ["hello", "version"]
