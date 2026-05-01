"""Smoke tests — ensure the skeleton is importable and the CLI runs."""

from __future__ import annotations


def test_ais_core_import() -> None:
    import ais_core

    assert ais_core.__version__ == "0.0.1"


def test_ais_engines_import() -> None:
    import ais_engines  # noqa: F401


def test_ais_cli_import() -> None:
    import ais_cli  # noqa: F401


def test_aisctl_main_with_no_args_exits_with_usage() -> None:
    """Bare ``aisctl`` (no subcommand) must exit non-zero with usage to stderr."""
    import pytest

    from ais_cli.__main__ import main

    with pytest.raises(SystemExit) as ei:
        main([])
    assert ei.value.code == 2  # argparse missing-arg exit code


def test_aisctl_list_subcommand_returns_zero() -> None:
    from ais_cli.__main__ import main

    rc = main(["list"])
    assert rc == 0


def test_as_subcmd_register_signature() -> None:
    """The plugin registration entry point must be a callable taking 2 args."""
    import argparse
    import inspect

    from ais_cli.as_subcmd import register

    sig = inspect.signature(register)
    assert len(sig.parameters) == 2

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    commands: dict[str, object] = {}
    register(subparsers, commands)
    assert "engine" in commands
