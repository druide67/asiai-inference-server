"""Tests for ais_cli.as_subcmd — versioned sub-CLI registration contract."""

from __future__ import annotations

import argparse

import pytest

from ais_cli.as_subcmd import (
    PLUGIN_API_VERSION,
    IncompatiblePluginError,
    RegistrationContext,
    register,
    validate_asiai_compat,
)


def _fresh_parser() -> tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    parser = argparse.ArgumentParser(prog="asiai-test")
    sub = parser.add_subparsers(dest="cmd")
    return parser, sub


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------


def test_plugin_api_version_is_int_one() -> None:
    assert isinstance(PLUGIN_API_VERSION, int)
    assert PLUGIN_API_VERSION == 1


def test_validate_compat_accepts_matching_version() -> None:
    validate_asiai_compat(PLUGIN_API_VERSION)


def test_validate_compat_rejects_lower_version() -> None:
    with pytest.raises(IncompatiblePluginError, match="major versions"):
        validate_asiai_compat(0)


def test_validate_compat_rejects_higher_version() -> None:
    with pytest.raises(IncompatiblePluginError):
        validate_asiai_compat(99)


# ---------------------------------------------------------------------------
# Modern dataclass calling convention
# ---------------------------------------------------------------------------


def test_register_with_dataclass_attaches_engine_subparser() -> None:
    parser, subparsers = _fresh_parser()
    commands: dict = {}
    ctx = RegistrationContext(
        api_version=PLUGIN_API_VERSION,
        subparsers=subparsers,
        commands=commands,
        asiai_version="1.6.0",
    )

    register(ctx)

    assert "engine" in commands
    parsed = parser.parse_args(["engine", "list"])
    assert parsed.cmd == "engine"
    assert parsed.engine_cmd == "list"


def test_register_with_dataclass_kwarg() -> None:
    parser, subparsers = _fresh_parser()
    ctx = RegistrationContext(api_version=PLUGIN_API_VERSION, subparsers=subparsers, commands={})
    register(context=ctx)
    parser.parse_args(["engine", "purge", "--dry-run"])


def test_register_rejects_incompatible_context() -> None:
    _parser, subparsers = _fresh_parser()
    ctx = RegistrationContext(api_version=99, subparsers=subparsers, commands={})
    with pytest.raises(IncompatiblePluginError):
        register(ctx)


# ---------------------------------------------------------------------------
# Legacy positional calling convention
# ---------------------------------------------------------------------------


def test_register_accepts_legacy_subparsers_commands_pair() -> None:
    """Old asiai versions pass (subparsers, commands) directly."""
    _parser, subparsers = _fresh_parser()
    commands: dict = {}
    register(subparsers, commands)
    assert "engine" in commands


def test_register_with_unknown_signature_raises_typeerror() -> None:
    with pytest.raises(TypeError, match="register"):
        register(42)


# ---------------------------------------------------------------------------
# Subcommand surface — every aisctl subcommand reachable as `engine <sub>`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["engine", "list"],
        ["engine", "list", "--json"],
        ["engine", "status"],
        ["engine", "status", "ollama", "--json"],
        ["engine", "status", "ollama", "--deep", "--json"],
        ["engine", "install", "ollama", "--firewall", "lan-only", "--user", "jmn"],
        ["engine", "install", "ollama", "--dry-run", "--force"],
        ["engine", "install", "llamacpp-aux-1", "--preset", "qwen3-4b-instruct-hermes-aux-1"],
        ["engine", "list-presets", "--json"],
        ["engine", "reinstall", "ollama"],
        ["engine", "uninstall", "ollama", "--dry-run"],
        ["engine", "start", "ollama"],
        ["engine", "stop", "ollama", "--dry-run"],
        ["engine", "restart", "ollama"],
        ["engine", "disable", "ollama"],
        ["engine", "enable", "ollama", "--start"],
        ["engine", "upgrade", "ollama", "--dry-run"],
        ["engine", "unload", "ollama"],
        ["engine", "unload", "ollama", "llama3.2"],
        ["engine", "purge"],
        ["engine", "purge", "--dry-run", "--force", "--json"],
        ["engine", "repair", "--json"],
        ["engine", "bootstrap", "--install-sudoers", "--dry-run"],
    ],
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    parser, subparsers = _fresh_parser()
    register(
        RegistrationContext(api_version=PLUGIN_API_VERSION, subparsers=subparsers, commands={})
    )
    parser.parse_args(argv)  # raises SystemExit if invalid


def test_engine_subcommand_surface_matches_aisctl() -> None:
    """Parity guard: every verb `aisctl` exposes must be reachable through
    `asiai engine <verb>` too (serve/fleet/install-service stay
    aisctl-only — they are host-level daemon plumbing, not engine ops)."""
    from ais_cli.__main__ import build_parser

    aisctl_only = {"serve", "fleet", "install-service", "uninstall-service"}
    aisctl_actions = next(
        a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)
    )
    _parser, subparsers = _fresh_parser()
    register(
        RegistrationContext(api_version=PLUGIN_API_VERSION, subparsers=subparsers, commands={})
    )
    engine_parser = subparsers.choices["engine"]
    engine_actions = next(
        a for a in engine_parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    missing = set(aisctl_actions.choices) - set(engine_actions.choices) - aisctl_only
    assert not missing, f"aisctl verbs missing from `asiai engine`: {sorted(missing)}"


def test_engine_without_subcommand_is_rejected() -> None:
    parser, subparsers = _fresh_parser()
    register(
        RegistrationContext(api_version=PLUGIN_API_VERSION, subparsers=subparsers, commands={})
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["engine"])


# ---------------------------------------------------------------------------
# Lazy handler — cold-start (Copilot Q4)
# ---------------------------------------------------------------------------


def test_handlers_are_not_imported_at_register_time() -> None:
    """Importing ais_cli.commands at register time would defeat cold-start.

    We assert that argparse's set_defaults stored a thunk, not the actual
    handler function — so a tree of imports (subprocess, urllib, fcntl, ...)
    only loads when the user actually runs ``asiai engine <subcmd>``.
    """
    parser, subparsers = _fresh_parser()
    register(
        RegistrationContext(api_version=PLUGIN_API_VERSION, subparsers=subparsers, commands={})
    )
    parsed = parser.parse_args(["engine", "list", "--json"])
    # Lazy thunks have a name like "lazy_cmd_list", not the real handler name.
    assert parsed.func.__name__.startswith("lazy_")
