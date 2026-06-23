"""Tests for ais_cli.commands helpers.

Covers _resolve_install_user — the D4 (story 2.2) fail-closed account resolution shared by
cmd_install and cmd_reinstall: --user wins, then $SUDO_USER, then $USER, NEVER a "root"
fallback (an engine daemon must run under a non-root account, I2)."""

from __future__ import annotations

import pytest

from ais_cli.commands import _resolve_install_user


class TestResolveInstallUser:
    def test_explicit_user_wins(self, monkeypatch):
        monkeypatch.setenv("SUDO_USER", "bob")
        monkeypatch.setenv("USER", "carol")
        assert _resolve_install_user("alice") == "alice"

    def test_sudo_user_preferred_over_user(self, monkeypatch):
        monkeypatch.setenv("SUDO_USER", "bob")
        monkeypatch.setenv("USER", "carol")
        assert _resolve_install_user(None) == "bob"

    def test_falls_back_to_user(self, monkeypatch):
        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setenv("USER", "carol")
        assert _resolve_install_user(None) == "carol"

    def test_refuses_explicit_root(self, monkeypatch):
        monkeypatch.setenv("SUDO_USER", "bob")  # must not rescue an explicit --user root
        with pytest.raises(SystemExit, match="refusing to install as root"):
            _resolve_install_user("root")

    def test_refuses_root_from_env(self, monkeypatch):
        """run as root via bare launchd/cron: no SUDO_USER, USER=root -> fail closed."""
        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.setenv("USER", "root")
        with pytest.raises(SystemExit, match="refusing to install as root"):
            _resolve_install_user(None)

    def test_refuses_when_unresolvable(self, monkeypatch):
        monkeypatch.delenv("SUDO_USER", raising=False)
        monkeypatch.delenv("USER", raising=False)
        with pytest.raises(SystemExit, match="refusing to install as root"):
            _resolve_install_user(None)
