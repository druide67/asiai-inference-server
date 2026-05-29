"""Tests for the upgrade whitelist + the native ``aisctl upgrade`` command."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import patch

import pytest

from ais_cli.__main__ import main
from ais_core.upgrade import UPGRADE_FORMULAS, upgrade_argv

# --- whitelist -------------------------------------------------------------


def test_upgrade_argv_whitelisted():
    argv = upgrade_argv("llamacpp")
    assert argv[-2:] == ["upgrade", "llama.cpp"]


def test_upgrade_argv_rejects_unlisted():
    with pytest.raises(ValueError, match="not whitelisted"):
        upgrade_argv("coreutils")


def test_whitelist_covers_expected_engines():
    assert set(UPGRADE_FORMULAS) >= {"ollama", "llamacpp", "lmstudio", "rapidmlx", "turboquant"}


# --- cmd_upgrade -----------------------------------------------------------


def _completed(returncode=0, stdout="upgraded", stderr=""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


def test_upgrade_dry_run_does_not_execute(capsys):
    with patch("ais_cli.commands.subprocess.run") as run:
        rc = main(["upgrade", "ollama", "--dry-run", "--json"])
    assert rc == 0
    run.assert_not_called()
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["argv"][-2:] == ["upgrade", "ollama"]


def test_upgrade_unlisted_engine_returns_2(capsys):
    # mlx-lm has a manifest but is NOT in the upgrade whitelist.
    with patch("ais_cli.commands.subprocess.run") as run:
        rc = main(["upgrade", "mlx-lm", "--json"])
    assert rc == 2
    run.assert_not_called()
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert "not whitelisted" in out["error"]


def test_upgrade_success_without_restart_hints(capsys):
    with (
        patch("ais_cli.commands.subprocess.run", return_value=_completed(0)),
        patch("ais_cli.commands.memory.OperationsLock") as lock,
        patch("ais_cli.commands.lifecycle.restart") as restart,
    ):
        lock.return_value.__enter__.return_value = lock.return_value
        rc = main(["upgrade", "ollama", "--json"])
    assert rc == 0
    restart.assert_not_called()
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert "hint" in out  # reminds the operator to restart


def test_upgrade_with_restart_reconciles(capsys):
    with (
        patch("ais_cli.commands.subprocess.run", return_value=_completed(0)),
        patch("ais_cli.commands.memory.OperationsLock") as lock,
        patch("ais_cli.commands.lifecycle.restart") as restart,
        patch("ais_cli.commands.lifecycle.wait_for_health", return_value=True),
    ):
        lock.return_value.__enter__.return_value = lock.return_value
        rc = main(["upgrade", "ollama", "--restart", "--json"])
    assert rc == 0
    restart.assert_called_once()
    out = json.loads(capsys.readouterr().out)
    assert out["restarted"] is True


def test_upgrade_brew_failure_returns_2(capsys):
    with (
        patch("ais_cli.commands.subprocess.run", return_value=_completed(1, stderr="brew boom")),
        patch("ais_cli.commands.memory.OperationsLock") as lock,
        patch("ais_cli.commands.lifecycle.restart") as restart,
    ):
        lock.return_value.__enter__.return_value = lock.return_value
        rc = main(["upgrade", "ollama", "--json"])
    assert rc == 2
    restart.assert_not_called()
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is False
    assert out["exit_code"] == 1


def test_serve_upgrade_uses_shared_whitelist():
    # The loopback command server must build the same argv as the native CLI.
    from ais_cli.serve import _build_argv

    argv = _build_argv("upgrade", {"engine": "llamacpp"})
    assert argv == upgrade_argv("llamacpp")
