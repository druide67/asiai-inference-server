"""Tests for the ``aisctl`` CLI surface — argument parsing + dispatch."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from ais_cli import commands
from ais_cli.__main__ import build_parser, main
from ais_core import install_state
from ais_core.manifest import load_manifest, manifest_source_path

# ---------------------------------------------------------------------------
# parser shape
# ---------------------------------------------------------------------------


def test_parser_requires_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_accepts_each_subcommand() -> None:
    parser = build_parser()
    expected_subcommands = [
        "list",
        "status",
        "install",
        "reinstall",
        "uninstall",
        "start",
        "stop",
        "restart",
        "upgrade",
        "unload",
        "purge",
        "repair",
        "bootstrap",
    ]
    for cmd in expected_subcommands:
        # Subcommands that take an engine name need one to parse cleanly.
        argv = [cmd]
        if cmd in {
            "install",
            "reinstall",
            "uninstall",
            "start",
            "stop",
            "restart",
            "upgrade",
            "unload",
        }:
            argv.append("ollama")
        parser.parse_args(argv)


def test_version_flag_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(["--version"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "aisctl" in out


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_human_output(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "ollama" in out
    assert "lmstudio" in out
    assert "omlx" in out
    assert "turboquant" in out
    assert "llamacpp" in out
    for i in range(1, 6):
        assert f"llamacpp-aux-{i}" in out
    assert "vmlx" in out
    assert "mlx-lm" in out
    assert "rapidmlx" in out


def test_list_json_output(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["list", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert set(payload["engines"]) == {
        "ollama",
        "lmstudio",
        "omlx",
        "turboquant",
        "llamacpp",
        "llamacpp-aux-1",
        "llamacpp-aux-2",
        "llamacpp-aux-3",
        "llamacpp-aux-4",
        "llamacpp-aux-5",
        "vmlx",
        "mlx-lm",
        "rapidmlx",
    }


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def test_status_unknown_engine_exits_with_message() -> None:
    """Unknown engine surfaces a clear SystemExit with the known list."""
    with pytest.raises(SystemExit, match="unknown engine"):
        main(["status", "zzz_does_not_exist"])


def test_status_single_engine_json(capsys: pytest.CaptureFixture[str]) -> None:
    fake_state = MagicMock()
    fake_state.value = "stopped"
    with patch("ais_cli.commands.lifecycle.current_state", return_value=fake_state):
        rc = main(["status", "ollama", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload == {
        "engines": [
            {
                "engine": "ollama",
                "state": "stopped",
                "port": 11434,
                "plist": "com.asiai.ollama",
                "preset": None,
            },
        ]
    }


def test_status_all_engines_when_no_arg(capsys: pytest.CaptureFixture[str]) -> None:
    fake_state = MagicMock()
    fake_state.value = "not_installed"
    with patch("ais_cli.commands.lifecycle.current_state", return_value=fake_state):
        rc = main(["status", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    # 9 baseline engines (ollama, lmstudio, omlx, turboquant, llamacpp,
    # llamacpp-aux-1, vmlx, mlx-lm, rapidmlx) + 4 extra aux-N siblings
    # (aux-2/3/4/5) = 13
    assert len(payload["engines"]) == 13


# ---------------------------------------------------------------------------
# install / uninstall — verify orchestration calls
# ---------------------------------------------------------------------------


def test_install_invokes_lifecycle_install_with_firewall() -> None:
    fake_result = {
        "engine": "ollama",
        "binary": "/x",
        "plist": "/y",
        "anchor": "/z",
        "health_ok": True,
        "dry_run": False,
    }
    with (
        patch("ais_cli.commands.lifecycle.install", return_value=fake_result) as m,
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["install", "ollama", "--firewall", "lan-only", "--user", "jmn", "--json"])

    assert rc == 0
    kwargs = m.call_args.kwargs
    assert kwargs["enable_firewall"] is True
    assert kwargs["user"] == "jmn"
    assert kwargs["dry_run"] is False


def test_install_dry_run_returns_zero() -> None:
    fake_result = {
        "engine": "ollama",
        "binary": "/x",
        "plist": "/y",
        "anchor": None,
        "health_ok": None,
        "dry_run": True,
    }
    with (
        patch("ais_cli.commands.lifecycle.install", return_value=fake_result),
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["install", "ollama", "--dry-run", "--user", "jmn"])
    assert rc == 0


def test_install_unhealthy_returns_nonzero() -> None:
    fake_result = {
        "engine": "ollama",
        "binary": "/x",
        "plist": "/y",
        "anchor": None,
        "health_ok": False,
        "dry_run": False,
    }
    with (
        patch("ais_cli.commands.lifecycle.install", return_value=fake_result),
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["install", "ollama", "--user", "jmn"])
    assert rc == 2


def test_uninstall_passes_dry_run() -> None:
    with (
        patch("ais_cli.commands.lifecycle.uninstall", return_value={}) as m,
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["uninstall", "ollama", "--dry-run"])
    assert rc == 0
    assert m.call_args.kwargs["dry_run"] is True
    assert "keep_logs" not in m.call_args.kwargs


def test_uninstall_no_keep_logs_flag() -> None:
    """--keep-logs was removed in v0.1; ensure argparse rejects it."""
    with patch("ais_cli.commands.memory.OperationsLock"), pytest.raises(SystemExit):
        main(["uninstall", "ollama", "--keep-logs"])


# ---------------------------------------------------------------------------
# install records / reinstall (issue #6)
# ---------------------------------------------------------------------------

_PRESET = "qwen3-4b-instruct-hermes-aux-1"
_ENGINE = "llamacpp-aux-1"


def _fake_install_result(engine: str, health: bool = True) -> dict:
    return {
        "engine": engine,
        "binary": "/x",
        "plist": "/y",
        "anchor": None,
        "health_ok": health,
        "dry_run": False,
    }


def _install_with_preset() -> None:
    with (
        patch("ais_cli.commands.lifecycle.install", return_value=_fake_install_result(_ENGINE)),
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        assert main(["install", _ENGINE, "--preset", _PRESET, "--user", "jmn", "--json"]) == 0


def test_install_preset_is_recorded_and_survives_uninstall() -> None:
    _install_with_preset()
    rec = install_state.read_install(_ENGINE)
    assert rec is not None
    assert rec.preset == _PRESET

    with (
        patch("ais_cli.commands.lifecycle.uninstall", return_value={}),
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        assert main(["uninstall", _ENGINE]) == 0
    # The record outlives uninstall — that's what powers the plain-install guard.
    assert install_state.read_install(_ENGINE) is not None


def test_plain_install_over_preset_install_is_refused() -> None:
    _install_with_preset()
    with patch("ais_cli.commands.memory.OperationsLock"), pytest.raises(SystemExit) as ei:
        main(["install", _ENGINE, "--user", "jmn"])
    assert "reinstall" in str(ei.value)


def test_plain_install_with_force_overrides_and_rerecords() -> None:
    _install_with_preset()
    with (
        patch("ais_cli.commands.lifecycle.install", return_value=_fake_install_result(_ENGINE)),
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        assert main(["install", _ENGINE, "--user", "jmn", "--force"]) == 0
    rec = install_state.read_install(_ENGINE)
    assert rec is not None
    assert rec.preset is None


def test_reinstall_replays_recorded_preset() -> None:
    _install_with_preset()
    with (
        patch("ais_cli.commands.lifecycle.uninstall", return_value={}) as m_un,
        patch(
            "ais_cli.commands.lifecycle.install", return_value=_fake_install_result(_ENGINE)
        ) as m_in,
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        rc = main(["reinstall", _ENGINE, "--user", "jmn", "--json"])
    assert rc == 0
    # Acceptance (issue #6): reinstall regenerates from the SAME preset
    # manifest the original install used — hence an identical plist.
    expected = load_manifest(_ENGINE, preset=_PRESET)
    assert m_in.call_args.args[0] == expected
    assert m_un.call_args.args[0] == expected


def test_reinstall_without_record_refuses() -> None:
    with pytest.raises(SystemExit) as ei:
        main(["reinstall", _ENGINE])
    assert "no install record" in str(ei.value)


def test_reinstall_reuses_recorded_firewall_mode() -> None:
    src = manifest_source_path(_ENGINE, _PRESET)
    assert src is not None
    install_state.record_install(_ENGINE, preset=_PRESET, manifest_path=src, firewall="lan-only")
    with (
        patch("ais_cli.commands.lifecycle.uninstall", return_value={}),
        patch(
            "ais_cli.commands.lifecycle.install", return_value=_fake_install_result(_ENGINE)
        ) as m_in,
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        assert main(["reinstall", _ENGINE, "--user", "jmn"]) == 0
    assert m_in.call_args.kwargs["enable_firewall"] is True


def test_status_json_shows_recorded_preset(capsys: pytest.CaptureFixture[str]) -> None:
    _install_with_preset()
    capsys.readouterr()  # drop the install output
    with patch(
        "ais_cli.commands.lifecycle.current_state",
        return_value=commands.lifecycle.EngineState.STOPPED,
    ):
        assert main(["status", _ENGINE, "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engines"][0]["preset"] == _PRESET


# ---------------------------------------------------------------------------
# unload
# ---------------------------------------------------------------------------


def test_unload_with_model_calls_driver_unload() -> None:
    fake_outcome = MagicMock(success=True)
    fake_outcome.engine = "ollama"
    fake_outcome.method = "api"
    fake_driver = MagicMock()
    fake_driver.unload.return_value = fake_outcome

    factory = MagicMock(return_value=fake_driver)
    with (
        patch.dict("ais_cli.commands.DRIVER_FACTORIES", {"ollama": factory}, clear=False),
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["unload", "ollama", "llama3.2"])

    fake_driver.unload.assert_called_once_with("llama3.2")
    assert rc == 0


def test_unload_without_model_passes_none() -> None:
    """Omitting the model arg means 'restart everything'."""
    fake_driver = MagicMock()
    fake_driver.unload.return_value = MagicMock(success=True)

    factory = MagicMock(return_value=fake_driver)
    with (
        patch.dict("ais_cli.commands.DRIVER_FACTORIES", {"ollama": factory}, clear=False),
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        main(["unload", "ollama"])

    fake_driver.unload.assert_called_once_with(None)


def test_unload_failure_returns_nonzero() -> None:
    fake_outcome = MagicMock(success=False)
    fake_driver = MagicMock()
    fake_driver.unload.return_value = fake_outcome

    factory = MagicMock(return_value=fake_driver)
    with (
        patch.dict("ais_cli.commands.DRIVER_FACTORIES", {"ollama": factory}, clear=False),
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["unload", "ollama", "foo"])

    assert rc == 2


# ---------------------------------------------------------------------------
# purge / repair
# ---------------------------------------------------------------------------


def test_purge_dry_run_emits_report() -> None:
    fake_report = MagicMock()
    fake_report.before = MagicMock()
    fake_report.after = MagicMock()
    fake_report.pressure_after = "normal"
    fake_report.elapsed_s = 0.0

    with (
        patch("ais_cli.commands.memory.purge_memory", return_value=fake_report) as m,
        patch("ais_cli.commands.memory.OperationsLock") as mock_lock,
    ):
        mock_lock.return_value.__enter__.return_value = mock_lock.return_value
        rc = main(["purge", "--dry-run"])

    assert rc == 0
    assert m.call_args.kwargs == {"dry_run": True}


def test_repair_runs_without_lock() -> None:
    """repair must NOT take the operations lock — that's the whole point."""
    with (
        patch("ais_cli.commands.memory.repair") as m_repair,
        patch("ais_cli.commands.memory.OperationsLock") as m_lock,
    ):
        m_repair.return_value = MagicMock(stale_lock_cleared=False, orphan_plists=[])
        rc = main(["repair", "--dry-run", "--json"])

    assert rc == 0
    assert m_lock.call_count == 0


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_no_flag_prints_help() -> None:
    rc = main(["bootstrap"])
    assert rc == 0


def test_bootstrap_dry_run_prints_content(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["bootstrap", "--install-sudoers", "--dry-run"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "/usr/sbin/purge" in out
    assert "com.asiai." in out


def test_bootstrap_install_invokes_install_sudoers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("ais_cli.commands.sudoers.validate_content"),
        patch(
            "ais_cli.commands.sudoers.install_sudoers",
            return_value="/etc/sudoers.d/asiai-inference",
        ) as m_install,
    ):
        rc = main(["bootstrap", "--install-sudoers"])

    assert rc == 0
    m_install.assert_called_once()
    out = capsys.readouterr().out
    assert "/etc/sudoers.d/asiai-inference" in out


def test_bootstrap_install_returns_2_on_validation_failure() -> None:
    with patch(
        "ais_cli.commands.sudoers.validate_content",
        side_effect=commands.sudoers.SudoersError("syntax error"),
    ):
        rc = main(["bootstrap", "--install-sudoers"])
    assert rc == 2
