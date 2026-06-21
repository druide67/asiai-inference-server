"""Tests for the ``aisctl`` CLI surface — argument parsing + dispatch."""

from __future__ import annotations

import json
from pathlib import Path
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
        "disable",
        "enable",
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
            "disable",
            "enable",
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
    with patch("ais_cli.commands.lifecycle.probe_state", return_value=(fake_state, None)):
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
    with patch("ais_cli.commands.lifecycle.probe_state", return_value=(fake_state, None)):
        rc = main(["status", "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    # 9 baseline engines (ollama, lmstudio, omlx, turboquant, llamacpp,
    # llamacpp-aux-1, vmlx, mlx-lm, rapidmlx) + 4 extra aux-N siblings
    # (aux-2/3/4/5) = 13
    assert len(payload["engines"]) == 13


def test_status_deep_surfaces_degraded_state_and_gen_verdict(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`status --deep` exposes the gen probe: a GPU-OOM zombie shows up as
    state=degraded + gen=zombie in the JSON payload."""
    from ais_core.lifecycle import EngineState, GenVerdict

    with patch(
        "ais_cli.commands.lifecycle.probe_state",
        return_value=(EngineState.DEGRADED, GenVerdict.ZOMBIE),
    ) as mock_probe:
        rc = main(["status", "ollama", "--deep", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    row = payload["engines"][0]
    assert row["state"] == "degraded"
    assert row["gen"] == "zombie"
    assert mock_probe.call_args.kwargs == {"deep": True}


def test_status_shallow_has_no_gen_key(capsys: pytest.CaptureFixture[str]) -> None:
    fake_state = MagicMock()
    fake_state.value = "running"
    with patch("ais_cli.commands.lifecycle.probe_state", return_value=(fake_state, None)):
        rc = main(["status", "ollama", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    row = json.loads(out)["engines"][0]
    assert "gen" not in row


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
# disable / enable (issue #8)
# ---------------------------------------------------------------------------


def test_disable_dispatches_to_lifecycle() -> None:
    with (
        patch(
            "ais_cli.commands.lifecycle.disable",
            return_value={"engine": "ollama", "disabled": True, "dry_run": False},
        ) as m,
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        rc = main(["disable", "ollama", "--json"])
    assert rc == 0
    assert m.call_args.kwargs["dry_run"] is False


def test_enable_without_start_returns_zero_without_health_wait() -> None:
    with (
        patch(
            "ais_cli.commands.lifecycle.enable",
            return_value={"engine": "ollama", "enabled": True, "started": False},
        ) as m,
        patch("ais_cli.commands.lifecycle.wait_for_health") as m_health,
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        rc = main(["enable", "ollama"])
    assert rc == 0
    assert m.call_args.kwargs["start_now"] is False
    m_health.assert_not_called()


def test_enable_with_start_waits_for_health() -> None:
    with (
        patch(
            "ais_cli.commands.lifecycle.enable",
            return_value={"engine": "ollama", "enabled": True, "started": True},
        ),
        patch("ais_cli.commands.lifecycle.wait_for_health", return_value=True),
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        assert main(["enable", "ollama", "--start"]) == 0


def test_enable_with_start_unhealthy_returns_2() -> None:
    with (
        patch(
            "ais_cli.commands.lifecycle.enable",
            return_value={"engine": "ollama", "enabled": True, "started": True},
        ),
        patch("ais_cli.commands.lifecycle.wait_for_health", return_value=False),
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        assert main(["enable", "ollama", "--start"]) == 2


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


def test_reinstall_replays_recorded_preset(capsys: pytest.CaptureFixture[str]) -> None:
    _install_with_preset()
    capsys.readouterr()  # drop the install output
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
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_changed_since_install"] is False


def test_reinstall_flags_manifest_drift(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A record whose digest no longer matches the manifest file on disk
    must surface manifest_changed_since_install=true."""
    stale = tmp_path / "stale-manifest.toml"
    stale.write_text("# content that does not match the bundled preset\n")
    install_state.record_install(_ENGINE, preset=_PRESET, manifest_path=stale, firewall="none")
    with (
        patch("ais_cli.commands.lifecycle.uninstall", return_value={}),
        patch("ais_cli.commands.lifecycle.install", return_value=_fake_install_result(_ENGINE)),
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        rc = main(["reinstall", _ENGINE, "--user", "jmn", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["manifest_changed_since_install"] is True


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
        "ais_cli.commands.lifecycle.probe_state",
        return_value=(commands.lifecycle.EngineState.STOPPED, None),
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


def test_family_factory_uses_manifest_when_given() -> None:
    """`aisctl unload llamacpp-aux-N` used to crash in FileNotFoundError:
    the family factory bound the manifest positional to its *name*
    parameter. The factory must honor the standard contract."""
    m = load_manifest("llamacpp-aux-1")
    driver = commands._driver_for(m)
    assert driver.manifest is m


def test_start_dry_run_does_not_touch_launchctl(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("ais_core.lifecycle.subprocess.run") as mock_run:
        rc = main(["start", "ollama", "--dry-run", "--json"])
    mock_run.assert_not_called()
    assert rc == 0
    out = capsys.readouterr().out  # "[dry-run] would start ..." line, then the JSON
    assert json.loads(out[out.index("{") :])["dry_run"] is True


def test_restart_dry_run_does_not_touch_launchctl(capsys: pytest.CaptureFixture[str]) -> None:
    with (
        patch("ais_core.lifecycle.subprocess.run") as mock_run,
        patch("ais_cli.commands.memory.OperationsLock"),
    ):
        rc = main(["restart", "ollama", "--dry-run", "--json"])
    mock_run.assert_not_called()
    assert rc == 0
    out = capsys.readouterr().out
    assert json.loads(out[out.index("{") :])["dry_run"] is True


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
    assert "NOPASSWD: /Library/PrivilegedHelperTools/asiai-priv" in out  # helper-only
    assert "com.asiai." not in out  # the wildcard surface is gone


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


def test_bootstrap_full_install_runs_in_strict_order(capsys: pytest.CaptureFixture[str]) -> None:
    """--install = strict order: I0 fleet chain check -> install helper -> install sudoers."""
    order: list[str] = []

    def _i0() -> None:
        order.append("i0")

    def _helper() -> str:
        order.append("helper")
        return "/Library/PrivilegedHelperTools/asiai-priv"

    def _sud() -> str:
        order.append("sud")
        return "/etc/sudoers.d/asiai-inference"

    with (
        patch("ais_cli.commands.bootstrap.assert_fleet_chain_locked", side_effect=_i0),
        patch("ais_cli.commands.bootstrap.install_helper", side_effect=_helper),
        patch("ais_cli.commands.sudoers.install_sudoers", side_effect=_sud),
    ):
        rc = main(["bootstrap", "--install"])
    assert rc == 0
    assert order == ["i0", "helper", "sud"]  # I0 before any write, helper before sudoers
    assert "/Library/PrivilegedHelperTools/asiai-priv" in capsys.readouterr().out


def test_bootstrap_full_install_dry_run_previews_both(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with (
        patch("ais_cli.commands.bootstrap.assert_fleet_chain_locked") as m_i0,
        patch("ais_cli.commands.bootstrap.install_helper") as m_helper,
        patch("ais_cli.commands.sudoers.install_sudoers") as m_sud,
    ):
        rc = main(["bootstrap", "--install", "--dry-run"])
    assert rc == 0
    m_i0.assert_not_called()  # dry-run previews; no live I0 walk
    m_helper.assert_called_once_with(dry_run=True)
    m_sud.assert_called_once_with(dry_run=True)


def test_bootstrap_full_install_aborts_on_i0_failure() -> None:
    """I0 gate: a locked-chain failure stops BEFORE any helper/sudoers write."""
    with (
        patch(
            "ais_cli.commands.bootstrap.assert_fleet_chain_locked",
            side_effect=commands.bootstrap.BootstrapError("chain not locked"),
        ),
        patch("ais_cli.commands.bootstrap.install_helper") as m_helper,
        patch("ais_cli.commands.sudoers.install_sudoers") as m_sud,
    ):
        rc = main(["bootstrap", "--install"])
    assert rc == 2
    m_helper.assert_not_called()
    m_sud.assert_not_called()
