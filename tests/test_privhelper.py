"""Tests for ais_core.privhelper — the thin invoker of the root-owned helper.

No real ``sudo`` is run: ``subprocess.run`` is mocked. These assert the argv shape
(invariant #3: the installed root:wheel path, prefixed by ``sudo``), the exit-code → reason
mapping, dry-run, and timeout handling.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from ais_core import privhelper, sudoers


def _proc(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    return MagicMock(returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_invokes_sudo_installed_helper_path() -> None:
    with patch("ais_core.privhelper.subprocess.run", return_value=_proc()) as mock_run:
        privhelper.run("start-daemon", "--label", "com.asiai.llamacpp-aux-1")

    argv = mock_run.call_args.args[0]
    # invariant #3: the EXECUTED helper is the installed root:wheel copy, not site-packages.
    assert argv == [
        "sudo",
        sudoers.PRIVILEGED_HELPER_PATH,
        "start-daemon",
        "--label",
        "com.asiai.llamacpp-aux-1",
    ]


def test_helper_path_is_the_installed_copy() -> None:
    assert privhelper.helper_path() == sudoers.PRIVILEGED_HELPER_PATH
    assert privhelper.helper_path().startswith("/Library/PrivilegedHelperTools/")


def test_run_dry_run_does_not_execute(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("ais_core.privhelper.subprocess.run") as mock_run:
        result = privhelper.run("purge", dry_run=True)
    mock_run.assert_not_called()
    assert result is None
    assert "purge" in capsys.readouterr().out


def test_run_raises_on_refused_with_reason() -> None:
    with (
        patch("ais_core.privhelper.subprocess.run", return_value=_proc(returncode=2)),
        pytest.raises(privhelper.PrivHelperError, match="refused"),
    ):
        privhelper.run("install-daemon", "--label", "com.asiai.web")


def test_run_raises_on_internal_error_with_reason() -> None:
    with (
        patch("ais_core.privhelper.subprocess.run", return_value=_proc(returncode=1)),
        pytest.raises(privhelper.PrivHelperError, match="internal error"),
    ):
        privhelper.run("start-daemon", "--label", "com.asiai.x")


def test_run_check_false_returns_failed_process() -> None:
    with patch("ais_core.privhelper.subprocess.run", return_value=_proc(returncode=2)):
        proc = privhelper.run("uninstall-daemon", "--label", "com.asiai.x", check=False)
    assert proc is not None
    assert proc.returncode == 2  # best-effort: caller inspects, no raise


def test_run_timeout_wrapped() -> None:
    with (
        patch(
            "ais_core.privhelper.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="asiai-priv", timeout=30),
        ),
        pytest.raises(privhelper.PrivHelperError, match="timed out"),
    ):
        privhelper.run("purge", timeout=30)


def test_run_success_returns_process() -> None:
    with patch("ais_core.privhelper.subprocess.run", return_value=_proc(returncode=0)):
        proc = privhelper.run("enable-daemon", "--label", "com.asiai.x")
    assert proc is not None
    assert proc.returncode == 0
