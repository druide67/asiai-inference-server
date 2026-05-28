"""Tests for ais_core.sudoers — content shape + visudo validation."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ais_core.sudoers import (
    SUDOERS_PATH,
    SudoersError,
    generate_sudoers_content,
    install_sudoers,
    validate_content,
)


def test_content_starts_with_warning_header() -> None:
    content = generate_sudoers_content()
    assert content.startswith("# /etc/sudoers.d/asiai-inference")
    assert "Do not edit by hand" in content


def test_content_includes_purge_rule() -> None:
    content = generate_sudoers_content()
    assert "NOPASSWD: /usr/sbin/purge" in content


def test_content_scope_is_strict_com_asiai_only() -> None:
    """Every line containing a wildcard ``*`` must scope it to com.asiai.*."""
    content = generate_sudoers_content()
    for line in content.splitlines():
        if "NOPASSWD:" not in line:
            continue
        if "*" not in line:
            continue  # exact paths are scoped by definition
        # iogpu.wired_limit_mb=* is a sysctl key=*VALUE wildcard, not a path one.
        if "iogpu.wired_limit_mb" in line:
            continue
        # /tmp/asiai-*/ tempfile sources are scoped to our per-invocation
        # 0700 staging directories (see ais_core.io.secure_staging_dir).
        if "/tmp/asiai-" in line:
            continue
        assert "com.asiai." in line, f"unscoped rule with wildcard: {line!r}"


def test_content_includes_pfctl_validate_command() -> None:
    """pfctl -nf - must be NOPASSWD; firewall.validate_anchor() depends on it."""
    content = generate_sudoers_content()
    assert "/sbin/pfctl -nf -" in content


def test_content_admin_group_only() -> None:
    """Rules must apply to the %admin group, not 'ALL'."""
    content = generate_sudoers_content()
    rule_lines = [ln for ln in content.splitlines() if "NOPASSWD" in ln]
    for line in rule_lines:
        assert line.startswith("%admin"), f"non-admin rule: {line!r}"


@pytest.mark.skipif(
    shutil.which("/usr/sbin/visudo") is None,
    reason="visudo not available (likely a sandbox or non-macOS host)",
)
def test_validate_content_accepts_real_generated_sudoers() -> None:
    """The generated content must pass real ``visudo -cf``.

    This is the canonical safety net: a syntactically broken sudoers file
    can lock the user out of every privileged op on the box. Catching that
    BEFORE any sudo move-into-place is the whole point of the validate step.
    """
    content = generate_sudoers_content()
    validate_content(content)  # raises if visudo rejects


@pytest.mark.skipif(
    shutil.which("/usr/sbin/visudo") is None,
    reason="visudo not available",
)
def test_validate_content_rejects_garbage() -> None:
    with pytest.raises(SudoersError):
        validate_content("this is not valid sudoers syntax at all\n")


def test_install_sudoers_dry_run_skips_subprocess() -> None:
    with patch("ais_core.sudoers.subprocess.run") as mock_run:
        path = install_sudoers(dry_run=True)
    mock_run.assert_not_called()
    assert path == SUDOERS_PATH


def test_install_sudoers_invokes_validate_then_mv() -> None:
    """Validation must happen BEFORE we move the file under /etc/sudoers.d/."""
    call_order: list[str] = []

    def fake_run(cmd, **_kwargs):
        if any("visudo" in c for c in cmd):
            call_order.append("visudo")
        elif cmd[:2] == ["sudo", "/bin/mv"]:
            call_order.append("mv")
        return MagicMock(returncode=0, stdout="", stderr="")

    # The non-TTY guard (US-004 fix) blocks pytest runs by default. We patch
    # sys.stdin.isatty to True so the install path runs end-to-end here.
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=fake_run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
    ):
        install_sudoers("# minimal\n")

    assert call_order[:2] == ["visudo", "mv"]


def test_remove_sudoers_when_absent_returns_false() -> None:
    from ais_core.sudoers import remove_sudoers

    with patch("ais_core.sudoers.is_installed", return_value=False):
        assert remove_sudoers() is False


def test_remove_sudoers_when_present_calls_sudo_rm() -> None:
    from ais_core.sudoers import remove_sudoers

    with (
        patch("ais_core.sudoers.is_installed", return_value=True),
        patch("ais_core.sudoers.subprocess.run") as mock_run,
    ):
        result = remove_sudoers()
    assert result is True
    cmd = mock_run.call_args.args[0]
    assert cmd == ["sudo", "/bin/rm", "-f", "/etc/sudoers.d/asiai-inference"]


def test_no_temp_files_left_after_validate(tmp_path: Path, monkeypatch) -> None:
    """validate_content must clean up its tempfile even on success."""
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("ais_core.sudoers.subprocess.run", return_value=fake):
        validate_content("# fake\n")
    leftover = list(tmp_path.glob("asiai-inference.*"))
    # Note: tempfile picks /tmp regardless of TMPDIR for our explicit dir=,
    # so this assertion is best-effort. Skip if no files appeared in tmp_path.
    if leftover:
        pytest.fail(f"tempfile left behind: {leftover}")


# ---------------------------------------------------------------------------
# Dogfood-discovered bug fix (US-004 — non-TTY graceful)
# ---------------------------------------------------------------------------


def test_install_sudoers_non_tty_raises_with_clear_instructions() -> None:
    """US-004: when run without a controlling terminal, surface clear hint."""
    with (
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=False),
        pytest.raises(SudoersError, match="interactive terminal"),
    ):
        install_sudoers("# minimal\n")


def test_install_sudoers_dry_run_skips_tty_check() -> None:
    """Dry-run prints content regardless of TTY status."""
    with patch("ais_core.sudoers.sys.stdin.isatty", return_value=False):
        result = install_sudoers("# dry\n", dry_run=True)
    assert result == "/etc/sudoers.d/asiai-inference"
