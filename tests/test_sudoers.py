"""Tests for ais_core.sudoers — content shape + visudo validation."""

from __future__ import annotations

import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ais_core.sudoers import (
    ADMIN_GROUP,
    PRIVILEGED_HELPER_PATH,
    SUDOERS_PATH,
    SUDOERS_STAGED,
    SudoersError,
    generate_sudoers_content,
    install_sudoers,
    validate_content,
)


def test_content_starts_with_warning_header() -> None:
    content = generate_sudoers_content()
    assert content.startswith("# /etc/sudoers.d/asiai-inference")
    assert "Do not edit by hand" in content


def test_content_grants_helper_only() -> None:
    """The single NOPASSWD rule targets the privileged helper (any args — self-validated)."""
    content = generate_sudoers_content()
    rule = f"{ADMIN_GROUP} ALL=(root) NOPASSWD: {PRIVILEGED_HELPER_PATH}"
    assert rule in content
    nopasswd = [ln for ln in content.splitlines() if "NOPASSWD:" in ln]
    assert nopasswd == [rule]  # exactly one rule, nothing else


def test_content_has_no_wildcards_or_raw_commands() -> None:
    """No '*' and no raw launchctl/pfctl/mv/chown/chmod/rm/install on ANY directive line."""
    content = generate_sudoers_content()
    raw = ("launchctl", "pfctl", "/bin/mv", "chown", "chmod", "/bin/rm", "/usr/bin/install")
    directives = [ln for ln in content.splitlines() if ln and not ln.lstrip().startswith("#")]
    for line in directives:
        assert "*" not in line, f"wildcard rule survived: {line!r}"
        for cmd in raw:
            assert cmd not in line, f"raw command rule survived: {line!r}"


def test_content_env_reset_no_sudo_keep() -> None:
    """env_reset re-asserted for the helper; SUDO_* never env_kept (I2 anti-spoof)."""
    content = generate_sudoers_content()
    assert f"Defaults!{PRIVILEGED_HELPER_PATH} env_reset" in content
    # env_keep / SUDO_ may appear only in the explanatory comment, never in a directive.
    directives = [ln for ln in content.splitlines() if ln and not ln.lstrip().startswith("#")]
    assert all("env_keep" not in ln for ln in directives)
    assert all("SUDO_" not in ln for ln in directives)


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


def test_install_sudoers_secure_publish_sequence() -> None:
    """All owner/mode work on the DOTTED staged name; the final name appears via one atomic
    rename; visudo -c validates the whole tree last. No invoker-owned / parsed-wrong-mode
    window, and cp (not install) leaves no non-dotted temp sudo could parse.
    """
    sudo_calls: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        if cmd[0] == "sudo":  # validate_content's `visudo -cf <tmp>` is NOT a sudo call
            sudo_calls.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    # The non-TTY guard (US-004 fix) blocks pytest runs by default. We patch
    # sys.stdin.isatty to True so the install path runs end-to-end here.
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=fake_run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
    ):
        install_sudoers("# minimal\n")

    assert [c[1] for c in sudo_calls] == [
        "/bin/rm",  # clear any stale/pre-positioned staged inode before cp (no symlink-follow)
        "/bin/cp",
        "/usr/sbin/chown",
        "/bin/chmod",
        "/bin/mv",
        "/usr/sbin/visudo",
    ]
    # rm + cp/chown/chmod operate on the dot-prefixed staged name; only mv touches the final name.
    assert sudo_calls[0] == ["sudo", "/bin/rm", "-f", SUDOERS_STAGED]
    assert sudo_calls[1][-1] == SUDOERS_STAGED  # cp writes the dotted staged file
    assert sudo_calls[2] == ["sudo", "/usr/sbin/chown", "root:wheel", SUDOERS_STAGED]
    assert sudo_calls[3] == ["sudo", "/bin/chmod", "0440", SUDOERS_STAGED]
    assert sudo_calls[4] == ["sudo", "/bin/mv", SUDOERS_STAGED, SUDOERS_PATH]
    assert sudo_calls[5] == ["sudo", "/usr/sbin/visudo", "-c"]


@pytest.mark.parametrize("fail_at", ["/bin/cp", "/bin/mv"])
def test_install_sudoers_cleans_staged_on_failure(fail_at: str) -> None:
    """A failure at cp OR at the publish-rename removes the dot-prefixed staging file.

    The mv-fail case is the sharp one: cp already succeeded, so a real dot-prefixed leftover
    exists and must be cleaned (and being dot-prefixed, sudo ignores it meanwhile).
    """
    import subprocess as _sp

    removed: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        if any("visudo" in c for c in cmd):
            return MagicMock(returncode=0, stdout="", stderr="")
        if cmd[:2] == ["sudo", fail_at]:
            raise _sp.CalledProcessError(1, cmd)
        if cmd[:3] == ["sudo", "/bin/rm", "-f"]:
            removed.append(cmd)
            return MagicMock(returncode=0)
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=fake_run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        pytest.raises(SudoersError),
    ):
        install_sudoers("# minimal\n")

    assert removed and removed[0][-1].endswith("/.asiai-inference.tmp")


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


def test_no_temp_files_left_after_validate(monkeypatch) -> None:
    """validate_content must clean up its staging dir even on success."""
    import ais_core.sudoers as sudoers_mod

    captured: list[Path] = []
    real_staging = sudoers_mod.secure_staging_dir

    @contextmanager
    def spying_staging():
        with real_staging() as staging:
            captured.append(staging)
            yield staging

    monkeypatch.setattr(sudoers_mod, "secure_staging_dir", spying_staging)
    fake = MagicMock(returncode=0, stdout="", stderr="")
    with patch("ais_core.sudoers.subprocess.run", return_value=fake):
        validate_content("# fake\n")
    assert len(captured) == 1
    assert not captured[0].exists(), f"staging dir left behind: {captured[0]}"


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
