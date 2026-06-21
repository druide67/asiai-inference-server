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
    SUDOERS_ABSENT_MARKER,
    SUDOERS_BACKUP_PATH,
    SUDOERS_BACKUP_STAGED,
    SUDOERS_PATH,
    SUDOERS_STAGED,
    SudoersError,
    _sudo_exists,
    backup_existing_sudoers,
    generate_sudoers_content,
    install_sudoers,
    restore_sudoers,
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


# ---------------------------------------------------------------------------
# Rollback support — backup + restore (FR8, story 2.3)
# ---------------------------------------------------------------------------


def _sudo_driver(
    existence: dict[str, bool],
    *,
    visudo_cf_rc: int = 0,
    grep_rc: int = 1,
    raise_on: str | None = None,
):
    """Build a fake ``subprocess.run`` driven by a path->exists map, plus a recording list.

    ``_sudo_exists`` issues ``sudo /bin/ls -d -- <path>``; we answer rc 0 (exists) or rc 1 with the
    real "No such file or directory" stderr (absent). ``visudo -cf`` returns ``visudo_cf_rc``. The
    baseline-poisoning ``grep -qF`` returns ``grep_rc`` (default 1 = the live fragment is NOT the
    helper model — the normal pre-cutover case). ``raise_on`` (a basename like "/bin/cp") makes that
    sudo verb raise CalledProcessError.
    """
    import subprocess as _sp

    calls: list[list[str]] = []

    def run(cmd, **_kw):
        calls.append(cmd)
        if raise_on is not None and cmd[:2] == ["sudo", raise_on]:
            raise _sp.CalledProcessError(1, cmd)
        if cmd[:4] == ["sudo", "/bin/ls", "-d", "--"]:
            path = cmd[4]
            if existence.get(path, False):
                return MagicMock(returncode=0, stdout=path + "\n", stderr="")
            return MagicMock(
                returncode=1, stdout="", stderr=f"ls: {path}: No such file or directory"
            )
        if cmd[:3] == ["sudo", "/usr/bin/grep", "-qF"]:
            return MagicMock(returncode=grep_rc, stdout="", stderr="")
        if cmd[:3] == ["sudo", "/usr/sbin/visudo", "-cf"]:
            return MagicMock(returncode=visudo_cf_rc, stdout="", stderr="parse error")
        return MagicMock(returncode=0, stdout="", stderr="")

    return run, calls


def test_sudo_exists_three_states() -> None:
    run, _ = _sudo_driver({"/exists": True})
    with patch("ais_core.sudoers.subprocess.run", side_effect=run):
        assert _sudo_exists("/exists") is True
        assert _sudo_exists("/gone") is False

    # An indeterminate result (neither rc 0 nor a clean "No such file") fails CLOSED.
    def weird(cmd, **_kw):
        return MagicMock(returncode=2, stdout="", stderr="sudo: a password is required")

    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=weird),
        pytest.raises(SudoersError, match="cannot determine existence"),
    ):
        _sudo_exists("/x")


def test_backup_skips_when_already_recorded() -> None:
    """If either marker already exists, the original pre-helper state is left untouched."""
    run, calls = _sudo_driver({SUDOERS_BACKUP_PATH: True})  # backup already present
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
    ):
        assert backup_existing_sudoers() is None
    # only the existence probe(s) ran — no cp/mv/touch clobbering the recorded state
    verbs = [c[1] for c in calls]
    assert "/bin/cp" not in verbs and "/bin/mv" not in verbs and "/usr/bin/touch" not in verbs


def test_backup_copies_prior_fragment_when_present() -> None:
    """No markers yet + a genuine (non-helper) prior fragment -> copy it to the backup via atomic mv."""
    existence = {SUDOERS_BACKUP_PATH: False, SUDOERS_ABSENT_MARKER: False, SUDOERS_PATH: True}
    run, calls = _sudo_driver(existence, grep_rc=1)  # grep_rc=1 -> not the helper model
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
    ):
        result = backup_existing_sudoers()
    assert result == SUDOERS_BACKUP_PATH
    seq = [c[1:] for c in calls if c[1] in ("/bin/cp", "/usr/sbin/chown", "/bin/chmod", "/bin/mv")]
    assert seq[0][:2] == ["/bin/cp", SUDOERS_PATH]  # copy the prior fragment...
    assert seq[0][-1] == SUDOERS_BACKUP_STAGED  # ...to the dotted staged name
    assert ["/bin/mv", SUDOERS_BACKUP_STAGED, SUDOERS_BACKUP_PATH] in seq  # atomic publish
    assert not any(c[1] == "/usr/bin/touch" for c in calls)  # no absent-marker path taken


def test_backup_refuses_to_record_a_helper_model_fragment() -> None:
    """HIGH (anti-poisoning): no markers + live fragment ALREADY references the helper -> refuse,
    never freeze the helper-only model as the 'pre-bootstrap' baseline (would defeat FR8)."""
    existence = {SUDOERS_BACKUP_PATH: False, SUDOERS_ABSENT_MARKER: False, SUDOERS_PATH: True}
    run, calls = _sudo_driver(existence, grep_rc=0)  # grep_rc=0 -> IS the helper model
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
        pytest.raises(SudoersError, match="refusing to record the baseline"),
    ):
        backup_existing_sudoers()
    # refused BEFORE recording anything
    assert not any(
        c[1] in ("/bin/cp", "/bin/mv", "/usr/bin/touch") for c in calls if c[0] == "sudo"
    )


def test_fragment_is_helper_model_three_states() -> None:
    from ais_core.sudoers import _fragment_is_helper_model

    with patch("ais_core.sudoers.subprocess.run", return_value=MagicMock(returncode=0)):
        assert _fragment_is_helper_model("/x") is True
    with patch("ais_core.sudoers.subprocess.run", return_value=MagicMock(returncode=1)):
        assert _fragment_is_helper_model("/x") is False
    with (
        patch(
            "ais_core.sudoers.subprocess.run",
            return_value=MagicMock(returncode=2, stderr="boom"),
        ),
        pytest.raises(SudoersError, match="cannot inspect"),
    ):
        _fragment_is_helper_model("/x")


def test_backup_writes_absent_marker_when_no_prior() -> None:
    """No markers yet + NO live fragment -> stage an empty marker, then atomic mv (symmetric)."""
    existence = {SUDOERS_BACKUP_PATH: False, SUDOERS_ABSENT_MARKER: False, SUDOERS_PATH: False}
    run, calls = _sudo_driver(existence)
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
    ):
        result = backup_existing_sudoers()
    assert result == SUDOERS_ABSENT_MARKER
    # absent branch is now atomic+symmetric: touch the STAGED name, then mv it onto the marker
    assert ["sudo", "/usr/bin/touch", SUDOERS_BACKUP_STAGED] in calls
    assert ["sudo", "/bin/mv", SUDOERS_BACKUP_STAGED, SUDOERS_ABSENT_MARKER] in calls
    assert not any(c[1] == "/bin/cp" for c in calls)  # nothing to copy
    # the live marker name is never written directly (no rm-f-before-write skip, no symlink-follow)
    assert not any(c[-1] == SUDOERS_ABSENT_MARKER and c[1] == "/usr/bin/touch" for c in calls)


def test_backup_dry_run_and_non_tty() -> None:
    with patch("ais_core.sudoers.subprocess.run") as m:
        assert backup_existing_sudoers(dry_run=True) is None
    m.assert_not_called()
    with (
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=False),
        pytest.raises(SudoersError, match="interactive terminal"),
    ):
        backup_existing_sudoers()


def test_backup_cleans_staged_on_failure() -> None:
    """A cp failure mid-capture removes the dotted staged file and surfaces a SudoersError."""
    existence = {SUDOERS_BACKUP_PATH: False, SUDOERS_ABSENT_MARKER: False, SUDOERS_PATH: True}
    run, calls = _sudo_driver(existence, raise_on="/bin/cp")
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
        pytest.raises(SudoersError, match="Failed to back up"),
    ):
        backup_existing_sudoers()
    # best-effort cleanup rm of the staged name ran (the last rm targets the staged backup)
    assert any(c == ["sudo", "/bin/rm", "-f", SUDOERS_BACKUP_STAGED] for c in calls)


def test_restore_from_backup_validates_then_publishes() -> None:
    """backup present -> stage, visudo -cf the staged backup, atomic mv, visudo -c the tree."""
    run, calls = _sudo_driver({SUDOERS_BACKUP_PATH: True})
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
    ):
        desc = restore_sudoers()
    assert "restored" in desc and SUDOERS_BACKUP_PATH in desc
    privileged = [c[1:] for c in calls if c[0] == "sudo" and c[1] != "/bin/ls"]
    # cp backup -> staged ; validate the STAGED file BEFORE it becomes live ; mv ; then whole-tree
    assert ["/bin/cp", SUDOERS_BACKUP_PATH, SUDOERS_STAGED] in privileged
    assert ["/usr/sbin/visudo", "-cf", SUDOERS_STAGED] in privileged
    mv_i = privileged.index(["/bin/mv", SUDOERS_STAGED, SUDOERS_PATH])
    cf_i = privileged.index(["/usr/sbin/visudo", "-cf", SUDOERS_STAGED])
    c_i = privileged.index(["/usr/sbin/visudo", "-c"])
    assert cf_i < mv_i < c_i  # validate-before-publish, then whole-tree check last
    # the recorded baseline is INTENTIONALLY retained (re-bootstrap must round-trip to it)
    assert not any(["/bin/rm", "-f", SUDOERS_BACKUP_PATH] == c for c in privileged)


def test_restore_refuses_backup_that_fails_visudo() -> None:
    """ANTI-LOCKOUT: a backup that fails visudo -cf is NOT published; the live fragment stays."""
    run, calls = _sudo_driver({SUDOERS_BACKUP_PATH: True}, visudo_cf_rc=1)
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
        pytest.raises(SudoersError, match="visudo -cf"),
    ):
        restore_sudoers()
    privileged = [c[1:] for c in calls if c[0] == "sudo"]
    assert ["/bin/mv", SUDOERS_STAGED, SUDOERS_PATH] not in privileged  # never went live
    assert ["/bin/rm", "-f", SUDOERS_STAGED] in privileged  # staged backup cleaned up


def test_restore_absent_marker_removes_fragment() -> None:
    """absent-marker present (no backup) -> remove the live fragment, then visudo -c."""
    run, calls = _sudo_driver({SUDOERS_BACKUP_PATH: False, SUDOERS_ABSENT_MARKER: True})
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
    ):
        desc = restore_sudoers()
    assert "removed the live fragment" in desc
    privileged = [c[1:] for c in calls if c[0] == "sudo"]
    assert ["/bin/rm", "-f", SUDOERS_PATH] in privileged
    assert ["/usr/sbin/visudo", "-c"] in privileged
    # the absent-marker is INTENTIONALLY retained, so a re-bootstrap still round-trips to "absent"
    assert not any(["/bin/rm", "-f", SUDOERS_ABSENT_MARKER] == c for c in privileged)


def test_restore_refuses_when_no_markers() -> None:
    """Neither backup nor absent-marker -> refuse (never guess the prior state)."""
    run, calls = _sudo_driver({SUDOERS_BACKUP_PATH: False, SUDOERS_ABSENT_MARKER: False})
    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
        pytest.raises(SudoersError, match="no recorded pre-bootstrap state"),
    ):
        restore_sudoers()
    # refusal happens before any mutating verb
    assert not any(c[1] in ("/bin/cp", "/bin/mv", "/bin/rm") for c in calls if c[0] == "sudo")


def test_restore_dry_run_and_non_tty() -> None:
    with patch("ais_core.sudoers.subprocess.run") as m:
        assert restore_sudoers(dry_run=True) == "dry-run"
    m.assert_not_called()
    with (
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=False),
        pytest.raises(SudoersError, match="interactive terminal"),
    ):
        restore_sudoers()


def test_fr8_roundtrip_first_capture_wins_survives_two_cycles() -> None:
    """FR8 durability: install -> rollback -> re-install -> rollback. The backup is captured ONCE
    (the original pre-helper fragment) and SURVIVES both cycles, so the 2nd rollback still targets
    the ORIGINAL baseline — modelled with a stateful filesystem map driving the real functions."""
    fs: dict[str, bool] = {
        SUDOERS_BACKUP_PATH: False,
        SUDOERS_ABSENT_MARKER: False,
        SUDOERS_PATH: True,
    }
    backup_captures: list[str] = []

    def run(cmd, **_kw):
        if cmd[:4] == ["sudo", "/bin/ls", "-d", "--"]:
            p = cmd[4]
            return (
                MagicMock(returncode=0, stdout=p, stderr="")
                if fs.get(p, False)
                else MagicMock(returncode=1, stdout="", stderr="No such file or directory")
            )
        if cmd[:3] == ["sudo", "/usr/bin/grep", "-qF"]:
            return MagicMock(returncode=1)  # the prior fragment is NOT the helper model
        if cmd[:2] == ["sudo", "/bin/cp"]:
            if cmd[-1] == SUDOERS_BACKUP_STAGED:  # only the backup-capture cp, not restore's cp
                backup_captures.append(cmd[2])  # source = what got recorded as the baseline
            fs[cmd[-1]] = True
            return MagicMock(returncode=0)
        if cmd[:2] == ["sudo", "/bin/mv"]:
            fs[cmd[-1]] = True
            fs[cmd[-2]] = False
            return MagicMock(returncode=0)
        if cmd[:3] == ["sudo", "/bin/rm", "-f"]:
            for t in cmd[3:]:
                fs[t] = False
            return MagicMock(returncode=0)
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("ais_core.sudoers.subprocess.run", side_effect=run),
        patch("ais_core.sudoers.sys.stdin.isatty", return_value=True),
        patch("ais_core.bootstrap.assert_chain_locked"),
    ):
        # cycle 1: capture the ORIGINAL pre-helper fragment, then roll back
        assert backup_existing_sudoers() == SUDOERS_BACKUP_PATH
        assert fs[SUDOERS_BACKUP_PATH] is True
        restore_sudoers()
        # cycle 2: re-bootstrap must be a NO-OP (backup already recorded), not re-capture
        assert backup_existing_sudoers() is None
        restore_sudoers()

    # the cp-as-baseline ran EXACTLY once, on the ORIGINAL live fragment — never on a later one
    assert backup_captures == [SUDOERS_PATH]
    assert fs[SUDOERS_BACKUP_PATH] is True  # baseline still there after both cycles
