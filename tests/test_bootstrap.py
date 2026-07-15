"""Tests for ais_core.bootstrap — the I0 dir-chain permission check.

The real chain (``/``, ``/Library``, ...) is root-owned and cannot be chmod'd in a test, so
``os.lstat`` is mocked to return controlled ``stat_result``s for a synthetic chain. This
exercises the LOGIC (root-owned + not group/other-writable + not a symlink, walked from /)
without needing root.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import plistlib
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from ais_core import bootstrap as bootstrap_mod
from ais_core import plist as plist_mod
from ais_core import sudoers
from ais_core.bootstrap import LOCKED_CHAIN, BootstrapError, assert_chain_locked


def _st(uid: int, mode: int, *, kind: int = stat.S_IFDIR, gid: int = 0) -> os.stat_result:
    # (mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime)
    return os.stat_result((kind | mode, 0, 0, 1, uid, gid, 0, 0, 0, 0))


def test_assert_chain_locked_passes_when_root_owned_and_not_writable(monkeypatch):
    monkeypatch.setattr(os, "lstat", lambda _p: _st(0, 0o755))
    assert_chain_locked("/Library/Logs/asiai")  # must not raise


def test_assert_chain_locked_rejects_non_absolute():
    with pytest.raises(BootstrapError):
        assert_chain_locked("Library/Logs/asiai")


def test_assert_chain_locked_rejects_group_writable_ancestor(monkeypatch):
    def fake(p):
        return _st(0, 0o775) if str(p) == "/Library" else _st(0, 0o755)  # /Library group-writable

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="group/other-writable"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_rejects_other_writable_ancestor(monkeypatch):
    def fake(p):
        return _st(0, 0o757) if str(p) == "/Library" else _st(0, 0o755)  # /Library other-writable

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="group/other-writable"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_rejects_non_root_owner(monkeypatch):
    def fake(p):
        return _st(501, 0o755) if str(p) == "/Library/Logs" else _st(0, 0o755)  # non-root ancestor

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="not root:wheel"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_rejects_non_wheel_group(monkeypatch):
    def fake(p):
        return _st(0, 0o755, gid=20) if str(p) == "/Library" else _st(0, 0o755)  # root:staff

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="not root:wheel"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_rejects_setgid_dir(monkeypatch):
    """A root-owned, NOT group-writable setgid dir (0o2755) still propagates its group to
    children root creates below — must be rejected even though the writable bits are clear."""

    def fake(p):
        return _st(0, 0o2755) if str(p) == "/Library/Logs" else _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="setuid/setgid"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_allows_sticky_dir(monkeypatch):
    """Sticky (S_ISVTX) is benign on a dir — /Library/PrivilegedHelperTools ships drwxr-xr-t."""
    monkeypatch.setattr(os, "lstat", lambda _p: _st(0, 0o1755))
    assert_chain_locked("/Library/PrivilegedHelperTools")  # must not raise


def test_assert_chain_locked_rejects_dotdot(monkeypatch):
    monkeypatch.setattr(os, "lstat", lambda _p: _st(0, 0o755))
    with pytest.raises(BootstrapError, match=r"\.\."):
        assert_chain_locked("/Library/Logs/../Logs/asiai")


def test_assert_chain_locked_rejects_non_directory_ancestor(monkeypatch):
    def fake(p):
        return _st(0, 0o644, kind=stat.S_IFREG) if str(p) == "/Library" else _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="not a directory"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_rejects_non_root_symlink_leaf(monkeypatch):
    """A NON-root symlink at the write target is a tamper vector and is refused."""

    def fake(p):
        if str(p) == "/Library/PrivilegedHelperTools":
            return _st(501, 0o777, kind=stat.S_IFLNK)  # attacker-owned symlink
        return _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="non-root symlink"):
        assert_chain_locked("/Library/PrivilegedHelperTools")


def test_assert_chain_locked_rejects_non_root_symlink_component(monkeypatch):
    def fake(p):
        if str(p) == "/Library/Logs":
            return _st(501, 0o777, kind=stat.S_IFLNK)  # attacker-owned symlinked component
        return _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    with pytest.raises(BootstrapError, match="non-root symlink"):
        assert_chain_locked("/Library/Logs/asiai")


def test_assert_chain_locked_allows_root_owned_symlink_component(monkeypatch):
    """A ROOT-owned symlink (its parent already verified root-owned + non-writable, so only root
    could have created it) is NOT a tamper vector — this is the macOS /etc -> /private/etc
    reality. Accept it and walk through to the resolved component."""

    def fake(p):
        if str(p) == "/Library/Logs":
            return _st(0, 0o755, kind=stat.S_IFLNK)  # root-owned symlink, like /etc
        return _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    assert_chain_locked("/Library/Logs/asiai")  # must NOT raise


def test_assert_chain_locked_passes_on_real_etc_sudoers_d():
    """Regression (the bug the mocked tests missed): macOS ships /etc as a root-owned symlink
    into /private/etc, so the strict no-symlink rule refused /etc and broke install_sudoers on
    EVERY Mac. The real /etc/sudoers.d chain (root-owned on a healthy host) must verify."""
    import sys

    if sys.platform != "darwin":
        pytest.skip("macOS-specific /etc symlink")
    assert_chain_locked("/etc/sudoers.d")  # must not raise on a healthy macOS host


def test_assert_chain_locked_allows_missing_leaf(monkeypatch):
    """A not-yet-existing leaf is fine — the bootstrap creates it root-owned."""

    def fake(p):
        if str(p) == "/Library/Logs/asiai":
            raise FileNotFoundError(p)
        return _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    assert_chain_locked("/Library/Logs/asiai")  # must not raise


def test_assert_chain_locked_walks_full_chain_from_root(monkeypatch):
    """Every component from / down is inspected (a deep writable ancestor is still caught)."""
    seen: list[str] = []

    def fake(p):
        seen.append(str(p))
        return _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake)
    assert_chain_locked("/Library/PrivilegedHelperTools")
    assert seen == ["/", "/Library", "/Library/PrivilegedHelperTools"]


def test_locked_chain_covers_all_root_write_targets():
    # /etc/sudoers.d included — the sudoers fragment is the most root-equivalent write.
    assert set(LOCKED_CHAIN) == {
        "/Library/PrivilegedHelperTools",
        "/Library/LaunchDaemons",
        "/Library/Logs/asiai",
        "/etc/sudoers.d",
    }


def test_locked_chain_sourced_from_canonical_constants():
    from ais_core import plist, sudoers

    assert sudoers.SUDOERS_DIR in LOCKED_CHAIN
    assert plist.LAUNCH_DAEMONS_DIR in LOCKED_CHAIN
    assert str(Path(sudoers.PRIVILEGED_HELPER_PATH).parent) in LOCKED_CHAIN


def test_assert_fleet_chain_locked_checks_every_target(monkeypatch):
    checked: list[str] = []
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: checked.append(p))
    bootstrap_mod.assert_fleet_chain_locked()
    assert checked == list(LOCKED_CHAIN)


# ---------------------------------------------------------------------------
# install_helper — the privileged copy (mocked subprocess; real sudo needs root).
# ---------------------------------------------------------------------------


def test_bundled_helper_path_finds_source():
    p = bootstrap_mod._bundled_helper_path()
    assert p.is_file() and p.name == "asiai_priv.py"


def test_install_helper_dry_run_touches_nothing(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    dest = bootstrap_mod.install_helper(dry_run=True)
    assert dest == sudoers.PRIVILEGED_HELPER_PATH
    assert calls == []  # dry-run runs no sudo


def test_install_helper_non_tty_raises(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(bootstrap_mod, "_bundled_helper_path", lambda: Path("/SRC/asiai_priv.py"))
    with pytest.raises(BootstrapError, match="interactive terminal"):
        bootstrap_mod.install_helper()


def test_install_helper_sequence_and_invariant3(monkeypatch):
    """I0 check on dest BEFORE any write, then mkdir -> cp -> chown -> chmod 0755 -> atomic mv
    onto PRIVILEGED_HELPER_PATH (invariant #3: the executed copy, never site-packages)."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    checks: list[str] = []
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: checks.append(p))
    monkeypatch.setattr(bootstrap_mod, "_bundled_helper_path", lambda: Path("/SRC/asiai_priv.py"))
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    dest = bootstrap_mod.install_helper()

    assert dest == sudoers.PRIVILEGED_HELPER_PATH
    # I0 on the destination twice: before any write, and again AFTER mkdir (a freshly created
    # dir must also be verified locked — closes the loose-umask window).
    assert checks == [sudoers.PRIVILEGED_HELPER_PATH, sudoers.PRIVILEGED_HELPER_PATH]
    staged = bootstrap_mod.HELPER_STAGED
    dest_dir = str(Path(sudoers.PRIVILEGED_HELPER_PATH).parent)
    assert calls == [
        ["sudo", "/bin/mkdir", "-p", dest_dir],
        ["sudo", "/bin/rm", "-f", staged],  # clear any stale/planted staged inode before cp
        ["sudo", "/bin/cp", "/SRC/asiai_priv.py", staged],
        ["sudo", "/usr/sbin/chown", "root:wheel", staged],
        ["sudo", "/bin/chmod", "0755", staged],
        ["sudo", "/bin/mv", staged, sudoers.PRIVILEGED_HELPER_PATH],
    ]


def test_install_helper_cleans_staged_on_failure(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: None)
    monkeypatch.setattr(bootstrap_mod, "_bundled_helper_path", lambda: Path("/SRC/asiai_priv.py"))
    calls: list[list[str]] = []

    def fake_run(argv, **k):
        calls.append(argv)
        if argv[:2] == ["sudo", "/bin/cp"]:
            raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(bootstrap_mod.subprocess, "run", fake_run)
    with pytest.raises(BootstrapError, match="Failed to install helper"):
        bootstrap_mod.install_helper()
    assert ["sudo", "/bin/rm", "-f", bootstrap_mod.HELPER_STAGED] in calls  # best-effort cleanup


def test_install_helper_dest_is_the_invoked_path():
    """invariant #3 closure: what install_helper writes == what privhelper invokes."""
    from ais_core import privhelper

    assert bootstrap_mod.install_helper(dry_run=True) == privhelper.helper_path()


# ---------------------------------------------------------------------------
# remove_helper — the rollback half on the helper side (FR8, story 2.3).
# ---------------------------------------------------------------------------


def test_remove_helper_dry_run_touches_nothing(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    targets = bootstrap_mod.remove_helper(dry_run=True)
    assert targets == [sudoers.PRIVILEGED_HELPER_PATH, bootstrap_mod.HELPER_SHA256_PATH]
    assert calls == []


def test_remove_helper_non_tty_raises(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: False)
    with pytest.raises(BootstrapError, match="interactive terminal"):
        bootstrap_mod.remove_helper()


def test_remove_helper_rms_helper_and_sidecar_after_i0(monkeypatch):
    """I0-checked, then a single rm -f over BOTH the helper and its signature (idempotent)."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    checks: list[str] = []
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: checks.append(p))
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    targets = bootstrap_mod.remove_helper()

    assert checks == [sudoers.PRIVILEGED_HELPER_PATH]  # I0 before the privileged rm
    assert calls == [
        [
            "sudo",
            "/bin/rm",
            "-f",
            sudoers.PRIVILEGED_HELPER_PATH,
            bootstrap_mod.HELPER_SHA256_PATH,
        ]
    ]
    assert targets == [sudoers.PRIVILEGED_HELPER_PATH, bootstrap_mod.HELPER_SHA256_PATH]


def test_remove_helper_wraps_rm_failure(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: None)

    def fake_run(argv, **k):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(bootstrap_mod.subprocess, "run", fake_run)
    with pytest.raises(BootstrapError, match="Failed to remove the helper"):
        bootstrap_mod.remove_helper()


# ---------------------------------------------------------------------------
# write_helper_signature / verify_helper — NFR11 SHA-256 sidecar.
# ---------------------------------------------------------------------------


def test_write_helper_signature_dry_run(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    out = bootstrap_mod.write_helper_signature(dry_run=True)
    assert out == bootstrap_mod.HELPER_SHA256_PATH
    assert calls == []


def test_write_helper_signature_requires_installed_helper(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(tmp_path / "absent"))
    with pytest.raises(BootstrapError, match="not installed"):
        bootstrap_mod.write_helper_signature()


def test_write_helper_signature_sequence(monkeypatch, tmp_path):
    helper = tmp_path / "asiai-priv"
    helper.write_text("#!/usr/bin/python3 -I\n")
    sidecar = tmp_path / "asiai-priv.sha256"
    staged = tmp_path / ".asiai-priv.sha256.tmp"
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir()
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(helper))
    monkeypatch.setattr(bootstrap_mod, "HELPER_SHA256_PATH", str(sidecar))
    monkeypatch.setattr(bootstrap_mod, "SIDECAR_STAGED", str(staged))
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: None)

    @contextlib.contextmanager
    def fake_staging():
        yield staging_dir

    monkeypatch.setattr(bootstrap_mod, "secure_staging_dir", fake_staging)
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    out = bootstrap_mod.write_helper_signature()

    assert out == str(sidecar)
    tmp_content = (staging_dir / "asiai-priv.sha256").read_text()
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    assert tmp_content == f"{digest}  asiai-priv\n"  # shasum -a 256 -c format
    assert calls == [
        ["sudo", "/bin/rm", "-f", str(staged)],
        ["sudo", "/bin/cp", str(staging_dir / "asiai-priv.sha256"), str(staged)],
        ["sudo", "/usr/sbin/chown", "root:wheel", str(staged)],
        ["sudo", "/bin/chmod", "0644", str(staged)],
        ["sudo", "/bin/mv", str(staged), str(sidecar)],
    ]


def test_verify_helper_match(monkeypatch, tmp_path):
    helper = tmp_path / "asiai-priv"
    helper.write_text("payload\n")
    sidecar = tmp_path / "asiai-priv.sha256"
    digest = hashlib.sha256(helper.read_bytes()).hexdigest()
    sidecar.write_text(f"{digest}  asiai-priv\n")
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(helper))
    monkeypatch.setattr(bootstrap_mod, "HELPER_SHA256_PATH", str(sidecar))
    assert bootstrap_mod.verify_helper() is True


def test_verify_helper_mismatch(monkeypatch, tmp_path):
    helper = tmp_path / "asiai-priv"
    helper.write_text("payload\n")
    sidecar = tmp_path / "asiai-priv.sha256"
    sidecar.write_text(f"{'0' * 64}  asiai-priv\n")  # wrong hash
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(helper))
    monkeypatch.setattr(bootstrap_mod, "HELPER_SHA256_PATH", str(sidecar))
    assert bootstrap_mod.verify_helper() is False


def test_verify_helper_missing_sidecar(monkeypatch, tmp_path):
    helper = tmp_path / "asiai-priv"
    helper.write_text("x")
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(helper))
    monkeypatch.setattr(bootstrap_mod, "HELPER_SHA256_PATH", str(tmp_path / "absent.sha256"))
    with pytest.raises(BootstrapError, match="no signature sidecar"):
        bootstrap_mod.verify_helper()


def test_verify_helper_malformed_sidecar(monkeypatch, tmp_path):
    helper = tmp_path / "asiai-priv"
    helper.write_text("x")
    sidecar = tmp_path / "asiai-priv.sha256"
    sidecar.write_text("not-a-hash\n")
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(helper))
    monkeypatch.setattr(bootstrap_mod, "HELPER_SHA256_PATH", str(sidecar))
    with pytest.raises(BootstrapError, match="malformed"):
        bootstrap_mod.verify_helper()


def test_verify_helper_rejects_64_char_non_hex_sidecar(monkeypatch, tmp_path):
    """A 64-char token that isn't hex is 'malformed', not a silent mismatch."""
    helper = tmp_path / "asiai-priv"
    helper.write_text("x")
    sidecar = tmp_path / "asiai-priv.sha256"
    sidecar.write_text(f"{'z' * 64}  asiai-priv\n")  # 64 chars but not hex
    monkeypatch.setattr(bootstrap_mod.sudoers, "PRIVILEGED_HELPER_PATH", str(helper))
    monkeypatch.setattr(bootstrap_mod, "HELPER_SHA256_PATH", str(sidecar))
    with pytest.raises(BootstrapError, match="malformed"):
        bootstrap_mod.verify_helper()


# ---------------------------------------------------------------------------
# create_dedicated_user / role-uid helpers — NFR12 (_aisrv).
# ---------------------------------------------------------------------------


def test_create_dedicated_user_dry_run(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    out = bootstrap_mod.create_dedicated_user(dry_run=True)
    assert out["created"] is None
    assert calls == []  # dry-run runs nothing


def test_create_dedicated_user_creates_role_account(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    uids = iter([None, 450])  # existence check -> absent; post-create verify -> 450
    monkeypatch.setattr(bootstrap_mod, "_account_uid", lambda n: next(uids))
    monkeypatch.setattr(bootstrap_mod, "_free_role_uid", lambda: 450)
    monkeypatch.setattr(bootstrap_mod, "_assert_role_account", lambda n: None)  # checked separately
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    out = bootstrap_mod.create_dedicated_user("_aisrv")

    assert out == {"user": "_aisrv", "created": True, "uid": 450}
    assert calls == [
        ["sudo", "/usr/sbin/sysadminctl", "-addUser", "_aisrv", "-UID", "450", "-roleAccount"]
    ]


def test_create_dedicated_user_idempotent_reverifies_shape(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod, "_account_uid", lambda n: 450)
    asserted: list[str] = []
    monkeypatch.setattr(bootstrap_mod, "_assert_role_account", lambda n: asserted.append(n))

    def _no_run(*a, **k):
        pytest.fail("no sysadminctl on an idempotent no-op")

    monkeypatch.setattr(bootstrap_mod.subprocess, "run", _no_run)
    out = bootstrap_mod.create_dedicated_user("_aisrv")
    assert out == {"user": "_aisrv", "created": False, "uid": 450}
    assert asserted == ["_aisrv"]  # full NFR12 shape re-verified before adopting


@pytest.mark.parametrize("uid", [0, 501])
def test_create_dedicated_user_refuses_uid_out_of_range(monkeypatch, uid):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod, "_account_uid", lambda n: uid)

    def _no_run(*a, **k):
        pytest.fail("must not run sysadminctl on an out-of-range existing account")

    monkeypatch.setattr(bootstrap_mod.subprocess, "run", _no_run)
    with pytest.raises(BootstrapError, match="outside the role range"):
        bootstrap_mod.create_dedicated_user("_aisrv")


def test_create_dedicated_user_refuses_existing_failing_shape(monkeypatch):
    """A uid-in-range existing account that fails the full shape check is refused, not adopted."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(bootstrap_mod, "_account_uid", lambda n: 450)

    def _boom(_n):
        raise BootstrapError("_aisrv: is in the admin group — the daemon account must not be")

    monkeypatch.setattr(bootstrap_mod, "_assert_role_account", _boom)
    with pytest.raises(BootstrapError, match="admin group"):
        bootstrap_mod.create_dedicated_user("_aisrv")


@pytest.mark.parametrize("bad", ["_aisrv/x", "../admin", "a b", "_AISRV", "-x", ""])
def test_create_dedicated_user_rejects_invalid_name(bad):
    with pytest.raises(BootstrapError, match="invalid account name"):
        bootstrap_mod.create_dedicated_user(bad)


def test_create_dedicated_user_non_tty_raises(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: False)
    with pytest.raises(BootstrapError, match="interactive terminal"):
        bootstrap_mod.create_dedicated_user("_aisrv")


def test_free_role_uid_picks_first_unused_in_range(monkeypatch):
    listing = "root 0\n_mbsetupuser 248\n_aisrvtest 450\nsomeone 451\njmn 501\n"

    def fake_run(argv, **k):
        assert argv[:4] == ["/usr/bin/dscl", ".", "-list", "/Users"]
        return type("P", (), {"returncode": 0, "stdout": listing})()

    monkeypatch.setattr(bootstrap_mod.subprocess, "run", fake_run)
    assert bootstrap_mod._free_role_uid() == 452  # 450 + 451 taken -> 452


def test_free_role_uid_raises_when_range_full(monkeypatch):
    listing = "".join(f"u{n} {n}\n" for n in range(450, 500))  # 450..499 all taken

    def fake_run(argv, **k):
        return type("P", (), {"returncode": 0, "stdout": listing})()

    monkeypatch.setattr(bootstrap_mod.subprocess, "run", fake_run)
    with pytest.raises(BootstrapError, match="no free uid"):
        bootstrap_mod._free_role_uid()


def test_free_role_uid_raises_on_dscl_error(monkeypatch):
    monkeypatch.setattr(
        bootstrap_mod, "_run_ro", lambda argv: type("P", (), {"returncode": 1, "stdout": ""})()
    )
    with pytest.raises(BootstrapError, match="cannot enumerate uids"):
        bootstrap_mod._free_role_uid()


# --- _in_admin_group: fail-CLOSED tri-state -------------------------------


@pytest.mark.parametrize(("rc", "expected"), [(0, True), (67, False)])
def test_in_admin_group_decisive(monkeypatch, rc, expected):
    monkeypatch.setattr(bootstrap_mod, "_run_ro", lambda argv: type("P", (), {"returncode": rc})())
    assert bootstrap_mod._in_admin_group("_aisrv") is expected


def test_in_admin_group_indeterminate_fails_closed(monkeypatch):
    # any rc that is neither 0 (member) nor 67 (confirmed not-member) -> refuse, never assume
    monkeypatch.setattr(bootstrap_mod, "_run_ro", lambda argv: type("P", (), {"returncode": 64})())
    with pytest.raises(BootstrapError, match="cannot determine admin membership"):
        bootstrap_mod._in_admin_group("_aisrv")


# --- _assert_role_account: full NFR12 shape -------------------------------


def _patch_role_attrs(
    monkeypatch, *, shell="/usr/bin/false", home="/var/empty", hidden="1", auth=False, admin=False
):
    vals = {"UserShell": shell, "NFSHomeDirectory": home, "IsHidden": hidden}
    monkeypatch.setattr(bootstrap_mod, "_dscl_value", lambda n, k: vals.get(k))
    monkeypatch.setattr(bootstrap_mod, "_has_auth_authority", lambda n: auth)
    monkeypatch.setattr(bootstrap_mod, "_in_admin_group", lambda n: admin)


def test_assert_role_account_accepts_conforming(monkeypatch):
    _patch_role_attrs(monkeypatch)
    bootstrap_mod._assert_role_account("_aisrv")  # must not raise


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"shell": "/bin/zsh"}, "UserShell"),
        ({"home": "/Users/_aisrv"}, "NFSHomeDirectory"),
        ({"hidden": "0"}, "IsHidden"),
        ({"auth": True}, "AuthenticationAuthority"),
        ({"admin": True}, "admin group"),
    ],
)
def test_assert_role_account_refuses_nonconforming(monkeypatch, kw, match):
    _patch_role_attrs(monkeypatch, **kw)
    with pytest.raises(BootstrapError, match=match):
        bootstrap_mod._assert_role_account("_aisrv")


# --- _has_auth_authority: fail-CLOSED (symmetry with _in_admin_group) ------


def _ro(returncode: int, stdout: str = "", stderr: str = ""):
    proc = type("P", (), {"returncode": returncode, "stdout": stdout, "stderr": stderr})()
    return lambda _argv: proc


def test_has_auth_authority_present(monkeypatch):
    # present: rc 0, the value on stdout (dscl -read returns rc 0 here)
    monkeypatch.setattr(bootstrap_mod, "_run_ro", _ro(0, stdout="AuthenticationAuthority: ;X;"))
    assert bootstrap_mod._has_auth_authority("x") is True


def test_has_auth_authority_absent_no_such_key(monkeypatch):
    # REAL macOS 26 behaviour: an ABSENT key still returns rc 0, with empty stdout and
    # "No such key" on STDERR. The decision must be made on the OUTPUT, never on rc==0
    # (which previously made _has_auth_authority wrongly report every account as login-capable,
    # breaking --dedicated-user against a perfectly valid role account).
    monkeypatch.setattr(
        bootstrap_mod, "_run_ro", _ro(0, stderr="No such key: AuthenticationAuthority")
    )
    assert bootstrap_mod._has_auth_authority("x") is False


def test_has_auth_authority_indeterminate_fails_closed(monkeypatch):
    # a non-zero rc that is NOT "No such key" (transient DS error) -> refuse, never assume absent
    monkeypatch.setattr(bootstrap_mod, "_run_ro", _ro(5, stderr="DS connection error"))
    with pytest.raises(BootstrapError, match="cannot read AuthenticationAuthority"):
        bootstrap_mod._has_auth_authority("x")


def test_has_auth_authority_rc0_but_empty_is_indeterminate(monkeypatch):
    # rc 0 with neither a value NOR "No such key" is unexpected -> fail-closed, not "present"
    monkeypatch.setattr(bootstrap_mod, "_run_ro", _ro(0, stdout="", stderr=""))
    with pytest.raises(BootstrapError, match="cannot read AuthenticationAuthority"):
        bootstrap_mod._has_auth_authority("x")


# ---------------------------------------------------------------------------
# SA audit #3b/#2: bootstrap owns the helper's log dir + audit log
# ---------------------------------------------------------------------------


def test_ensure_log_dir_dry_run_touches_nothing(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    assert bootstrap_mod.ensure_log_dir(dry_run=True) == bootstrap_mod.LOG_DIR
    assert calls == []


def test_ensure_log_dir_non_tty_raises(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: False)
    with pytest.raises(bootstrap_mod.BootstrapError, match="interactive terminal"):
        bootstrap_mod.ensure_log_dir()


def test_ensure_log_dir_sequence(monkeypatch):
    """I0 before AND after the mkdir (freshly created dir re-verified), root:wheel 0755."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    checks: list[str] = []
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: checks.append(p))
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    assert bootstrap_mod.ensure_log_dir() == bootstrap_mod.LOG_DIR
    assert checks == [bootstrap_mod.LOG_DIR, bootstrap_mod.LOG_DIR]
    assert calls == [
        ["sudo", "/bin/mkdir", "-p", bootstrap_mod.LOG_DIR],
        ["sudo", "/usr/sbin/chown", "root:wheel", bootstrap_mod.LOG_DIR],
        ["sudo", "/bin/chmod", "0755", bootstrap_mod.LOG_DIR],
    ]


def test_ensure_audit_log_dry_run_touches_nothing(monkeypatch):
    calls: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: calls.append(a))
    assert bootstrap_mod.ensure_audit_log(dry_run=True) == bootstrap_mod.AUDIT_LOG_PATH
    assert calls == []


def test_ensure_audit_log_non_tty_raises(monkeypatch):
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: False)
    with pytest.raises(bootstrap_mod.BootstrapError, match="interactive terminal"):
        bootstrap_mod.ensure_audit_log()


def test_ensure_audit_log_sequence_0640_root_admin(monkeypatch):
    """Finding #2: touch (never truncates), then root:admin 0640 — group READ without
    sudo, write stays root-only. Group ownership set here, once; the helper's hot
    path never chowns. The I0 check targets the PARENT chain (LOG_DIR), not the leaf
    whose admin group is intentional (see idempotence regression below)."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)
    checks: list[str] = []
    monkeypatch.setattr(bootstrap_mod, "assert_chain_locked", lambda p: checks.append(p))
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    assert bootstrap_mod.ensure_audit_log() == bootstrap_mod.AUDIT_LOG_PATH
    assert checks == [bootstrap_mod.LOG_DIR]
    assert calls == [
        ["sudo", "/usr/bin/touch", bootstrap_mod.AUDIT_LOG_PATH],
        ["sudo", "/usr/sbin/chown", "root:admin", bootstrap_mod.AUDIT_LOG_PATH],
        ["sudo", "/bin/chmod", "0640", bootstrap_mod.AUDIT_LOG_PATH],
    ]


def test_ensure_audit_log_idempotent_over_existing_admin_leaf(monkeypatch):
    """Regression (M5 cutover rehearsal, 2026-07-03): re-running ``bootstrap --install``
    over an audit log that ALREADY exists as root:admin (gid 80, by design) must NOT be
    rejected. ensure_audit_log verifies the parent chain (LOG_DIR, root:wheel), never the
    leaf — otherwise the leaf's own admin group trips the root:wheel-only I0 check, making
    bootstrap non-idempotent and breaking the rollback→reinstall recovery path. Uses the
    REAL assert_chain_locked so a revert to checking the leaf would fail this test."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)

    def fake_lstat(p):
        # The exact leftover state: a prior bootstrap left the audit log root:ADMIN and
        # rollback does not remove it. If the leaf were (wrongly) walked, gid 80 != 0 aborts.
        if str(p) == bootstrap_mod.AUDIT_LOG_PATH:
            return _st(0, 0o640, kind=stat.S_IFREG, gid=80)
        return _st(0, 0o755)  # every parent-chain component is a root:wheel dir

    monkeypatch.setattr(os, "lstat", fake_lstat)
    calls: list[list[str]] = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda argv, **k: calls.append(argv))

    assert bootstrap_mod.ensure_audit_log() == bootstrap_mod.AUDIT_LOG_PATH  # must not raise
    assert ["sudo", "/usr/sbin/chown", "root:admin", bootstrap_mod.AUDIT_LOG_PATH] in calls


def test_ensure_audit_log_still_rejects_tampered_parent(monkeypatch):
    """The fix must not weaken I0: a group-writable LOG_DIR parent still aborts before any write."""
    monkeypatch.setattr(bootstrap_mod.sys.stdin, "isatty", lambda: True)

    def fake_lstat(p):
        if str(p) == bootstrap_mod.LOG_DIR:
            return _st(0, 0o775)  # group-writable parent → must be caught
        return _st(0, 0o755)

    monkeypatch.setattr(os, "lstat", fake_lstat)
    ran: list = []
    monkeypatch.setattr(bootstrap_mod.subprocess, "run", lambda *a, **k: ran.append(a))
    with pytest.raises(bootstrap_mod.BootstrapError, match="group/other-writable"):
        bootstrap_mod.ensure_audit_log()
    assert ran == []  # aborted before any sudo write


# ---------------------------------------------------------------------------
# Daemon logging surface (--logs-only): discovery + pre-creation logic.
# The real /Library paths are root-owned; LAUNCH_DAEMONS_DIR and LOG_DIR are
# monkeypatched to tmp_path so the LOGIC is exercised without root.
# ---------------------------------------------------------------------------


def _write_plist(dirpath, label, *, user="alice", out=None, err=None, **extra):
    data = {"Label": label, **extra}
    if user is not None:
        data["UserName"] = user
    if out is not None:
        data["StandardOutPath"] = out
    if err is not None:
        data["StandardErrorPath"] = err
    p = dirpath / f"{label}.plist"
    with p.open("wb") as fh:
        plistlib.dump(data, fh)
    return p


@pytest.fixture
def log_surface(tmp_path, monkeypatch):
    daemons = tmp_path / "LaunchDaemons"
    logs = tmp_path / "Logs" / "asiai"
    daemons.mkdir()
    logs.mkdir(parents=True)
    monkeypatch.setattr(plist_mod, "LAUNCH_DAEMONS_DIR", str(daemons))
    monkeypatch.setattr(bootstrap_mod, "LOG_DIR", str(logs))
    return daemons, logs


def test_log_specs_discovers_both_std_paths(log_surface):
    daemons, logs = log_surface
    _write_plist(
        daemons,
        "com.asiai.demo",
        out=f"{logs}/com.asiai.demo.out",
        err=f"{logs}/com.asiai.demo.err",
    )
    specs = bootstrap_mod.installed_daemon_log_specs()
    assert [(str(p), u) for p, u in specs] == [
        (f"{logs}/com.asiai.demo.err", "alice"),
        (f"{logs}/com.asiai.demo.out", "alice"),
    ] or [(str(p), u) for p, u in specs] == [
        (f"{logs}/com.asiai.demo.out", "alice"),
        (f"{logs}/com.asiai.demo.err", "alice"),
    ]


def test_log_specs_ignores_paths_outside_log_dir(log_surface, tmp_path):
    daemons, _logs = log_surface
    _write_plist(daemons, "com.asiai.evil", out=str(tmp_path / "elsewhere" / "x.out"))
    assert bootstrap_mod.installed_daemon_log_specs() == []


def test_log_specs_skips_plist_without_username(log_surface):
    daemons, logs = log_surface
    _write_plist(daemons, "com.asiai.nouser", user=None, out=f"{logs}/com.asiai.nouser.out")
    assert bootstrap_mod.installed_daemon_log_specs() == []


def test_log_specs_skips_foreign_and_malformed(log_surface, tmp_path):
    daemons, logs = log_surface
    _write_plist(daemons, "com.other.thing", out=f"{logs}/x.out")  # not com.asiai.*
    (daemons / "com.asiai.broken.plist").write_bytes(b"not a plist")
    assert bootstrap_mod.installed_daemon_log_specs() == []


def test_missing_daemon_log_files_only_reports_absent(log_surface):
    daemons, logs = log_surface
    _write_plist(
        daemons,
        "com.asiai.demo",
        out=f"{logs}/com.asiai.demo.out",
        err=f"{logs}/com.asiai.demo.err",
    )
    (logs / "com.asiai.demo.out").touch()
    missing = bootstrap_mod.missing_daemon_log_files()
    assert [str(p) for p, _ in missing] == [f"{logs}/com.asiai.demo.err"]


def test_ensure_daemon_log_files_dry_run_touches_nothing(log_surface, capsys):
    daemons, logs = log_surface
    _write_plist(daemons, "com.asiai.demo", out=f"{logs}/com.asiai.demo.out")
    created = bootstrap_mod.ensure_daemon_log_files(dry_run=True)
    assert created == [f"{logs}/com.asiai.demo.out"]
    assert not (logs / "com.asiai.demo.out").exists()
    assert "[dry-run]" in capsys.readouterr().out


def test_ensure_daemon_log_files_noop_when_all_present(log_surface):
    daemons, logs = log_surface
    _write_plist(daemons, "com.asiai.demo", out=f"{logs}/com.asiai.demo.out")
    (logs / "com.asiai.demo.out").touch()
    # no TTY available in tests: reaching the isatty gate would raise, so a
    # clean [] proves the no-op short-circuits BEFORE any sudo/TTY need.
    assert bootstrap_mod.ensure_daemon_log_files(dry_run=False) == []


def test_ensure_daemon_log_files_requires_tty_when_work_pending(log_surface, monkeypatch):
    daemons, logs = log_surface
    _write_plist(daemons, "com.asiai.demo", out=f"{logs}/com.asiai.demo.out")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(BootstrapError, match="interactive terminal"):
        bootstrap_mod.ensure_daemon_log_files(dry_run=False)


def test_ensure_daemon_log_files_creates_via_sudo_and_chowns(log_surface, monkeypatch):
    daemons, logs = log_surface
    _write_plist(daemons, "com.asiai.demo", user="alice", out=f"{logs}/com.asiai.demo.out")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    fake_pw = type("PW", (), {"pw_gid": 20})()
    monkeypatch.setattr(bootstrap_mod.pwd, "getpwnam", lambda name: fake_pw)
    calls = []
    monkeypatch.setattr(
        bootstrap_mod.subprocess,
        "run",
        lambda argv, check: calls.append(argv) or subprocess.CompletedProcess(argv, 0),
    )
    created = bootstrap_mod.ensure_daemon_log_files(dry_run=False)
    assert created == [f"{logs}/com.asiai.demo.out"]
    assert calls == [
        ["sudo", "/usr/bin/touch", f"{logs}/com.asiai.demo.out"],
        ["sudo", "/usr/sbin/chown", "alice:20", f"{logs}/com.asiai.demo.out"],
        ["sudo", "/bin/chmod", "0640", f"{logs}/com.asiai.demo.out"],
    ]


def test_ensure_daemon_log_files_skips_unknown_user(log_surface, monkeypatch, capsys):
    daemons, logs = log_surface
    _write_plist(daemons, "com.asiai.ghost", user="nosuchuser", out=f"{logs}/com.asiai.ghost.out")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

    def _raise(name):
        raise KeyError(name)

    monkeypatch.setattr(bootstrap_mod.pwd, "getpwnam", _raise)
    assert bootstrap_mod.ensure_daemon_log_files(dry_run=False) == []
    assert "does not exist" in capsys.readouterr().err
