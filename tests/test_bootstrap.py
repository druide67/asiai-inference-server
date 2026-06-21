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
import stat
import subprocess
from pathlib import Path

import pytest

from ais_core import bootstrap as bootstrap_mod
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
