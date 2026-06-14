"""Tests for the asiai-priv privileged-helper foundation (story 1.1).

The helper is a standalone binary (``python3 -I``), not an importable package
module. It is loaded via ``importlib`` (the ``-I`` shebang only affects direct
execution, not import) and the ``AUDIT_LOG`` constant is monkeypatched to
``tmp_path``. In production ``-I`` blocks any external override, so the
hardcoded value is the production value.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pwd
import stat
from pathlib import Path

import pytest

_HELPER_PATH = Path(__file__).resolve().parents[1] / "data" / "helpers" / "asiai_priv.py"


def _load_helper():
    spec = importlib.util.spec_from_file_location("asiai_priv_under_test", _HELPER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def helper(tmp_path, monkeypatch):
    mod = _load_helper()
    audit = tmp_path / "audit.log"
    monkeypatch.setattr(mod, "AUDIT_LOG", str(audit))
    return mod, audit


def _read_audit(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _silence_syslog(mod, monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(mod.syslog, "syslog", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(mod.syslog, "openlog", lambda *a, **k: None)
    monkeypatch.setattr(mod.syslog, "closelog", lambda: None)
    return calls


def test_unknown_action_refused(helper):
    mod, audit = helper
    with pytest.raises(SystemExit) as exc:
        mod.main(["nope"])
    assert exc.value.code == mod.EXIT_REFUSED
    assert any(r["verdict"] == "refused" for r in _read_audit(audit))


def test_missing_action_refused(helper):
    mod, audit = helper
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert exc.value.code == mod.EXIT_REFUSED
    assert any(r["verdict"] == "refused" for r in _read_audit(audit))


def test_help_is_not_a_refusal(helper):
    mod, audit = helper
    with pytest.raises(SystemExit) as exc:
        mod.main(["-h"])
    assert exc.value.code == 0
    assert _read_audit(audit) == []  # --help must never be audited as a refusal


def test_stub_action_returns_internal_error(helper):
    mod, audit = helper
    assert mod.main(["purge"]) == mod.EXIT_INTERNAL
    rows = _read_audit(audit)
    assert any(r["action"] == "purge" and r["verdict"] == "error" for r in rows)


def test_audit_is_append_only(helper):
    mod, audit = helper
    mod.main(["purge"])
    mod.main(["purge"])
    assert len(_read_audit(audit)) == 2


def test_audit_refuses_symlink(helper, monkeypatch, tmp_path):
    mod, _ = helper
    target = tmp_path / "target.log"
    target.write_text("")
    Path(mod.AUDIT_LOG).symlink_to(target)  # AUDIT_LOG is a symlink
    calls = _silence_syslog(mod, monkeypatch)
    mod._audit("purge", "error", reason="x")
    assert target.read_text() == ""  # the symlink was NOT followed (O_NOFOLLOW)
    assert calls  # syslog fallback was invoked


def test_audit_enospc_fails_open(helper, monkeypatch):
    mod, _ = helper
    calls = _silence_syslog(mod, monkeypatch)

    def _boom(*_a, **_k):
        raise OSError(28, "No space left on device")  # ENOSPC

    monkeypatch.setattr(mod.os, "write", _boom)
    mod._audit("purge", "error")  # must not raise
    assert calls


def test_audit_missing_dir_fails_open(helper, monkeypatch, tmp_path):
    mod, _ = helper
    calls = _silence_syslog(mod, monkeypatch)
    monkeypatch.setattr(mod, "AUDIT_LOG", str(tmp_path / "absent" / "audit.log"))
    mod._audit("purge", "error")  # missing directory -> fail-open
    assert calls


def test_audit_refuses_foreign_owned_file(helper, monkeypatch):
    """A regular file owned by another uid (pre-positioned) is refused."""
    mod, audit = helper
    calls = _silence_syslog(mod, monkeypatch)

    class _ForeignStat:
        st_mode = stat.S_IFREG | 0o600
        st_uid = os.geteuid() + 12345  # an owner other than us

    monkeypatch.setattr(mod.os, "fstat", lambda _fd: _ForeignStat())
    mod._audit("purge", "error")
    assert calls  # syslog fallback was invoked
    assert _read_audit(audit) == []  # nothing written to the suspect file


def test_run_never_uses_shell(helper, monkeypatch):
    mod, _ = helper
    captured: dict = {}

    def _fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return None

    monkeypatch.setattr(mod.subprocess, "run", _fake_run)
    mod._run(["echo", "x"], check=True)
    assert captured["argv"] == ["echo", "x"]
    assert captured["kwargs"].get("shell", False) is False


# ---------------------------------------------------------------------------
# Story 1.2 — strict parameter validation. Negative tests are first-class ACs.
# ---------------------------------------------------------------------------


def _fake_pw(name: str, uid: int) -> pwd.struct_passwd:
    return pwd.struct_passwd((name, "*", uid, uid, name, f"/Users/{name}", "/bin/zsh"))


# --- I2: identity derivation (the security-critical one) -------------------


def test_resolve_user_refuses_literal_root(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._resolve_user("root")  # real getpwnam('root') -> pw_uid 0


def test_resolve_user_decides_on_object_not_string(helper, monkeypatch):
    """The refusal keys on the resolved pw_uid, never the literal string.

    getpwnam is case-insensitive on macOS, so 'Root'/'ROOT' all resolve to uid 0.
    Here getpwnam is mocked to return uid 0 for a name that is NOT 'root' — the
    helper must still refuse, proving the decision is on the object.
    """
    mod, _ = helper
    monkeypatch.setattr(mod.pwd, "getpwnam", lambda name: _fake_pw("not-literally-root", 0))
    with pytest.raises(mod._Refused):
        mod._resolve_user("not-literally-root")


def test_resolve_user_derives_from_sudo_uid(helper, monkeypatch):
    mod, _ = helper
    fake = _fake_pw("alice", 1234)

    def fake_getpwuid(uid):
        if uid == 1234:
            return fake
        raise KeyError(uid)

    monkeypatch.setattr(mod.pwd, "getpwuid", fake_getpwuid)
    monkeypatch.setenv("SUDO_UID", "1234")
    pw = mod._resolve_user(None)
    assert pw.pw_uid == 1234 and pw.pw_name == "alice"


def test_resolve_user_refuses_sudo_uid_zero(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setenv("SUDO_UID", "0")  # original invoker was root -> refuse
    with pytest.raises(mod._Refused):
        mod._resolve_user(None)


def test_resolve_user_refuses_missing_sudo_uid(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.delenv("SUDO_UID", raising=False)
    with pytest.raises(mod._Refused):
        mod._resolve_user(None)


def test_resolve_user_refuses_nonnumeric_sudo_uid(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setenv("SUDO_UID", "not-a-number")
    with pytest.raises(mod._Refused):
        mod._resolve_user(None)


def test_resolve_user_refuses_nonascii_digit_sudo_uid(helper, monkeypatch):
    """Arabic-Indic '5' passes str.isdigit() but is non-canonical -> refused (not crash)."""
    mod, _ = helper
    monkeypatch.setenv("SUDO_UID", "\u0665")  # Arabic-Indic 5: isdigit() True, isascii() False
    with pytest.raises(mod._Refused):
        mod._resolve_user(None)


def test_resolve_user_refuses_unknown_name(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._resolve_user("nonexistent-user-zzz-9999")


# --- I9: label ------------------------------------------------------------


def test_validate_label_accepts_canonical(helper):
    mod, _ = helper
    assert mod._validate_label("com.asiai.web") == "com.asiai.web"
    assert mod._validate_label("com.asiai.llamacpp-aux-1") == "com.asiai.llamacpp-aux-1"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "com.asiai.",
        "com.asiai.WEB",  # uppercase
        "com.evil.x",  # wrong prefix
        "com.asiai.web\n",  # trailing control char
        "com.asiai.a b",  # space
        "com.asiai.x\x00",  # NUL
        "prefixcom.asiai.web",  # not anchored at start
    ],
)
def test_validate_label_rejects(helper, bad):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._validate_label(bad)


# --- I5: binary -----------------------------------------------------------


def test_resolve_binary_accepts_under_prefix(helper, monkeypatch, tmp_path):
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    target = bindir / "llama-server"
    target.write_text("x")
    real_prefix = os.path.realpath(str(bindir)) + os.sep
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (real_prefix,))
    assert mod._resolve_binary(str(target)) == os.path.realpath(str(target))


def test_resolve_binary_rejects_outside_prefix(helper, monkeypatch, tmp_path):
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    target = tmp_path / "rogue"
    target.write_text("x")
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (os.path.realpath(str(bindir)) + os.sep,))
    with pytest.raises(mod._Refused):
        mod._resolve_binary(str(target))


def test_resolve_binary_rejects_symlink_final(helper, monkeypatch, tmp_path):
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    real = bindir / "llama-server"
    real.write_text("x")
    link = bindir / "llama-link"
    link.symlink_to(real)  # target is valid AND under prefix, but the leaf is a symlink
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (os.path.realpath(str(bindir)) + os.sep,))
    with pytest.raises(mod._Refused):
        mod._resolve_binary(str(link))


def test_resolve_binary_rejects_prefix_sibling(helper, monkeypatch, tmp_path):
    """A path sharing the prefix string but not the directory boundary is refused."""
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    evil = tmp_path / "binEVIL"
    evil.mkdir()
    target = evil / "llama-server"
    target.write_text("x")
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (os.path.realpath(str(bindir)) + os.sep,))
    with pytest.raises(mod._Refused):
        mod._resolve_binary(str(target))


def test_resolve_binary_rejects_relative_and_missing(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._resolve_binary("bin/llama-server")  # relative
    with pytest.raises(mod._Refused):
        mod._resolve_binary("/opt/homebrew/bin/does-not-exist-zzz")


def test_resolve_binary_rejects_nul(helper):
    """A NUL byte makes os.lstat raise ValueError (not OSError) -> must still refuse."""
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._resolve_binary("/opt/homebrew/bin/llama\x00server")


# --- I3: environment ------------------------------------------------------


def test_validate_env_accepts(helper):
    mod, _ = helper
    assert mod._validate_env(["FOO=bar", "PATH=/usr/bin"]) == {"FOO": "bar", "PATH": "/usr/bin"}


@pytest.mark.parametrize(
    "bad",
    [
        "DYLD_INSERT_LIBRARIES=/evil.dylib",
        "DYLD_LIBRARY_PATH=/x",
        "LD_PRELOAD=/x",
        "1FOO=x",  # key starts with a digit
        "FO O=x",  # space in key
        "FOO",  # no '='
        "FOO=a\nb",  # control char in value
    ],
)
def test_validate_env_rejects(helper, bad):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._validate_env([bad])


# --- port -----------------------------------------------------------------


@pytest.mark.parametrize("ok", ["1024", "8080", "65535"])
def test_validate_port_accepts(helper, ok):
    mod, _ = helper
    assert mod._validate_port(ok) == int(ok)


@pytest.mark.parametrize("bad", ["80", "0", "1023", "65536", "70000", "abc", ""])
def test_validate_port_rejects(helper, bad):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._validate_port(bad)


# --- user-home paths ------------------------------------------------------


def test_resolve_user_path_accepts_under_home(helper, tmp_path):
    mod, _ = helper
    home = os.path.realpath(str(tmp_path))
    models = tmp_path / "models"
    models.mkdir()
    gguf = models / "m.gguf"
    gguf.write_text("x")
    assert mod._resolve_user_path("~/models/m.gguf", home) == os.path.realpath(str(gguf))


def test_resolve_user_path_rejects_outside_home(helper, tmp_path):
    mod, _ = helper
    home = os.path.realpath(str(tmp_path / "home"))
    os.makedirs(home)
    outside = tmp_path / "outside.gguf"
    outside.write_text("x")
    with pytest.raises(mod._Refused):
        mod._resolve_user_path(str(outside), home)


def test_resolve_user_path_rejects_parent_traversal(helper, tmp_path):
    mod, _ = helper
    sub = tmp_path / "home"
    sub.mkdir()
    home = os.path.realpath(str(sub))
    secret = tmp_path / "secret.gguf"
    secret.write_text("x")  # exists, but ~/../secret.gguf escapes home
    with pytest.raises(mod._Refused):
        mod._resolve_user_path("~/../secret.gguf", home)


def test_resolve_user_path_rejects_symlink_final(helper, tmp_path):
    mod, _ = helper
    home = os.path.realpath(str(tmp_path))
    real = tmp_path / "real.gguf"
    real.write_text("x")
    link = tmp_path / "link.gguf"
    link.symlink_to(real)
    with pytest.raises(mod._Refused):
        mod._resolve_user_path(str(link), home)


def test_resolve_user_path_rejects_nul(helper, tmp_path):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        mod._resolve_user_path("~/m\x00.gguf", os.path.realpath(str(tmp_path)))


def test_resolve_user_path_rejects_degenerate_home(helper, tmp_path):
    """home '/' must not void confinement (would otherwise accept any absolute path)."""
    mod, _ = helper
    target = tmp_path / "x.gguf"
    target.write_text("x")
    with pytest.raises(mod._Refused):
        mod._resolve_user_path(str(target), "/")


# --- root guard -----------------------------------------------------------


def test_require_root_refuses_non_root(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(mod.os, "geteuid", lambda: 1000)
    with pytest.raises(mod._Refused):
        mod._require_root()


def test_require_root_passes_as_root(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(mod.os, "geteuid", lambda: 0)
    mod._require_root()  # must not raise


# --- main: refusal & internal-error hardening -----------------------------


def test_main_audits_refused_from_handler(helper, monkeypatch):
    mod, audit = helper

    def boom(_args):
        raise mod._Refused("bad param")

    monkeypatch.setitem(mod._DISPATCH, "purge", boom)
    assert mod.main(["purge"]) == mod.EXIT_REFUSED
    rows = _read_audit(audit)
    assert any(r["verdict"] == "refused" and "bad param" in r["reason"] for r in rows)


def test_main_internal_error_logs_type_not_message(helper, monkeypatch):
    """A bug in a handler must not leak its message (paths/secrets) to the audit."""
    mod, audit = helper
    secret = "/Users/secret/leak/path"

    def boom(_args):
        raise RuntimeError(secret)

    monkeypatch.setitem(mod._DISPATCH, "purge", boom)
    assert mod.main(["purge"]) == mod.EXIT_INTERNAL
    content = audit.read_text() if audit.exists() else ""
    assert "RuntimeError" in content
    assert secret not in content  # only the exception type is logged, never the message


# --- audit hardening: partial write + PIPE_BUF bound ----------------------


def test_audit_writes_full_line_on_partial_write(helper, monkeypatch):
    mod, audit = helper
    real_write = os.write

    def short_write(fd, data):
        return real_write(fd, data[:1])  # one byte per call -> exercises the loop

    monkeypatch.setattr(mod.os, "write", short_write)
    mod._audit("purge", "error", reason="hello-world")
    rows = _read_audit(audit)
    assert rows and rows[0]["reason"] == "hello-world"


def test_audit_truncates_long_field(helper):
    mod, audit = helper
    mod._audit("purge", "error", reason="r", argv=["x" * 5000])
    raw = audit.read_text()
    assert len(raw.encode("utf-8")) < 4096  # below PIPE_BUF
    assert "x" * 600 not in raw  # the 5000-char arg was clipped


def test_audit_truncates_oversized_record_to_skeleton(helper):
    mod, audit = helper
    mod._audit("purge", "error", reason="r", argv=["aaaa"] * 2000)
    rows = _read_audit(audit)
    assert rows and rows[0].get("truncated") is True
    assert len(audit.read_text().encode("utf-8")) < 4096
