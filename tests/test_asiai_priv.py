"""Tests for the asiai-priv privileged-helper foundation (story 1.1).

The helper is a standalone binary (``python3 -I``), not an importable package
module. It is loaded via ``importlib`` (the ``-I`` shebang only affects direct
execution, not import) and the ``AUDIT_LOG`` constant is monkeypatched to
``tmp_path``. In production ``-I`` blocks any external override, so the
hardcoded value is the production value.
"""

from __future__ import annotations

import grp
import importlib.util
import json
import os
import plistlib
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


def _fake_pw(name: str, uid: int, gid: int | None = None) -> pwd.struct_passwd:
    gid = uid if gid is None else gid
    return pwd.struct_passwd((name, "*", uid, gid, name, f"/Users/{name}", "/bin/zsh"))


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
        "com.asiai.-web",  # leading hyphen
        "com.asiai.web-",  # trailing hyphen
        "com.asiai.-",  # bare hyphen
        "com.asiai.--",  # all hyphens
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
    # Returns the ORIGINAL path (validated via realpath), not the realpath itself.
    assert mod._resolve_binary(str(target)) == str(target)


def test_resolve_binary_rejects_outside_prefix(helper, monkeypatch, tmp_path):
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    target = tmp_path / "rogue"
    target.write_text("x")
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (os.path.realpath(str(bindir)) + os.sep,))
    with pytest.raises(mod._Refused):
        mod._resolve_binary(str(target))


def test_resolve_binary_accepts_symlink_into_prefix(helper, monkeypatch, tmp_path):
    """homebrew ships bin/<engine> as a symlink into Cellar/. A symlink whose REALPATH
    lands under an allowlisted prefix is now ACCEPTED and resolved (the refusal was removed —
    it protected nothing and broke every brew engine)."""
    mod, _ = helper
    cellar = tmp_path / "Cellar" / "llama.cpp" / "9700" / "bin"
    cellar.mkdir(parents=True)
    real = cellar / "llama-server"
    real.write_text("x")  # the real binary inside Cellar
    bindir = tmp_path / "bin"
    bindir.mkdir()
    link = bindir / "llama-server"
    link.symlink_to(real)  # the stable brew symlink the plist references
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (os.path.realpath(str(cellar.parent)) + os.sep,))
    # Validated via the realpath (target under prefix) but RETURNS the stable symlink path the
    # caller passed (launchd re-resolves it at exec → survives brew upgrade), not the
    # versioned Cellar realpath.
    assert mod._resolve_binary(str(link)) == str(link)


def test_resolve_binary_rejects_symlink_escaping_prefix(helper, monkeypatch, tmp_path):
    """The realpath check still pins the real target: a symlink UNDER the prefix that points
    OUTSIDE every allowlisted prefix is refused (relaxing S_ISLNK did not open an escape)."""
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    rogue = tmp_path / "rogue"
    rogue.write_text("x")  # real file OUTSIDE the allowlist
    link = bindir / "llama-server"
    link.symlink_to(rogue)  # symlink sits under the prefix but escapes it on resolution
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (os.path.realpath(str(bindir)) + os.sep,))
    with pytest.raises(mod._Refused, match="outside allowlisted prefixes"):
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


# ---------------------------------------------------------------------------
# Story 1.3 — plist generation (generate-don't-validate). The critical invariant
# is that the generated plist forces a non-root UserName and can never emit a
# dangerous key. Negative tests are first-class ACs.
# ---------------------------------------------------------------------------


def _real_pw() -> pwd.struct_passwd:
    # The test runner's own (non-root) account; _build_plist_dict re-resolves the name
    # via getpwnam, so the user_pw it receives must be a real, resolvable entry.
    return pwd.getpwuid(os.getuid())


def _plist(mod, **overrides):
    kwargs = {
        "label": "com.asiai.web",
        "binary": "/opt/homebrew/bin/llama-server",
        "user_pw": _real_pw(),
    }
    kwargs.update(overrides)
    return mod._build_plist_dict(**kwargs)


def test_plist_only_allowlisted_keys(helper):
    mod, _ = helper
    plist = _plist(
        mod,
        keep_alive={"Crashed": True},
        throttle_interval=10,
        timeout=30,
        nice=0,
        env={"FOO": "bar"},
    )
    assert set(plist) <= set(mod._PLIST_KEYS)


def test_plist_never_emits_dangerous_keys(helper):
    mod, _ = helper
    plist = _plist(mod)
    for dangerous in ("RootDirectory", "GroupName", "Sockets", "MachServices", "LaunchEvents"):
        assert dangerous not in plist


def test_plist_forces_nonroot_username(helper):
    mod, _ = helper
    pw = _real_pw()
    plist = _plist(mod, user_pw=pw)
    assert plist["UserName"] == pw.pw_name


def test_plist_refuses_root_pw(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, user_pw=_fake_pw("root", 0))


def test_plist_refuses_root_by_name_even_if_struct_uid_nonzero(helper):
    """I2 self-defending: launchd resolves by NAME, so a struct claiming uid 502 but
    name 'root' must still be refused (getpwnam('root').pw_uid == 0)."""
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, user_pw=_fake_pw("root", 502))


def test_plist_refuses_unknown_username(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, user_pw=_fake_pw("nonexistent-user-zzz-9999", 1234))


def test_plist_forces_standard_paths_under_log_dir(helper):
    mod, _ = helper
    plist = _plist(mod, label="com.asiai.aux-1")
    assert plist["StandardOutPath"] == "/Library/Logs/asiai/com.asiai.aux-1.out"
    assert plist["StandardErrorPath"] == "/Library/Logs/asiai/com.asiai.aux-1.err"


def test_plist_forces_home_and_path(helper):
    mod, _ = helper
    pw = _real_pw()
    plist = _plist(mod, user_pw=pw)
    env = plist["EnvironmentVariables"]
    assert env["HOME"] == pw.pw_dir
    assert env["PATH"].startswith("/usr/bin:/bin:")  # system dirs first (anti-masking)
    assert env["PATH"].index("/usr/bin") < env["PATH"].index("/opt/homebrew/bin")
    assert pw.pw_dir not in env["PATH"]  # daemon PATH never includes the home


def test_plist_forced_env_overrides_caller(helper):
    mod, _ = helper
    pw = _real_pw()
    plist = _plist(mod, user_pw=pw, env={"HOME": "/evil", "PATH": "/evil/bin", "OK": "1"})
    env = plist["EnvironmentVariables"]
    assert env["HOME"] == pw.pw_dir  # forced; caller /evil ignored
    assert "/evil" not in env["PATH"]
    assert env["OK"] == "1"


def test_plist_refuses_dyld_ld_env(helper):
    """Self-defending: DYLD_/LD_ keys are refused, not silently stripped."""
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, env={"DYLD_INSERT_LIBRARIES": "/x.dylib"})
    with pytest.raises(mod._Refused):
        _plist(mod, env={"LD_PRELOAD": "/y"})


def test_plist_refuses_bad_keep_alive(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, keep_alive={"PathState": {"/x": True}})  # activation directive smuggling
    with pytest.raises(mod._Refused):
        _plist(mod, keep_alive={"Crashed": "yes"})  # non-bool subvalue


def test_plist_refuses_bad_label(helper):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, label="com.evil.x")  # would also poison the log path


def test_plist_optional_keys_omitted_when_none(helper):
    mod, _ = helper
    plist = _plist(mod)  # no keep_alive/throttle/timeout/nice
    for optional in ("KeepAlive", "ThrottleInterval", "TimeOut", "Nice"):
        assert optional not in plist


def test_render_plist_xml_roundtrips(helper):
    mod, _ = helper
    plist = _plist(mod, keep_alive={"Crashed": True, "SuccessfulExit": False})
    xml = mod._render_plist_xml(plist)
    assert isinstance(xml, bytes)
    assert b"<key>UserName</key>" in xml
    parsed = plistlib.loads(xml)
    assert parsed == plist  # lossless round-trip


def test_precreate_log_leaves_creates_owned_files(helper, tmp_path):
    mod, _ = helper
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chmod(str(log_dir), 0o700)  # not group/other-writable -> passes the dir check
    pw = _fake_pw("me", os.getuid(), os.getgid())  # chown to our own uid/gid: allowed non-root
    mod._precreate_log_leaves("com.asiai.web", pw, log_dir=str(log_dir))
    assert (log_dir / "com.asiai.web.out").exists()
    assert (log_dir / "com.asiai.web.err").exists()


def test_precreate_log_leaves_refuses_symlink_leaf(helper, tmp_path):
    mod, _ = helper
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chmod(str(log_dir), 0o700)
    decoy = tmp_path / "decoy"
    decoy.write_text("")
    (log_dir / "com.asiai.web.out").symlink_to(decoy)  # pre-positioned symlink leaf
    pw = _fake_pw("me", os.getuid(), os.getgid())
    with pytest.raises(OSError):  # openat O_NOFOLLOW -> ELOOP
        mod._precreate_log_leaves("com.asiai.web", pw, log_dir=str(log_dir))
    assert decoy.read_text() == ""  # the symlink target was not written


def test_precreate_log_leaves_refuses_symlinked_dir(helper, tmp_path):
    mod, _ = helper
    realdir = tmp_path / "real"
    realdir.mkdir()
    os.chmod(str(realdir), 0o700)
    linkdir = tmp_path / "logs"
    linkdir.symlink_to(realdir)  # log_dir itself is a symlink
    pw = _fake_pw("me", os.getuid(), os.getgid())
    with pytest.raises(OSError):  # O_DIRECTORY|O_NOFOLLOW refuses the symlinked dir
        mod._precreate_log_leaves("com.asiai.web", pw, log_dir=str(linkdir))


def test_precreate_log_leaves_accepts_prior_daemon_owned_leaf(helper, monkeypatch, tmp_path):
    """Re-install idempotency (FR7): a leaf left by a PRIOR install — owned by the target daemon
    account, NOT root — must be accepted and re-chowned, not refused. The dir is root-only so
    such a leaf can only be legitimate. (Caught empirically: a 2nd install of an
    already-installed engine failed on the daemon-user-owned leaf the first install left, since
    uninstall deliberately keeps logs.) ``os.fstat`` is stubbed to report a non-root leaf owner
    while the dir stays real (owned by our euid -> dir check passes); ``fchown`` is mocked
    (chowning to an arbitrary uid needs root)."""
    mod, _ = helper
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chmod(str(log_dir), 0o700)
    daemon_uid = 9999  # != our euid -> only the NEW clause (uid == daemon) can accept it
    real_fstat = os.fstat

    def fake_fstat(fd):
        st = real_fstat(fd)
        if stat.S_ISREG(st.st_mode):  # a leaf: pretend a prior install left it owned by daemon
            return os.stat_result((st.st_mode, 0, 0, 1, daemon_uid, st.st_gid, 0, 0, 0, 0))
        return st  # the dir: real (owned by our euid)

    monkeypatch.setattr(mod.os, "fstat", fake_fstat)
    chowns = _mock_fchown(mod, monkeypatch)
    pw = _fake_pw("aisrv", daemon_uid, os.getgid())
    mod._precreate_log_leaves("com.asiai.web", pw, log_dir=str(log_dir))  # must NOT raise
    assert chowns == [(daemon_uid, os.getgid()), (daemon_uid, os.getgid())]  # .out + .err


def test_precreate_log_leaves_refuses_leaf_owned_by_third_party(helper, monkeypatch, tmp_path):
    """A leaf owned by NEITHER root NOR the target daemon account is still refused (the fix
    only widened acceptance to the daemon account, nothing else)."""
    mod, _ = helper
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    os.chmod(str(log_dir), 0o700)
    real_fstat = os.fstat

    def fake_fstat(fd):
        st = real_fstat(fd)
        if stat.S_ISREG(st.st_mode):
            return os.stat_result((st.st_mode, 0, 0, 1, 12345, st.st_gid, 0, 0, 0, 0))  # stranger
        return st

    monkeypatch.setattr(mod.os, "fstat", fake_fstat)
    _mock_fchown(mod, monkeypatch)
    pw = _fake_pw("aisrv", 9999, os.getgid())  # daemon uid 9999 != 12345
    with pytest.raises(mod._Refused, match="suspect log leaf"):
        mod._precreate_log_leaves("com.asiai.web", pw, log_dir=str(log_dir))


# ---------------------------------------------------------------------------
# Story 1.4 — lifecycle actions (the integration point). The plist WRITE and the
# operation ORDER are the crux; fchown(0,0) is mocked (root-only, not testable non-root).
# ---------------------------------------------------------------------------


def _mock_fchown(mod, monkeypatch) -> list:
    calls: list = []
    monkeypatch.setattr(mod.os, "fchown", lambda fd, uid, gid: calls.append((uid, gid)))
    return calls


def _daemons_dir(tmp_path) -> str:
    d = tmp_path / "LaunchDaemons"
    d.mkdir()
    os.chmod(str(d), 0o755)  # root-equivalent on the target; not group/other-writable
    return str(d)


# --- _write_plist_atomic ---------------------------------------------------


def test_write_plist_atomic_writes_root_wheel_0644(helper, monkeypatch, tmp_path):
    mod, _ = helper
    d = _daemons_dir(tmp_path)
    calls = _mock_fchown(mod, monkeypatch)
    path = mod._write_plist_atomic("com.asiai.web", b"<plist/>\n", daemons_dir=d)
    assert path == f"{d}/com.asiai.web.plist"
    leaf = Path(d) / "com.asiai.web.plist"
    assert leaf.read_bytes() == b"<plist/>\n"
    assert (leaf.stat().st_mode & 0o777) == 0o644
    assert (0, 0) in calls  # fchown(fd, 0, 0) -> root:wheel
    assert not (Path(d) / "com.asiai.web.plist.new").exists()  # temp gone


def test_write_plist_atomic_overwrites_atomically(helper, monkeypatch, tmp_path):
    mod, _ = helper
    d = _daemons_dir(tmp_path)
    (Path(d) / "com.asiai.web.plist").write_bytes(b"OLD")
    _mock_fchown(mod, monkeypatch)
    mod._write_plist_atomic("com.asiai.web", b"NEW\n", daemons_dir=d)
    assert (Path(d) / "com.asiai.web.plist").read_bytes() == b"NEW\n"


def test_write_plist_atomic_refuses_group_writable_dir(helper, monkeypatch, tmp_path):
    mod, _ = helper
    d = _daemons_dir(tmp_path)
    os.chmod(d, 0o775)  # group-writable -> I0 runtime refusal
    _mock_fchown(mod, monkeypatch)
    with pytest.raises(mod._Refused):
        mod._write_plist_atomic("com.asiai.web", b"x", daemons_dir=d)


def test_write_plist_atomic_refuses_symlinked_dir(helper, monkeypatch, tmp_path):
    mod, _ = helper
    real = tmp_path / "real"
    real.mkdir()
    os.chmod(str(real), 0o755)
    link = tmp_path / "LaunchDaemons"
    link.symlink_to(real)
    _mock_fchown(mod, monkeypatch)
    with pytest.raises(OSError):  # O_DIRECTORY|O_NOFOLLOW
        mod._write_plist_atomic("com.asiai.web", b"x", daemons_dir=str(link))


def test_unlink_plist_removes_and_is_idempotent(helper, tmp_path):
    mod, _ = helper
    d = _daemons_dir(tmp_path)
    leaf = os.path.join(d, "com.asiai.web.plist")
    with open(leaf, "w") as fh:
        fh.write("x")
    mod._unlink_plist("com.asiai.web", daemons_dir=d)
    assert not os.path.exists(leaf)
    mod._unlink_plist("com.asiai.web", daemons_dir=d)  # idempotent: no error when absent


# --- handlers via main() (mock _require_root + _run) ------------------------


def test_install_daemon_order_and_bootstrap(helper, monkeypatch):
    """Order is: validate -> write -> precreate logs -> bootstrap (never bootstrap first)."""
    mod, audit = helper
    seq: list = []
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_binary", lambda b: b)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())

    def fake_write(label, xml, **_k):
        seq.append("write")
        return f"/Library/LaunchDaemons/{label}.plist"

    def fake_precreate(label, pw, **_k):
        seq.append("precreate")

    def fake_run(argv, *, check):
        seq.append(("run", tuple(argv)))

    monkeypatch.setattr(mod, "_write_plist_atomic", fake_write)
    monkeypatch.setattr(mod, "_precreate_log_leaves", fake_precreate)
    monkeypatch.setattr(mod, "_run", fake_run)

    rc = mod.main(
        [
            "install-daemon",
            "--label",
            "com.asiai.aux-1",
            "--binary",
            "/opt/homebrew/bin/llama-server",
            "--program-arg",
            "serve",
        ]
    )
    assert rc == mod.EXIT_OK
    assert seq[0] == "write"
    assert seq[1] == "precreate"
    assert seq[2] == (
        "run",
        ("/bin/launchctl", "bootstrap", "system", "/Library/LaunchDaemons/com.asiai.aux-1.plist"),
    )
    assert any(
        r["action"] == "install-daemon" and r["verdict"] == "accepted" for r in _read_audit(audit)
    )


@pytest.mark.parametrize(
    "action,expected",
    [
        ("start-daemon", ("/bin/launchctl", "kickstart", "-k", "system/com.asiai.aux-1")),
        ("stop-daemon", ("/bin/launchctl", "kill", "SIGTERM", "system/com.asiai.aux-1")),
        ("enable-daemon", ("/bin/launchctl", "enable", "system/com.asiai.aux-1")),
        ("disable-daemon", ("/bin/launchctl", "disable", "system/com.asiai.aux-1")),
    ],
)
def test_lifecycle_action_launchctl_argv(helper, monkeypatch, action, expected):
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    captured: dict = {}
    monkeypatch.setattr(
        mod, "_run", lambda argv, *, check: captured.setdefault("argv", tuple(argv))
    )
    rc = mod.main([action, "--label", "com.asiai.aux-1"])
    assert rc == mod.EXIT_OK
    assert captured["argv"] == expected
    assert any(r["verdict"] == "accepted" for r in _read_audit(audit))


def test_purge_runs_purge(helper, monkeypatch):
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    captured: dict = {}
    monkeypatch.setattr(
        mod, "_run", lambda argv, *, check: captured.setdefault("argv", tuple(argv))
    )
    rc = mod.main(["purge"])
    assert rc == mod.EXIT_OK
    assert captured["argv"] == ("/usr/sbin/purge",)
    assert any(r["action"] == "purge" and r["verdict"] == "accepted" for r in _read_audit(audit))


def test_uninstall_bootout_then_unlink(helper, monkeypatch):
    mod, _ = helper
    seq: list = []
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_run", lambda argv, *, check: seq.append(("run", tuple(argv), check)))
    monkeypatch.setattr(mod, "_unlink_plist", lambda label, **_k: seq.append(("unlink", label)))
    rc = mod.main(["uninstall-daemon", "--label", "com.asiai.aux-1"])
    assert rc == mod.EXIT_OK
    assert seq[0] == ("run", ("/bin/launchctl", "bootout", "system/com.asiai.aux-1"), False)
    assert seq[1] == ("unlink", "com.asiai.aux-1")


@pytest.mark.parametrize("action", ["disable-daemon", "uninstall-daemon", "stop-daemon"])
@pytest.mark.parametrize("label", ["com.asiai.web", "com.asiai.aisctl-serve"])
def test_reserved_label_refused(helper, monkeypatch, action, label):
    """I9: destructive actions on a fixed asiai service are hard-refused, no launchctl runs."""
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    ran: list = []
    monkeypatch.setattr(mod, "_run", lambda argv, *, check: ran.append(argv))
    monkeypatch.setattr(mod, "_unlink_plist", lambda *a, **k: ran.append(("unlink",)))
    rc = mod.main([action, "--label", label])
    assert rc == mod.EXIT_REFUSED
    assert not ran  # nothing privileged ran
    assert any(r["verdict"] == "refused" for r in _read_audit(audit))


@pytest.mark.parametrize("label", ["com.asiai.web", "com.asiai.aisctl-serve"])
def test_install_reserved_label_refused(helper, monkeypatch, label):
    """I9: install overwrites via renameat -> hijack; reserved labels are refused pre-write."""
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    did: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: did.append("write") or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: did.append("run"))
    rc = mod.main(
        [
            "install-daemon",
            "--label",
            label,
            "--binary",
            "/opt/homebrew/bin/llama-server",
            "--program-arg",
            "serve",
        ]
    )
    assert rc == mod.EXIT_REFUSED
    assert not did  # refused before any write/launchctl
    assert any(r["verdict"] == "refused" for r in _read_audit(audit))


def test_action_refused_when_not_root(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(mod.os, "geteuid", lambda: 1000)  # not root
    ran: list = []
    monkeypatch.setattr(mod, "_run", lambda argv, *, check: ran.append(argv))
    assert mod.main(["purge"]) == mod.EXIT_REFUSED
    assert not ran  # _require_root refused before any privileged op


@pytest.mark.parametrize(
    "kw", [{"nice": 99}, {"nice": -99}, {"timeout": -1}, {"throttle_interval": -5}]
)
def test_plist_refuses_out_of_range_timing(helper, kw):
    mod, _ = helper
    with pytest.raises(mod._Refused):
        _plist(mod, **kw)


def test_install_rolls_back_on_bootstrap_failure(helper, monkeypatch):
    """A failed bootstrap after the write must remove the plist (no stray reboot-load)."""
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_binary", lambda b: b)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(
        mod, "_write_plist_atomic", lambda label, xml, **_k: f"/Library/LaunchDaemons/{label}.plist"
    )
    monkeypatch.setattr(mod, "_precreate_log_leaves", lambda label, pw, **_k: None)
    unlinked: list = []
    monkeypatch.setattr(mod, "_unlink_plist", lambda label, **_k: unlinked.append(label))

    def fake_run(argv, *, check):
        if "bootstrap" in argv:
            raise mod.subprocess.CalledProcessError(1, argv)  # bootstrap fails
        # bootout (rollback) succeeds

    monkeypatch.setattr(mod, "_run", fake_run)
    rc = mod.main(
        [
            "install-daemon",
            "--label",
            "com.asiai.aux-1",
            "--binary",
            "/opt/homebrew/bin/llama-server",
            "--program-arg",
            "serve",
        ]
    )
    assert rc == mod.EXIT_INTERNAL
    assert unlinked == ["com.asiai.aux-1"]  # rolled back
    assert not any(r.get("verdict") == "accepted" for r in _read_audit(audit))  # no false accept


# ===========================================================================
# Reserved-service content-validated install (story 2.2). The action GENERATES
# the whole argv from _RESERVED_SERVICE_SPECS; the caller supplies only a service
# enum, a bounded port, and (web only) a host from a closed set.
# ===========================================================================


class _FakeStat:
    """Minimal stat-like object: _resolve_reserved_binary only reads st_mode/st_uid,
    and stat.S_ISLNK/S_ISREG are pure functions of the mode int."""

    def __init__(self, mode: int, uid: int) -> None:
        self.st_mode = mode
        self.st_uid = uid


def _make_reserved_binary(tmp_path, basename: str = "asiai"):
    """Build a clean ``<home>/.local/bin/<basename>`` chain owned by us, 0755, no symlink.

    Returns ``(home, binpath, owner_pw)`` where owner_pw.pw_dir == home and
    owner_pw.pw_uid == our uid (so the {root, owner} ownership check passes)."""
    home = tmp_path / "home"
    bindir = home / ".local" / "bin"
    bindir.mkdir(parents=True)
    binpath = bindir / basename
    binpath.write_text("#!/bin/sh\nexec true\n")
    for p in (home, home / ".local", bindir, binpath):
        p.chmod(0o755)
    pw = pwd.struct_passwd(("owner", "*", os.getuid(), os.getgid(), "owner", str(home), "/bin/zsh"))
    return home, binpath, pw


# --- _resolve_reserved_binary ----------------------------------------------


def test_resolve_reserved_binary_accepts_clean_chain(helper, tmp_path):
    mod, _ = helper
    _home, binpath, pw = _make_reserved_binary(tmp_path)
    assert mod._resolve_reserved_binary("asiai", pw) == str(binpath)


def test_resolve_reserved_binary_rejects_symlink_leaf(helper, tmp_path):
    mod, _ = helper
    _home, binpath, pw = _make_reserved_binary(tmp_path)
    binpath.unlink()
    real = tmp_path / "real_target"
    real.write_text("#!/bin/sh\n")
    binpath.symlink_to(real)  # leaf is now a symlink -> lateral-write redirection primitive
    with pytest.raises(mod._Refused, match="symlink"):
        mod._resolve_reserved_binary("asiai", pw)


def test_resolve_reserved_binary_rejects_group_writable_component(helper, tmp_path):
    mod, _ = helper
    home, _binpath, pw = _make_reserved_binary(tmp_path)
    (home / ".local" / "bin").chmod(0o775)  # group-writable dir in the chain
    with pytest.raises(mod._Refused, match="group/other-writable"):
        mod._resolve_reserved_binary("asiai", pw)


def test_resolve_reserved_binary_rejects_other_writable_leaf(helper, tmp_path):
    mod, _ = helper
    _home, binpath, pw = _make_reserved_binary(tmp_path)
    binpath.chmod(0o757)  # world-writable binary
    with pytest.raises(mod._Refused, match="group/other-writable"):
        mod._resolve_reserved_binary("asiai", pw)


def test_resolve_reserved_binary_rejects_foreign_owner(helper, monkeypatch, tmp_path):
    mod, _ = helper
    _home, binpath, pw = _make_reserved_binary(tmp_path)
    real_lstat = mod.os.lstat
    foreign = os.getuid() + 99999

    def fake_lstat(p):
        st = real_lstat(p)
        if str(p) == str(binpath):  # the binary itself is owned by a third party
            return _FakeStat(st.st_mode, foreign)
        return st

    monkeypatch.setattr(mod.os, "lstat", fake_lstat)
    with pytest.raises(mod._Refused, match="not owned by"):
        mod._resolve_reserved_binary("asiai", pw)


@pytest.mark.parametrize("bad", ["ais/ctl", "..", ".", "", "sub/dir/asiai"])
def test_resolve_reserved_binary_rejects_bad_basename(helper, tmp_path, bad):
    mod, _ = helper
    _home, _binpath, pw = _make_reserved_binary(tmp_path)
    with pytest.raises(mod._Refused):
        mod._resolve_reserved_binary(bad, pw)


@pytest.mark.parametrize("home", ["/var/empty", "/dev/null", "", "relative/home"])
def test_resolve_reserved_binary_rejects_non_home(helper, home):
    mod, _ = helper
    pw = pwd.struct_passwd(("svc", "*", os.getuid(), os.getgid(), "svc", home, "/usr/bin/false"))
    with pytest.raises(mod._Refused):
        mod._resolve_reserved_binary("asiai", pw)


def test_resolve_reserved_binary_rejects_missing(helper, tmp_path):
    mod, _ = helper
    _home, binpath, pw = _make_reserved_binary(tmp_path)
    binpath.unlink()  # nothing at ~/.local/bin/asiai
    with pytest.raises(mod._Refused, match="missing"):
        mod._resolve_reserved_binary("asiai", pw)


# --- _validate_host --------------------------------------------------------


def test_validate_host_web_default_is_lan(helper):
    mod, _ = helper
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    assert mod._validate_host(spec, None) == "0.0.0.0"
    assert mod._validate_host(spec, "127.0.0.1") == "127.0.0.1"


def test_validate_host_web_rejects_outside_set(helper):
    mod, _ = helper
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    with pytest.raises(mod._Refused, match="not allowed"):
        mod._validate_host(spec, "10.0.0.1")


def test_validate_host_serve_pinned_loopback(helper):
    mod, _ = helper
    spec = mod._RESERVED_SERVICE_SPECS["aisctl-serve"]
    assert mod._validate_host(spec, None) == "127.0.0.1"
    with pytest.raises(mod._Refused, match="not allowed"):
        mod._validate_host(spec, "0.0.0.0")  # loopback companion can never be LAN-facing


# --- _resolve_reserved_runas -----------------------------------------------


def test_resolve_reserved_runas_dedicated_fail_closed_when_absent(helper, monkeypatch):
    mod, _ = helper

    def no_such_user(name):
        raise KeyError(name)

    monkeypatch.setattr(mod.pwd, "getpwnam", no_such_user)
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    with pytest.raises(mod._Refused, match="dedicated non-admin account"):
        mod._resolve_reserved_runas(spec)  # never an admin fallback


def test_resolve_reserved_runas_dedicated_rejects_root(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(
        mod.pwd,
        "getpwnam",
        lambda name: pwd.struct_passwd((name, "*", 0, 0, name, "/var/empty", "/usr/bin/false")),
    )
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    with pytest.raises(mod._Refused, match="must not be root"):
        mod._resolve_reserved_runas(spec)


def test_resolve_reserved_runas_invoker_uses_sudo_uid(helper, monkeypatch):
    mod, _ = helper
    sentinel = _real_pw()
    monkeypatch.setattr(mod, "_resolve_user", lambda u: sentinel)
    spec = mod._RESERVED_SERVICE_SPECS["aisctl-serve"]
    assert mod._resolve_reserved_runas(spec) is sentinel


# --- install-reserved-service (integration) --------------------------------


def _run_reserved_install(
    mod, monkeypatch, service, extra_argv=(), *, binary="/Users/owner/.local/bin/asiai"
):
    captured: dict = {}
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_runas", lambda spec: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_binary", lambda basename, pw: binary)

    def fake_write(label, xml, **_k):
        captured["label"] = label
        captured["plist"] = plistlib.loads(xml)
        return f"/Library/LaunchDaemons/{label}.plist"

    monkeypatch.setattr(mod, "_write_plist_atomic", fake_write)
    monkeypatch.setattr(mod, "_precreate_log_leaves", lambda *a, **k: None)
    runs: list = []
    monkeypatch.setattr(mod, "_run", lambda argv, *, check: runs.append(tuple(argv)))
    captured["rc"] = mod.main(["install-reserved-service", "--service", service, *extra_argv])
    captured["runs"] = runs
    return captured


def test_install_reserved_web_generates_full_argv(helper, monkeypatch):
    mod, audit = helper
    binary = "/Users/owner/.local/bin/asiai"
    cap = _run_reserved_install(mod, monkeypatch, "asiai-web", binary=binary)
    assert cap["rc"] == mod.EXIT_OK
    plist = cap["plist"]
    assert plist["Label"] == "com.asiai.web"
    assert plist["ProgramArguments"] == [binary, "web", "--host", "0.0.0.0", "--port", "8899"]
    assert plist["UserName"] == _real_pw().pw_name
    # D2/D3 hardening: GroupName + WorkingDirectory dropped; ThrottleInterval present.
    assert "GroupName" not in plist
    assert "WorkingDirectory" not in plist
    assert plist["ThrottleInterval"] == mod._RESERVED_THROTTLE_INTERVAL
    # PATH must NOT carry the user-writable ~/.local/bin.
    assert ".local/bin" not in plist["EnvironmentVariables"]["PATH"]
    # bootstrap (not load), and audited accepted.
    assert cap["runs"] == [
        ("/bin/launchctl", "bootstrap", "system", "/Library/LaunchDaemons/com.asiai.web.plist")
    ]
    assert any(
        r["action"] == "install-reserved-service" and r["verdict"] == "accepted"
        for r in _read_audit(audit)
    )


def test_install_reserved_serve_omits_host(helper, monkeypatch):
    mod, _ = helper
    binary = "/Users/owner/.local/bin/aisctl"
    cap = _run_reserved_install(mod, monkeypatch, "aisctl-serve", binary=binary)
    assert cap["rc"] == mod.EXIT_OK
    # serve binds loopback itself -> NO --host emitted.
    assert cap["plist"]["ProgramArguments"] == [binary, "serve", "--port", "8898"]
    assert cap["plist"]["Label"] == "com.asiai.aisctl-serve"


def test_install_reserved_custom_port(helper, monkeypatch):
    mod, _ = helper
    cap = _run_reserved_install(mod, monkeypatch, "asiai-web", ["--port", "9000"])
    assert cap["plist"]["ProgramArguments"][-1] == "9000"


def test_install_reserved_rejects_extra_program_arg(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    # argparse on the closed surface has no --program-arg -> parse error -> EXIT_REFUSED.
    with pytest.raises(SystemExit) as exc:
        mod.main(["install-reserved-service", "--service", "asiai-web", "--program-arg", "x"])
    assert exc.value.code == mod.EXIT_REFUSED


@pytest.mark.parametrize("flag", ["--label", "--binary", "--env"])
def test_install_reserved_rejects_free_form_flags(helper, flag):
    mod, _ = helper
    with pytest.raises(SystemExit) as exc:
        mod.main(["install-reserved-service", "--service", "asiai-web", flag, "x"])
    assert exc.value.code == mod.EXIT_REFUSED


def test_install_reserved_web_bad_host_refused(helper, monkeypatch):
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_runas", lambda spec: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_binary", lambda b, pw: "/Users/o/.local/bin/asiai")
    wrote: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: wrote.append(1) or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: wrote.append("run"))
    rc = mod.main(["install-reserved-service", "--service", "asiai-web", "--host", "10.0.0.1"])
    assert rc == mod.EXIT_REFUSED
    assert not wrote  # refused before any write/launchctl
    assert any(r["verdict"] == "refused" for r in _read_audit(audit))


def test_install_reserved_serve_lan_host_refused(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_runas", lambda spec: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_binary", lambda b, pw: "/Users/o/.local/bin/aisctl")
    wrote: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: wrote.append(1) or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: wrote.append("run"))
    rc = mod.main(["install-reserved-service", "--service", "aisctl-serve", "--host", "0.0.0.0"])
    assert rc == mod.EXIT_REFUSED
    assert not wrote


def test_install_reserved_unknown_service_refused(helper):
    mod, _ = helper
    with pytest.raises(SystemExit) as exc:
        mod.main(["install-reserved-service", "--service", "com.asiai.evil"])
    assert exc.value.code == mod.EXIT_REFUSED


def test_install_reserved_fails_closed_without_dedicated_account(helper, monkeypatch):
    mod, _ = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(mod.pwd, "getpwnam", lambda name: (_ for _ in ()).throw(KeyError(name)))
    wrote: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: wrote.append(1) or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: wrote.append("run"))
    rc = mod.main(["install-reserved-service", "--service", "asiai-web"])
    assert rc == mod.EXIT_REFUSED
    assert not wrote


# --- uninstall-reserved-service --------------------------------------------


def test_uninstall_reserved_service_bootout_then_unlink(helper, monkeypatch):
    mod, audit = helper
    seq: list = []
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_run", lambda argv, *, check: seq.append(("run", tuple(argv), check)))
    monkeypatch.setattr(mod, "_unlink_plist", lambda label, **_k: seq.append(("unlink", label)))
    rc = mod.main(["uninstall-reserved-service", "--service", "asiai-web"])
    assert rc == mod.EXIT_OK
    assert seq[0] == ("run", ("/bin/launchctl", "bootout", "system/com.asiai.web"), False)
    assert seq[1] == ("unlink", "com.asiai.web")
    assert any(
        r["action"] == "uninstall-reserved-service" and r["verdict"] == "accepted"
        for r in _read_audit(audit)
    )


# --- invariants: generic path unchanged, PLIST_KEYS frozen, PATH flag -------


def test_generic_install_daemon_still_refuses_reserved_labels(helper, monkeypatch):
    """The content-validated path must NOT weaken the I9 blanket refusal on install-daemon."""
    mod, _ = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    did: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: did.append(1) or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: did.append("run"))
    rc = mod.main(
        ["install-daemon", "--label", "com.asiai.web", "--binary", "/opt/homebrew/bin/llama-server"]
    )
    assert rc == mod.EXIT_REFUSED
    assert not did


def test_plist_keys_allowlist_frozen():
    """A drift in the emitted-key allowlist must be a deliberate, reviewed change (no GroupName,
    no WorkingDirectory, no RootDirectory)."""
    mod = _load_helper()
    assert mod._PLIST_KEYS == (
        "Label",
        "ProgramArguments",
        "UserName",
        "EnvironmentVariables",
        "StandardOutPath",
        "StandardErrorPath",
        "RunAtLoad",
        "KeepAlive",
        "ThrottleInterval",
        "TimeOut",
        "Nice",
    )


def test_build_plist_omits_binary_dir_when_flag_false(helper):
    mod, _ = helper
    plist = mod._build_plist_dict(
        label="com.asiai.web",
        binary="/Users/owner/.local/bin/asiai",
        user_pw=_real_pw(),
        include_binary_dir_in_path=False,
    )
    assert ".local/bin" not in plist["EnvironmentVariables"]["PATH"]


def test_build_plist_includes_binary_dir_by_default(helper, monkeypatch, tmp_path):
    mod, _ = helper
    bindir = tmp_path / "bin"
    bindir.mkdir()
    target = bindir / "llama-server"
    target.write_text("x")
    monkeypatch.setattr(mod, "_BINARY_PREFIXES", (str(bindir) + "/",))
    plist = mod._build_plist_dict(
        label="com.asiai.aux-1",
        binary=str(target),
        user_pw=_real_pw(),
    )
    assert str(bindir) in plist["EnvironmentVariables"]["PATH"]  # engines keep binary_dir


# --- post-review F6/F3/F5 fixes (auto-review wf_62e97817-146) -----------------


def test_resolve_reserved_binary_rejects_intermediate_foreign_owner(helper, monkeypatch, tmp_path):
    """F6: the per-component ownership walk must refuse a third-party-owned INTERMEDIATE
    component (~/.local), not only the leaf binary."""
    mod, _ = helper
    home, _binpath, pw = _make_reserved_binary(tmp_path)
    intermediate = str(home / ".local")
    real_lstat = mod.os.lstat
    foreign = os.getuid() + 99999

    def fake_lstat(p):
        st = real_lstat(p)
        if str(p) == intermediate:  # a parent dir owned by someone else
            return _FakeStat(st.st_mode, foreign)
        return st

    monkeypatch.setattr(mod.os, "lstat", fake_lstat)
    with pytest.raises(mod._Refused, match="not owned by"):
        mod._resolve_reserved_binary("asiai", pw)


@pytest.mark.parametrize("port", ["80", "1023", "70000", "0"])
def test_install_reserved_rejects_out_of_range_port(helper, monkeypatch, port):
    """F6: a privileged/out-of-range --port is refused before any plist write or launchctl."""
    mod, audit = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_runas", lambda spec: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_binary", lambda b, pw: "/Users/o/.local/bin/asiai")
    wrote: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: wrote.append(1) or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: wrote.append("run"))
    rc = mod.main(["install-reserved-service", "--service", "asiai-web", "--port", port])
    assert rc == mod.EXIT_REFUSED
    assert not wrote
    assert any(r["verdict"] == "refused" for r in _read_audit(audit))


def test_install_reserved_serve_rejects_host_flag(helper, monkeypatch):
    """F5: --host is a web-only knob; the loopback companion refuses it outright (even
    127.0.0.1) rather than silently dropping it."""
    mod, _ = helper
    monkeypatch.setattr(mod, "_require_root", lambda: None)
    monkeypatch.setattr(mod, "_resolve_user", lambda u: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_runas", lambda spec: _real_pw())
    monkeypatch.setattr(mod, "_resolve_reserved_binary", lambda b, pw: "/Users/o/.local/bin/aisctl")
    wrote: list = []
    monkeypatch.setattr(mod, "_write_plist_atomic", lambda *a, **k: wrote.append(1) or "/x")
    monkeypatch.setattr(mod, "_run", lambda *a, **k: wrote.append("run"))
    rc = mod.main(["install-reserved-service", "--service", "aisctl-serve", "--host", "127.0.0.1"])
    assert rc == mod.EXIT_REFUSED
    assert not wrote


def _fake_pwnam(name: str, uid: int, gid: int):
    return lambda n: pwd.struct_passwd((n, "*", uid, gid, n, "/var/empty", "/usr/bin/false"))


def _fake_admin_group(gid: int = 80, members=()):
    return lambda g: grp.struct_group((g, "*", gid, list(members)))


def test_resolve_reserved_runas_dedicated_rejects_admin_primary_gid(helper, monkeypatch):
    """F3: a dedicated account whose PRIMARY gid is admin (80) is refused at the choke point."""
    mod, _ = helper
    monkeypatch.setattr(mod.pwd, "getpwnam", _fake_pwnam("_aisweb", 460, 80))
    monkeypatch.setattr(mod.grp, "getgrnam", _fake_admin_group(gid=80, members=[]))
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    with pytest.raises(mod._Refused, match="must not be admin"):
        mod._resolve_reserved_runas(spec)


def test_resolve_reserved_runas_dedicated_rejects_admin_supplementary(helper, monkeypatch):
    """F3: admin membership via the supplementary member list is also refused."""
    mod, _ = helper
    monkeypatch.setattr(mod.pwd, "getpwnam", _fake_pwnam("_aisweb", 460, 20))  # primary staff
    monkeypatch.setattr(mod.grp, "getgrnam", _fake_admin_group(gid=80, members=["_aisweb"]))
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    with pytest.raises(mod._Refused, match="must not be admin"):
        mod._resolve_reserved_runas(spec)


def test_resolve_reserved_runas_dedicated_accepts_nonadmin(helper, monkeypatch):
    """F3: a non-root, non-admin dedicated account is accepted (the happy path)."""
    mod, _ = helper
    monkeypatch.setattr(mod.pwd, "getpwnam", _fake_pwnam("_aisweb", 460, 20))
    monkeypatch.setattr(mod.grp, "getgrnam", _fake_admin_group(gid=80, members=[]))
    spec = mod._RESERVED_SERVICE_SPECS["asiai-web"]
    pw = mod._resolve_reserved_runas(spec)
    assert pw.pw_name == "_aisweb" and pw.pw_uid == 460


def test_is_admin_fail_closed_when_group_unresolvable(helper, monkeypatch):
    """F3: if the admin group cannot be resolved we cannot prove non-admin -> refuse."""
    mod, _ = helper

    def boom(_g):
        raise KeyError(_g)

    monkeypatch.setattr(mod.grp, "getgrnam", boom)
    pw = pwd.struct_passwd(("x", "*", 460, 20, "x", "/var/empty", "/usr/bin/false"))
    with pytest.raises(mod._Refused, match="cannot resolve"):
        mod._is_admin(pw)


# ---------------------------------------------------------------------------
# SA audit #3: missing log dir is a CLEAR refusal, not an opaque internal error
# ---------------------------------------------------------------------------


def test_precreate_log_leaves_missing_dir_refused_with_bootstrap_hint(helper, tmp_path):
    """ENOENT on the log dir → _Refused naming the dir and `aisctl bootstrap` (finding #3).
    Used to bubble as a bare FileNotFoundError → generic EXIT_INTERNAL ('internal error')."""
    mod, _ = helper
    pw = pwd.getpwuid(os.getuid())
    with pytest.raises(mod._Refused, match="aisctl bootstrap"):
        mod._precreate_log_leaves("com.asiai.test", pw, log_dir=str(tmp_path / "absent"))


def test_precreate_log_leaves_notadir_refused(helper, tmp_path):
    """ENOTDIR (a FILE at the log-dir path) also gets the clear refusal, same narrow catch."""
    mod, _ = helper
    pw = pwd.getpwuid(os.getuid())
    not_a_dir = tmp_path / "file"
    not_a_dir.write_text("x")
    with pytest.raises(mod._Refused, match="log dir missing"):
        mod._precreate_log_leaves("com.asiai.test", pw, log_dir=str(not_a_dir / "leaf"))


# ---------------------------------------------------------------------------
# SA audit #2 (degraded path): _audit O_CREAT mode is 0640, never wider
# ---------------------------------------------------------------------------


def test_audit_creates_file_0640(helper):
    """The degraded re-create (log deleted): group-readable 0640, no group/other write.
    The canonical root:admin group is set by the bootstrap, not here (hot path never chowns)."""
    mod, audit = helper
    old_umask = os.umask(0o022)
    try:
        mod._audit("purge", "accepted")
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(os.stat(audit).st_mode)
    assert mode == 0o640
