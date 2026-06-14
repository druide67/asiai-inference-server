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
