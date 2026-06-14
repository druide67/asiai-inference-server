#!/usr/bin/python3 -I
"""asiai-priv — privileged helper for asiai-inference-server (foundation, story 1.1).

Root-owned helper invoked through a single ``sudoers`` NOPASSWD rule. It runs under
an isolated system interpreter (``python3 -I``): no ``PYTHON*`` environment, no user
site-packages, stdlib only, never a shell. Because of ``-I`` it cannot import
``ais_core`` — everything here is self-contained.

This module is the FOUNDATION only: the closed action CLI, default-deny dispatch, and the
append-only audit log. The privileged operations themselves are stubs that raise
``NotImplementedError``; they land in later stories (parameter validation 1.2, plist
generation 1.3, lifecycle actions 1.4, sudoers generator 1.5).

Security invariants realised here:
  * I6 — append-only, root-owned audit log, opened ``O_NOFOLLOW``, fail-open + syslog.
  * I7 — refuse by default: any unknown or missing action is refused and logged.
  * NFR10 — ``subprocess`` is always called with an argument list, never ``shell=True``.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import syslog
import time

# Audit log: root-owned, OUTSIDE any home directory (I6). Hardcoded on purpose: the
# tests monkeypatch this module attribute, but production runs under ``-I`` so no
# external override (env, sys.path) is possible — the hardcoded value is the prod value.
AUDIT_LOG = "/Library/Logs/asiai/asiai-priv-audit.log"

# Stable exit-code contract.
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_REFUSED = 2

# Closed allowlist of actions. Handlers are stubs in story 1.1.
_ACTIONS: tuple[str, ...] = (
    "install-daemon",
    "uninstall-daemon",
    "start-daemon",
    "stop-daemon",
    "enable-daemon",
    "disable-daemon",
    "purge",
)


def _syslog_fallback(action: str | None, verdict: str, reason: str, exc: OSError) -> None:
    """Emit one syslog line when the audit file cannot be written. Never raises."""
    try:
        syslog.openlog("asiai-priv", syslog.LOG_PID, syslog.LOG_AUTHPRIV)
        syslog.syslog(
            syslog.LOG_WARNING,
            f"audit fallback: action={action} verdict={verdict} reason={reason} err={exc}",
        )
        syslog.closelog()
    except OSError:
        pass  # logging must never block the operation


def _audit(action: str | None, verdict: str, reason: str = "", **fields: object) -> None:
    """Append one JSON record to the audit log. Never raises (fail-open + syslog).

    The file is opened ``O_WRONLY|O_APPEND|O_CREAT|O_NOFOLLOW`` so a symlink
    pre-positioned at the log path is refused by the kernel (``ELOOP``). An
    ``fstat`` on the resulting fd then refuses a *regular* file pre-positioned by
    another owner (or made group/other-writable) — so audit integrity does not
    depend on the directory hardening that story 2.1 will add (I6). Any failure —
    missing dir, ``ENOSPC``, symlink, failed integrity check — is downgraded to
    ``syslog`` and the operation is never blocked.
    """
    record: dict[str, object] = {
        "ts": time.time(),
        "action": action,
        "verdict": verdict,
        "reason": reason,
        "pid": os.getpid(),
    }
    record.update(fields)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    try:
        fd = os.open(AUDIT_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        _syslog_fallback(action, verdict, reason, exc)
        return
    try:
        st = os.fstat(fd)
        # O_NOFOLLOW already blocked a symlink; this blocks a *regular* file
        # pre-positioned by someone else, or made group/other-writable. The
        # owner must be our effective uid (root in production). Check is on the
        # fd, not the path (anti-TOCTOU). Do not depend on story 2.1 hardening.
        if not (
            stat.S_ISREG(st.st_mode)
            and st.st_uid == os.geteuid()
            and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            _syslog_fallback(
                action,
                verdict,
                reason,
                OSError(f"audit integrity check failed: uid={st.st_uid} mode={st.st_mode:#o}"),
            )
            return
        os.write(fd, line.encode("utf-8"))
    except OSError as exc:
        _syslog_fallback(action, verdict, reason, exc)
    finally:
        os.close(fd)


def _run(argv: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    """Run a system command as an argument list. Never uses a shell (NFR10).

    Posed in the foundation so every later story routes privileged commands through one
    audited, shell-free entry point.
    """
    return subprocess.run(argv, check=check, capture_output=True, text=True)


def _make_stub(action: str):
    """Return a handler that audits the attempt then refuses to pretend it worked."""

    def handler(_args: argparse.Namespace) -> None:
        _audit(action, "error", reason="not implemented: foundation story 1.1")
        raise NotImplementedError(f"{action}: implemented in a later story")

    return handler


# Dispatch table. Closed set, default-deny everywhere else (I7).
_DISPATCH = {action: _make_stub(action) for action in _ACTIONS}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asiai-priv",
        description="Privileged helper for asiai-inference-server.",
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="action", required=True)
    for action in _ACTIONS:
        sub.add_parser(action, allow_abbrev=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse one action and dispatch it. Returns the process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 0 on --help (not a refusal) and 2 on a parse error.
        if exc.code == 0:
            raise
        # action=None: nothing was validated. The raw input is preserved in `argv`
        # so the `action` field only ever carries a known action or null.
        _audit(None, "refused", reason="unknown or missing action", argv=argv)
        raise SystemExit(EXIT_REFUSED) from exc

    handler = _DISPATCH.get(args.action)
    if handler is None:  # defence-in-depth: argparse already restricts the choices
        _audit(args.action, "refused", reason="no handler", argv=argv)
        return EXIT_REFUSED
    try:
        handler(args)
    except NotImplementedError:
        return EXIT_INTERNAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
