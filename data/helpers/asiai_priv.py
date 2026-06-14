#!/usr/bin/python3 -I
"""asiai-priv — privileged helper for asiai-inference-server (foundation, story 1.1).

Root-owned helper invoked through a single ``sudoers`` NOPASSWD rule. It runs under
an isolated system interpreter (``python3 -I``): no ``PYTHON*`` environment, no user
site-packages, stdlib only, never a shell. Because of ``-I`` it cannot import
``ais_core`` — everything here is self-contained.

This module holds the foundation (1.1), strict parameter validation (1.2), and the plist
generator (1.3). The lifecycle operations themselves are still stubs that raise
``NotImplementedError``; the real actions (1.4) — which wire the argparse options and call
the validators / generator below — land in a later story.

Security invariants realised here:
  * I1 — generate-don't-validate: the helper BUILDS the plist from bounded fields with a
    CLOSED key allowlist; it never accepts a caller-supplied plist.
  * I2 — the daemon run-as account is forced non-root, decided on the resolved ``pwd``
    object (``pw_uid``), derived from ``SUDO_UID``, never ``$USER`` and never a root fallback.
  * I3 — ``DYLD_*``/``LD_*`` environment keys are rejected; dangerous launchd keys
    (``RootDirectory``, ``GroupName``, ``Sockets``, ...) are never emitted.
  * I5 — the daemon binary must live under a hardcoded prefix allowlist (secondary defence).
  * I6 — append-only, root-owned audit log, opened ``O_NOFOLLOW``, fail-open + syslog.
  * I7 — refuse by default: any unknown or missing action is refused and logged.
  * I8 — log leaves are root-created (``O_NOFOLLOW``) then chowned to the daemon account.
  * I9 — labels are constrained to ``com.asiai.<name>``.
  * NFR10 — ``subprocess`` is always called with an argument list, never ``shell=True``.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import pwd
import re
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

# Keep an audit record small (one short O_APPEND write) so a concurrent writer is
# unlikely to interleave; oversized records are truncated (see ``_audit``). Best-effort
# on a regular file — not a hard POSIX atomicity guarantee. 4096 ~ PIPE_BUF as a yardstick.
_AUDIT_MAX_BYTES = 3072
_AUDIT_FIELD_MAX = 512


class _Refused(Exception):
    """A parameter failed validation: hostile or non-conforming input.

    Named ``_Refused`` (not ``*Error``) on purpose: this is the expected refusal
    verdict, not an internal error.

    Raised by the validators below and caught in ``main`` (audited ``refused`` →
    ``EXIT_REFUSED``). Distinct from ``NotImplementedError`` (a 1.1 stub) and from a
    bare ``Exception`` (an internal bug — audited ``error`` → ``EXIT_INTERNAL``).
    """


# I9 / label: only ``com.asiai.<name>``. Anchored implicitly by ``fullmatch`` (no ``^``/``$``
# — those would let a trailing newline slip through ``$``; fullmatch requires the whole string).
_LABEL_RE = re.compile(r"com\.asiai\.[a-z0-9-]+")

# I3 / environment: keys that let a non-root daemon load attacker code are forbidden.
_FORBIDDEN_ENV_PREFIXES = ("DYLD_", "LD_")
_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# I5 / binary: the daemon binary (``ProgramArguments[0]``) must resolve under one of these
# hardcoded system prefixes. Calibrated on the real engine manifests (llama-server, ollama,
# mlx-lm-server-start, llama-server-turboquant all live here). SECONDARY defence: the binary
# runs under the non-root ``UserName`` (I2), which is the actual privilege barrier.
_BINARY_PREFIXES: tuple[str, ...] = ("/opt/homebrew/bin/", "/usr/local/bin/")

# I1 / plist: CLOSED allowlist of emitted keys. Anything outside this set is NEVER emitted —
# in particular the dangerous keys RootDirectory (root pivot), GroupName (privileged group),
# Sockets / MachServices / LaunchEvents (activation surface) are absent by construction (I3).
# StartInterval is intentionally absent: engine daemons are long-running servers (KeepAlive),
# never the periodic run-and-exit pattern.
_PLIST_KEYS: tuple[str, ...] = (
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

# I8 / logs: root-owned, outside any home. Leaves are pre-created root then chowned to the
# daemon account (launchd opens Standard*Path AFTER dropping to UserName — VERIF-4).
_LOG_DIR = "/Library/Logs/asiai"

# KeepAlive: a bool, or a dict whose subkeys are confined to this closed set (I1) — keeps
# the only caller-supplied sub-tree bounded (no PathState/OtherJobEnabled activation tricks).
_KEEP_ALIVE_KEYS: tuple[str, ...] = ("Crashed", "SuccessfulExit")

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


def _trunc(value: object, limit: int = _AUDIT_FIELD_MAX) -> object:
    """Bound a field for the audit line. Strings are clipped; lists element-wise."""
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    if isinstance(value, (list, tuple)):
        return [_trunc(item, limit) for item in value]
    return value


def _write_all(fd: int, data: bytes) -> None:
    """Write every byte, looping over partial writes (POSIX ``write`` may be short).

    A 0-byte return is treated as an error (raised, then caught by the caller's
    fail-open path) rather than spinning a privileged process forever.
    """
    offset = 0
    while offset < len(data):
        written = os.write(fd, data[offset:])
        if written == 0:
            raise OSError("audit write returned 0")
        offset += written


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
        "reason": _trunc(reason),
        "pid": os.getpid(),
    }
    for key, value in fields.items():
        record[key] = _trunc(value)
    data = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
    if len(data) > _AUDIT_MAX_BYTES:
        # Last resort: drop the variable fields and keep a bounded skeleton so the
        # O_APPEND write stays atomic (< PIPE_BUF) even with a pathological argv.
        skeleton = {
            "ts": record["ts"],
            "action": _trunc(action, 256),
            "verdict": _trunc(verdict, 256),
            "reason": _trunc(reason, 256),
            "pid": record["pid"],
            "truncated": True,
        }
        data = (json.dumps(skeleton, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
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
        _write_all(fd, data)
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


# ---------------------------------------------------------------------------
# Parameter validation (story 1.2). Each validator canonicalises its input and
# raises ``_Refused`` on anything not explicitly allowed. They are pure and
# unit-tested directly; the argparse wiring that feeds them lands in 1.3/1.4.
# ---------------------------------------------------------------------------


def _require_root() -> None:
    """Refuse a privileged action unless running as root (euid 0).

    Posed here; wired at the start of the real lifecycle actions in story 1.4 (the
    1.1 stubs raise ``NotImplementedError`` and need no privilege).
    """
    if os.geteuid() != 0:
        raise _Refused("must run as root (euid 0)")


def _validate_label(label: str) -> str:
    """I9: accept only ``com.asiai.<name>`` (lowercase, digits, hyphen)."""
    if not _LABEL_RE.fullmatch(label):
        raise _Refused(f"invalid label: {label!r}")
    return label


def _resolve_user(requested: str | None) -> pwd.struct_passwd:
    """I2: resolve the daemon's run-as account to a non-root ``pwd`` object.

    The decision is made on the resolved object (``pw_uid``), never on the
    caller-supplied string: ``getpwnam`` is case-insensitive on macOS, so
    ``"Root"``/``"ROOT"`` resolve to uid 0 and must be refused. With no explicit
    request the identity is derived from ``SUDO_UID`` (the original invoker) —
    never ``$USER`` and never ``getpwuid(getuid())`` (root under sudo) — with no
    root fallback.
    """
    if requested is not None:
        try:
            pw = pwd.getpwnam(requested)
        except KeyError:
            raise _Refused(f"unknown user: {requested!r}") from None
    else:
        sudo_uid = os.environ.get("SUDO_UID")
        # isascii() rules out non-ASCII codepoints that str.isdigit() accepts but int()
        # then rejects (e.g. U+00B2 'SUPERSCRIPT TWO') or folds ambiguously (Arabic-Indic
        # digits). sudo always sets SUDO_UID to the invoker's ASCII decimal real uid.
        if not (sudo_uid and sudo_uid.isascii() and sudo_uid.isdigit()):
            raise _Refused(f"SUDO_UID unset or non-canonical: {sudo_uid!r}")
        try:
            pw = pwd.getpwuid(int(sudo_uid))
        except (KeyError, OverflowError):
            raise _Refused(f"unknown SUDO_UID: {sudo_uid}") from None
    if pw.pw_uid == 0:
        raise _Refused(f"run-as user must not be root (uid 0): {pw.pw_name!r}")
    return pw


def _is_within(path: str, prefix: str) -> bool:
    """True if ``path`` is ``prefix`` itself or strictly under it (component-wise).

    A degenerate empty/root prefix returns False (refused) rather than matching every
    absolute path — otherwise an account with home ``/`` would void the residence check.
    """
    prefix = prefix.rstrip(os.sep)
    if not prefix:
        return False
    return path == prefix or path.startswith(prefix + os.sep)


def _resolve_binary(path: str) -> str:
    """I5: canonicalise the daemon binary and require it under a hardcoded prefix.

    The final component must not be a symlink (``lstat``, anti-TOCTOU); ``realpath``
    then resolves parent symlinks before the prefix check. Secondary defence: the
    binary runs under the non-root ``UserName`` (I2).
    """
    if "\x00" in path:
        raise _Refused("NUL byte in binary path")
    if not path.startswith("/"):
        raise _Refused(f"binary path must be absolute: {path!r}")
    try:
        st = os.lstat(path)
    except OSError as exc:
        raise _Refused(f"binary not found: {path!r} ({exc.strerror})") from None
    if stat.S_ISLNK(st.st_mode):
        raise _Refused(f"binary must not be a symlink: {path!r}")
    real = os.path.realpath(path)
    if not any(_is_within(real, prefix) for prefix in _BINARY_PREFIXES):
        raise _Refused(f"binary outside allowlisted prefixes: {real!r}")
    return real


def _validate_env(pairs: list[str]) -> dict[str, str]:
    """I3: parse ``KEY=VALUE`` pairs, reject ``DYLD_*``/``LD_*`` and bad keys/values."""
    env: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise _Refused(f"malformed --env (expected KEY=VALUE): {pair!r}")
        if not _ENV_KEY_RE.fullmatch(key):
            raise _Refused(f"invalid env key: {key!r}")
        if key.startswith(_FORBIDDEN_ENV_PREFIXES):
            raise _Refused(f"forbidden env key: {key!r}")
        if any(c in value for c in ("\x00", "\n", "\r")):
            raise _Refused(f"control character in env value for {key!r}")
        env[key] = value
    return env


def _validate_port(value: str) -> int:
    """Accept an integer port in the unprivileged range ``[1024, 65535]``."""
    try:
        port = int(value)
    except (TypeError, ValueError):
        raise _Refused(f"port not an integer: {value!r}") from None
    if not 1024 <= port <= 65535:
        raise _Refused(f"port out of range [1024,65535]: {port}")
    return port


def _resolve_user_path(raw: str, home: str) -> str:
    """Resolve a model/template/mmproj path under the target account's home.

    ``~`` expands to ``home`` (never ``$HOME``/root); the final component must not
    be a symlink; ``realpath`` then resolves parent symlinks and ``..`` before the
    residence check, closing traversal.
    """
    if "\x00" in raw:
        raise _Refused("NUL byte in path")
    if raw == "~":
        raw = home
    elif raw.startswith("~/"):
        raw = home.rstrip(os.sep) + raw[1:]
    if not raw.startswith("/"):
        raise _Refused(f"path must be absolute: {raw!r}")
    try:
        st = os.lstat(raw)
    except OSError as exc:
        raise _Refused(f"path not found: {raw!r} ({exc.strerror})") from None
    if stat.S_ISLNK(st.st_mode):
        raise _Refused(f"path must not be a symlink: {raw!r}")
    real = os.path.realpath(raw)
    if not _is_within(real, home.rstrip(os.sep)):
        raise _Refused(f"path escapes target home: {real!r}")
    return real


# ---------------------------------------------------------------------------
# Plist generation (story 1.3) — generate-don't-validate (I1). The helper never
# receives a plist; it BUILDS one from bounded fields. Dangerous keys cannot be
# emitted because they are not in ``_PLIST_KEYS``. The install handler that wires
# argparse and writes the file lands in story 1.4.
# ---------------------------------------------------------------------------


def _validate_keep_alive(value: bool | dict) -> bool | dict:
    """Constrain KeepAlive to a bool or a dict of allowlisted boolean subkeys (I1).

    Stops a caller smuggling launchd activation directives (``PathState``,
    ``OtherJobEnabled``, ...) through the only sub-tree the generator passes down.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        bad = set(value) - set(_KEEP_ALIVE_KEYS)
        if bad:
            raise _Refused(f"KeepAlive subkeys not allowed: {sorted(bad)}")
        if not all(isinstance(v, bool) for v in value.values()):
            raise _Refused("KeepAlive subvalues must be booleans")
        return value
    raise _Refused(f"KeepAlive must be bool or dict, got {type(value).__name__}")


def _build_plist_dict(
    *,
    label: str,
    binary: str,
    user_pw: pwd.struct_passwd,
    program_args: list[str] | None = None,
    env: dict[str, str] | None = None,
    run_at_load: bool = True,
    keep_alive: bool | dict | None = None,
    throttle_interval: int | None = None,
    timeout: int | None = None,
    nice: int | None = None,
) -> dict:
    """I1/I2/I3/I8: build a daemon plist from bounded inputs (never a caller-supplied one).

    Only ``_PLIST_KEYS`` are emitted; the dangerous keys (``RootDirectory``,
    ``GroupName``, ``Sockets``, ``MachServices``, ``LaunchEvents``) are absent by
    construction. ``UserName`` is forced to the validated non-root account and root is
    refused; ``Standard*Path`` are forced under ``/Library/Logs/asiai``; ``HOME``/``PATH``
    are forced and ``DYLD_*``/``LD_*`` excluded even if the caller supplied them.
    """
    _validate_label(label)  # the label composes the log path — re-check, anti-poisoning

    # I2 rampart, self-defending: launchd resolves UserName by NAME, so re-resolve the
    # exact name we will emit and refuse root. Do NOT trust the caller's pw_uid field
    # (decoupled from the emitted pw_name). getpwnam("")/unknown -> KeyError -> refused.
    name = user_pw.pw_name
    try:
        resolved_uid = pwd.getpwnam(name).pw_uid
    except KeyError:
        raise _Refused(f"unknown run-as user: {name!r}") from None
    if resolved_uid == 0:
        raise _Refused("refuse to generate a plist that runs as root (uid 0)")

    program = list(program_args or [])
    for arg in program:
        if not isinstance(arg, str) or "\x00" in arg:
            raise _Refused(f"invalid program argument: {arg!r}")

    # Forced environment, self-defending: re-validate caller keys/values (DYLD_/LD_ and
    # malformed keys refused, not silently kept) then force HOME/PATH. PATH excludes the
    # user's home — the binary is found via the absolute ProgramArguments[0]; PATH only
    # serves its subprocesses.
    binary_dir = os.path.dirname(binary)
    environment: dict[str, str] = {}
    for key, value in (env or {}).items():
        if key in ("HOME", "PATH"):
            continue  # forced below; any caller-supplied value is ignored
        if not _ENV_KEY_RE.fullmatch(key) or key.startswith(_FORBIDDEN_ENV_PREFIXES):
            raise _Refused(f"invalid or forbidden env key: {key!r}")
        if any(c in value for c in ("\x00", "\n", "\r")):
            raise _Refused(f"control character in env value for {key!r}")
        environment[key] = value
    environment["HOME"] = user_pw.pw_dir
    # System dirs FIRST so a trojaned binary in an admin-writable prefix (homebrew /
    # usr-local — where binary_dir itself always lives, per I5) cannot mask a system
    # binary a daemon subprocess calls. The daemon's own binary is found via the absolute
    # ProgramArguments[0], so binary_dir need not lead. dict.fromkeys dedups, order-preserving.
    path_dirs = [
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
        binary_dir,
        "/opt/homebrew/bin",
        "/usr/local/bin",
    ]
    # dict.fromkeys dedups order-preserving, collapsing binary_dir when it equals a literal
    # prefix. (Assumes realpath did not cross a symlinked prefix parent — the I5 prefixes are
    # stable and non-symlinked on the fleet; a stale duplicate would be harmless, system leads.)
    environment["PATH"] = ":".join(dict.fromkeys(path_dirs))

    # GroupName is intentionally omitted: launchd runs the daemon with the non-root
    # UserName's primary group (non-privileged). Emitting it would only add surface.
    plist: dict[str, object] = {
        "Label": label,
        "ProgramArguments": [binary, *program],
        "UserName": name,
        "EnvironmentVariables": environment,
        "StandardOutPath": f"{_LOG_DIR}/{label}.out",
        "StandardErrorPath": f"{_LOG_DIR}/{label}.err",
        "RunAtLoad": run_at_load,
    }
    if keep_alive is not None:
        plist["KeepAlive"] = _validate_keep_alive(keep_alive)
    if throttle_interval is not None:
        plist["ThrottleInterval"] = throttle_interval
    if timeout is not None:
        plist["TimeOut"] = timeout
    if nice is not None:
        plist["Nice"] = nice
    return plist


def _render_plist_xml(plist: dict) -> bytes:
    """Serialise a plist dict to XML bytes via plistlib (stdlib, available under -I)."""
    return plistlib.dumps(plist, fmt=plistlib.FMT_XML, sort_keys=True)


def _precreate_log_leaves(
    label: str, user_pw: pwd.struct_passwd, *, log_dir: str = _LOG_DIR
) -> None:
    """I8: create the daemon's stdout/stderr leaves as root, then chown to the daemon.

    launchd opens ``Standard*Path`` AFTER dropping to ``UserName`` (VERIF-4), so the
    leaves must be writable by that account. The log dir is opened first with
    ``O_DIRECTORY|O_NOFOLLOW`` and verified (real dir we own, not group/other-writable),
    then each leaf is opened *relative to that dir fd* (``openat``) with ``O_NOFOLLOW``.
    A symlinked ``log_dir``, a symlinked final component, and a symlinked / pre-positioned /
    hardlinked leaf are all refused. (``O_NOFOLLOW`` guards the final dir component and the
    leaves; an *intermediate* symlink in ``log_dir``'s path is still followed, but the dir
    fstat below independently requires the resolved dir be root-owned and non-writable, so
    swapping a component already needs root.) Local safety does NOT depend on the 2.1 lock.
    Each leaf fd is then
    ``fstat``'d (exactly like ``_audit``) to refuse a pre-positioned regular file, a
    hardlink (``st_nlink``), or a group/other-writable file BEFORE ``fchown`` — otherwise
    root would hand the daemon ownership of an attacker-chosen inode. ``_require_root`` is
    enforced by the caller (install handler, 1.4), so this stays unit-testable non-root.
    """
    _validate_label(label)
    dir_fd = os.open(log_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        dst = os.fstat(dir_fd)
        if not (
            stat.S_ISDIR(dst.st_mode)
            and dst.st_uid == os.geteuid()
            and not dst.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        ):
            raise _Refused(
                f"refuse to use a suspect log dir: {log_dir!r} "
                f"(uid={dst.st_uid} mode={dst.st_mode:#o})"
            )
        for suffix in ("out", "err"):
            name = f"{label}.{suffix}"
            fd = os.open(
                name,
                os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW,
                0o640,
                dir_fd=dir_fd,
            )
            try:
                st = os.fstat(fd)
                if not (
                    stat.S_ISREG(st.st_mode)
                    and st.st_uid == os.geteuid()
                    and st.st_nlink == 1
                    and not st.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
                ):
                    raise _Refused(
                        f"refuse to chown a suspect log leaf: {name!r} "
                        f"(uid={st.st_uid} nlink={st.st_nlink} mode={st.st_mode:#o})"
                    )
                os.fchown(fd, user_pw.pw_uid, user_pw.pw_gid)
            finally:
                os.close(fd)
    finally:
        os.close(dir_fd)


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
    except _Refused as exc:
        _audit(args.action, "refused", reason=str(exc), argv=argv)
        return EXIT_REFUSED
    except NotImplementedError:
        return EXIT_INTERNAL
    except Exception as exc:
        # Never let a root process crash with a traceback on stderr.
        # Log only the exception *type*, never its message: a stray path or secret
        # in the message would otherwise land in the audit log, and a raw traceback
        # on a root process's stderr leaks filesystem layout.
        _audit(args.action, "error", reason=f"internal: {type(exc).__name__}")
        return EXIT_INTERNAL
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
