"""``aisctl install-service`` — provision the Phase 2 LaunchDaemons.

Two services exist:

- ``asiai-web`` — the LAN-facing ``asiai web`` dashboard + write API
  (port 8899 by default, bound on ``0.0.0.0`` for fleet access).
- ``aisctl-serve`` — the loopback companion that runs the write
  commands (port 8898 on ``127.0.0.1`` only).

Both run as the calling user (``UserName``) so that ``~/.config/asiai``
and ``~/.local/state/asiai`` resolve to the same HOME on both sides
of the proxy hop. The sudoers fragment shipped by ``aisctl bootstrap``
must include the ``launchctl bootstrap system /Library/LaunchDaemons/
com.asiai.{web,aisctl-serve}.plist`` lines for this to work without a
password prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

LAUNCHDAEMONS_DIR = Path("/Library/LaunchDaemons")

SUPPORTED_SERVICES = ("asiai-web", "aisctl-serve")


def _user_local_bin(user: str) -> str:
    """Return ``/Users/<user>/.local/bin`` regardless of whether the
    calling process can ``expanduser`` for that user."""
    if user == os.environ.get("USER", "") or user == os.environ.get("LOGNAME", ""):
        return os.path.expanduser("~/.local/bin")
    return f"/Users/{user}/.local/bin"


def build_plist(
    service: str,
    *,
    user: str,
    port: int | None = None,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Return the plist dict for one of the two Phase 2 services."""
    user_home = f"/Users/{user}"
    user_local_bin = _user_local_bin(user)
    path_value = ":".join(
        [user_local_bin, "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    )

    env: dict[str, str] = {
        "HOME": user_home,
        "PATH": path_value,
    }
    if extra_env:
        env.update(extra_env)

    if service == "asiai-web":
        bind_port = port or 8899
        return {
            "Label": "com.asiai.web",
            "Comment": "asiai web dashboard + Phase 2 fleet write API — managed by aisctl",
            "ProgramArguments": [
                f"{user_local_bin}/asiai",
                "web",
                "--host",
                "0.0.0.0",
                "--port",
                str(bind_port),
            ],
            "UserName": user,
            "GroupName": "staff",
            "WorkingDirectory": user_home,
            "EnvironmentVariables": env,
            "RunAtLoad": True,
            "KeepAlive": {"Crashed": True, "SuccessfulExit": False},
            "StandardOutPath": f"{user_home}/Library/Logs/asiai-web.log",
            "StandardErrorPath": f"{user_home}/Library/Logs/asiai-web.err.log",
        }

    if service == "aisctl-serve":
        bind_port = port or 8898
        return {
            "Label": "com.asiai.aisctl-serve",
            "Comment": "aisctl serve loopback companion for asiai Phase 2 — managed by aisctl",
            "ProgramArguments": [
                f"{user_local_bin}/aisctl",
                "serve",
                "--port",
                str(bind_port),
            ],
            "UserName": user,
            "GroupName": "staff",
            "WorkingDirectory": user_home,
            "EnvironmentVariables": env,
            "RunAtLoad": True,
            "KeepAlive": {"Crashed": True, "SuccessfulExit": False},
            "StandardOutPath": f"{user_home}/Library/Logs/asiai-aisctl-serve.log",
            "StandardErrorPath": f"{user_home}/Library/Logs/asiai-aisctl-serve.err.log",
        }

    raise ValueError(f"unknown service: {service} (allowed: {', '.join(SUPPORTED_SERVICES)})")


def _plist_path(service: str) -> Path:
    return LAUNCHDAEMONS_DIR / f"com.asiai.{'web' if service == 'asiai-web' else service}.plist"


def write_plist(service: str, plist: dict[str, Any], dest: Path) -> Path:
    """Write the plist atomically to a tmp file. Returns the tmp path.

    The caller is responsible for the privileged ``install`` step that
    moves the file to ``/Library/LaunchDaemons/``.
    """
    fd, tmp_path = tempfile.mkstemp(prefix=f"{service}-", suffix=".plist")
    with os.fdopen(fd, "wb") as f:
        plistlib.dump(plist, f, sort_keys=False)
    return Path(tmp_path)


def _ensure_logs_dir(user: str) -> None:
    """Create ``~/Library/Logs`` for the user if missing."""
    log_dir = Path(f"/Users/{user}/Library/Logs")
    log_dir.mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str], *, dry_run: bool) -> int:
    if dry_run:
        print(f"+ {' '.join(cmd)}")
        return 0
    result = subprocess.run(cmd, check=False)
    return result.returncode


def _install_one(
    service: str,
    *,
    user: str,
    port: int | None,
    dry_run: bool,
    force: bool,
    as_json: bool,
) -> dict[str, Any]:
    target = _plist_path(service)
    if target.exists() and not force:
        msg = (
            f"{target} already exists — pass --force to overwrite "
            "(or 'aisctl uninstall-service ...' first)"
        )
        return {"service": service, "ok": False, "error": "already_installed", "detail": msg}

    plist = build_plist(service, user=user, port=port)
    tmp = write_plist(service, plist, target)

    _ensure_logs_dir(user)

    install_cmd = [
        "sudo",
        "-n",
        "/usr/bin/install",
        "-m",
        "0644",
        "-o",
        "root",
        "-g",
        "wheel",
        str(tmp),
        str(target),
    ]
    rc = _run(install_cmd, dry_run=dry_run)
    if rc != 0:
        tmp.unlink(missing_ok=True)
        return {
            "service": service,
            "ok": False,
            "error": "install_failed",
            "detail": (
                f"sudo install returned {rc}. Either run 'aisctl bootstrap "
                "--install-sudoers' to grant non-interactive sudo, or rerun "
                "as root."
            ),
        }
    tmp.unlink(missing_ok=True)

    # If already loaded (re-install case), bootout first.
    label = plist["Label"]
    bootout_cmd = ["sudo", "-n", "/bin/launchctl", "bootout", f"system/{label}"]
    _run(bootout_cmd, dry_run=dry_run)

    bootstrap_cmd = ["sudo", "-n", "/bin/launchctl", "bootstrap", "system", str(target)]
    rc2 = _run(bootstrap_cmd, dry_run=dry_run)
    if rc2 != 0:
        return {
            "service": service,
            "ok": False,
            "error": "bootstrap_failed",
            "detail": f"launchctl bootstrap returned {rc2}",
            "plist": str(target),
        }

    return {
        "service": service,
        "ok": True,
        "plist": str(target),
        "label": label,
        "user": user,
        "port": port or (8899 if service == "asiai-web" else 8898),
    }


def cmd_install_service(args: argparse.Namespace) -> int:
    """``aisctl install-service <service> [--user U] [--port P]``."""
    service = args.service
    if service not in SUPPORTED_SERVICES:
        msg = f"unknown service '{service}' (allowed: {', '.join(SUPPORTED_SERVICES)})"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 2

    user = args.user or os.environ.get("SUDO_USER") or os.environ.get("USER") or "root"
    if user == "root":
        msg = (
            "refusing to run as root: pass --user <name> or rerun as a regular user. "
            "Daemons that own ~/.config/asiai must run as a real account."
        )
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 2

    result = _install_one(
        service,
        user=user,
        port=args.port,
        dry_run=args.dry_run,
        force=args.force,
        as_json=args.json,
    )
    if args.json:
        print(json.dumps(result))
    else:
        if result.get("ok"):
            print(f"✓ {result['service']} installed: {result['plist']}")
            print(f"  label: {result['label']}")
            print(f"  user:  {result['user']}")
            print(f"  port:  {result['port']}")
        else:
            print(f"✗ {result.get('error')}: {result.get('detail', '')}", file=sys.stderr)
    return 0 if result.get("ok") else 1


def cmd_uninstall_service(args: argparse.Namespace) -> int:
    """``aisctl uninstall-service <service>`` — symmetric removal."""
    service = args.service
    if service not in SUPPORTED_SERVICES:
        msg = f"unknown service '{service}' (allowed: {', '.join(SUPPORTED_SERVICES)})"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 2

    target = _plist_path(service)
    label = f"com.asiai.{'web' if service == 'asiai-web' else service}"

    if not target.exists() and not args.force:
        msg = f"{target} not found (already uninstalled?)"
        if args.json:
            print(json.dumps({"ok": False, "error": "not_installed", "detail": msg}))
        else:
            print(msg, file=sys.stderr)
        return 0

    bootout_cmd = ["sudo", "-n", "/bin/launchctl", "bootout", f"system/{label}"]
    _run(bootout_cmd, dry_run=args.dry_run)

    rm_cmd = ["sudo", "-n", "/bin/rm", "-f", str(target)]
    rc = _run(rm_cmd, dry_run=args.dry_run)

    payload = {
        "service": service,
        "ok": rc == 0,
        "plist": str(target),
        "label": label,
    }
    if args.json:
        print(json.dumps(payload))
    else:
        if rc == 0:
            print(f"✓ {service} uninstalled ({target})")
        else:
            print(f"✗ rm returned {rc}", file=sys.stderr)
    return 0 if rc == 0 else 1


def add_install_service_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register ``install-service`` and ``uninstall-service`` parsers."""
    p_install = subparsers.add_parser(
        "install-service",
        help="Provision a Phase 2 LaunchDaemon (asiai-web / aisctl-serve).",
    )
    p_install.add_argument("service", choices=SUPPORTED_SERVICES)
    p_install.add_argument(
        "--user",
        default=None,
        help="macOS user the daemon runs as. Default: $SUDO_USER or $USER.",
    )
    p_install.add_argument(
        "--port", type=int, default=None, help="Override the default port for this service."
    )
    p_install.add_argument("--force", action="store_true", help="Overwrite if already installed.")
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument("--json", action="store_true")
    p_install.set_defaults(func=cmd_install_service)

    p_uninstall = subparsers.add_parser(
        "uninstall-service",
        help="Remove a Phase 2 LaunchDaemon (asiai-web / aisctl-serve).",
    )
    p_uninstall.add_argument("service", choices=SUPPORTED_SERVICES)
    p_uninstall.add_argument("--force", action="store_true")
    p_uninstall.add_argument("--dry-run", action="store_true")
    p_uninstall.add_argument("--json", action="store_true")
    p_uninstall.set_defaults(func=cmd_uninstall_service)
