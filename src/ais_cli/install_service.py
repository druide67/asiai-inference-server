"""``aisctl install-service`` — provision the Phase 2 LaunchDaemons via the privileged helper.

Two services exist:

- ``aisctl-serve`` — the loopback companion (``127.0.0.1:8898``) that runs the fleet write
  commands. It runs as the calling admin (it is the one that calls the helper). **Shipped.**
- ``asiai-web`` — the LAN/mesh-facing ``asiai web`` dashboard (port 8899). Running it under
  the dedicated non-admin ``_aisweb`` account is cross-repo work tied to the **fleet manager**
  (loopback-token cross-user, DB/config XDG resolution, trusted-hosts — all in the main
  ``asiai`` repo), so this command reports it as **deferred** rather than installing a daemon
  that would resolve ``~`` to ``/var/empty`` and fail at runtime. For an ad-hoc dashboard now,
  run ``asiai web`` directly.

Post-AEP-01 every privileged daemon op routes through the root-owned helper (``asiai-priv``)
via :mod:`ais_core.privhelper`: this module no longer does raw ``sudo mv`` / ``launchctl`` —
it calls the content-validated ``install-reserved-service`` action, which GENERATES the entire
plist (run-as account, binary, argv) from a hardcoded spec, so a reserved label cannot be
hijacked and no caller value reaches ``ProgramArguments``.
"""

from __future__ import annotations

import argparse
import json
import sys

from ais_core import privhelper

SUPPORTED_SERVICES = ("asiai-web", "aisctl-serve")

# asiai-web (the mesh-exposed dashboard daemon, non-admin _aisweb account) is deferred to the
# fleet-manager work. The helper already fail-closes without _aisweb; the client surfaces the
# status cleanly here rather than emitting a cryptic "account missing" refusal.
_DEFERRED_WEB_MSG = (
    "asiai-web daemon is deferred to the fleet-manager work — it must run as the non-admin "
    "_aisweb account, which needs cross-repo work in the main asiai repo (loopback "
    "token cross-user, DB/config resolution, trusted-hosts). For an ad-hoc dashboard "
    "now, run `asiai web` directly."
)


def _emit(payload: dict, *, as_json: bool, ok_prefix: str, err_prefix: str) -> None:
    if as_json:
        print(json.dumps(payload))
        return
    if payload.get("ok"):
        print(f"✓ {ok_prefix}")
    else:
        print(f"✗ {err_prefix}: {payload.get('detail', payload.get('error', ''))}", file=sys.stderr)


def cmd_install_service(args: argparse.Namespace) -> int:
    """``aisctl install-service <service> [--port P] [--host H]``."""
    service = args.service
    if service not in SUPPORTED_SERVICES:
        msg = f"unknown service '{service}' (allowed: {', '.join(SUPPORTED_SERVICES)})"
        _emit({"ok": False, "error": msg}, as_json=args.json, ok_prefix="", err_prefix=msg)
        return 2

    if service == "asiai-web":
        # Documented deferral, not a failure: exit 0 so scripts don't treat it as an error.
        _emit(
            {"service": service, "ok": False, "error": "deferred", "detail": _DEFERRED_WEB_MSG},
            as_json=args.json,
            ok_prefix="",
            err_prefix="deferred",
        )
        return 0

    # aisctl-serve → the content-validated helper action. The helper resolves the run-as
    # (the SUDO_UID admin), the binary (under the invoker's ~/.local/bin, perms-chain checked),
    # and GENERATES the loopback plist. No --user / --host: serve is 127.0.0.1-pinned in root.
    helper_args = ["--service", service]
    if args.port is not None:
        helper_args += ["--port", str(args.port)]
    try:
        privhelper.run("install-reserved-service", *helper_args, dry_run=args.dry_run)
    except privhelper.PrivHelperError as exc:
        _emit(
            {"service": service, "ok": False, "error": "install_failed", "detail": str(exc)},
            as_json=args.json,
            ok_prefix="",
            err_prefix="install_failed",
        )
        return 1
    port = args.port or 8898
    _emit(
        {"service": service, "ok": True, "label": "com.asiai.aisctl-serve", "port": port},
        as_json=args.json,
        ok_prefix=f"{service} installed via asiai-priv (loopback :{port})",
        err_prefix="",
    )
    return 0


def cmd_uninstall_service(args: argparse.Namespace) -> int:
    """``aisctl uninstall-service <service>`` — symmetric removal via the helper."""
    service = args.service
    if service not in SUPPORTED_SERVICES:
        msg = f"unknown service '{service}' (allowed: {', '.join(SUPPORTED_SERVICES)})"
        _emit({"ok": False, "error": msg}, as_json=args.json, ok_prefix="", err_prefix=msg)
        return 2

    if service == "asiai-web":
        _emit(
            {
                "service": service,
                "ok": True,
                "detail": "asiai-web daemon is not installed (deferred to the fleet-manager work)",
            },
            as_json=args.json,
            ok_prefix="asiai-web daemon not installed (deferred to the fleet-manager work)",
            err_prefix="",
        )
        return 0

    try:
        privhelper.run("uninstall-reserved-service", "--service", service, dry_run=args.dry_run)
    except privhelper.PrivHelperError as exc:
        _emit(
            {"service": service, "ok": False, "error": "uninstall_failed", "detail": str(exc)},
            as_json=args.json,
            ok_prefix="",
            err_prefix="uninstall_failed",
        )
        return 1
    _emit(
        {"service": service, "ok": True, "label": "com.asiai.aisctl-serve"},
        as_json=args.json,
        ok_prefix=f"{service} uninstalled via asiai-priv",
        err_prefix="",
    )
    return 0


def add_install_service_subparsers(subparsers: argparse._SubParsersAction) -> None:
    """Register ``install-service`` and ``uninstall-service`` parsers."""
    p_install = subparsers.add_parser(
        "install-service",
        help="Provision a Phase 2 LaunchDaemon (aisctl-serve; asiai-web deferred to Fleet).",
    )
    p_install.add_argument("service", choices=SUPPORTED_SERVICES)
    p_install.add_argument(
        "--port", type=int, default=None, help="Override the default port for this service."
    )
    p_install.add_argument(
        "--host",
        default=None,
        help="Bind host — applies to asiai-web only (deferred); aisctl-serve is loopback-pinned.",
    )
    p_install.add_argument("--dry-run", action="store_true")
    p_install.add_argument("--json", action="store_true")
    p_install.set_defaults(func=cmd_install_service)

    p_uninstall = subparsers.add_parser(
        "uninstall-service",
        help="Remove a Phase 2 LaunchDaemon (aisctl-serve).",
    )
    p_uninstall.add_argument("service", choices=SUPPORTED_SERVICES)
    p_uninstall.add_argument("--dry-run", action="store_true")
    p_uninstall.add_argument("--json", action="store_true")
    p_uninstall.set_defaults(func=cmd_uninstall_service)
