"""``aisctl fleet`` — orchestrator-side write commands (Phase 2).

Phase 1 (read-only) lives in ``asiai`` (the observability tool); this
module adds the missing write half on top of the same ``fleet.json``
configuration. The split is intentional: orchestrator-side reads
ship with the OSS observability tool, writes ship with the inference
manager so that nodes with no engines installed can still be observed.

Subcommands
-----------
``aisctl fleet push <nickname> <command> [--engine E] [--model M]``
    POSTs a single command to a remote node's ``asiai web``.
``aisctl fleet info <nickname>``
    Print the resolved node (without echoing the token) for debugging.

The auth token is read verbatim from ``fleet.json`` (Phase 1 schema's
reserved ``auth_token`` field) — populate it via
``asiai fleet add <nick> --url ... --auth-token <secret>`` or by
hand-editing the file.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from asiai.fleet import config as fleet_config

# Mirror of ``asiai.web.routes.fleet.COMMAND_TIMEOUTS`` (LAN-facing).
# Tighter than the upstream loopback timeouts so the client fails fast.
COMMAND_TIMEOUTS: dict[str, float] = {
    "purge": 20.0,
    "unload": 45.0,
    "stop": 45.0,
    "start": 90.0,
    "restart": 90.0,
    "install": 240.0,
    "uninstall": 90.0,
    "upgrade": 420.0,
}

_MAX_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MB cap on response body


def _resolve_node(nickname: str) -> dict[str, Any] | None:
    """Return the fleet.json entry for ``nickname`` or None."""
    return fleet_config.find_node(nickname)


def _do_push(
    node: dict[str, Any],
    command: str,
    args: dict[str, Any],
    *,
    timeout: float | None = None,
) -> tuple[int, dict[str, Any]]:
    """POST a command to the remote node. Returns ``(http_status, body)``."""
    url = node.get("asiai_url", "").rstrip("/")
    token = node.get("auth_token") or ""
    if not url:
        return (0, {"error": "node has no asiai_url"})
    if not token:
        return (
            0,
            {
                "error": (
                    "node has no auth_token in fleet.json; "
                    "set it with `asiai fleet add <nick> --url ... --auth-token <secret>`"
                )
            },
        )

    nickname = urllib.parse.quote(str(node.get("nickname", "_")), safe="")
    endpoint = f"{url}/api/v1/fleet/{nickname}/command"
    payload = json.dumps({"command": command, "args": args}).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "Origin": url,
            "User-Agent": "aisctl-fleet-push/1",
        },
    )
    eff_timeout = timeout if timeout is not None else COMMAND_TIMEOUTS.get(command, 60.0)
    try:
        # nosec B310 — URL is built from validated fleet.json (http/https scheme enforced).
        with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
            raw = resp.read(_MAX_RESPONSE_BYTES + 1)
            if len(raw) > _MAX_RESPONSE_BYTES:
                return (resp.status, {"error": "response_oversized"})
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return (resp.status, {"error": "response_not_json"})
            return (resp.status, body if isinstance(body, dict) else {"data": body})
    except urllib.error.HTTPError as e:
        try:
            raw = e.read(_MAX_RESPONSE_BYTES + 1)
            body = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(body, dict):
                body = {"data": body}
        except (ValueError, UnicodeDecodeError, OSError):
            body = {"error": "remote_http_error"}
        return (e.code, body)
    except urllib.error.URLError as e:
        return (0, {"error": "unreachable", "detail": str(e.reason)})
    except TimeoutError:
        return (0, {"error": "timeout"})
    except OSError as e:
        return (0, {"error": "io_error", "detail": str(e)})


def _cmd_push(args: argparse.Namespace) -> int:
    """``aisctl fleet push <nickname> <command> [--engine ...] [--model ...]``."""
    node = _resolve_node(args.nickname)
    if node is None:
        msg = f"no node named '{args.nickname}' in fleet.json"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 1

    if args.command not in COMMAND_TIMEOUTS:
        allowed = ", ".join(sorted(COMMAND_TIMEOUTS))
        msg = f"unknown command '{args.command}' (allowed: {allowed})"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 2

    cmd_args: dict[str, Any] = {}
    if args.engine:
        cmd_args["engine"] = args.engine
    if args.model:
        cmd_args["model"] = args.model

    status, body = _do_push(node, args.command, cmd_args, timeout=args.timeout)

    if args.json:
        print(json.dumps({"http_status": status, **body}))
    else:
        ok = isinstance(body, dict) and body.get("ok")
        if ok:
            print(f"✓ {args.command} on {args.nickname} (http={status})")
            if body.get("stdout"):
                print(body["stdout"].rstrip())
        else:
            err = body.get("error") if isinstance(body, dict) else None
            detail = body.get("detail") if isinstance(body, dict) else None
            print(
                f"✗ {args.command} on {args.nickname} failed (http={status}): {err}"
                + (f" — {detail}" if detail else ""),
                file=sys.stderr,
            )
            if isinstance(body, dict) and body.get("stderr"):
                print(body["stderr"].rstrip(), file=sys.stderr)

    return 0 if (200 <= status < 300 and body.get("ok")) else 3


def _cmd_info(args: argparse.Namespace) -> int:
    """Print the resolved node (auth_token redacted)."""
    node = _resolve_node(args.nickname)
    if node is None:
        print(f"✗ no node named '{args.nickname}'", file=sys.stderr)
        return 1
    public = fleet_config.redact_node(node)
    public["has_auth_token"] = bool(node.get("auth_token"))
    if args.json:
        print(json.dumps(public, indent=2))
    else:
        for k, v in public.items():
            print(f"{k}: {v}")
    return 0


def cmd_fleet(args: argparse.Namespace) -> int:
    """Dispatcher for ``aisctl fleet <action>``."""
    action = getattr(args, "fleet_action", None)
    if action == "push":
        return _cmd_push(args)
    if action == "info":
        return _cmd_info(args)
    print("usage: aisctl fleet {push,info} ...", file=sys.stderr)
    return 2


def add_fleet_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``fleet`` parser on ``subparsers``."""
    fleet_parser = subparsers.add_parser(
        "fleet",
        help="Push write commands to remote nodes (Phase 2).",
    )
    fleet_sub = fleet_parser.add_subparsers(dest="fleet_action", metavar="<action>")

    p_push = fleet_sub.add_parser(
        "push",
        help="POST a single command to a remote node's asiai web.",
    )
    p_push.add_argument("nickname", help="Node nickname (from fleet.json).")
    p_push.add_argument(
        "command",
        choices=sorted(COMMAND_TIMEOUTS),
        help="Write command to execute on the remote.",
    )
    p_push.add_argument("--engine", help="Engine name (required for everything except purge).")
    p_push.add_argument("--model", help="Model name (for unload).")
    p_push.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Override the per-command HTTP timeout (seconds).",
    )
    p_push.add_argument("--json", action="store_true")
    p_push.set_defaults(func=cmd_fleet, fleet_action="push")

    p_info = fleet_sub.add_parser(
        "info",
        help="Show the resolved node configuration (no secrets echoed).",
    )
    p_info.add_argument("nickname")
    p_info.add_argument("--json", action="store_true")
    p_info.set_defaults(func=cmd_fleet, fleet_action="info")
