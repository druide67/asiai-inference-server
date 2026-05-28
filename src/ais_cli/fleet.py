"""``aisctl fleet`` — orchestrator-side write commands (Phase 2).

Phase 1 (read-only) lives in ``asiai`` (the observability tool); this
module adds the missing write half on top of the same ``fleet.json``
configuration. The split is intentional: orchestrator-side reads
ship with the OSS observability tool, writes ship with the inference
manager so that nodes with no engines installed can still be observed.

Subcommands
-----------
``aisctl fleet push <target> <command> [--engine E] [--model M]``
    POSTs ``command`` to one or many nodes. ``target`` may be:

    - a literal nickname (``studio``) — single-node push
    - ``@all`` — broadcast to every node in ``fleet.json``
    - ``@role:dev`` — broadcast to every node whose ``role == "dev"``

    Broadcast pushes run in parallel (one thread per node) and return a
    non-zero exit code if at least one target failed.

``aisctl fleet info <nickname>``
    Print the resolved node (without echoing the token) for debugging.

The auth token is read verbatim from ``fleet.json`` (Phase 1 schema's
reserved ``auth_token`` field) — populate it via
``asiai fleet add <nick> --url ... --auth-token <secret>`` or by
hand-editing the file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from asiai.fleet import config as fleet_config

_BROADCAST_MAX_WORKERS = 16

# Mirror of ``asiai.web.routes.fleet.COMMAND_TIMEOUTS`` (LAN-facing).
# Tighter than the upstream loopback timeouts so the client fails fast.
COMMAND_TIMEOUTS: dict[str, float] = {
    "purge": 20.0,
    "load": 240.0,
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


def _resolve_targets(target: str) -> tuple[list[dict[str, Any]], str | None]:
    """Expand a target selector into a list of nodes.

    ``target`` accepts:

    - a plain nickname (most common case) → list of 1 node, or empty
    - ``@all`` → every node in fleet.json
    - ``@role:<value>`` → every node whose ``role`` field equals ``value``

    Returns ``(nodes, error)``. ``error`` is set when the selector is
    malformed; ``nodes`` may be empty if the selector matched zero
    nodes (caller decides if that's an error).
    """
    if not target.startswith("@"):
        node = _resolve_node(target)
        return ([node] if node else [], None)
    if target == "@all":
        return (fleet_config.get_nodes(), None)
    if target.startswith("@role:"):
        role = target[len("@role:") :].strip()
        if not role:
            return ([], "empty role after '@role:'")
        nodes = [n for n in fleet_config.get_nodes() if n.get("role") == role]
        return (nodes, None)
    return ([], f"unknown selector '{target}' (use a nickname, @all, or @role:<value>)")


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


def _format_single_result(
    nickname: str, command: str, status: int, body: dict[str, Any], *, prefix: str = ""
) -> bool:
    """Render one node's push result. Returns True if the call succeeded."""
    ok = isinstance(body, dict) and body.get("ok")
    if ok:
        print(f"{prefix}✓ {command} on {nickname} (http={status})")
        if body.get("stdout"):
            stdout_text = body["stdout"].rstrip()
            if stdout_text:
                print(stdout_text)
        return True
    err = body.get("error") if isinstance(body, dict) else None
    detail = body.get("detail") if isinstance(body, dict) else None
    line = f"{prefix}✗ {command} on {nickname} failed (http={status}): {err}"
    if detail:
        line += f" — {detail}"
    print(line, file=sys.stderr)
    if isinstance(body, dict) and body.get("stderr"):
        stderr_text = body["stderr"].rstrip()
        if stderr_text:
            print(stderr_text, file=sys.stderr)
    return False


def _cmd_push(args: argparse.Namespace) -> int:
    """``aisctl fleet push <target> <command> [--engine ...] [--model ...]``.

    ``target`` may be a literal nickname, ``@all``, or ``@role:<value>``.
    """
    if args.command not in COMMAND_TIMEOUTS:
        allowed = ", ".join(sorted(COMMAND_TIMEOUTS))
        msg = f"unknown command '{args.command}' (allowed: {allowed})"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 2

    nodes, target_err = _resolve_targets(args.nickname)
    if target_err:
        msg = target_err
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 1
    if not nodes:
        msg = (
            f"no node named '{args.nickname}' in fleet.json"
            if not args.nickname.startswith("@")
            else f"selector '{args.nickname}' matched zero nodes"
        )
        if args.json:
            print(json.dumps({"ok": False, "error": msg}))
        else:
            print(f"✗ {msg}", file=sys.stderr)
        return 1

    cmd_args: dict[str, Any] = {}
    if args.engine:
        cmd_args["engine"] = args.engine
    if args.model:
        cmd_args["model"] = args.model
    if getattr(args, "keep_alive", None):
        cmd_args["keep_alive"] = args.keep_alive

    # Single-node path stays simple (matches Phase 2.0 behavior).
    if len(nodes) == 1:
        node = nodes[0]
        status, body = _do_push(node, args.command, cmd_args, timeout=args.timeout)
        if args.json:
            print(json.dumps({"http_status": status, "nickname": node.get("nickname"), **body}))
            ok = isinstance(body, dict) and body.get("ok")
        else:
            ok = _format_single_result(node.get("nickname", "?"), args.command, status, body)
        return 0 if (200 <= status < 300 and ok) else 3

    # Broadcast: fan out in parallel, then aggregate.
    def _one(node: dict[str, Any]) -> tuple[dict[str, Any], int, dict[str, Any]]:
        st, bd = _do_push(node, args.command, cmd_args, timeout=args.timeout)
        return (node, st, bd)

    results: list[tuple[dict[str, Any], int, dict[str, Any]]] = []
    workers = min(_BROADCAST_MAX_WORKERS, len(nodes))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for fut in concurrent.futures.as_completed(ex.submit(_one, n) for n in nodes):
            try:
                results.append(fut.result())
            except Exception as e:  # pragma: no cover — defensive
                results.append(({"nickname": "?"}, 0, {"ok": False, "error": str(e)}))

    # Preserve fleet.json order in the rendered output for human readability.
    nick_order = {n.get("nickname"): i for i, n in enumerate(nodes)}
    results.sort(key=lambda r: nick_order.get(r[0].get("nickname"), 1_000_000))

    succeeded = 0
    failed = 0
    if args.json:
        payload = {
            "target": args.nickname,
            "command": args.command,
            "results": [
                {
                    "nickname": node.get("nickname"),
                    "http_status": status,
                    **body,
                }
                for node, status, body in results
            ],
        }
        print(json.dumps(payload))
        for _node, status, body in results:
            ok = isinstance(body, dict) and body.get("ok") and 200 <= status < 300
            if ok:
                succeeded += 1
            else:
                failed += 1
    else:
        print(f"=== broadcast {args.command} → {len(nodes)} node(s) ({args.nickname}) ===")
        for node, status, body in results:
            ok = _format_single_result(
                node.get("nickname", "?"), args.command, status, body, prefix="  "
            )
            if ok:
                succeeded += 1
            else:
                failed += 1
        print(f"--- {succeeded} ok, {failed} failed ---", file=sys.stderr if failed else None)

    return 0 if failed == 0 else 3


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
    p_push.add_argument("--model", help="Model name (for unload / load).")
    p_push.add_argument(
        "--keep-alive",
        default=None,
        help="For the 'load' command: how long to keep the model resident "
        "(e.g. '5m', '30s'). Ollama only — ignored elsewhere.",
    )
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
