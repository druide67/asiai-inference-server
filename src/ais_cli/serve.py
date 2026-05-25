"""``aisctl serve`` — loopback HTTP server for fleet write commands.

Listens on ``127.0.0.1:8898`` and accepts ``POST /internal/v1/command``
with a Bearer token shared with ``asiai web``. Each request shells out
to ``aisctl <command> [args]`` and returns the JSON-shaped result.

Why a separate process?
-----------------------
The fleet write surface lives on the LAN (``asiai web`` on port 8899)
but the actual orchestration logic is in ``aisctl``. Co-locating both
in ``asiai web`` would force every node to install
``asiai-inference-server`` as a hard runtime dependency. Instead,
``aisctl serve`` is an opt-in companion: nodes that don't run it can
still be observed (Phase 1) but cannot receive writes (Phase 2).

Why loopback only?
------------------
The LAN auth surface is in ``asiai web`` (Bearer + rate limit + audit).
Exposing ``aisctl serve`` directly to the LAN would duplicate that
surface and complicate the threat model. Loopback + a per-startup
shared secret (see :mod:`asiai.auth.loopback`) keeps the trust boundary
in one place.

Concurrency
-----------
Threaded with a semaphore (``MAX_CONCURRENT``) so a slow command
(install/upgrade) cannot starve the queue, but a burst cannot fork
arbitrary numbers of subprocesses.
"""

from __future__ import annotations

import argparse
import contextlib
import http.server
import json
import logging
import re
import shutil
import signal
import socketserver
import subprocess
import sys
import threading
import time
from typing import Any

from asiai.auth import loopback as asiai_loopback

logger = logging.getLogger("aisctl.serve")

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8898
MAX_CONCURRENT = 4
SHUTDOWN_GRACE_SECONDS = 5.0

# Commands the loopback server forwards to the local ``aisctl`` binary.
# Values are upstream subprocess timeouts (seconds). Must stay in sync
# with ``asiai.web.routes.fleet.COMMAND_TIMEOUTS`` (the LAN-facing
# timeout there is intentionally tighter so we fail fast at the edge).
COMMAND_TIMEOUTS: dict[str, float] = {
    "purge": 30.0,
    "load": 300.0,
    "unload": 60.0,
    "stop": 60.0,
    "start": 120.0,
    "restart": 120.0,
    "install": 300.0,
    "uninstall": 120.0,
    "upgrade": 600.0,
}


def _aisctl_binary() -> str:
    """Locate the aisctl executable to invoke for command dispatch.

    Prefer the same Python's ``console_scripts`` entry-point if it lives
    next to the running interpreter (avoids PATH dependency on a host
    that has multiple aisctl installs). Fall back to ``shutil.which``.
    """
    py_dir = sys.prefix
    candidate = f"{py_dir}/bin/aisctl"
    import os

    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    found = shutil.which("aisctl")
    if found:
        return found
    # Last-resort: assume PATH at exec time. subprocess.run will raise
    # FileNotFoundError if it's wrong, which we surface as 503.
    return "aisctl"


def _build_argv(command: str, args: dict[str, Any]) -> list[str]:
    """Translate ``{command, args}`` into the aisctl argv vector.

    Strict whitelist; anything unexpected raises ValueError so the
    handler can return 400 without invoking subprocess.
    """
    if command not in COMMAND_TIMEOUTS:
        raise ValueError(f"unknown command: {command}")
    argv = [_aisctl_binary(), command, "--json"]
    engine = args.get("engine")
    if command in {"start", "stop", "restart", "install", "uninstall", "unload", "load"}:
        if not isinstance(engine, str) or not engine:
            raise ValueError(f"command {command} requires args.engine (string)")
        argv.append(engine)
    if command in ("unload", "load"):
        model = args.get("model")
        if command == "load" and not model:
            raise ValueError("command load requires args.model (string)")
        if model is not None:
            if not isinstance(model, str) or not model:
                raise ValueError("args.model must be a non-empty string when set")
            argv.append(model)
        keep_alive = args.get("keep_alive")
        if command == "load" and keep_alive:
            if not isinstance(keep_alive, str) or not re.match(r"^[0-9]+[smh]?$", keep_alive):
                raise ValueError("args.keep_alive must match [0-9]+[smh]?")
            argv.extend(["--keep-alive", keep_alive])
    if command == "upgrade":
        # upgrade isn't a built-in aisctl subcommand yet; route via brew
        # under the engine's well-known formula name. Until aisrv ships
        # a native ``aisctl upgrade``, this is a thin shim with a fixed
        # formula whitelist to prevent argv injection.
        if not isinstance(engine, str) or not engine:
            raise ValueError("upgrade requires args.engine")
        argv = _upgrade_argv(engine)
    return argv


# Whitelist of Homebrew formulas an upgrade may target. Anything not in
# this set is rejected at validation time — defends against an attacker
# who somehow bypasses the engine regex from passing ``coreutils`` or a
# tap that runs arbitrary post-install hooks.
UPGRADE_FORMULAS: dict[str, str] = {
    "ollama": "ollama",
    "llamacpp": "llama.cpp",
    "lmstudio": "lm-studio",
    "rapidmlx": "rapid-mlx",
    "turboquant": "turboquant",
}


def _upgrade_argv(engine: str) -> list[str]:
    formula = UPGRADE_FORMULAS.get(engine)
    if not formula:
        raise ValueError(
            f"upgrade is not whitelisted for engine '{engine}' "
            f"(allowed: {', '.join(sorted(UPGRADE_FORMULAS))})"
        )
    brew = shutil.which("brew") or "/opt/homebrew/bin/brew"
    return [brew, "upgrade", formula]


def _execute(argv: list[str], timeout: float) -> dict[str, Any]:
    """Run argv, capture output, return a normalized response dict."""
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as e:
        return {
            "ok": False,
            "exit_code": 127,
            "stdout": "",
            "stderr": str(e),
            "duration_ms": int((time.monotonic() - started) * 1000),
            "error": "aisctl_binary_not_found",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": "",
            "stderr": f"command timed out after {timeout:.0f}s",
            "duration_ms": int((time.monotonic() - started) * 1000),
            "error": "timeout",
        }
    return {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-16384:],
        "stderr": proc.stderr[-16384:],
        "duration_ms": int((time.monotonic() - started) * 1000),
    }


def _semaphore_acquire(sem: threading.Semaphore, timeout: float = 1.0) -> bool:
    """Acquire with timeout (Python's Semaphore.acquire(timeout) is fine)."""
    return sem.acquire(timeout=timeout)


class _Handler(http.server.BaseHTTPRequestHandler):
    server_version = "aisctl-serve/1"
    sys_version = ""

    # Per-server attributes injected after construction (see ``serve_forever``).
    expected_token: str = ""
    semaphore: threading.Semaphore | None = None

    def log_message(self, fmt: str, *args: Any) -> None:
        # Route stdlib logging through our named logger instead of stderr.
        logger.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        with contextlib.suppress(OSError):
            self.wfile.write(payload)

    def _read_body(self, max_bytes: int = 64 * 1024) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None
        if length < 0 or length > max_bytes:
            return None
        try:
            return self.rfile.read(length)
        except OSError:
            return None

    def _check_auth(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.lower().startswith("bearer "):
            return False
        token = header[7:].strip()
        if not token or not self.expected_token:
            return False
        # Constant-time compare on bytes.
        import hmac

        return hmac.compare_digest(token.encode("utf-8"), self.expected_token.encode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/internal/v1/health":
            self._json(200, {"ok": True, "service": "aisctl-serve"})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        # Reject anything other than the one documented endpoint.
        if self.path != "/internal/v1/command":
            self._json(404, {"error": "not_found"})
            return

        # Auth before reading the body so we don't even bother spending
        # bytes on unauthenticated callers.
        if not self._check_auth():
            self._json(401, {"error": "unauthorized"})
            return

        raw = self._read_body()
        if raw is None:
            self._json(413, {"error": "body_too_large_or_unreadable"})
            return
        try:
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, UnicodeDecodeError):
            self._json(400, {"error": "invalid_json"})
            return
        if not isinstance(payload, dict):
            self._json(400, {"error": "body_must_be_object"})
            return
        command = payload.get("command")
        args = payload.get("args") or {}
        if not isinstance(command, str) or not isinstance(args, dict):
            self._json(400, {"error": "bad_shape"})
            return
        try:
            argv = _build_argv(command, args)
        except ValueError as e:
            self._json(400, {"error": "bad_args", "detail": str(e)})
            return

        # Concurrency cap: refuse if all workers are busy.
        sem = self.semaphore
        if sem is None or not _semaphore_acquire(sem, timeout=0.05):
            self._json(503, {"error": "busy", "detail": "all workers in use"})
            return
        try:
            timeout = COMMAND_TIMEOUTS.get(command, 60.0)
            result = _execute(argv, timeout=timeout)
        finally:
            sem.release()

        status = 200 if result.get("ok") else 500
        self._json(status, result)


class _ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    request_queue_size = 10
    allow_reuse_address = True


def _build_server(
    port: int, token: str, *, max_concurrent: int = MAX_CONCURRENT
) -> _ThreadedServer:
    """Construct the threaded server with the given loopback token."""
    handler_class = type(
        "BoundHandler",
        (_Handler,),
        {
            "expected_token": token,
            "semaphore": threading.Semaphore(max_concurrent),
        },
    )
    return _ThreadedServer((LOOPBACK_HOST, port), handler_class)


def cmd_serve(args: argparse.Namespace) -> int:
    """``aisctl serve`` entry point. Blocks until SIGINT/SIGTERM."""
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        token = asiai_loopback.write_token()
    except OSError as e:
        print(f"aisctl serve: failed to write loopback token: {e}", file=sys.stderr)
        return 1

    try:
        server = _build_server(args.port, token, max_concurrent=args.max_concurrent)
    except OSError as e:
        print(
            f"aisctl serve: bind {LOOPBACK_HOST}:{args.port} failed: {e}",
            file=sys.stderr,
        )
        asiai_loopback.remove_token()
        return 1

    stop_evt = threading.Event()

    def _shutdown(*_args: Any) -> None:
        if stop_evt.is_set():
            return
        stop_evt.set()
        logger.info("shutting down")
        # server.shutdown() must run from a thread other than the one
        # blocked on serve_forever; signal handlers run on the main
        # thread, so we spin a one-shot helper.
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(
        f"aisctl serve listening on http://{LOOPBACK_HOST}:{args.port} "
        f"(loopback only, max_concurrent={args.max_concurrent})",
        file=sys.stderr,
    )

    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        with contextlib.suppress(Exception):
            server.server_close()
        asiai_loopback.remove_token()

    return 0


def add_serve_subparser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``serve`` parser on ``subparsers``."""
    p = subparsers.add_parser(
        "serve",
        help="Run the loopback HTTP server that ``asiai web`` proxies fleet writes to.",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port on 127.0.0.1 (default: {DEFAULT_PORT}).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max parallel commands (default: {MAX_CONCURRENT}).",
    )
    p.add_argument("--verbose", action="store_true", help="DEBUG-level logs to stderr.")
    p.set_defaults(func=cmd_serve)
