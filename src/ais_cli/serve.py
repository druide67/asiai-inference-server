"""``aisctl serve`` — loopback HTTP server for fleet write commands.

Listens on ``127.0.0.1:8898`` and accepts ``POST /internal/v1/command``
with a Bearer token shared with ``asiai web``. Each request shells out
to ``aisctl <command> [args]`` and returns the JSON-shaped result.

It also answers ``GET /internal/v1/engines-state`` (same Bearer): the
rich :class:`ais_core.lifecycle.EngineState` of every manifest, so the
co-located ``asiai web`` can surface stopped/disabled/unhealthy engines
in its snapshot instead of the poor reachable/unreachable split HTTP
detection alone can offer.

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
from urllib.parse import parse_qs, urlsplit

from asiai.auth import loopback as asiai_loopback
from asiai.fleet.command_spec import ALLOWED_COMMANDS, inner_tool_timeout, loopback_timeout

logger = logging.getLogger("aisctl.serve")

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8898
MAX_CONCURRENT = 4
SHUTDOWN_GRACE_SECONDS = 5.0

# Shape gate for preset names arriving over HTTP — the SAME pattern the
# install funnel applies in ``_build_argv`` below, so the read (plan) and
# write (install --preset) surfaces accept exactly the same names.
PRESET_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}$")

# Command whitelist + subprocess-kill timeouts come from the single shared
# source (``asiai.fleet.command_spec``). This loopback layer is the INNERMOST /
# authoritative budget: ``loopback_timeout(cmd)`` == the work budget itself. The
# edge and client derive LONGER deadlines from it (budget + margin per hop), so
# no outer layer abandons a command still running here. Replaces the old
# hand-synced map + the "edge is intentionally tighter" note that was the SB bug.


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
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"unknown command: {command}")
    argv = [_aisctl_binary(), command, "--json"]
    engine = args.get("engine")
    if command in {
        "start",
        "stop",
        "restart",
        "install",
        "uninstall",
        "unload",
        "load",
        "enable",
        "disable",
    }:
        if not isinstance(engine, str) or not engine:
            raise ValueError(f"command {command} requires args.engine (string)")
        argv.append(engine)
    if command == "install":
        # Optional tuned-manifest preset. Shape-checked here (400, no
        # subprocess) and validated against the bundled preset registry
        # inside cmd_install (_resolve_manifest raises on unknown names).
        preset = args.get("preset")
        if preset is not None:
            if not isinstance(preset, str) or not PRESET_NAME_RE.match(preset):
                raise ValueError("args.preset must match [a-zA-Z0-9][a-zA-Z0-9._-]{0,63}")
            argv.extend(["--preset", preset])
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
        # Routed through ``aisctl upgrade`` like every other command, so the
        # OperationsLock and the JSON envelope apply (calling brew directly
        # used to skip both). The whitelist check still runs HERE so a
        # non-whitelisted engine is a 400, not a failed subprocess; it is
        # enforced again inside cmd_upgrade. The inner timeout stays under
        # this server's 600s envelope so aisctl reports its own failure
        # instead of being killed mid-write.
        if not isinstance(engine, str) or not engine:
            raise ValueError("upgrade requires args.engine")
        from ais_core.upgrade import upgrade_argv

        upgrade_argv(engine)  # validation only — raises ValueError if not whitelisted
        # Inner tool deadline derived from the shared spec (work budget - headroom)
        # so ``aisctl upgrade`` reports its own failure before this server's
        # subprocess-kill fires. Single-sourced with the loopback budget.
        inner = int(inner_tool_timeout("upgrade"))
        argv = [_aisctl_binary(), "upgrade", engine, "--json", "--timeout", str(inner)]
    return argv


# Cap on the human-readable failure detail extracted from a failed
# subprocess's output. Large enough for aisctl's multi-line error messages,
# small enough to stay readable in a dashboard toast or an audit log line.
FAILURE_DETAIL_MAX_CHARS = 800


def _failure_detail(stderr: str, stdout: str) -> str:
    """Extract the actionable tail of a failed command's output.

    Prefer stderr (where aisctl prints its error messages); fall back to
    stdout (some failures emit their envelope there). Keep the LAST lines:
    the root-cause message ends the output, after any progress noise. The
    full streams still travel in the ``stdout``/``stderr`` fields — this is
    the short, display-ready summary consumers can show verbatim.
    """
    text = stderr.strip() or stdout.strip()
    if not text:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    detail = "\n".join(lines)
    if len(detail) > FAILURE_DETAIL_MAX_CHARS:
        detail = detail[-FAILURE_DETAIL_MAX_CHARS:]
        # Drop the partial first line of the retained window when the tail
        # spans several lines (a single oversized line is kept as-is).
        cut = detail.find("\n")
        if cut != -1:
            detail = detail[cut + 1 :]
    return detail


def _execute(argv: list[str], timeout: float) -> dict[str, Any]:
    """Run argv, capture output, return a normalized response dict.

    On failure (non-zero exit), the dict additionally carries
    ``error: "command_failed"`` plus a bounded ``detail`` string with the
    last useful lines of stderr (stdout fallback), so HTTP consumers get an
    actionable message in the body instead of a bare status code. Existing
    fields are unchanged — the enrichment is strictly additive.
    """
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
            "detail": str(e)[:FAILURE_DETAIL_MAX_CHARS],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": "",
            "stderr": f"command timed out after {timeout:.0f}s",
            "duration_ms": int((time.monotonic() - started) * 1000),
            "error": "timeout",
            "detail": f"command timed out after {timeout:.0f}s",
        }
    result: dict[str, Any] = {
        "ok": proc.returncode == 0,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-16384:],
        "stderr": proc.stderr[-16384:],
        "duration_ms": int((time.monotonic() - started) * 1000),
    }
    if proc.returncode != 0:
        # Stable machine code (like "timeout"/"aisctl_binary_not_found"
        # above) + bounded human-readable detail. Without these, dashboard
        # clients that key on ``error`` show an opaque "http_500" even
        # though aisctl printed an explicit, actionable message.
        result["error"] = "command_failed"
        detail = _failure_detail(proc.stderr, proc.stdout)
        if detail:
            result["detail"] = detail
    return result


def _semaphore_acquire(sem: threading.Semaphore, timeout: float = 1.0) -> bool:
    """Acquire with timeout (Python's Semaphore.acquire(timeout) is fine)."""
    return sem.acquire(timeout=timeout)


# --- engines-state read surface ---------------------------------------------
#
# ``asiai web`` polls this on every snapshot (its own cache is ~10s); probing
# every manifest costs launchctl+HTTP round-trips, so answers are cached a few
# seconds. Two racing callers may both collect once — harmless, the cache
# absorbs the rest.

ENGINES_STATE_CACHE_TTL = 5.0
_PROBE_POOL_SIZE = 8

_state_cache_lock = threading.Lock()
_state_cache: dict[str, Any] = {"ts": 0.0, "payload": None}


def _collect_engines_state() -> dict[str, Any]:
    """Probe every manifest's lifecycle state (parallel, never raises per-engine)."""
    from concurrent.futures import ThreadPoolExecutor

    from ais_core import install_state, lifecycle
    from ais_core import manifest as manifest_mod

    def probe_one(name: str) -> dict[str, Any] | None:
        try:
            # As INSTALLED (preset overlay): the baseline port may differ
            # from the port the service was generated on — probing the
            # baseline would show a serving engine as stopped.
            m = install_state.load_installed_manifest(name)
            state, _ = lifecycle.probe_state(m)
            record = install_state.read_install(m.name)
            return {
                "name": m.name,
                "display": m.display,
                "port": m.network.port,
                "state": str(state),
                # What a provisioned engine WOULD serve (from its installed
                # plist) — lets the dashboard label non-running cards.
                "model": lifecycle.installed_model(m),
                # Recorded at install time — which tuned preset (if any)
                # the service was generated from. The cockpit shows it and
                # the install picker preselects it.
                "preset": record.preset if record else None,
            }
        except Exception as e:  # one broken manifest must never hide the others
            logger.warning("engines-state: probe failed for %s: %s", name, e)
            return None

    names = manifest_mod.list_manifests()
    with ThreadPoolExecutor(max_workers=_PROBE_POOL_SIZE) as pool:
        engines = [r for r in pool.map(probe_one, names) if r]
    return {"engines": engines, "ts": int(time.time())}


def _engines_state_cached() -> dict[str, Any]:
    now = time.monotonic()
    with _state_cache_lock:
        cached = _state_cache["payload"]
        if cached is not None and now - _state_cache["ts"] < ENGINES_STATE_CACHE_TTL:
            return cached
    payload = _collect_engines_state()
    with _state_cache_lock:
        _state_cache["ts"] = time.monotonic()
        _state_cache["payload"] = payload
    return payload


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
        # allow_nan=False: NaN/Infinity are not valid JSON tokens; a producer
        # bug must fail loudly server-side, never emit unparseable output
        # onto the wire.
        payload = json.dumps(body, allow_nan=False).encode("utf-8")
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
        if self.path == "/internal/v1/engines-state":
            # Unlike /health this discloses manifest names/ports/states:
            # same Bearer gate as the write endpoint.
            if not self._check_auth():
                self._json(401, {"error": "unauthorized"})
                return
            try:
                self._json(200, _engines_state_cached())
            except Exception as e:  # fail as JSON, never a traceback page
                logger.error("engines-state collection failed: %s", e)
                self._json(500, {"error": "engines_state_failed"})
            return
        if self.path == "/internal/v1/presets":
            # Bundled tuned-manifest presets, for the cockpit's install
            # picker. Same Bearer gate: preset names reveal the tuning
            # surface of the node.
            if not self._check_auth():
                self._json(401, {"error": "unauthorized"})
                return
            try:
                from ais_core.manifest import list_presets, preset_summary

                self._json(200, {"presets": [preset_summary(n) for n in list_presets()]})
            except Exception as e:  # fail as JSON, never a traceback page
                logger.error("presets listing failed: %s", e)
                self._json(500, {"error": "presets_failed"})
            return
        if urlsplit(self.path).path == "/internal/v1/plan":
            # Advisory memory-cost estimate for one preset (the "cost" half
            # of the UMA plan contract; ``asiai web`` owns the verdict).
            # Same Bearer gate: the estimate discloses preset names and the
            # node's tuning surface.
            if not self._check_auth():
                self._json(401, {"error": "unauthorized"})
                return
            query = parse_qs(urlsplit(self.path).query)
            presets = query.get("preset", [])
            preset = presets[0] if len(presets) == 1 else None
            # Same shape gate as the install funnel (PRESET_NAME_RE): the
            # read and write surfaces must accept exactly the same names.
            if not preset or not PRESET_NAME_RE.match(preset):
                self._json(400, {"error": "bad_preset", "detail": "preset= query required"})
                return
            try:
                from ais_core.plan import plan_for_preset

                cost = plan_for_preset(preset)
            except FileNotFoundError:
                self._json(404, {"error": "unknown_preset", "preset": preset})
                return
            except Exception as e:  # fail as JSON, never a traceback page
                logger.error("plan estimation failed for %s: %s", preset, e)
                self._json(500, {"error": "plan_failed"})
                return
            self._json(200, cost.payload())
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
            timeout = loopback_timeout(command)
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
