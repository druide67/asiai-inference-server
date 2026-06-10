"""Tests for ais_core.lifecycle — state machine + health backoff (mocked)."""

from __future__ import annotations

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest

from ais_core import lifecycle
from ais_core.lifecycle import (
    EngineState,
    probe_health,
    process_alive,
    wait_for_health,
)
from ais_core.manifest import load_manifest

# ---------------------------------------------------------------------------
# probe_health / wait_for_health — real HTTP server, no mocks
# ---------------------------------------------------------------------------


class _Echo200(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args: object) -> None:
        pass


class _DelayedServer(BaseHTTPRequestHandler):
    """Refuses connections until ``delay_until`` has elapsed.

    Set via class attribute so the helper at module scope can flip it.
    """

    delay_until: float = 0.0

    def do_GET(self) -> None:
        if time.monotonic() < self.delay_until:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args: object) -> None:
        pass


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def healthy_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _Echo200)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


def _manifest_pointing_to_port(port: int):
    """Return a manifest tweaked so its health URL hits the test server."""
    from dataclasses import replace

    m = load_manifest("ollama")
    return replace(
        m,
        network=replace(
            m.network,
            port=port,
            bind="127.0.0.1",
            health_endpoint="/",
            health_timeout=5,
        ),
    )


def test_probe_health_returns_true_on_2xx(healthy_server: int) -> None:
    m = _manifest_pointing_to_port(healthy_server)
    assert probe_health(m) is True


def test_probe_health_returns_false_on_unreachable_host() -> None:
    """No server bound to the chosen port → probe must NOT raise."""
    m = _manifest_pointing_to_port(_free_port())
    assert probe_health(m) is False


def test_wait_for_health_returns_quickly_when_already_up(
    healthy_server: int,
) -> None:
    m = _manifest_pointing_to_port(healthy_server)
    t0 = time.monotonic()
    ok = wait_for_health(m, timeout=2)
    elapsed = time.monotonic() - t0
    assert ok is True
    assert elapsed < 1.0  # exponential backoff hits on first probe


def test_wait_for_health_times_out_when_endpoint_silent() -> None:
    m = _manifest_pointing_to_port(_free_port())
    t0 = time.monotonic()
    ok = wait_for_health(m, timeout=1)
    elapsed = time.monotonic() - t0
    assert ok is False
    assert 0.9 < elapsed < 2.0  # roughly the timeout, not a runaway loop


def test_wait_for_health_succeeds_after_delayed_start() -> None:
    """Endpoint flips healthy mid-poll → wait_for_health must catch it."""
    port = _free_port()
    _DelayedServer.delay_until = time.monotonic() + 0.7
    server = HTTPServer(("127.0.0.1", port), _DelayedServer)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        m = _manifest_pointing_to_port(port)
        t0 = time.monotonic()
        ok = wait_for_health(m, timeout=5, initial_delay=0.1, cap_delay=0.5)
        elapsed = time.monotonic() - t0
        assert ok is True
        assert elapsed >= 0.7
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# process_alive — pgrep mock
# ---------------------------------------------------------------------------


def test_process_alive_true_when_pgrep_returns_zero() -> None:
    m = load_manifest("ollama")
    fake = MagicMock(returncode=0, stdout=b"", stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake) as mock_run:
        assert process_alive(m) is True
    assert mock_run.call_args.args[0][:2] == ["pgrep", "-f"]


def test_process_alive_false_when_pgrep_returns_nonzero() -> None:
    m = load_manifest("ollama")
    fake = MagicMock(returncode=1, stdout=b"", stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake):
        assert process_alive(m) is False


# ---------------------------------------------------------------------------
# current_state — composes everything above
# ---------------------------------------------------------------------------


def test_current_state_not_installed_when_plist_absent(tmp_path) -> None:
    m = load_manifest("ollama")
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "missing.plist")):
        assert lifecycle.current_state(m) == EngineState.NOT_INSTALLED


# ---------------------------------------------------------------------------
# start / stop — verify exact subprocess args (no real launchctl)
# ---------------------------------------------------------------------------


def test_start_invokes_launchctl_load_w() -> None:
    m = load_manifest("ollama")
    with patch("ais_core.lifecycle.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        lifecycle.start(m)
    cmd = mock_run.call_args.args[0]
    assert cmd[:5] == [
        "sudo",
        "/bin/launchctl",
        "load",
        "-w",
        "/Library/LaunchDaemons/com.asiai.ollama.plist",
    ]


def test_stop_calls_stop_then_unload_then_pkill_only_if_alive() -> None:
    m = load_manifest("ollama")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        # Every subprocess call (launchctl stop, unload, pgrep, pkill) returns 0
        # so the test exercises the full happy path including the pkill branch.
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with (
        patch("ais_core.lifecycle.subprocess.run", side_effect=fake_run),
        patch("ais_core.lifecycle.time.sleep"),
    ):
        lifecycle.stop(m)

    [c[0] if isinstance(c, list) else c for c in calls]
    assert ["sudo", "/bin/launchctl", "stop", "com.asiai.ollama"] in calls
    assert any(c[:4] == ["sudo", "/bin/launchctl", "unload", "-w"] for c in calls)
    assert any(c[0] == "pkill" for c in calls)


# ---------------------------------------------------------------------------
# Dogfood-discovered bug fixes (US-003 / US-015 / US-017)
# ---------------------------------------------------------------------------


def test_install_dry_run_does_not_require_binary_present() -> None:
    """US-003: dry-run must not raise when the binary candidate is absent."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.manifest.BinarySpec.resolve", return_value=None),
        patch("ais_core.lifecycle.stop_existing"),
        patch("ais_core.lifecycle.plist.write_plist", return_value="/fake/plist"),
    ):
        result = lifecycle.install(m, user="jmn", dry_run=True)
    assert result["dry_run"] is True
    assert result["binary"] in m.binary.candidates
    assert result["health_ok"] is None


def test_install_creates_log_dir_before_launchctl(tmp_path) -> None:
    """US-015: log dir must be created (user-space, no sudo) before start."""
    m = load_manifest("ollama")
    fake_logs_dir = tmp_path / "Library" / "Logs" / "asiai" / "ollama"

    with (
        patch(
            "ais_core.manifest.LogSpec.expanded_dir",
            new_callable=lambda: property(lambda self: str(fake_logs_dir)),
        ),
        patch("ais_core.manifest.BinarySpec.resolve", return_value="/opt/homebrew/bin/ollama"),
        patch("ais_core.lifecycle.stop_existing"),
        patch("ais_core.lifecycle.plist.write_plist", return_value="/fake/plist"),
        patch("ais_core.lifecycle.start"),
        patch("ais_core.lifecycle.wait_for_health", return_value=True),
    ):
        lifecycle.install(m, user="jmn", dry_run=False)
    assert fake_logs_dir.is_dir()


def test_current_state_running_when_health_ok_even_if_launchctl_silent() -> None:
    """US-017: probe_health is the source of truth, not launchctl list.

    `launchctl list` for system daemons (in /Library/LaunchDaemons/) requires
    sudo from a non-root caller. The state machine must trust the network
    probe so the user sees RUNNING when the daemon is actually serving.
    """
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.Path.exists", return_value=True),
        patch("ais_core.lifecycle.probe_health", return_value=True),
    ):
        state = lifecycle.current_state(m)
    assert state == EngineState.RUNNING


def test_current_state_unhealthy_when_process_but_no_health() -> None:
    """US-017: process running but health probe fails → UNHEALTHY."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.Path.exists", return_value=True),
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.process_alive", return_value=True),
    ):
        state = lifecycle.current_state(m)
    assert state == EngineState.UNHEALTHY


def test_current_state_stopped_when_plist_present_but_silent() -> None:
    """US-017: plist exists but nothing else → STOPPED (terminal fallback)."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.Path.exists", return_value=True),
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.process_alive", return_value=False),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
    ):
        state = lifecycle.current_state(m)
    assert state == EngineState.STOPPED


# ---------------------------------------------------------------------------
# gen_probe / DEGRADED state — real HTTP server, POST /v1/chat/completions
# ---------------------------------------------------------------------------


class _GenHandler(BaseHTTPRequestHandler):
    """POST handler whose behaviour is driven by the ``mode`` class attribute.

    Modes mirror what llama-server actually does:
      ok         → 200 with a choices payload
      zombie     → 500 {"error": {"message": "Compute error."}} (post-OOM Metal)
      zombie200  → 200 body carrying "Compute error" (belt and braces)
      notfound   → 404 (no OpenAI-compatible endpoint)
      hang       → sleep beyond the client timeout (slots busy)
    """

    mode: str = "ok"

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.mode == "hang":
            time.sleep(2.0)
        if self.mode == "ok":
            body = b'{"choices":[{"message":{"role":"assistant","content":"!"}}]}'
            self.send_response(200)
        elif self.mode == "zombie":
            body = b'{"error":{"code":500,"message":"Compute error.","type":"server_error"}}'
            self.send_response(500)
        elif self.mode == "zombie200":
            body = b'{"error":"Compute error."}'
            self.send_response(200)
        elif self.mode == "notfound":
            body = b'{"error":"file not found"}'
            self.send_response(404)
        else:  # hang fell through after sleeping
            body = b'{"choices":[]}'
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # health endpoint stays green — that's the point
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def gen_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _GenHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        _GenHandler.mode = "ok"
        server.shutdown()


class TestGenProbe:
    def test_ok(self, gen_server: int) -> None:
        _GenHandler.mode = "ok"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.OK

    def test_zombie_on_http_500_compute_error(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.ZOMBIE

    def test_zombie_marker_in_200_body(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie200"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.ZOMBIE

    def test_unsupported_on_404(self, gen_server: int) -> None:
        _GenHandler.mode = "notfound"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.UNSUPPORTED

    def test_busy_on_timeout(self, gen_server: int) -> None:
        _GenHandler.mode = "hang"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m, request_timeout=0.3) is lifecycle.GenVerdict.BUSY

    def test_down_on_unreachable(self) -> None:
        m = _manifest_pointing_to_port(_free_port())  # nobody listens
        assert lifecycle.gen_probe(m, request_timeout=1.0) is lifecycle.GenVerdict.DOWN


class TestDeepState:
    def test_running_health_but_zombie_gen_is_degraded(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie"
        m = _manifest_pointing_to_port(gen_server)
        with patch("ais_core.lifecycle.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            assert lifecycle.current_state(m, deep=True) is EngineState.DEGRADED
            # shallow check still says RUNNING — that's the lie --deep exists for
            assert lifecycle.current_state(m) is EngineState.RUNNING

    def test_busy_engine_stays_running(self, gen_server: int) -> None:
        _GenHandler.mode = "ok"
        m = _manifest_pointing_to_port(gen_server)
        with (
            patch("ais_core.lifecycle.Path") as mock_path,
            patch("ais_core.lifecycle.gen_probe", return_value=lifecycle.GenVerdict.BUSY),
        ):
            mock_path.return_value.exists.return_value = True
            assert lifecycle.current_state(m, deep=True) is EngineState.RUNNING


class TestWaitForHealthGenCheck:
    def test_gen_check_optin_requires_generation(self, gen_server: int) -> None:
        from dataclasses import replace

        _GenHandler.mode = "zombie"
        m = _manifest_pointing_to_port(gen_server)
        m = replace(m, network=replace(m.network, gen_check=True, health_timeout=1))
        assert wait_for_health(m, timeout=1) is False

        _GenHandler.mode = "ok"
        assert wait_for_health(m, timeout=3) is True

    def test_without_optin_health_2xx_suffices(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie"  # gen broken, but gen_check is off
        m = _manifest_pointing_to_port(gen_server)
        assert wait_for_health(m, timeout=3) is True
