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
    assert cmd[:5] == ["sudo", "/bin/launchctl", "load", "-w",
                       "/Library/LaunchDaemons/com.asiai.ollama.plist"]


def test_stop_calls_stop_then_unload_then_pkill_only_if_alive() -> None:
    m = load_manifest("ollama")
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(cmd)
        # Every subprocess call (launchctl stop, unload, pgrep, pkill) returns 0
        # so the test exercises the full happy path including the pkill branch.
        return MagicMock(returncode=0, stdout=b"", stderr=b"")

    with patch("ais_core.lifecycle.subprocess.run", side_effect=fake_run), \
         patch("ais_core.lifecycle.time.sleep"):
        lifecycle.stop(m)

    [c[0] if isinstance(c, list) else c for c in calls]
    assert ["sudo", "/bin/launchctl", "stop", "com.asiai.ollama"] in calls
    assert any(c[:4] == ["sudo", "/bin/launchctl", "unload", "-w"] for c in calls)
    assert any(c[0] == "pkill" for c in calls)
