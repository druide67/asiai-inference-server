"""Tests for ais_core.lifecycle — state machine + health backoff (mocked)."""

from __future__ import annotations

import getpass
import importlib.util
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ais_core import lifecycle
from ais_core.lifecycle import (
    EngineState,
    probe_health,
    process_alive,
    wait_for_health,
)
from ais_core.manifest import list_manifests, load_manifest

# The privileged helper is a standalone ``python3 -I`` binary (not an importable package
# module); load it via importlib so the composition tests below can feed _install_args output
# through its REAL argparse — the seam the per-layer mocks skip.
_HELPER_PATH = Path(__file__).resolve().parents[1] / "data" / "helpers" / "asiai_priv.py"


@pytest.fixture(autouse=True)
def _isolate_host_bundle_state(monkeypatch, tmp_path):
    """Tests must never see THIS host's installed bundle app or active
    manifests — the dev machine runs the real thing (/Applications/Asiai.app
    embeds com.asiai.* labels, ~/.local/share/.../active/ holds real
    presets), which would flip provisioned-state and installed_model
    assertions depending on where the suite runs."""
    monkeypatch.setenv("ASIAI_BUNDLE_APP", str(tmp_path / "no-bundle.app"))
    monkeypatch.setenv("ASIAI_LAUNCH_MANIFEST_DIR", str(tmp_path / "no-active"))


def _load_priv_helper():
    spec = importlib.util.spec_from_file_location("asiai_priv_compose", _HELPER_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


class _Auth200(BaseHTTPRequestHandler):
    """Answers 200 only with the right Bearer key — MTPLX-style middleware
    that covers EVERY route, /health included."""

    expected_key = "sekret"

    def do_GET(self) -> None:
        if self.headers.get("Authorization") != f"Bearer {self.expected_key}":
            self.send_response(401)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def auth_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _Auth200)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield port
    finally:
        server.shutdown()


def _manifest_with_api_key_file(port: int, key_file: str):
    from dataclasses import replace

    m = _manifest_pointing_to_port(port)
    return replace(m, network=replace(m.network, api_key_file=key_file))


def test_probe_health_sends_bearer_from_api_key_file(auth_server: int, tmp_path) -> None:
    """[network].api_key_file → the probe authenticates (engines like MTPLX
    gate /health itself behind the key on non-localhost binds)."""
    key_file = tmp_path / "api-key"
    key_file.write_text("sekret\n")  # trailing newline must be stripped
    m = _manifest_with_api_key_file(auth_server, str(key_file))
    assert probe_health(m) is True


def test_probe_health_without_key_reads_401_as_unhealthy(auth_server: int) -> None:
    """No api_key_file declared → no header → the 401 is reported honestly."""
    m = _manifest_pointing_to_port(auth_server)
    assert probe_health(m) is False


def test_probe_health_with_missing_key_file_degrades_to_no_header(
    auth_server: int, tmp_path
) -> None:
    """An unreadable key file must not crash the probe — it degrades to an
    unauthenticated request (→ 401 → False), surfacing the misconfiguration
    as unhealthy instead of masking it."""
    m = _manifest_with_api_key_file(auth_server, str(tmp_path / "does-not-exist"))
    assert probe_health(m) is False


def test_probe_health_with_key_still_true_on_keyless_engine(healthy_server: int, tmp_path) -> None:
    """A declared key against an engine that ignores auth stays healthy —
    the header is additive, never harmful."""
    key_file = tmp_path / "api-key"
    key_file.write_text("sekret")
    m = _manifest_with_api_key_file(healthy_server, str(key_file))
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
    fake = MagicMock(returncode=0, stdout=b"123 ollama serve\n", stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake) as mock_run:
        assert process_alive(m) is True
    assert mock_run.call_args.args[0][:2] == ["pgrep", "-fl"]


def test_process_alive_disambiguates_shared_binary_by_port() -> None:
    """Aux presets share one binary: a neighbour's --port match must not
    count as OUR process (the weeks-long false UNHEALTHY on turboquant)."""
    m = load_manifest("llamacpp")  # port 8080
    neighbours = b"742 llama-server-turboquant --model x --port 8090\n"
    fake = MagicMock(returncode=0, stdout=neighbours, stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake):
        assert process_alive(m) is False

    ours = neighbours + b"901 llama-server-turboquant --model y --port 8080\n"
    fake_ours = MagicMock(returncode=0, stdout=ours, stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake_ours):
        assert process_alive(m) is True

    # Port-less matches (wrapper scripts, ollama-style launchers) keep the
    # historical pattern-only behaviour.
    portless = neighbours + b"555 turboquant-start\n"
    fake_mixed = MagicMock(returncode=0, stdout=portless, stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake_mixed):
        assert process_alive(m) is True


def test_process_alive_false_when_pgrep_returns_nonzero() -> None:
    m = load_manifest("ollama")
    fake = MagicMock(returncode=1, stdout=b"", stderr=b"")
    with patch("ais_core.lifecycle.subprocess.run", return_value=fake):
        assert process_alive(m) is False


# ---------------------------------------------------------------------------
# current_state — composes everything above
# ---------------------------------------------------------------------------


def test_current_state_not_installed_when_binary_also_absent(tmp_path) -> None:
    """Nothing provisioned AND the software itself is absent = NOT_INSTALLED."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "missing.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch.object(type(m.binary), "resolve", return_value=None),
    ):
        assert lifecycle.current_state(m) == EngineState.NOT_INSTALLED


def test_current_state_available_when_binary_present_but_unprovisioned(tmp_path) -> None:
    """Software installed (brew) but never provisioned as a managed service:
    AVAILABLE, not NOT_INSTALLED — the operator must not be told to install
    something that is already there (dogfooding report: ollama)."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "missing.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch.object(type(m.binary), "resolve", return_value="/opt/homebrew/bin/ollama"),
    ):
        assert lifecycle.current_state(m) == EngineState.AVAILABLE


def test_current_state_running_when_bundle_managed_without_legacy_plist(tmp_path) -> None:
    """Bundle-managed engine (SMAppService): no /Library/LaunchDaemons plist,
    but launchd knows the label and health answers -> RUNNING, not
    NOT_INSTALLED (deploy-M5 finding: the migrated 27B showed not_installed)."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "missing.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=True),
        patch("ais_core.lifecycle.probe_health", return_value=True),
    ):
        assert lifecycle.current_state(m) == EngineState.RUNNING


# ---------------------------------------------------------------------------
# start / stop — verify exact subprocess args (no real launchctl)
# ---------------------------------------------------------------------------


def test_start_enables_then_kickstarts() -> None:
    """Modern model: start = enable-daemon (clear any disable, parity with legacy load -w)
    then start-daemon (kickstart -k), both via the helper."""
    m = load_manifest("ollama")
    with patch("ais_core.lifecycle.privhelper.run") as mock_run:
        lifecycle.start(m)
    actions = [c.args[0] for c in mock_run.call_args_list]
    assert actions == ["enable-daemon", "start-daemon"]
    for call in mock_run.call_args_list:
        assert "--label" in call.args and m.plist.name in call.args


def test_stop_kills_via_helper_no_pkill() -> None:
    """stop = stop-daemon (kill SIGTERM), best-effort (check=False), and crucially NO pkill —
    a SIGKILL on a still-bootstrapped KeepAlive daemon would be read as a crash and respawned."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.privhelper.run") as mock_run,
        patch("ais_core.lifecycle.subprocess.run") as mock_sub,
    ):
        lifecycle.stop(m)
    assert mock_run.call_count == 1
    assert mock_run.call_args.args[0] == "stop-daemon"
    assert mock_run.call_args.kwargs.get("check") is False
    # No pkill (nor any raw subprocess) — the whole point of the lean stop.
    pkills = [
        c for c in mock_sub.call_args_list if c.args and c.args[0] and c.args[0][0] == "pkill"
    ]
    assert pkills == []


def test_restart_is_enable_then_kickstart() -> None:
    """restart = start (kickstart -k restarts atomically: kill current + start fresh)."""
    m = load_manifest("ollama")
    with patch("ais_core.lifecycle.privhelper.run") as mock_run:
        lifecycle.restart(m)
    actions = [c.args[0] for c in mock_run.call_args_list]
    assert actions == ["enable-daemon", "start-daemon"]


# ---------------------------------------------------------------------------
# Dogfood-discovered bug fixes (US-003 / US-015 / US-017)
# ---------------------------------------------------------------------------


def test_install_dry_run_does_not_require_binary_present() -> None:
    """US-003: dry-run must not raise when the binary candidate is absent."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.manifest.BinarySpec.resolve", return_value=None),
        patch("ais_core.lifecycle.stop_existing"),
        patch("ais_core.lifecycle.privhelper.run"),
    ):
        result = lifecycle.install(m, user=getpass.getuser(), dry_run=True)
    assert result["dry_run"] is True
    assert result["binary"] in m.binary.candidates
    assert result["health_ok"] is None


def test_install_invokes_helper_install_daemon() -> None:
    """Install routes through the helper (generate-don't-validate): it no longer writes the
    plist or mkdir's a user log dir — the helper generates the plist + creates the root-owned
    /Library/Logs/asiai leaves + bootstraps (which starts it), so no separate start() either."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.manifest.BinarySpec.resolve", return_value="/opt/homebrew/bin/ollama"),
        patch("ais_core.lifecycle.stop_existing"),
        patch("ais_core.lifecycle.privhelper.run") as mock_run,
        patch("ais_core.lifecycle.start") as mock_start,
        patch("ais_core.lifecycle.wait_for_health", return_value=True),
    ):
        lifecycle.install(m, user=getpass.getuser(), dry_run=False)
    assert mock_run.call_count == 1
    flags = mock_run.call_args.args
    assert flags[0] == "install-daemon"
    # flags are joined as --flag=value (so argparse never reads a value as an option)
    assert f"--label={m.plist.name}" in flags
    assert "--binary=/opt/homebrew/bin/ollama" in flags
    assert any(f.startswith("--user=") for f in flags)
    # bootstrap already starts it — no separate start() call
    mock_start.assert_not_called()


def test_current_state_running_when_health_ok_even_if_launchctl_silent() -> None:
    """US-017: probe_health is the source of truth, not launchctl list.

    `launchctl list` for system daemons (in /Library/LaunchDaemons/) requires
    sudo from a non-root caller. The state machine must trust the network
    probe so the user sees RUNNING when the daemon is actually serving.
    """
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.Path") as mock_path,
        patch("ais_core.lifecycle.probe_health", return_value=True),
    ):
        mock_path.return_value.exists.return_value = True
        state = lifecycle.current_state(m)
    assert state == EngineState.RUNNING


def test_current_state_unhealthy_when_process_but_no_health() -> None:
    """US-017: process running but health probe fails → UNHEALTHY."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.Path") as mock_path,
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.process_alive", return_value=True),
    ):
        mock_path.return_value.exists.return_value = True
        state = lifecycle.current_state(m)
    assert state == EngineState.UNHEALTHY


def test_current_state_stopped_when_plist_present_but_silent() -> None:
    """US-017: plist exists but nothing else → STOPPED (terminal fallback)."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.Path") as mock_path,
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.process_alive", return_value=False),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch("ais_core.lifecycle.is_disabled", return_value=False),
    ):
        mock_path.return_value.exists.return_value = True
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
      zombie200choices → 200 body with BOTH a choices array and the marker
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
        elif self.mode == "zombie200choices":
            body = b'{"choices":[],"error":"Compute error."}'
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

    def test_gen_probe_sends_bearer_from_api_key_file(self, tmp_path) -> None:
        """Same auth contract as probe_health: a key-gated engine (MTPLX)
        must see the Bearer header on the generation probe too."""

        class _AuthGen(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                if self.headers.get("Authorization") != "Bearer sekret":
                    body = b'{"error":{"message":"missing or invalid API key"}}'
                    self.send_response(401)
                else:
                    body = b'{"choices":[{"message":{"role":"assistant","content":"!"}}]}'
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args: object) -> None:
                pass

        port = _free_port()
        server = HTTPServer(("127.0.0.1", port), _AuthGen)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            key_file = tmp_path / "api-key"
            key_file.write_text("sekret\n")
            m = _manifest_with_api_key_file(port, str(key_file))
            assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.OK
            # Without the declaration the 401 reads UNSUPPORTED (4xx) —
            # non-alarming, but not OK.
            m_bare = _manifest_pointing_to_port(port)
            assert lifecycle.gen_probe(m_bare) is lifecycle.GenVerdict.UNSUPPORTED
        finally:
            server.shutdown()

    def test_zombie_on_http_500_compute_error(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.ZOMBIE

    def test_zombie_marker_in_200_body(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie200"
        m = _manifest_pointing_to_port(gen_server)
        assert lifecycle.gen_probe(m) is lifecycle.GenVerdict.ZOMBIE

    def test_zombie_marker_wins_over_choices_in_200_body(self, gen_server: int) -> None:
        """A 2xx body carrying both a choices array and the compute-error
        marker is a broken backend — the marker takes precedence."""
        _GenHandler.mode = "zombie200choices"
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

    def test_probe_state_returns_verdict_with_state(self, gen_server: int) -> None:
        """probe_state exposes the single gen probe's verdict so display
        callers (aisctl status --deep) never have to probe twice."""
        _GenHandler.mode = "zombie"
        m = _manifest_pointing_to_port(gen_server)
        with patch("ais_core.lifecycle.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            state, verdict = lifecycle.probe_state(m, deep=True)
            assert state is EngineState.DEGRADED
            assert verdict is lifecycle.GenVerdict.ZOMBIE
            # shallow: no probe, no verdict
            assert lifecycle.probe_state(m) == (EngineState.RUNNING, None)


class TestDisableEnable:
    def test_disable_disables_then_stops(self) -> None:
        m = load_manifest("ollama")
        # disable-daemon (persistent override) BEFORE stop-daemon (SIGTERM) is the safety
        # property: a KeepAlive daemon must not respawn in the gap.
        with patch("ais_core.lifecycle.privhelper.run") as mock_run:
            result = lifecycle.disable(m)
        actions = [c.args[0] for c in mock_run.call_args_list]
        assert actions == ["disable-daemon", "stop-daemon"]
        for call in mock_run.call_args_list:
            assert "--label" in call.args and m.plist.name in call.args
        # stop is best-effort (kill on an already-down service is a no-op, not a failure)
        assert mock_run.call_args_list[1].kwargs.get("check") is False
        assert result["disabled"] is True

    def test_disable_dry_run_forwards_dry_run(self) -> None:
        m = load_manifest("ollama")
        with patch("ais_core.lifecycle.privhelper.run") as mock_run:
            result = lifecycle.disable(m, dry_run=True)
        assert mock_run.call_count == 2
        assert all(c.kwargs.get("dry_run") is True for c in mock_run.call_args_list)
        assert result["dry_run"] is True

    def test_enable_without_start_does_not_start(self) -> None:
        m = load_manifest("ollama")
        with patch("ais_core.lifecycle.privhelper.run") as mock_run:
            result = lifecycle.enable(m)
        actions = [c.args[0] for c in mock_run.call_args_list]
        assert actions == ["enable-daemon"]  # no start-daemon
        assert result["enabled"] is True
        assert result["started"] is False

    def test_enable_with_start_starts(self) -> None:
        m = load_manifest("ollama")
        with patch("ais_core.lifecycle.privhelper.run") as mock_run:
            result = lifecycle.enable(m, start_now=True)
        actions = [c.args[0] for c in mock_run.call_args_list]
        assert actions == ["enable-daemon", "start-daemon"]
        assert result["started"] is True


class TestIsDisabled:
    def _run_result(self, stdout: str, returncode: int = 0) -> MagicMock:
        return MagicMock(returncode=returncode, stdout=stdout)

    def test_modern_output_disabled(self) -> None:
        m = load_manifest("ollama")
        out = f'disabled services = {{\n\t"{m.plist.name}" => disabled\n}}\n'
        with patch("ais_core.lifecycle.subprocess.run", return_value=self._run_result(out)):
            assert lifecycle.is_disabled(m) is True

    def test_legacy_output_true(self) -> None:
        m = load_manifest("ollama")
        out = f'disabled services = {{\n\t"{m.plist.name}" => true\n}}\n'
        with patch("ais_core.lifecycle.subprocess.run", return_value=self._run_result(out)):
            assert lifecycle.is_disabled(m) is True

    def test_absent_label_is_not_disabled(self) -> None:
        m = load_manifest("ollama")
        out = 'disabled services = {\n\t"com.other.svc" => disabled\n}\n'
        with patch("ais_core.lifecycle.subprocess.run", return_value=self._run_result(out)):
            assert lifecycle.is_disabled(m) is False

    def test_command_failure_degrades_to_false(self) -> None:
        m = load_manifest("ollama")
        with patch(
            "ais_core.lifecycle.subprocess.run", return_value=self._run_result("", returncode=1)
        ):
            assert lifecycle.is_disabled(m) is False

    def test_probe_state_reports_disabled(self) -> None:
        m = load_manifest("ollama")
        with (
            patch("ais_core.lifecycle.Path") as mock_path,
            patch("ais_core.lifecycle.probe_health", return_value=False),
            patch("ais_core.lifecycle.process_alive", return_value=False),
            patch("ais_core.lifecycle.is_loaded", return_value=False),
            patch("ais_core.lifecycle.is_disabled", return_value=True),
        ):
            mock_path.return_value.exists.return_value = True
            assert lifecycle.probe_state(m) == (EngineState.DISABLED, None)


class TestWaitForHealthGenCheck:
    def test_gen_check_optin_requires_generation(self, gen_server: int) -> None:
        from dataclasses import replace

        _GenHandler.mode = "zombie"
        m = _manifest_pointing_to_port(gen_server)
        m = replace(m, network=replace(m.network, gen_check=True, health_timeout=1))
        assert wait_for_health(m, timeout=1) is False

        _GenHandler.mode = "ok"
        assert wait_for_health(m, timeout=3) is True

    def test_hung_generation_cannot_overshoot_deadline(self, gen_server: int) -> None:
        """The gen probe's timeout is clamped to the remaining budget: a hung
        server (answers after 2s) must NOT turn a timeout=1 wait into a
        2s+ success — the wait fails within its deadline."""
        from dataclasses import replace

        _GenHandler.mode = "hang"
        m = _manifest_pointing_to_port(gen_server)
        m = replace(m, network=replace(m.network, gen_check=True, health_timeout=1))
        started = time.monotonic()
        assert wait_for_health(m, timeout=1) is False
        assert time.monotonic() - started < 1.9  # hang answers at 2.0s

    def test_without_optin_health_2xx_suffices(self, gen_server: int) -> None:
        _GenHandler.mode = "zombie"  # gen broken, but gen_check is off
        m = _manifest_pointing_to_port(gen_server)
        assert wait_for_health(m, timeout=3) is True


# ---------------------------------------------------------------------------
# Composition guard: _install_args output MUST parse through the REAL helper
# argparse. The per-layer unit tests mock privhelper.run (asserting only the
# flag list) and the helper tests pass hand-built argv — neither feeds one into
# the other, which let a parse-fatal bug ship green. This closes that seam.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list_manifests())
def test_install_args_parse_through_real_helper(name: str) -> None:
    """For every bundled manifest, lifecycle._install_args(...) must parse cleanly through the
    real asiai-priv parser (no SystemExit) — catching the dash-prefixed program-arg class
    (--flash-attn, --mlock, --host…) that action='append' rejects unless joined with '='."""
    m = load_manifest(name)
    argv = lifecycle._install_args(m, user="someuser", binary_path="/opt/homebrew/bin/engine")
    ns = _load_priv_helper()._build_parser().parse_args(["install-daemon", *argv])

    assert ns.action == "install-daemon"
    assert ns.label == m.plist.name
    expected_binary = m.wrapper.install_path if m.wrapper.needed else "/opt/homebrew/bin/engine"
    assert ns.binary == expected_binary
    if not m.wrapper.needed:
        # every manifest program-arg round-trips verbatim, dash-prefixed flags included
        for pa in m.binary.program_args:
            assert pa in ns.program_arg
        # --host/--port emitted together iff the manifest binds (parity with the old plist)
        if m.network.bind:
            assert ns.port == str(m.network.port)
            assert "--host" in ns.program_arg and m.network.bind in ns.program_arg
        else:
            assert ns.port is None


def test_install_args_mtplx_hermes_preset_parses_through_real_helper() -> None:
    """The mtplx Hermes preset brings two firsts the guard above (baselines
    only) doesn't cover: a POSITIONAL subcommand token (``quickstart``) and
    a tilde path as a program-arg VALUE (``--api-key-file ~/.mtplx/api-key``,
    expanded by MTPLX against the daemon's $HOME, deliberately NOT by us)."""
    m = load_manifest("mtplx", preset="qwen3.6-27b-mtplx-hermes-agent")
    argv = lifecycle._install_args(m, user="someuser", binary_path="/opt/homebrew/bin/mtplx")
    ns = _load_priv_helper()._build_parser().parse_args(["install-daemon", *argv])

    assert ns.label == "com.asiai.mtplx"
    assert ns.program_arg[0] == "quickstart"
    # tilde path round-trips verbatim (no client-side expansion)
    assert "~/.mtplx/api-key" in ns.program_arg
    # LAN bind → --host/--port emitted
    assert ns.port == "8080"
    assert "--host" in ns.program_arg and "0.0.0.0" in ns.program_arg
    # the repo id goes through --program-arg, never --model-path
    assert ns.model_path is None
    assert "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed" in ns.program_arg


def test_install_args_ollama_no_port_llamacpp_has_port_and_dash_flags() -> None:
    """Pin the two reproduced regressions: ollama (bind='') must get NO --port (else
    'ollama serve --port' crash-loops); llamacpp must get --port AND its dash-prefixed flags
    joined with '=' so the helper parser accepts them."""
    args_o = lifecycle._install_args(
        load_manifest("ollama"), user="u", binary_path="/opt/homebrew/bin/ollama"
    )
    assert not any(a.startswith("--port") for a in args_o)

    args_l = lifecycle._install_args(
        load_manifest("llamacpp"), user="u", binary_path="/opt/homebrew/bin/llama-server"
    )
    assert any(a.startswith("--port=") for a in args_l)
    assert any(a.startswith("--program-arg=--") for a in args_l)  # dash-prefixed flag, joined


# ---------------------------------------------------------------------------
# S0 (2026-07-01 half-uninstall retex): pf preflight ordering + keep_firewall
# ---------------------------------------------------------------------------


def test_uninstall_preflight_fails_before_daemon_teardown() -> None:
    """The retex bug: pf removal died AFTER the daemon was booted out. The preflight
    must raise BEFORE the helper is invoked, leaving the engine untouched."""
    from ais_core.firewall import FirewallError

    m = load_manifest("llamacpp")
    with (
        patch("ais_core.lifecycle.firewall.anchor_present", return_value=True),
        patch(
            "ais_core.lifecycle.firewall.preflight_sudo",
            side_effect=FirewallError("no TTY"),
        ),
        patch("ais_core.lifecycle.privhelper.run") as mock_helper,
        patch("ais_core.lifecycle.firewall.remove_anchor") as mock_remove,
        pytest.raises(FirewallError),
    ):
        lifecycle.uninstall(m)
    mock_helper.assert_not_called()
    mock_remove.assert_not_called()


def test_uninstall_keep_firewall_skips_pf_entirely() -> None:
    """keep_firewall=True (reinstall with an unchanged anchor): no preflight, no removal."""
    m = load_manifest("llamacpp")
    with (
        patch("ais_core.lifecycle.firewall.preflight_sudo") as mock_pre,
        patch("ais_core.lifecycle.privhelper.run") as mock_helper,
        patch("ais_core.lifecycle.firewall.remove_anchor") as mock_remove,
    ):
        result = lifecycle.uninstall(m, keep_firewall=True)
    mock_pre.assert_not_called()
    mock_remove.assert_not_called()
    mock_helper.assert_called_once()
    assert result["firewall_kept"] is True
    assert result["firewall_removed"] is False


def test_uninstall_without_anchor_needs_no_preflight() -> None:
    """No anchor trace on disk/pf.conf → the password-gated path is never consulted."""
    m = load_manifest("llamacpp")
    with (
        patch("ais_core.lifecycle.firewall.anchor_present", return_value=False),
        patch("ais_core.lifecycle.firewall.preflight_sudo") as mock_pre,
        patch("ais_core.lifecycle.privhelper.run") as mock_helper,
        patch("ais_core.lifecycle.firewall.remove_anchor") as mock_remove,
    ):
        lifecycle.uninstall(m)
    mock_pre.assert_not_called()
    mock_remove.assert_not_called()
    mock_helper.assert_called_once()


def test_install_preflight_fails_before_any_mutation_when_anchor_stale() -> None:
    """enable_firewall + stale/missing anchor + no sudo → refuse before stop/install."""
    from ais_core.firewall import FirewallError

    m = load_manifest("llamacpp")
    with (
        patch("ais_core.manifest.BinarySpec.resolve", return_value="/opt/homebrew/bin/x"),
        patch("ais_core.lifecycle.firewall.anchor_up_to_date", return_value=False),
        patch(
            "ais_core.lifecycle.firewall.preflight_sudo",
            side_effect=FirewallError("no TTY"),
        ),
        patch("ais_core.lifecycle.stop_existing") as mock_stop,
        patch("ais_core.lifecycle.privhelper.run") as mock_helper,
        pytest.raises(FirewallError),
    ):
        lifecycle.install(m, user=getpass.getuser(), enable_firewall=True)
    mock_stop.assert_not_called()
    mock_helper.assert_not_called()


# ---------------------------------------------------------------------------
# SA audit #1: client-side symlink resolution of model/template paths
# ---------------------------------------------------------------------------


def test_resolve_user_file_resolves_absolute_symlink(tmp_path) -> None:
    """The active.gguf convention: the symlink is resolved HERE so the helper (which
    refuses a symlink final component by design) receives the pinned real target."""
    real = tmp_path / "real-model.gguf"
    real.write_bytes(b"gguf")
    link = tmp_path / "active.gguf"
    link.symlink_to(real)
    assert lifecycle._resolve_user_file(str(link)) == str(real.resolve())


def test_resolve_user_file_passes_tilde_through_untouched() -> None:
    """Tilde paths expand against the DAEMON home inside the helper, never ours."""
    assert lifecycle._resolve_user_file("~/llms/gguf/active.gguf") == "~/llms/gguf/active.gguf"


def test_probe_state_running_via_network_even_without_plist(tmp_path) -> None:
    """Network first: an engine answering /health is RUNNING even when no
    plist exists and launchd never heard of it (desktop app, hand-launched
    server, bundle daemon invisible from a user domain)."""
    m = load_manifest("ollama")
    with (
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "missing.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch("ais_core.lifecycle.probe_health", return_value=True),
    ):
        assert lifecycle.current_state(m) == EngineState.RUNNING


def test_installed_model_reads_plist_program_arguments(tmp_path) -> None:
    import plistlib as _plistlib

    m = load_manifest("llamacpp")
    plist_file = tmp_path / "com.asiai.llamacpp.plist"
    with open(plist_file, "wb") as f:
        _plistlib.dump(
            {
                "Label": "com.asiai.llamacpp",
                "ProgramArguments": [
                    "/opt/homebrew/bin/llama-server",
                    "--model",
                    "/Users/x/llms/Qwen3-27B-Q8.gguf",
                    "--port",
                    "8080",
                ],
            },
            f,
        )
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(plist_file)):
        assert lifecycle.installed_model(m) == "Qwen3-27B-Q8.gguf"


def test_installed_model_resolves_preset_symlink(tmp_path) -> None:
    """A preset's stable active.gguf symlink must resolve to the real file:
    'preset: active.gguf' tells the operator nothing."""
    import plistlib as _plistlib

    real = tmp_path / "Qwen3-4B-Instruct-UD-Q5_K_XL.gguf"
    real.write_bytes(b"gguf")
    link = tmp_path / "active.gguf"
    link.symlink_to(real)
    m = load_manifest("llamacpp")
    plist_file = tmp_path / "com.asiai.llamacpp.plist"
    with open(plist_file, "wb") as f:
        _plistlib.dump({"ProgramArguments": ["bin", "--model", str(link)]}, f)
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(plist_file)):
        assert lifecycle.installed_model(m) == "Qwen3-4B-Instruct-UD-Q5_K_XL.gguf"


def test_installed_model_none_for_wrapper_or_missing(tmp_path) -> None:
    m = load_manifest("llamacpp")
    # missing plist
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "nope.plist")):
        assert lifecycle.installed_model(m) is None
    # wrapper plist without --model
    import plistlib as _plistlib

    plist_file = tmp_path / "wrapper.plist"
    with open(plist_file, "wb") as f:
        _plistlib.dump({"ProgramArguments": ["/usr/local/bin/turboquant-start"]}, f)
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(plist_file)):
        assert lifecycle.installed_model(m) is None


# ---------------------------------------------------------------------------
# bundle-aware states + model (deploy-M4 findings, 0.8.2)
# ---------------------------------------------------------------------------


def _fake_bundle_app(tmp_path: Path, label: str) -> Path:
    app = tmp_path / "Asiai.app"
    daemons = app / "Contents" / "Library" / "LaunchDaemons"
    daemons.mkdir(parents=True)
    (daemons / f"{label}.plist").write_bytes(b"<plist/>")
    return app


def test_probe_state_bundle_dormant_is_stopped(tmp_path, monkeypatch) -> None:
    """A bundle-provisioned service whose label is unloaded is STOPPED, not
    AVAILABLE: with no legacy plist on disk, the embedded plist inside the
    installed app is the only provisioning evidence (deploy-M4 finding —
    dormant bundle services looked like never-provisioned software)."""
    m = load_manifest("llamacpp")
    monkeypatch.setenv("ASIAI_BUNDLE_APP", str(_fake_bundle_app(tmp_path, m.plist.name)))
    with (
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "no.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch("ais_core.lifecycle.process_alive", return_value=False),
        patch("ais_core.lifecycle.is_disabled", return_value=False),
    ):
        assert lifecycle.current_state(m) == EngineState.STOPPED


def test_probe_state_bundle_dormant_disabled(tmp_path, monkeypatch) -> None:
    """Same dormant bundle service with a launchd override: DISABLED (the
    cold-standby truth an operator acts on — Start vs re-enable)."""
    m = load_manifest("llamacpp")
    monkeypatch.setenv("ASIAI_BUNDLE_APP", str(_fake_bundle_app(tmp_path, m.plist.name)))
    with (
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "no.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch("ais_core.lifecycle.process_alive", return_value=False),
        patch("ais_core.lifecycle.is_disabled", return_value=True),
    ):
        assert lifecycle.current_state(m) == EngineState.DISABLED


def test_probe_state_other_bundle_service_stays_available(tmp_path, monkeypatch) -> None:
    """The installed app embeds OTHER labels only: this engine is still just
    AVAILABLE — bundle evidence is per-label, not per-host."""
    m = load_manifest("ollama")
    monkeypatch.setenv(
        "ASIAI_BUNDLE_APP", str(_fake_bundle_app(tmp_path, "com.asiai.something-else"))
    )
    with (
        patch("ais_core.lifecycle.probe_health", return_value=False),
        patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "no.plist")),
        patch("ais_core.lifecycle.is_loaded", return_value=False),
        patch.object(type(m.binary), "resolve", return_value="/opt/homebrew/bin/ollama"),
    ):
        assert lifecycle.current_state(m) == EngineState.AVAILABLE


def test_installed_model_falls_back_to_active_manifest(tmp_path, monkeypatch) -> None:
    """Bundle-managed engines have no legacy plist to read argv from: the
    active manifest published by ``aisctl bundle activate`` carries the
    model, symlink resolved like the plist path (deploy-M4 finding —
    model=None wiped 'preset: X' off every dormant card)."""
    real = tmp_path / "Qwen3-4B-Instruct-UD-Q5_K_XL.gguf"
    real.write_bytes(b"gguf")
    link = tmp_path / "active.gguf"
    link.symlink_to(real)
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    (active_dir / "llamacpp.toml").write_text(f'[binary]\nmodel_path = "{link}"\n')
    monkeypatch.setenv("ASIAI_LAUNCH_MANIFEST_DIR", str(active_dir))
    m = load_manifest("llamacpp")
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(tmp_path / "no.plist")):
        assert lifecycle.installed_model(m) == "Qwen3-4B-Instruct-UD-Q5_K_XL.gguf"


def test_installed_model_legacy_plist_wins_over_active_manifest(tmp_path, monkeypatch) -> None:
    """When both exist the INSTALLED plist is the truth — it is what launchd
    actually execs on a legacy service; the active manifest may hold a
    fresher preset not yet applied."""
    import plistlib as _plistlib

    active_dir = tmp_path / "active"
    active_dir.mkdir()
    (active_dir / "llamacpp.toml").write_text('[binary]\nmodel_path = "/x/from-active.gguf"\n')
    monkeypatch.setenv("ASIAI_LAUNCH_MANIFEST_DIR", str(active_dir))
    plist_file = tmp_path / "com.asiai.llamacpp.plist"
    with open(plist_file, "wb") as f:
        _plistlib.dump({"ProgramArguments": ["bin", "--model", "/x/from-plist.gguf"]}, f)
    m = load_manifest("llamacpp")
    with patch("ais_core.lifecycle.plist.plist_path", return_value=str(plist_file)):
        assert lifecycle.installed_model(m) == "from-plist.gguf"
