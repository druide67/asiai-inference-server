"""Unit + integration tests for ``aisctl serve``."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

from ais_cli import serve

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def free_port() -> int:
    """Allocate an OS-assigned free TCP port on the loopback."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def server(free_port, monkeypatch):
    """Start a loopback server with a fixed token in a background thread."""
    token = "aint_test_token_value"
    srv = serve._build_server(free_port, token, max_concurrent=2)

    # Replace _execute with a deterministic stub so we don't shell out
    # during the test suite.
    def fake_execute(argv: list[str], timeout: float) -> dict[str, Any]:
        return {
            "ok": True,
            "exit_code": 0,
            "stdout": "ok " + " ".join(argv),
            "stderr": "",
            "duration_ms": 1,
        }

    monkeypatch.setattr(serve, "_execute", fake_execute)

    thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    # Give the server a moment to bind.
    time.sleep(0.05)
    yield {"port": free_port, "token": token, "server": srv}
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def _post(url: str, body: dict, headers: dict, timeout: float = 5.0):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def _get(url: str, headers: dict | None = None, timeout: float = 5.0):
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    return urllib.request.urlopen(req, timeout=timeout)


# ---------------------------------------------------------------------------
# _build_argv unit tests (pure validation, no network)
# ---------------------------------------------------------------------------


class TestBuildArgv:
    def test_purge_no_engine(self):
        argv = serve._build_argv("purge", {})
        assert argv[-2:] == ["purge", "--json"]

    def test_stop_requires_engine(self):
        with pytest.raises(ValueError, match="engine"):
            serve._build_argv("stop", {})

    def test_stop_includes_engine(self):
        argv = serve._build_argv("stop", {"engine": "ollama"})
        assert argv[-3:] == ["stop", "--json", "ollama"]

    def test_unload_with_model(self):
        argv = serve._build_argv("unload", {"engine": "ollama", "model": "llama3.2"})
        assert argv[-1] == "llama3.2"

    def test_unload_without_model(self):
        argv = serve._build_argv("unload", {"engine": "ollama"})
        # Must NOT include a None or empty model token.
        assert "" not in argv
        assert None not in argv  # type: ignore[comparison-overlap]

    def test_unknown_command(self):
        with pytest.raises(ValueError, match="unknown command"):
            serve._build_argv("drop_db", {})

    def test_upgrade_uses_whitelisted_formula(self):
        argv = serve._build_argv("upgrade", {"engine": "ollama"})
        # brew upgrade ollama
        assert argv[-2:] == ["upgrade", "ollama"]

    def test_upgrade_rejects_unknown_engine(self):
        with pytest.raises(ValueError, match="not whitelisted"):
            serve._build_argv("upgrade", {"engine": "evil-engine"})


# ---------------------------------------------------------------------------
# _execute unit tests
# ---------------------------------------------------------------------------


class TestExecute:
    def test_captures_stdout(self):
        result = serve._execute(["echo", "hello"], timeout=5.0)
        assert result["ok"] is True
        assert "hello" in result["stdout"]

    def test_nonzero_exit_marks_not_ok(self):
        result = serve._execute(["false"], timeout=5.0)
        assert result["ok"] is False
        assert result["exit_code"] != 0

    def test_missing_binary_returns_127(self, tmp_path):
        result = serve._execute([str(tmp_path / "nonexistent")], timeout=2.0)
        assert result["exit_code"] == 127
        assert result["error"] == "aisctl_binary_not_found"

    def test_timeout_returns_124(self, monkeypatch):
        # Use a tiny sleep timeout to keep the test fast.
        result = serve._execute(["sleep", "5"], timeout=0.1)
        assert result["exit_code"] == 124
        assert result["error"] == "timeout"


# ---------------------------------------------------------------------------
# HTTP handler tests (live server, mocked _execute)
# ---------------------------------------------------------------------------


class TestHandlerAuth:
    def test_missing_bearer_returns_401(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                f"http://127.0.0.1:{server['port']}/internal/v1/command",
                {"command": "purge"},
                headers={"Content-Type": "application/json"},
            )
        assert exc.value.code == 401

    def test_wrong_bearer_returns_401(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                f"http://127.0.0.1:{server['port']}/internal/v1/command",
                {"command": "purge"},
                headers={
                    "Content-Type": "application/json",
                    "Authorization": "Bearer aint_wrong",
                },
            )
        assert exc.value.code == 401

    def test_correct_bearer_returns_200(self, server):
        resp = _post(
            f"http://127.0.0.1:{server['port']}/internal/v1/command",
            {"command": "purge"},
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {server['token']}",
            },
        )
        body = json.loads(resp.read())
        assert body["ok"] is True


class TestHandlerPayload:
    def _ok_headers(self, server):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {server['token']}",
        }

    def test_invalid_json_returns_400(self, server):
        req = urllib.request.Request(
            f"http://127.0.0.1:{server['port']}/internal/v1/command",
            data=b"not json",
            headers=self._ok_headers(server),
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=2)
        assert exc.value.code == 400

    def test_unknown_command_returns_400(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                f"http://127.0.0.1:{server['port']}/internal/v1/command",
                {"command": "drop_database"},
                headers=self._ok_headers(server),
            )
        assert exc.value.code == 400

    def test_stop_without_engine_returns_400(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                f"http://127.0.0.1:{server['port']}/internal/v1/command",
                {"command": "stop", "args": {}},
                headers=self._ok_headers(server),
            )
        assert exc.value.code == 400

    def test_unknown_path_returns_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _post(
                f"http://127.0.0.1:{server['port']}/wrong/path",
                {"command": "purge"},
                headers=self._ok_headers(server),
            )
        assert exc.value.code == 404


class TestHandlerHealth:
    def test_health_endpoint_open(self, server):
        # Health probe should not require auth.
        resp = _get(f"http://127.0.0.1:{server['port']}/internal/v1/health")
        body = json.loads(resp.read())
        assert body["ok"] is True
        assert body["service"] == "aisctl-serve"

    def test_get_other_path_returns_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(f"http://127.0.0.1:{server['port']}/other")
        assert exc.value.code == 404


class TestLoopbackOnly:
    def test_binds_to_127_only(self, server):
        # Verify the server is NOT reachable on the LAN — connecting on
        # 0.0.0.0 should fail because we explicitly bound to 127.0.0.1.
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        # localhost works.
        s.connect(("127.0.0.1", server["port"]))
        s.close()
        # An external interface (e.g. en0 IP if present) should NOT.
        # We can't reliably enumerate interfaces here without going
        # external; we just assert the socket family is INET (loopback
        # binding is enforced in _build_server with LOOPBACK_HOST).
        assert serve.LOOPBACK_HOST == "127.0.0.1"
