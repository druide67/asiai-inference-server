"""Unit + integration tests for ``aisctl serve``."""

from __future__ import annotations

import json
import socket
import threading
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
    # No bind wait needed: _build_server binds synchronously before this point.
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

    @pytest.mark.parametrize(
        "command",
        ["install-service", "uninstall-service", "install-reserved-service", "bootstrap"],
    )
    def test_install_family_never_routable(self, command):
        """The loopback hop forwards only the fleet lifecycle whitelist (the shared
        ALLOWED_COMMANDS). Service provisioning / bootstrap is operator-only (privileged);
        it must NEVER be reachable over :8898, so a compromised asiai-web edge cannot
        drive an install."""
        assert command not in serve.ALLOWED_COMMANDS
        with pytest.raises(ValueError, match="unknown command"):
            serve._build_argv(command, {"service": "asiai-web", "engine": "ollama"})

    def test_upgrade_routes_through_aisctl(self):
        # Routed via `aisctl upgrade` so the OperationsLock + JSON envelope
        # apply (a direct brew argv used to skip both).
        argv = serve._build_argv("upgrade", {"engine": "ollama"})
        assert argv[0].endswith("aisctl")
        assert argv[1:3] == ["upgrade", "ollama"]
        assert "--json" in argv
        # Inner tool deadline is derived from the shared spec (work budget - headroom),
        # under the loopback subprocess-kill so aisctl reports its own timeout.
        from asiai.fleet.command_spec import inner_tool_timeout, loopback_timeout

        assert "--timeout" in argv
        inner = int(argv[argv.index("--timeout") + 1])
        assert inner == int(inner_tool_timeout("upgrade"))
        assert inner < loopback_timeout("upgrade")

    def test_enable_disable_route_with_engine(self):
        """Cold-standby pair (asiai>=1.18 command_spec): same engine-argv
        shape as start/stop — without the engine token the subprocess would
        die on argparse instead of a clean 400."""
        for cmd in ("enable", "disable"):
            argv = serve._build_argv(cmd, {"engine": "llamacpp-aux-4"})
            assert argv[-3:] == [cmd, "--json", "llamacpp-aux-4"]

    def test_enable_disable_require_engine(self):
        for cmd in ("enable", "disable"):
            with pytest.raises(ValueError, match="engine"):
                serve._build_argv(cmd, {})

    def test_upgrade_rejects_unknown_engine(self):
        with pytest.raises(ValueError, match="not whitelisted"):
            serve._build_argv("upgrade", {"engine": "evil-engine"})

    def test_install_with_preset(self):
        argv = serve._build_argv("install", {"engine": "llamacpp", "preset": "hermes-aux-1"})
        assert argv[-2:] == ["--preset", "hermes-aux-1"]
        assert argv[argv.index("--json") + 1] == "llamacpp"

    def test_install_without_preset_adds_no_flag(self):
        argv = serve._build_argv("install", {"engine": "llamacpp"})
        assert "--preset" not in argv

    @pytest.mark.parametrize(
        "bad",
        ["../evil", "a b", "x" * 65, "", 42, "-flag"],
    )
    def test_install_rejects_malformed_preset(self, bad):
        with pytest.raises(ValueError, match="preset"):
            serve._build_argv("install", {"engine": "llamacpp", "preset": bad})

    def test_preset_ignored_on_other_commands(self):
        """A preset smuggled onto a non-install verb must not reach argv."""
        argv = serve._build_argv("stop", {"engine": "ollama", "preset": "hermes-aux-1"})
        assert "--preset" not in argv


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
        # Assert the address the live server actually bound, not a constant.
        assert server["server"].server_address[0] == "127.0.0.1"


class TestEnginesState:
    @pytest.fixture(autouse=True)
    def _reset_cache(self):
        # Module-level cache: isolate every test from the previous one.
        serve._state_cache["ts"] = 0.0
        serve._state_cache["payload"] = None
        yield
        serve._state_cache["ts"] = 0.0
        serve._state_cache["payload"] = None

    def _url(self, server) -> str:
        return f"http://127.0.0.1:{server['port']}/internal/v1/engines-state"

    def test_requires_auth_unlike_health(self, server):
        # States disclose manifest names/ports — same Bearer gate as writes.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server))
        assert exc.value.code == 401

    def test_returns_probed_states(self, server, monkeypatch):
        payload = {
            "engines": [
                {
                    "name": "llamacpp-aux-1",
                    "display": "llama.cpp aux 1",
                    "port": 8090,
                    "state": "running",
                },
                {"name": "ollama", "display": "Ollama", "port": 11434, "state": "stopped"},
            ],
            "ts": 1700000000,
        }
        monkeypatch.setattr(serve, "_collect_engines_state", lambda: payload)
        resp = _get(
            self._url(server),
            headers={"Authorization": f"Bearer {server['token']}"},
        )
        body = json.loads(resp.read())
        assert body == payload

    def test_cache_serves_second_call(self, server, monkeypatch):
        calls = {"n": 0}

        def fake_collect():
            calls["n"] += 1
            return {"engines": [], "ts": calls["n"]}

        monkeypatch.setattr(serve, "_collect_engines_state", fake_collect)
        headers = {"Authorization": f"Bearer {server['token']}"}
        first = json.loads(_get(self._url(server), headers=headers).read())
        second = json.loads(_get(self._url(server), headers=headers).read())
        assert first == second
        assert calls["n"] == 1

    def test_collection_failure_is_json_500(self, server, monkeypatch):
        def boom():
            raise RuntimeError("manifest dir exploded")

        monkeypatch.setattr(serve, "_collect_engines_state", boom)
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(
                self._url(server),
                headers={"Authorization": f"Bearer {server['token']}"},
            )
        assert exc.value.code == 500
        assert json.loads(exc.value.read())["error"] == "engines_state_failed"

    def test_collect_skips_broken_manifest(self, monkeypatch):
        # One corrupt manifest must not hide the healthy ones.
        from ais_cli import serve as serve_mod

        class FakeManifest:
            def __init__(self, name, port):
                self.name = name
                self.display = name.upper()

                class _Net:
                    pass

                self.network = _Net()
                self.network.port = port

        def fake_load(name, preset=None):
            if name == "broken":
                raise ValueError("bad toml")
            return FakeManifest(name, 8090)

        import ais_core.lifecycle as lifecycle_mod
        import ais_core.manifest as manifest_mod

        monkeypatch.setattr(manifest_mod, "list_manifests", lambda: ["good", "broken"])
        monkeypatch.setattr(manifest_mod, "load_manifest", fake_load)
        monkeypatch.setattr(lifecycle_mod, "probe_state", lambda m, deep=False: ("running", None))
        monkeypatch.setattr(lifecycle_mod, "installed_model", lambda m: None)
        result = serve_mod._collect_engines_state()
        names = [e["name"] for e in result["engines"]]
        assert names == ["good"]
        assert result["engines"][0]["state"] == "running"


class TestPresetsEndpoint:
    def _url(self, server) -> str:
        return f"http://127.0.0.1:{server['port']}/internal/v1/presets"

    def test_requires_auth(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server))
        assert exc.value.code == 401

    def test_lists_bundled_presets(self, server, monkeypatch):
        from ais_core import manifest as manifest_mod

        monkeypatch.setattr(manifest_mod, "list_presets", lambda: ["hermes-aux-1"])
        monkeypatch.setattr(
            manifest_mod,
            "preset_summary",
            lambda n: {"preset": n, "engine": "llamacpp-aux-1", "display": "aux 1 tuned"},
        )
        resp = _get(
            self._url(server),
            headers={"Authorization": f"Bearer {server['token']}"},
        )
        body = json.loads(resp.read())
        assert body["presets"] == [
            {"preset": "hermes-aux-1", "engine": "llamacpp-aux-1", "display": "aux 1 tuned"}
        ]


class TestPlanEndpoint:
    def _url(self, server, preset: str | None) -> str:
        base = f"http://127.0.0.1:{server['port']}/internal/v1/plan"
        return base if preset is None else f"{base}?preset={preset}"

    def _auth(self, server) -> dict:
        return {"Authorization": f"Bearer {server['token']}"}

    def test_requires_auth(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server, "some-preset"))
        assert exc.value.code == 401

    def test_missing_preset_param_400(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server, None), headers=self._auth(server))
        assert exc.value.code == 400

    def test_malformed_preset_400(self, server):
        # Same shape gate as the install funnel: leading dash refused
        # before any preset resolution happens.
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server, "-flag"), headers=self._auth(server))
        assert exc.value.code == 400
        body = json.loads(exc.value.read())
        assert body["error"] == "bad_preset"

    def test_unknown_preset_404(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server, "definitely.not.a.preset"), headers=self._auth(server))
        assert exc.value.code == 404
        body = json.loads(exc.value.read())
        assert body["error"] == "unknown_preset"

    def test_valid_preset_returns_frozen_contract(self, server, monkeypatch):
        from ais_core import plan as plan_mod
        from ais_core.plan import CostComponent, PresetCost

        fake = PresetCost(
            preset="hermes-aux-1",
            total_mb_low=4000.0,
            total_mb_high=6000.0,
            confidence="declared",
            components={"weights": CostComponent(mb=3000.0, source="declared", detail="x")},
        )
        monkeypatch.setattr(plan_mod, "plan_for_preset", lambda preset: fake)
        resp = _get(self._url(server, "hermes-aux-1"), headers=self._auth(server))
        body = json.loads(resp.read())
        # The frozen wire contract, verbatim.
        assert body == {
            "preset": "hermes-aux-1",
            "cost": {
                "total_mb_low": 4000.0,
                "total_mb_high": 6000.0,
                "confidence": "declared",
                "components": {"weights": {"mb": 3000.0, "source": "declared", "detail": "x"}},
            },
        }

    def test_estimator_crash_is_json_500(self, server, monkeypatch):
        from ais_core import plan as plan_mod

        def boom(preset):
            raise RuntimeError("estimator exploded")

        monkeypatch.setattr(plan_mod, "plan_for_preset", boom)
        with pytest.raises(urllib.error.HTTPError) as exc:
            _get(self._url(server, "hermes-aux-1"), headers=self._auth(server))
        assert exc.value.code == 500
        assert json.loads(exc.value.read())["error"] == "plan_failed"

    def test_nan_payload_never_reaches_the_wire(self, server, monkeypatch):
        """allow_nan=False belt-and-braces: even if a NaN somehow survives
        the upstream guards, the server must fail rather than emit the
        (invalid-JSON) NaN token to a consumer."""
        import http.client

        from ais_core import plan as plan_mod
        from ais_core.plan import PresetCost

        fake = PresetCost(
            preset="hermes-aux-1",
            total_mb_low=float("nan"),
            total_mb_high=float("inf"),
            confidence="declared",
            components={},
        )
        monkeypatch.setattr(plan_mod, "plan_for_preset", lambda preset: fake)
        with pytest.raises((urllib.error.URLError, ConnectionError, http.client.HTTPException)):
            _get(self._url(server, "hermes-aux-1"), headers=self._auth(server))
