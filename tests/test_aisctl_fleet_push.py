"""Unit tests for ``aisctl fleet push``."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock

import pytest

from ais_cli import fleet as fleet_cli

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_fleet(tmp_path, monkeypatch):
    from asiai.fleet import config as fleet_config

    cfg_dir = tmp_path / "asiai"
    monkeypatch.setattr(fleet_config, "CONFIG_DIR", str(cfg_dir))
    monkeypatch.setattr(fleet_config, "CONFIG_PATH", str(cfg_dir / "fleet.json"))
    monkeypatch.setattr(fleet_config, "LOCK_PATH", str(cfg_dir / "fleet.lock"))
    fleet_config.upsert_node(
        "alpha", "http://192.0.2.1:8899", role="workstation", auth_token="asai_test_secret"
    )
    fleet_config.upsert_node("beta", "http://192.0.2.2:8899", role="spare")  # no token
    yield cfg_dir


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _mock_urlopen_ok(status: int, body: dict):
    class _Resp:
        status = 200

        def __init__(self, st, body):
            self._body = json.dumps(body).encode("utf-8")
            self.status = st

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, _n=None):
            return self._body

    return MagicMock(return_value=_Resp(status, body))


# ---------------------------------------------------------------------------
# Resolution + arg validation
# ---------------------------------------------------------------------------


class TestResolveNode:
    def test_finds_existing_node(self, tmp_fleet):
        node = fleet_cli._resolve_node("alpha")
        assert node is not None
        assert node["asiai_url"] == "http://192.0.2.1:8899"

    def test_returns_none_for_unknown(self, tmp_fleet):
        assert fleet_cli._resolve_node("unknown") is None


class TestCmdPush:
    def test_unknown_nickname_returns_1(self, tmp_fleet, capsys):
        ns = _ns(
            nickname="unknown",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 1
        err = capsys.readouterr().err
        assert "no node named" in err

    def test_unknown_command_returns_2(self, tmp_fleet, capsys):
        ns = _ns(
            nickname="alpha",
            command="drop_database",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 2

    def test_node_without_token_returns_3(self, tmp_fleet, capsys):
        ns = _ns(
            nickname="beta",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 3
        out = capsys.readouterr()
        assert "auth_token" in out.err

    def test_success_path_returns_0(self, tmp_fleet, monkeypatch, capsys):
        mock = _mock_urlopen_ok(
            200,
            {"ok": True, "exit_code": 0, "stdout": "freed 2 GB", "duration_ms": 100},
        )
        monkeypatch.setattr(urllib.request, "urlopen", mock)
        ns = _ns(
            nickname="alpha",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 0
        out = capsys.readouterr().out
        assert "freed 2 GB" in out

    def test_http_error_returns_3(self, tmp_fleet, monkeypatch):
        def raising(*_args, **_kwargs):
            raise urllib.error.HTTPError(
                "u",
                401,
                "Unauthorized",
                {},
                None,  # type: ignore[arg-type]
            )

        monkeypatch.setattr(urllib.request, "urlopen", raising)
        ns = _ns(
            nickname="alpha",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 3

    def test_timeout_returns_3(self, tmp_fleet, monkeypatch):
        def raising(*_args, **_kwargs):
            raise TimeoutError()

        monkeypatch.setattr(urllib.request, "urlopen", raising)
        ns = _ns(
            nickname="alpha",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 3

    def test_unreachable_returns_3(self, tmp_fleet, monkeypatch):
        def raising(*_args, **_kwargs):
            raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(urllib.request, "urlopen", raising)
        ns = _ns(
            nickname="alpha",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=False,
        )
        assert fleet_cli._cmd_push(ns) == 3

    def test_json_output(self, tmp_fleet, monkeypatch, capsys):
        mock = _mock_urlopen_ok(200, {"ok": True, "exit_code": 0})
        monkeypatch.setattr(urllib.request, "urlopen", mock)
        ns = _ns(
            nickname="alpha",
            command="purge",
            engine=None,
            model=None,
            timeout=None,
            json=True,
        )
        fleet_cli._cmd_push(ns)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["http_status"] == 200


class TestCmdInfo:
    def test_redacts_auth_token(self, tmp_fleet, capsys):
        ns = _ns(nickname="alpha", json=False)
        assert fleet_cli._cmd_info(ns) == 0
        out = capsys.readouterr().out
        # The secret must NEVER appear in the output.
        assert "asai_test_secret" not in out
        # But the boolean flag must be visible.
        assert "has_auth_token: True" in out

    def test_unknown_returns_1(self, tmp_fleet, capsys):
        ns = _ns(nickname="ghost", json=False)
        assert fleet_cli._cmd_info(ns) == 1


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------


class TestArgparse:
    def test_push_parser_accepts_all_commands(self, tmp_fleet):
        from ais_cli.__main__ import build_parser

        parser = build_parser()
        for cmd in fleet_cli.ALLOWED_COMMANDS:
            args = parser.parse_args(["fleet", "push", "alpha", cmd])
            assert args.command == cmd

    def test_push_parser_engine_and_model(self, tmp_fleet):
        from ais_cli.__main__ import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["fleet", "push", "alpha", "unload", "--engine", "ollama", "--model", "llama3.2"]
        )
        assert args.engine == "ollama"
        assert args.model == "llama3.2"
