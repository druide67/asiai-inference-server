"""Unit tests for the broadcast selector ``@all`` / ``@role:<value>``."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from unittest.mock import MagicMock

import pytest

from ais_cli import fleet as fleet_cli


@pytest.fixture
def tmp_fleet(tmp_path, monkeypatch):
    from asiai.fleet import config as fleet_config

    cfg_dir = tmp_path / "asiai"
    monkeypatch.setattr(fleet_config, "CONFIG_DIR", str(cfg_dir))
    monkeypatch.setattr(fleet_config, "CONFIG_PATH", str(cfg_dir / "fleet.json"))
    monkeypatch.setattr(fleet_config, "LOCK_PATH", str(cfg_dir / "fleet.lock"))
    fleet_config.upsert_node(
        "studio", "http://192.0.2.10:8899", role="prod", auth_token="asai_studio"
    )
    fleet_config.upsert_node(
        "laptop", "http://192.0.2.11:8899", role="dev", auth_token="asai_laptop"
    )
    fleet_config.upsert_node("spare", "http://192.0.2.12:8899", role="dev", auth_token="asai_spare")
    yield cfg_dir


def _ns(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(**kwargs)


def _mock_urlopen(status: int, body: dict):
    class _Resp:
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


class TestResolveTargets:
    def test_literal_nickname_returns_one(self, tmp_fleet):
        nodes, err = fleet_cli._resolve_targets("studio")
        assert err is None
        assert len(nodes) == 1
        assert nodes[0]["nickname"] == "studio"

    def test_at_all_returns_every_node(self, tmp_fleet):
        nodes, err = fleet_cli._resolve_targets("@all")
        assert err is None
        assert {n["nickname"] for n in nodes} == {"studio", "laptop", "spare"}

    def test_at_role_filters(self, tmp_fleet):
        nodes, err = fleet_cli._resolve_targets("@role:dev")
        assert err is None
        assert {n["nickname"] for n in nodes} == {"laptop", "spare"}

    def test_at_role_empty_value_errors(self, tmp_fleet):
        nodes, err = fleet_cli._resolve_targets("@role:")
        assert nodes == []
        assert err is not None
        assert "empty role" in err

    def test_unknown_selector_errors(self, tmp_fleet):
        nodes, err = fleet_cli._resolve_targets("@something-else")
        assert nodes == []
        assert err is not None
        assert "unknown selector" in err

    def test_unknown_nickname_returns_empty(self, tmp_fleet):
        nodes, err = fleet_cli._resolve_targets("ghost")
        assert nodes == []
        assert err is None


class TestBroadcastPush:
    def test_at_all_runs_against_every_node(self, tmp_fleet, monkeypatch, capsys):
        mock = _mock_urlopen(200, {"ok": True, "exit_code": 0, "stdout": ""})
        monkeypatch.setattr(urllib.request, "urlopen", mock)
        ns = _ns(
            nickname="@all",
            command="purge",
            engine=None,
            model=None,
            keep_alive=None,
            timeout=None,
            json=False,
        )
        rc = fleet_cli._cmd_push(ns)
        assert rc == 0
        # 3 nodes → 3 HTTP POSTs.
        assert mock.call_count == 3

    def test_at_role_runs_against_subset(self, tmp_fleet, monkeypatch):
        mock = _mock_urlopen(200, {"ok": True, "exit_code": 0, "stdout": ""})
        monkeypatch.setattr(urllib.request, "urlopen", mock)
        ns = _ns(
            nickname="@role:dev",
            command="purge",
            engine=None,
            model=None,
            keep_alive=None,
            timeout=None,
            json=False,
        )
        fleet_cli._cmd_push(ns)
        assert mock.call_count == 2  # laptop + spare

    def test_partial_failure_returns_3(self, tmp_fleet, monkeypatch):
        # First call succeeds, second raises. ThreadPool semantics mean
        # the order isn't deterministic but at least one fails → rc=3.
        call_state = {"n": 0}

        def flaky(*args, **kwargs):
            call_state["n"] += 1
            if call_state["n"] == 2:
                raise urllib.error.URLError("connection refused")
            return _mock_urlopen(200, {"ok": True, "exit_code": 0}).return_value

        monkeypatch.setattr(urllib.request, "urlopen", flaky)
        ns = _ns(
            nickname="@all",
            command="purge",
            engine=None,
            model=None,
            keep_alive=None,
            timeout=None,
            json=False,
        )
        rc = fleet_cli._cmd_push(ns)
        assert rc == 3

    def test_empty_match_returns_1(self, tmp_fleet, capsys):
        ns = _ns(
            nickname="@role:nonexistent",
            command="purge",
            engine=None,
            model=None,
            keep_alive=None,
            timeout=None,
            json=False,
        )
        rc = fleet_cli._cmd_push(ns)
        assert rc == 1
        out = capsys.readouterr()
        assert "zero nodes" in out.err or "matched zero" in out.err

    def test_json_aggregate_payload(self, tmp_fleet, monkeypatch, capsys):
        mock = _mock_urlopen(200, {"ok": True, "exit_code": 0, "stdout": ""})
        monkeypatch.setattr(urllib.request, "urlopen", mock)
        ns = _ns(
            nickname="@role:dev",
            command="purge",
            engine=None,
            model=None,
            keep_alive=None,
            timeout=None,
            json=True,
        )
        fleet_cli._cmd_push(ns)
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert payload["target"] == "@role:dev"
        assert payload["command"] == "purge"
        assert len(payload["results"]) == 2
        nicks = {r["nickname"] for r in payload["results"]}
        assert nicks == {"laptop", "spare"}
