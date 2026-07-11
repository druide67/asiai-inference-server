"""Install records — persistence of the preset an engine was installed with."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ais_core import install_state


@pytest.fixture
def manifest_file(tmp_path: Path) -> Path:
    f = tmp_path / "some-preset.toml"
    f.write_text('name = "llamacpp-aux-1"\n')
    return f


def test_record_read_roundtrip(manifest_file: Path) -> None:
    written = install_state.record_install(
        "llamacpp-aux-1",
        preset="my-preset",
        manifest_path=manifest_file,
        firewall="lan-only",
    )
    read = install_state.read_install("llamacpp-aux-1")
    assert read is not None
    assert read.engine == "llamacpp-aux-1"
    assert read.preset == "my-preset"
    assert read.firewall == "lan-only"
    assert read.manifest_sha256 == written.manifest_sha256
    assert read.manifest_sha256 == install_state.manifest_digest(manifest_file)


def test_base_install_records_none_preset(manifest_file: Path) -> None:
    install_state.record_install("ollama", preset=None, manifest_path=manifest_file)
    read = install_state.read_install("ollama")
    assert read is not None
    assert read.preset is None
    assert read.firewall == "none"


def test_read_missing_returns_none() -> None:
    assert install_state.read_install("never-installed") is None


def test_read_corrupt_record_degrades_to_none(manifest_file: Path) -> None:
    """The record is a safety net; a broken net must not block install/status."""
    install_state.record_install("ollama", preset="p", manifest_path=manifest_file)
    install_state._record_path("ollama").write_text("{not json")
    assert install_state.read_install("ollama") is None
    install_state._record_path("ollama").write_text(json.dumps([1, 2]))
    assert install_state.read_install("ollama") is None


def test_clear_install(manifest_file: Path) -> None:
    install_state.record_install("ollama", preset="p", manifest_path=manifest_file)
    assert install_state.clear_install("ollama") is True
    assert install_state.read_install("ollama") is None
    assert install_state.clear_install("ollama") is False


def test_digest_detects_manifest_drift(manifest_file: Path) -> None:
    rec = install_state.record_install("ollama", preset="p", manifest_path=manifest_file)
    manifest_file.write_text('name = "llamacpp-aux-1"\n# edited after install\n')
    assert install_state.manifest_digest(manifest_file) != rec.manifest_sha256


def test_state_dir_falls_back_to_xdg_state_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, manifest_file: Path
) -> None:
    monkeypatch.delenv("ASIAI_USER_STATE_DIR", raising=False)
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    install_state.record_install("ollama", preset="p", manifest_path=manifest_file)
    expected = tmp_path / "xdg-state" / "asiai-inference-server" / "installs" / "ollama.json"
    assert expected.is_file()


class TestLoadInstalledManifest:
    """The INSTALLED view: lifecycle verbs must resolve the preset recorded
    at install time — the baseline may live on another port entirely."""

    def test_overlays_recorded_preset(self, manifest_file: Path) -> None:
        install_state.record_install(
            "mtplx", preset="qwen3.6-27b-mtplx-hermes-agent", manifest_path=manifest_file
        )
        m = install_state.load_installed_manifest("mtplx")
        assert m.network.port == 8080  # preset port, not the 8005 baseline

    def test_no_record_resolves_baseline(self) -> None:
        m = install_state.load_installed_manifest("mtplx")
        assert m.network.port == 8005

    def test_vanished_preset_degrades_to_baseline(self, manifest_file: Path) -> None:
        install_state.record_install(
            "mtplx", preset="deleted-since-install", manifest_path=manifest_file
        )
        m = install_state.load_installed_manifest("mtplx")
        assert m.network.port == 8005
