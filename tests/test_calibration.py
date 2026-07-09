"""Tests for ais_core.calibration — ring buffer, weighted median, staleness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ais_core import calibration

SHA = "a" * 64
OTHER_SHA = "b" * 64


def _record(mb: float, *, source: str = "health", sha: str = SHA, engine: str = "llamacpp"):
    return calibration.record_sample(
        engine,
        preset="test-preset",
        manifest_sha256=sha,
        phys_footprint_mb=mb,
        source=source,
    )


def _lookup(*, sha: str = SHA, host: str | None = None, engine: str = "llamacpp"):
    return calibration.measured_footprint_mb(
        engine, preset="test-preset", manifest_sha256=sha, host=host
    )


class TestRecordSample:
    def test_round_trip(self, _isolated_install_state: Path):
        assert _record(3000.0) is True
        result = _lookup()
        assert result is not None
        median, n = result
        assert median == 3000.0
        assert n == 1
        # File lands in the calibration subtree of the state dir.
        path = _isolated_install_state / "calibration" / "llamacpp.json"
        assert path.is_file()
        raw = json.loads(path.read_text())
        ring = raw["samples"][f"test-preset@{SHA}"]
        assert ring[0]["source"] == "health"
        assert ring[0]["phys_footprint_mb"] == 3000.0
        assert "ts" in ring[0] and "host" in ring[0]

    def test_ring_caps_at_ten(self, _isolated_install_state: Path):
        for i in range(15):
            _record(1000.0 + i)
        raw = json.loads((_isolated_install_state / "calibration" / "llamacpp.json").read_text())
        ring = raw["samples"][f"test-preset@{SHA}"]
        assert len(ring) == calibration.RING_SIZE
        # Oldest samples were evicted, newest kept.
        assert ring[-1]["phys_footprint_mb"] == 1014.0
        assert ring[0]["phys_footprint_mb"] == 1005.0

    def test_negative_delta_discarded(self):
        """An unload freed_mb can go negative (other processes allocated
        during the window); such a sample must never enter the ring."""
        assert _record(-512.0, source="unload") is False
        assert _record(0.0, source="unload") is False
        assert _lookup() is None

    def test_unknown_source_raises(self):
        with pytest.raises(ValueError, match="unknown sample source"):
            _record(1000.0, source="guess")

    def test_invalid_engine_name_raises(self):
        with pytest.raises(ValueError, match="invalid engine name"):
            _record(1000.0, engine="../evil")

    def test_corrupt_file_degrades_to_empty(self, _isolated_install_state: Path):
        path = _isolated_install_state / "calibration" / "llamacpp.json"
        path.parent.mkdir(parents=True)
        path.write_text("{not json")
        assert _lookup() is None
        # And recording over the corrupt file recovers instead of raising.
        assert _record(2000.0) is True
        assert _lookup() == (2000.0, 1)


class TestStaleness:
    def test_different_sha_ignored_not_deleted(self, _isolated_install_state: Path):
        _record(3000.0, sha=OTHER_SHA)
        # A lookup under the current sha sees nothing…
        assert _lookup(sha=SHA) is None
        # …but the stale samples are still on disk (revert the edit → revive).
        assert _lookup(sha=OTHER_SHA) is not None

    def test_different_host_ignored(self):
        _record(3000.0)
        assert _lookup(host="not-this-host") is None
        assert _lookup() is not None  # default host = the recording host


class TestWeightedMedian:
    def test_health_outweighs_unload(self):
        """Two half-weight unload outliers must not drag the median off the
        two full-weight health samples."""
        _record(3000.0, source="health")
        _record(3010.0, source="health")
        _record(5000.0, source="unload")
        _record(5010.0, source="unload")
        result = _lookup()
        assert result is not None
        median, n = result
        assert n == 4
        assert median <= 3010.0

    def test_single_unload_sample_still_answers(self):
        _record(2500.0, source="unload")
        assert _lookup() == (2500.0, 1)

    def test_median_of_odd_health_samples(self):
        for mb in (1000.0, 9000.0, 3000.0):
            _record(mb)
        result = _lookup()
        assert result is not None
        assert result[0] == 3000.0


class TestLifecycleHooks:
    """The commands.py hooks feed the ring — best-effort by contract."""

    def _install_record(self, tmp_path: Path, preset: str | None = "some-preset") -> str:
        from ais_core import install_state

        manifest_file = tmp_path / "preset.toml"
        manifest_file.write_text('name = "llamacpp"\n')
        install_state.record_install("llamacpp", preset=preset, manifest_path=manifest_file)
        return install_state.manifest_digest(manifest_file)

    def test_health_hook_records_sample(self, monkeypatch, tmp_path):
        from ais_cli import commands
        from ais_core.manifest import load_manifest

        sha = self._install_record(tmp_path)
        m = load_manifest("llamacpp")
        monkeypatch.setattr(commands.calibration, "engine_rss_mb", lambda manifest: 4242.0)
        commands._record_health_calibration(m)
        assert calibration.measured_footprint_mb(
            "llamacpp", preset="some-preset", manifest_sha256=sha
        ) == (4242.0, 1)

    def test_health_hook_noop_without_preset_record(self, monkeypatch, tmp_path):
        from ais_cli import commands
        from ais_core.manifest import load_manifest

        sha = self._install_record(tmp_path, preset=None)  # base-manifest install
        m = load_manifest("llamacpp")
        monkeypatch.setattr(commands.calibration, "engine_rss_mb", lambda manifest: 4242.0)
        commands._record_health_calibration(m)
        assert (
            calibration.measured_footprint_mb("llamacpp", preset="some-preset", manifest_sha256=sha)
            is None
        )

    def test_health_hook_never_raises(self, monkeypatch, tmp_path):
        from ais_cli import commands
        from ais_core.manifest import load_manifest

        self._install_record(tmp_path)
        m = load_manifest("llamacpp")

        def boom(manifest):
            raise RuntimeError("ps exploded")

        monkeypatch.setattr(commands.calibration, "engine_rss_mb", boom)
        commands._record_health_calibration(m)  # must not raise

    def test_unload_hook_records_positive_delta(self, monkeypatch, tmp_path):
        from ais_cli import commands
        from ais_core.manifest import load_manifest
        from ais_core.memory import VmStat

        sha = self._install_record(tmp_path)
        m = load_manifest("llamacpp")
        page = 16384
        before = VmStat(
            page_size_bytes=page,
            pages_free=0,
            pages_active=200_000,
            pages_inactive=0,
            pages_wired=0,
            pages_compressed=0,
        )
        after = VmStat(
            page_size_bytes=page,
            pages_free=0,
            pages_active=100_000,
            pages_inactive=0,
            pages_wired=0,
            pages_compressed=0,
        )
        monkeypatch.setattr(commands.memory, "vm_stat_parse", lambda: after)
        commands._record_unload_calibration(m, before)
        result = calibration.measured_footprint_mb(
            "llamacpp", preset="some-preset", manifest_sha256=sha
        )
        assert result is not None
        freed_mb = (100_000 * page) // (1024 * 1024)
        assert result[0] == float(freed_mb)
