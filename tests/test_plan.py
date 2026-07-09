"""Tests for the preset memory-cost estimator (ais_core.plan) and its surfaces.

Covers the [memory] manifest section, the estimation ladder
(measured > declared > computed > unknown), the fail-closed property, the
ctx parsing rules (total budget, never multiplied by --parallel), and the
CLI ``aisctl plan`` handler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from ais_core import calibration, install_state, plan
from ais_core.manifest import ManifestError, MemorySpec, _from_dict, load_manifest

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PRESET_TEMPLATE = """\
name = "llamacpp"
display = "test preset"

[binary]
candidates = ["/opt/homebrew/bin/llama-server"]
process_pattern = "llms/test/active.gguf"
{model_path_line}
{mmproj_path_line}
program_args = [{program_args}]

[plist]
name = "com.asiai.llamacpp"
throttle_interval = 10
timeout = 30

[network]
port = 8080
bind = "0.0.0.0"
health_endpoint = "/health"
health_timeout = 30

[firewall]
supported = true
anchor_name = "com.asiai.llamacpp"

[logs]
dir = "~/Library/Logs/asiai/llamacpp"
stdout = "llamacpp.log"
stderr = "llamacpp.err"

[wrapper]
needed = false

[environment]
vars = []
{memory_section}
"""


def _write_preset(
    config_dir: Path,
    name: str = "test-plan-preset",
    *,
    model_path: str | None = None,
    mmproj_path: str | None = None,
    program_args: tuple[str, ...] = ("--ctx-size", "65536", "--parallel", "4"),
    memory_section: str = "",
) -> Path:
    preset_dir = config_dir / "engine_manifests" / "presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    path = preset_dir / f"{name}.toml"
    args = ", ".join(f'"{a}"' for a in program_args)
    path.write_text(
        _PRESET_TEMPLATE.format(
            model_path_line=f'model_path = "{model_path}"' if model_path else "",
            mmproj_path_line=f'mmproj_path = "{mmproj_path}"' if mmproj_path else "",
            program_args=args,
            memory_section=memory_section,
        )
    )
    return path


def _load(config_dir: Path, name: str = "test-plan-preset"):
    return load_manifest("llamacpp", preset=name), name


def _fake_model(tmp_path: Path, size_mb: int, name: str = "model.gguf") -> Path:
    f = tmp_path / name
    f.write_bytes(b"\x00" * (size_mb * 1024 * 1024))
    return f


# ---------------------------------------------------------------------------
# [memory] manifest section
# ---------------------------------------------------------------------------


class TestMemorySpecParsing:
    def _raw(self, memory: dict | None) -> dict:
        raw = {
            "name": "llamacpp",
            "display": "x",
            "binary": {
                "candidates": ["/opt/homebrew/bin/llama-server"],
                "process_pattern": "llama-server",
            },
            "plist": {"name": "com.asiai.llamacpp", "throttle_interval": 10, "timeout": 30},
            "network": {"port": 8080, "health_endpoint": "/health", "health_timeout": 30},
            "firewall": {"supported": True, "anchor_name": "com.asiai.llamacpp"},
            "logs": {"dir": "~/x", "stdout": "o.log", "stderr": "e.log"},
            "wrapper": {"needed": False},
        }
        if memory is not None:
            raw["memory"] = memory
        return raw

    def test_absent_section_yields_empty_spec(self):
        m = _from_dict(self._raw(None), source="test")
        assert m.memory == MemorySpec()
        assert m.memory.weights_mb is None

    def test_full_section_parses(self):
        m = _from_dict(
            self._raw(
                {
                    "weights_mb": 19456.0,
                    "kv_bytes_per_token": 32768,
                    "overhead_mb": 900,
                    "peak_extra_mb": 2048.5,
                }
            ),
            source="test",
        )
        assert m.memory.weights_mb == 19456.0
        assert m.memory.kv_bytes_per_token == 32768.0
        assert m.memory.overhead_mb == 900.0
        assert m.memory.peak_extra_mb == 2048.5

    @pytest.mark.parametrize(
        "bad",
        ["19GB", True, -1, 0, [1024], float("inf"), float("-inf"), float("nan")],
    )
    def test_malformed_value_fails_closed(self, bad):
        """Non-finite floats matter: TOML 1.0 accepts inf/nan literals, and
        NaN sails past a `<= 0` guard (every comparison is False) — it must
        be rejected here, not discovered as an invalid token on the wire."""
        with pytest.raises(ManifestError, match="weights_mb"):
            _from_dict(self._raw({"weights_mb": bad}), source="test")

    @pytest.mark.parametrize(
        ("key", "literal"),
        [
            ("kv_bytes_per_token", "inf"),
            ("kv_bytes_per_token", "nan"),
            ("weights_mb", "+inf"),
            ("overhead_mb", "-inf"),
        ],
    )
    def test_toml_nonfinite_literals_rejected_end_to_end(self, _isolated_user_config, key, literal):
        """The actual TOML literals (tomllib parses them into float inf/nan)
        must die in load_manifest with a ManifestError, never reach the
        estimator or the JSON surfaces."""
        _write_preset(
            _isolated_user_config,
            memory_section=f"[memory]\n{key} = {literal}\n",
        )
        with pytest.raises(ManifestError, match=key):
            load_manifest("llamacpp", preset="test-plan-preset")

    def test_unknown_key_fails_closed(self):
        """A typo'd key (resident_ram_mb…) must be a loud error, not silently
        ignored — the operator believes they declared a figure."""
        with pytest.raises(ManifestError, match="resident_ram_mb"):
            _from_dict(self._raw({"resident_ram_mb": 1024}), source="test")

    def test_non_table_section_fails_closed(self):
        with pytest.raises(ManifestError, match=r"\[memory\]"):
            _from_dict(self._raw(None) | {"memory": "1024"}, source="test")


class TestBundledPresetCuration:
    def test_27b_dense_declares_weights_and_kv(self):
        """The hybrid-attention preset carries the human-computed KV rate
        (16 of 64 layers only — the figure a GGUF parse would get 4x wrong)."""
        m = load_manifest("llamacpp", preset="qwen3.6-27b-dense-hermes-agent-64gb")
        assert m.memory.weights_mb == 19456.0
        assert m.memory.kv_bytes_per_token == 32768.0

    def test_all_bundled_presets_still_load(self):
        from ais_core.manifest import list_presets, preset_summary

        for name in list_presets():
            engine = preset_summary(name)["engine"]
            m = load_manifest(engine, preset=name)
            assert isinstance(m.memory, MemorySpec)


# ---------------------------------------------------------------------------
# ctx parsing
# ---------------------------------------------------------------------------


class TestParseCtxSize:
    def test_long_flag(self):
        assert plan._parse_ctx_size(("--ctx-size", "262144")) == 262144

    def test_short_flag(self):
        assert plan._parse_ctx_size(("-c", "4096")) == 4096

    def test_equals_form(self):
        assert plan._parse_ctx_size(("--ctx-size=131072",)) == 131072

    def test_absent(self):
        assert plan._parse_ctx_size(("--parallel", "2")) is None

    def test_malformed_value_is_none(self):
        assert plan._parse_ctx_size(("--ctx-size", "lots")) is None

    def test_last_occurrence_wins(self):
        """llama.cpp's CLI parser applies the LAST assignment; the estimate
        must price what would actually run."""
        assert plan._parse_ctx_size(("--ctx-size", "4096", "--ctx-size", "262144")) == 262144

    def test_short_flag_after_long_flag_wins(self):
        assert plan._parse_ctx_size(("--ctx-size", "4096", "-c", "8192")) == 8192

    def test_equals_form_after_long_flag_wins(self):
        assert plan._parse_ctx_size(("--ctx-size", "4096", "--ctx-size=16384")) == 16384

    def test_malformed_last_occurrence_fails_closed(self):
        """The runtime would reject the winning (last) assignment — an earlier
        valid value must not silently price a config that cannot start."""
        assert plan._parse_ctx_size(("--ctx-size", "4096", "--ctx-size", "lots")) is None

    def test_trailing_flag_is_none(self):
        assert plan._parse_ctx_size(("--parallel", "2", "--ctx-size")) is None

    @pytest.mark.parametrize("parallel", [None, "1", "4"])
    def test_parallel_never_multiplies(self, parallel):
        """--ctx-size is the TOTAL KV budget shared across slots."""
        args: tuple[str, ...] = ("--ctx-size", "65536")
        if parallel is not None:
            args += ("--parallel", parallel)
        assert plan._parse_ctx_size(args) == 65536


# ---------------------------------------------------------------------------
# estimate_preset_cost — decomposed paths
# ---------------------------------------------------------------------------


class TestEstimateDeclared:
    def test_declared_weights_and_kv(self, _isolated_user_config):
        _write_preset(
            _isolated_user_config,
            program_args=("--ctx-size", "65536", "--parallel", "4"),
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        # kv = 32768 B/token x 65536 tokens = 2048 MB — NOT multiplied by --parallel.
        assert cost.components["kv_cache"].mb == 2048.0
        assert cost.components["kv_cache"].source == "declared"
        assert cost.components["weights"].source == "declared"
        # overhead: llamacpp family default, computed source.
        assert cost.components["overhead"].mb == plan.LLAMACPP_OVERHEAD_MB
        assert cost.components["overhead"].source == "computed"
        # worst-of(declared, declared, computed) = computed
        assert cost.confidence == "computed"
        expected_low = (1000 + 2048) * 0.80 + 1024 * 0.65
        expected_high = (1000 + 2048) * 1.20 + 1024 * 1.35
        assert cost.total_mb_low == pytest.approx(expected_low, abs=0.1)
        assert cost.total_mb_high == pytest.approx(expected_high, abs=0.1)

    def test_declared_overhead_overrides_default(self, _isolated_user_config):
        _write_preset(
            _isolated_user_config,
            memory_section=(
                "[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\noverhead_mb = 700.0\n"
            ),
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["overhead"].mb == 700.0
        assert cost.components["overhead"].source == "declared"
        # All three components declared → global confidence declared.
        assert cost.confidence == "declared"

    def test_peak_extra_widens_high_bound_only(self, _isolated_user_config):
        base = "[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n"
        _write_preset(_isolated_user_config, memory_section=base)
        m, name = _load(_isolated_user_config)
        without = plan.estimate_preset_cost(m, preset=name)

        _write_preset(_isolated_user_config, memory_section=base + "peak_extra_mb = 500.0\n")
        m, name = _load(_isolated_user_config)
        with_peak = plan.estimate_preset_cost(m, preset=name)
        assert with_peak.total_mb_low == without.total_mb_low
        assert with_peak.total_mb_high == pytest.approx(without.total_mb_high + 500.0, abs=0.1)
        assert with_peak.components["peak_extra"].mb == 500.0


class TestEstimateComputed:
    def test_weights_from_disk_size(self, _isolated_user_config, tmp_path):
        model = _fake_model(tmp_path, 10)
        _write_preset(
            _isolated_user_config,
            model_path=str(model),
            memory_section="[memory]\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["weights"].mb == pytest.approx(10.0, abs=0.1)
        assert cost.components["weights"].source == "computed"
        assert cost.confidence == "computed"

    def test_weights_include_mmproj(self, _isolated_user_config, tmp_path):
        model = _fake_model(tmp_path, 10)
        mmproj = _fake_model(tmp_path, 3, name="mmproj.gguf")
        _write_preset(
            _isolated_user_config,
            model_path=str(model),
            mmproj_path=str(mmproj),
            memory_section="[memory]\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["weights"].mb == pytest.approx(13.0, abs=0.1)

    def test_weights_resolve_symlink(self, _isolated_user_config, tmp_path):
        """The active.gguf convention is a symlink; st_size must follow it."""
        real = _fake_model(tmp_path, 7)
        link = tmp_path / "active.gguf"
        link.symlink_to(real)
        _write_preset(
            _isolated_user_config,
            model_path=str(link),
            memory_section="[memory]\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["weights"].mb == pytest.approx(7.0, abs=0.1)

    def test_declared_weights_beat_disk_size(self, _isolated_user_config, tmp_path):
        model = _fake_model(tmp_path, 10)
        _write_preset(
            _isolated_user_config,
            model_path=str(model),
            memory_section="[memory]\nweights_mb = 999.0\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["weights"].mb == 999.0
        assert cost.components["weights"].source == "declared"


class TestEstimateFailClosed:
    def test_no_kv_rate_means_unknown(self, _isolated_user_config, tmp_path):
        """kv_bytes_per_token absent → whole estimate unknown with zeroed
        bounds — never a partial optimistic sum. (GGUF parsing is
        deliberately not attempted as a fallback.)"""
        model = _fake_model(tmp_path, 10)
        _write_preset(_isolated_user_config, model_path=str(model))
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["kv_cache"].source == "unknown"
        assert cost.confidence == "unknown"
        assert cost.total_mb_low == 0.0
        assert cost.total_mb_high == 0.0

    def test_kv_rate_without_ctx_flag_is_unknown(self, _isolated_user_config):
        _write_preset(
            _isolated_user_config,
            program_args=("--parallel", "2"),
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["kv_cache"].source == "unknown"
        assert cost.confidence == "unknown"

    def test_missing_model_file_is_unknown(self, _isolated_user_config, tmp_path):
        _write_preset(
            _isolated_user_config,
            model_path=str(tmp_path / "does-not-exist.gguf"),
            memory_section="[memory]\nkv_bytes_per_token = 32768.0\n",
        )
        m, name = _load(_isolated_user_config)
        cost = plan.estimate_preset_cost(m, preset=name)
        assert cost.components["weights"].source == "unknown"
        assert cost.confidence == "unknown"
        assert cost.total_mb_low == 0.0


# ---------------------------------------------------------------------------
# estimate_preset_cost — measured path (calibration)
# ---------------------------------------------------------------------------


class TestEstimateMeasured:
    _SHA = "a" * 64

    def _preset(self, config_dir):
        _write_preset(
            config_dir,
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        return _load(config_dir)

    def test_measured_beats_decomposition(self, _isolated_user_config):
        m, name = self._preset(_isolated_user_config)
        calibration.record_sample(
            m.name,
            preset=name,
            manifest_sha256=self._SHA,
            phys_footprint_mb=3000.0,
            source="health",
        )
        cost = plan.estimate_preset_cost(m, preset=name, manifest_sha256=self._SHA)
        assert cost.confidence == "measured"
        assert "measured_total" in cost.components
        # A measured sample covers the whole footprint: no weights/kv/overhead
        # components to double-count on top.
        assert "weights" not in cost.components
        assert cost.total_mb_low == pytest.approx(3000.0 * 0.9, abs=0.1)
        assert cost.total_mb_high == pytest.approx(3000.0 * 1.1, abs=0.1)

    def test_stale_sha_falls_back(self, _isolated_user_config):
        m, name = self._preset(_isolated_user_config)
        calibration.record_sample(
            m.name,
            preset=name,
            manifest_sha256="b" * 64,
            phys_footprint_mb=3000.0,
            source="health",
        )
        cost = plan.estimate_preset_cost(m, preset=name, manifest_sha256=self._SHA)
        assert cost.confidence != "measured"
        assert "measured_total" not in cost.components

    def test_other_host_falls_back(self, _isolated_user_config):
        m, name = self._preset(_isolated_user_config)
        calibration.record_sample(
            m.name,
            preset=name,
            manifest_sha256=self._SHA,
            phys_footprint_mb=3000.0,
            source="health",
        )
        cost = plan.estimate_preset_cost(
            m, preset=name, manifest_sha256=self._SHA, host="some-other-host"
        )
        assert cost.confidence != "measured"

    def test_measured_high_bound_includes_peak_extra(self, _isolated_user_config):
        _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\npeak_extra_mb = 400.0\n",
        )
        m, name = _load(_isolated_user_config)
        calibration.record_sample(
            m.name,
            preset=name,
            manifest_sha256=self._SHA,
            phys_footprint_mb=2000.0,
            source="health",
        )
        cost = plan.estimate_preset_cost(m, preset=name, manifest_sha256=self._SHA)
        assert cost.total_mb_high == pytest.approx(2000.0 * 1.1 + 400.0, abs=0.1)
        assert cost.total_mb_low == pytest.approx(2000.0 * 0.9, abs=0.1)


# ---------------------------------------------------------------------------
# plan_for_preset + wire contract
# ---------------------------------------------------------------------------


class TestPlanForPreset:
    def test_unknown_preset_raises(self):
        with pytest.raises(FileNotFoundError, match="no-such-preset"):
            plan.plan_for_preset("no-such-preset")

    def test_engine_resolved_from_preset(self, _isolated_user_config):
        _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        cost = plan.plan_for_preset("test-plan-preset")
        assert cost.preset == "test-plan-preset"
        assert cost.confidence in ("measured", "declared", "computed", "unknown")

    def test_payload_matches_frozen_contract(self, _isolated_user_config):
        _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        payload = plan.plan_for_preset("test-plan-preset").payload()
        assert payload["preset"] == "test-plan-preset"
        cost = payload["cost"]
        assert set(cost) == {"total_mb_low", "total_mb_high", "confidence", "components"}
        assert isinstance(cost["total_mb_low"], float)
        assert isinstance(cost["total_mb_high"], float)
        assert cost["confidence"] in ("measured", "declared", "computed", "unknown")
        assert isinstance(cost["components"], dict)
        # Payload must be JSON-serializable as-is (the endpoint sends it raw).
        json.dumps(payload)

    def test_calibration_keyed_on_real_manifest_sha(self, _isolated_user_config):
        """plan_for_preset must look up calibration under the digest of the
        actual preset file — the same key the lifecycle hooks record under."""
        path = _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        sha = install_state.manifest_digest(path)
        calibration.record_sample(
            "llamacpp",
            preset="test-plan-preset",
            manifest_sha256=sha,
            phys_footprint_mb=2500.0,
            source="health",
        )
        cost = plan.plan_for_preset("test-plan-preset")
        assert cost.confidence == "measured"


# ---------------------------------------------------------------------------
# CLI handler
# ---------------------------------------------------------------------------


class TestCmdPlan:
    def _ns(self, preset: str, *, engine: str | None = None, as_json: bool = False):
        return argparse.Namespace(preset=preset, engine=engine, json=as_json)

    def test_json_output_is_the_wire_contract(self, _isolated_user_config, capsys):
        from ais_cli import commands

        _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        rc = commands.cmd_plan(self._ns("test-plan-preset", as_json=True))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["preset"] == "test-plan-preset"
        assert set(payload["cost"]) == {
            "total_mb_low",
            "total_mb_high",
            "confidence",
            "components",
        }

    def test_human_output_shows_components_and_band(self, _isolated_user_config, capsys):
        from ais_cli import commands

        _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\nkv_bytes_per_token = 32768.0\n",
        )
        rc = commands.cmd_plan(self._ns("test-plan-preset"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "weights" in out
        assert "kv_cache" in out
        assert "confidence" in out
        assert "Advisory" in out

    def test_unknown_component_renders_unknown_total(self, _isolated_user_config, capsys):
        from ais_cli import commands

        _write_preset(_isolated_user_config)  # no [memory] at all
        rc = commands.cmd_plan(self._ns("test-plan-preset"))
        assert rc == 0
        out = capsys.readouterr().out
        assert "unknown" in out

    def test_unknown_preset_exits_with_search_dirs(self, capsys):
        from ais_cli import commands

        with pytest.raises(SystemExit, match="unknown preset"):
            commands.cmd_plan(self._ns("no-such-preset"))

    def test_emit_json_refuses_nan(self):
        """Belt-and-braces on the CLI output surface: NaN/Infinity are not
        valid JSON tokens — a producer bug must crash here, never print
        unparseable output for a consumer to choke on."""
        from ais_cli import commands

        with pytest.raises(ValueError):
            commands._emit({"total_mb_low": float("nan")}, as_json=True)

    def test_engine_mismatch_exits_cleanly(self, _isolated_user_config):
        from ais_cli import commands

        _write_preset(
            _isolated_user_config,
            memory_section="[memory]\nweights_mb = 1000.0\n",
        )
        # The preset targets llamacpp; pinning ollama must be a clean exit,
        # not a traceback.
        with pytest.raises(SystemExit):
            commands.cmd_plan(self._ns("test-plan-preset", engine="ollama"))
