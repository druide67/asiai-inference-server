"""Tests for ais_core.manifest — loading, validation, label policy."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from ais_core.manifest import (
    EngineManifest,
    ManifestError,
    _from_dict,
    is_valid_plist_label,
    list_manifests,
    load_manifest,
)

EXPECTED_ENGINES = {"ollama", "lmstudio", "omlx", "turboquant", "llamacpp", "llamacpp-aux"}


def test_list_manifests_returns_all_engines() -> None:
    names = list_manifests()
    assert set(names) == EXPECTED_ENGINES
    assert names == sorted(names)


@pytest.mark.parametrize("engine", sorted(EXPECTED_ENGINES))
def test_each_manifest_loads_and_validates(engine: str) -> None:
    m = load_manifest(engine)
    assert isinstance(m, EngineManifest)
    assert m.name == engine
    assert m.plist.name.startswith("com.asiai.")
    assert m.firewall.anchor_name.startswith("com.asiai.")
    assert 1 <= m.network.port <= 65535
    assert m.network.health_endpoint.startswith("/")
    assert m.binary.candidates  # non-empty


def test_ollama_specifics() -> None:
    m = load_manifest("ollama")
    assert m.network.port == 11434
    assert m.network.health_endpoint == "/api/version"
    assert "OLLAMA_HOST=0.0.0.0:11434" in m.env_vars
    assert not m.wrapper.needed
    assert not m.binary.builds_from_source


def test_lmstudio_specifics() -> None:
    m = load_manifest("lmstudio")
    assert m.network.port == 1234
    assert m.wrapper.needed is True
    assert m.wrapper.install_path == "/usr/local/bin/lmstudio-server-start"
    # Tilde-expansion is handled at resolve() time, so the manifest still carries it raw.
    assert any(c.startswith("~/") for c in m.binary.candidates)


def test_omlx_specifics() -> None:
    m = load_manifest("omlx")
    assert m.network.port == 8800
    assert m.options.get("log_level") == "info"


def test_turboquant_specifics() -> None:
    m = load_manifest("turboquant")
    assert m.binary.builds_from_source is True
    assert m.wrapper.needed is True
    assert m.wrapper.template == "wrapper-start.sh.tpl"
    # 70B cold-start is slow — manifest must reflect that.
    assert m.network.health_timeout >= 60


def test_llamacpp_baseline_specifics() -> None:
    """Baseline llamacpp.toml is the generic OSS-safe defaults.

    Tuned configurations (specific samplers, cache flags, parallel/ctx
    sizing) live in presets under data/engine_manifests/presets/, not in
    the baseline. See test_llamacpp_hermes_preset for the Qwen3.6 preset.
    """
    m = load_manifest("llamacpp")
    assert m.network.port == 8080
    assert m.network.health_endpoint == "/health"
    assert m.plist.name == "com.asiai.llamacpp"
    assert m.firewall.anchor_name == "com.asiai.llamacpp"
    assert m.binary.process_pattern == "llms/gguf/active.gguf"
    assert m.binary.model_path is not None
    assert m.binary.model_path.startswith("~/")
    assert m.binary.model_path.endswith(".gguf")
    assert not m.wrapper.needed
    assert not m.binary.builds_from_source
    # Generic baseline keeps only widely-applicable flags
    pa = list(m.binary.program_args)
    assert "--mlock" in pa
    assert "--cont-batching" in pa
    assert "--n-gpu-layers" in pa
    # Baseline uses the embedded template — no template_path override
    assert m.binary.template_path is None
    assert "--jinja" in pa
    # Baseline does NOT include workload-specific tuning (those live in presets)
    assert "--cache-reuse" not in pa
    assert "--slot-prompt-similarity" not in pa
    assert "--chat-template-kwargs" not in pa
    assert "--temp" not in pa
    # Health timeout accommodates large-GGUF load even in the generic case.
    assert m.network.health_timeout >= 60


def test_llamacpp_hermes_preset() -> None:
    """The Qwen3.6-35B-A3B Hermes preset carries the production tuning."""
    m = load_manifest("llamacpp", preset="qwen3.6-35b-a3b-hermes-agent-64gb")
    assert m.name == "llamacpp"  # preset targets the llamacpp engine
    pa = list(m.binary.program_args)
    # Hermes Agent class tuning
    assert pa[pa.index("--ctx-size") + 1] == "131072"
    assert pa[pa.index("--parallel") + 1] == "2"
    assert "--cache-reuse" in pa
    assert "--slot-prompt-similarity" in pa
    # froggeric chat template override (no --jinja)
    assert m.binary.template_path is not None
    assert m.binary.template_path.endswith(".jinja")
    assert "--jinja" not in pa
    # Qwen3.6 thinking-mode reco sampling
    assert pa[pa.index("--temp") + 1] == "0.6"
    assert pa[pa.index("--top-p") + 1] == "0.95"
    assert pa[pa.index("--top-k") + 1] == "20"
    assert "--chat-template-kwargs" in pa


def test_llamacpp_aux_baseline_specifics() -> None:
    """Baseline llamacpp-aux.toml is the generic OSS-safe defaults (no tuning)."""
    m = load_manifest("llamacpp-aux")
    assert m.network.port == 8082
    assert m.network.health_endpoint == "/health"
    assert m.plist.name == "com.asiai.llamacpp-aux"
    assert m.firewall.anchor_name == "com.asiai.llamacpp-aux"
    assert m.binary.process_pattern == "llms/gguf/aux/active.gguf"
    assert m.binary.model_path == "~/llms/gguf/aux/active.gguf"
    # Small-model health timeout: ≤30s
    assert m.network.health_timeout <= 30
    pa = list(m.binary.program_args)
    assert "--mlock" in pa
    assert "--cont-batching" in pa
    # No workload-specific tuning in the baseline
    assert "--cache-reuse" not in pa
    assert "--slot-prompt-similarity" not in pa


def test_llamacpp_aux_presets_target_engine_and_carry_tuning() -> None:
    """Both aux presets (1.7B and 4B) target llamacpp-aux and carry tuning."""
    # Per-preset expected parallel value (4B parallel=1 after PRISM ADR-004 D3,
    # 1.7B legacy parallel=2 kept for non-Hermes use cases <64K).
    expected_parallel = {
        "qwen3-1.7b-instruct-hermes-compression": "2",
        "qwen3-4b-instruct-hermes-compression": "1",
    }
    for preset, par in expected_parallel.items():
        m = load_manifest("llamacpp-aux", preset=preset)
        assert m.name == "llamacpp-aux", preset
        pa = list(m.binary.program_args)
        assert "--ctx-size" in pa, preset
        assert pa[pa.index("--parallel") + 1] == par, preset
        assert "--cache-reuse" in pa, preset
        # KV cache q8 keeps the aux footprint small
        assert "--cache-type-k" in pa, preset
        assert pa[pa.index("--cache-type-k") + 1] == "q8_0", preset


def test_llamacpp_and_aux_process_patterns_are_disjoint() -> None:
    """Critical invariant: pkill -f on one must never match the other's cmdline.

    The real semantic we care about is: given a representative cmdline for
    each instance (the kind ``pgrep -f`` actually sees in production), the
    pattern of one engine must not match the cmdline of the other. Testing
    only ``main not in aux`` (substring on the patterns themselves) misses
    that target — see code review M1.
    """
    main_pat = load_manifest("llamacpp").binary.process_pattern
    aux_pat = load_manifest("llamacpp-aux").binary.process_pattern

    main_cmdline = (
        "/opt/homebrew/bin/llama-server --ctx-size 131072 --parallel 2 "
        "--model /Users/jmn/llms/gguf/active.gguf --port 8080"
    )
    aux_cmdline = (
        "/opt/homebrew/bin/llama-server --ctx-size 32768 --parallel 4 "
        "--model /Users/jmn/llms/gguf/aux/active.gguf --port 8082"
    )

    assert main_pat in main_cmdline  # main matches its own cmdline
    assert aux_pat in aux_cmdline  # aux matches its own cmdline
    assert main_pat not in aux_cmdline  # main pattern must miss aux cmdline
    assert aux_pat not in main_cmdline  # aux pattern must miss main cmdline
    # And the ports differ.
    assert load_manifest("llamacpp").network.port != load_manifest("llamacpp-aux").network.port


def test_load_unknown_engine_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest("nonexistent_engine")


def test_health_url_property() -> None:
    m = load_manifest("ollama")
    # ollama leaves bind="" so health_url falls back to localhost.
    assert m.network.health_url == "http://127.0.0.1:11434/api/version"

    m2 = load_manifest("lmstudio")
    assert m2.network.health_url == "http://0.0.0.0:1234/v1/models"


def test_log_path_properties() -> None:
    """Manifests now store ~/Library/Logs/asiai/<engine>/ (US-015 fix).

    The properties expand ~ at access time so the absolute path works for
    both the daemon (UserName=jmn) and the install-time mkdir.
    """
    import os

    m = load_manifest("ollama")
    expected_dir = os.path.expanduser("~/Library/Logs/asiai/ollama")
    assert m.logs.stdout_path == f"{expected_dir}/ollama.log"
    assert m.logs.stderr_path == f"{expected_dir}/ollama.err"
    # The raw manifest still keeps the tilde (lisible côté human)
    assert m.logs.dir == "~/Library/Logs/asiai/ollama"


def test_invalid_plist_label_rejected(tmp_path: Path) -> None:
    """Plist labels outside com.asiai.* must fail validation hard."""
    raw = _read_minimal_dict()
    raw["plist"]["name"] = "com.evilcorp.malware"
    with pytest.raises(ManifestError, match=r"com\.asiai"):
        _from_dict(raw, source="<test>")


def test_invalid_anchor_name_rejected() -> None:
    raw = _read_minimal_dict()
    raw["firewall"]["anchor_name"] = "com.random.anchor"
    with pytest.raises(ManifestError, match=r"com\.asiai"):
        _from_dict(raw, source="<test>")


def test_invalid_port_rejected() -> None:
    raw = _read_minimal_dict()
    raw["network"]["port"] = 99999
    with pytest.raises(ManifestError, match="port"):
        _from_dict(raw, source="<test>")


def test_health_endpoint_must_start_with_slash() -> None:
    raw = _read_minimal_dict()
    raw["network"]["health_endpoint"] = "api/version"
    with pytest.raises(ManifestError, match="health_endpoint"):
        _from_dict(raw, source="<test>")


def test_wrapper_needed_without_install_path_rejected() -> None:
    raw = _read_minimal_dict()
    raw["wrapper"] = {"needed": True}
    with pytest.raises(ManifestError, match="install_path"):
        _from_dict(raw, source="<test>")


def test_env_var_must_be_key_value() -> None:
    raw = _read_minimal_dict()
    raw["environment"] = {"vars": ["NOT_A_KEY_VALUE_PAIR"]}
    with pytest.raises(ManifestError, match="KEY=VALUE"):
        _from_dict(raw, source="<test>")


# ---------------------------------------------------------------------------
# process_pattern validation (H2 — DoS via pgrep -f regex)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_pattern",
    [
        ".",  # matches every cmdline
        ".*",  # explicit greedy
        ".*sshd",  # would target sshd
        "(ollama|sshd)",  # alternation
        "ollama|sshd",  # alternation no parens
        "[a-z]+",  # char class
        "ollama\\s",  # backslash escape
        "ollama$",  # anchor
        "^ollama",  # anchor start
        "ollama?",  # quantifier
        "0lookups",  # leading digit (we require leading letter)
        "",  # empty
        " ollama",  # leading space
        "ollama ",  # trailing space
    ],
)
def test_dangerous_process_pattern_rejected(bad_pattern: str) -> None:
    raw = _read_minimal_dict()
    raw["binary"]["process_pattern"] = bad_pattern
    with pytest.raises(ManifestError, match="process_pattern"):
        _from_dict(raw, source="<test>")


@pytest.mark.parametrize(
    "good_pattern",
    [
        "ollama",
        "lmstudio-server-start",
        "llama-server-turboquant",
        "omlx serve",
        "a/b/c",
        "engine.bin",
    ],
)
def test_safe_process_pattern_accepted(good_pattern: str) -> None:
    raw = _read_minimal_dict()
    raw["binary"]["process_pattern"] = good_pattern
    _from_dict(raw, source="<test>")  # must not raise


def test_missing_required_section_rejected() -> None:
    raw = _read_minimal_dict()
    del raw["network"]
    with pytest.raises(ManifestError, match="network"):
        _from_dict(raw, source="<test>")


def test_is_valid_plist_label_predicate() -> None:
    assert is_valid_plist_label("com.asiai.ollama")
    assert is_valid_plist_label("com.asiai.my-engine-2")
    assert not is_valid_plist_label("com.evilcorp.x")
    assert not is_valid_plist_label("com.asiai.UPPERCASE")
    assert not is_valid_plist_label("com.asiai.")
    assert not is_valid_plist_label("com.asiai.with_underscore")


def _read_minimal_dict() -> dict:
    """Load the ollama manifest as a starting point for negative tests."""
    here = Path(__file__).resolve().parent.parent
    with (here / "data" / "engine_manifests" / "ollama.toml").open("rb") as f:
        return tomllib.load(f)
