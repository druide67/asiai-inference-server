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

EXPECTED_ENGINES = {
    "ollama",
    "lmstudio",
    "omlx",
    "turboquant",
    "llamacpp",
    "llamacpp-aux-1",
    "llamacpp-aux-2",
    "llamacpp-aux-3",
    "llamacpp-aux-4",
    "llamacpp-aux-5",
    "vmlx",
    "mlx-lm",
}


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


def test_vmlx_specifics() -> None:
    m = load_manifest("vmlx")
    assert m.network.port == 8003
    assert m.network.health_endpoint == "/v1/models"
    assert m.binary.process_pattern == "vmlx serve"
    assert m.plist.name == "com.asiai.vmlx"
    assert m.firewall.anchor_name == "com.asiai.vmlx"
    assert not m.wrapper.needed
    assert not m.binary.builds_from_source


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


@pytest.mark.parametrize(
    "name, port",
    [
        ("llamacpp-aux-1", 8090),
        ("llamacpp-aux-2", 8091),
        ("llamacpp-aux-3", 8092),
        ("llamacpp-aux-4", 8093),
        ("llamacpp-aux-5", 8094),
    ],
)
def test_llamacpp_aux_baseline_specifics(name: str, port: int) -> None:
    """Baseline llamacpp-aux-N.toml is the generic OSS-safe defaults (no tuning).

    Each manifest in the aux family declares its own port (8090-8093) and a
    distinct ``llms/gguf/auxN/active.gguf`` symlink path so the four
    instances coexist on the host without process-pattern cross-match.
    """
    m = load_manifest(name)
    assert m.network.port == port
    assert m.network.health_endpoint == "/health"
    assert m.plist.name == f"com.asiai.{name}"
    assert m.firewall.anchor_name == f"com.asiai.{name}"
    instance_suffix = name.rsplit("-", 1)[-1]  # "1", "2", "3", "4"
    assert m.binary.process_pattern == f"llms/gguf/aux{instance_suffix}/active.gguf"
    assert m.binary.model_path == f"~/llms/gguf/aux{instance_suffix}/active.gguf"
    pa = list(m.binary.program_args)
    assert "--mlock" in pa
    assert "--cont-batching" in pa
    # No workload-specific tuning in the baseline
    assert "--cache-reuse" not in pa
    assert "--slot-prompt-similarity" not in pa


@pytest.mark.parametrize(
    "name, preset, expected_parallel, expected_ctx_size, expected_v_quant",
    [
        # ctx-size sized so the slot (= ctx-size / parallel) stays >= 64K for
        # orchestrators that gate boot on the per-slot context window.
        # aux-1 + aux-5 use the TurboQuant V=turbo2 quant (validated 2026-05-22),
        # other aux instances stay on stock q8_0 (smaller KV footprint already).
        ("llamacpp-aux-1", "qwen3-4b-instruct-hermes-aux-1", "4", "262144", "turbo2"),
        ("llamacpp-aux-2", "qwen3-1.7b-instruct-hermes-aux-2", "2", "131072", "q8_0"),
        ("llamacpp-aux-3", "qwen3-0.6b-instruct-hermes-aux-3", "2", "131072", "q8_0"),
        ("llamacpp-aux-4", "qwen2.5-vl-7b-instruct-hermes-aux-4", "1", "65536", "q8_0"),
        ("llamacpp-aux-5", "qwen3-4b-instruct-hermes-aux-5-compression", "1", "262144", "turbo2"),
    ],
)
def test_llamacpp_aux_presets_target_engine_and_carry_tuning(
    name: str,
    preset: str,
    expected_parallel: str,
    expected_ctx_size: str,
    expected_v_quant: str,
) -> None:
    """Each aux-N preset targets its own manifest and carries Hermes-class tuning."""
    m = load_manifest(name, preset=preset)
    assert m.name == name
    pa = list(m.binary.program_args)
    assert "--ctx-size" in pa
    assert pa[pa.index("--ctx-size") + 1] == expected_ctx_size
    assert pa[pa.index("--parallel") + 1] == expected_parallel
    # Slot effective is ctx-size / parallel; orchestrators that check via
    # /v1/models read this as the usable window.
    assert int(expected_ctx_size) // int(expected_parallel) >= 65536
    assert "--cache-reuse" in pa
    # K stays q8_0 across all aux presets (K is more sensitive than V to
    # aggressive quants, validated by failed K=turbo3 / K=q4_0 benchs).
    assert pa[pa.index("--cache-type-k") + 1] == "q8_0"
    assert pa[pa.index("--cache-type-v") + 1] == expected_v_quant


def test_llamacpp_and_aux_process_patterns_are_disjoint() -> None:
    """Critical invariant: pkill -f on one must never match a peer's cmdline.

    The real semantic we care about is: given a representative cmdline for
    each instance (the kind ``pgrep -f`` actually sees in production), the
    pattern of one engine must not match the cmdline of any other. The
    aux-N suffixes mean each ``auxN/`` segment in the model path
    discriminates one instance against the four others (main + 3 peers).
    """
    main_pat = load_manifest("llamacpp").binary.process_pattern
    aux_pats = {
        f"llamacpp-aux-{i}": load_manifest(f"llamacpp-aux-{i}").binary.process_pattern
        for i in range(1, 6)
    }

    cmdlines = {
        "llamacpp": (
            "/opt/homebrew/bin/llama-server --ctx-size 131072 --parallel 2 "
            "--model /path/to/llms/gguf/active.gguf --port 8080"
        ),
        "llamacpp-aux-1": (
            "/opt/homebrew/bin/llama-server --ctx-size 65536 --parallel 4 "
            "--model /path/to/llms/gguf/aux1/active.gguf --port 8090"
        ),
        "llamacpp-aux-2": (
            "/opt/homebrew/bin/llama-server --ctx-size 65536 --parallel 2 "
            "--model /path/to/llms/gguf/aux2/active.gguf --port 8091"
        ),
        "llamacpp-aux-3": (
            "/opt/homebrew/bin/llama-server --ctx-size 65536 --parallel 2 "
            "--model /path/to/llms/gguf/aux3/active.gguf --port 8092"
        ),
        "llamacpp-aux-4": (
            "/opt/homebrew/bin/llama-server --ctx-size 65536 --parallel 1 "
            "--model /path/to/llms/gguf/aux4/active.gguf --port 8093"
        ),
        "llamacpp-aux-5": (
            "/opt/homebrew/bin/llama-server --ctx-size 262144 --parallel 1 "
            "--model /path/to/llms/gguf/aux5/active.gguf --port 8094"
        ),
    }

    # Each pattern matches its own cmdline.
    assert main_pat in cmdlines["llamacpp"]
    for name, pat in aux_pats.items():
        assert pat in cmdlines[name], name

    # Main pattern misses every aux cmdline.
    for name in aux_pats:
        assert main_pat not in cmdlines[name], name

    # Each aux pattern misses every peer cmdline (main + 3 sibling aux).
    for name, pat in aux_pats.items():
        for other_name, other_cmdline in cmdlines.items():
            if other_name == name:
                continue
            assert pat not in other_cmdline, (name, other_name)

    # All five ports are distinct.
    ports = {
        "llamacpp": load_manifest("llamacpp").network.port,
        **{name: load_manifest(name).network.port for name in aux_pats},
    }
    assert len(set(ports.values())) == len(ports), ports


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
        "vmlx serve",
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
