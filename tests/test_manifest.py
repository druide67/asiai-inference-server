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

EXPECTED_ENGINES = {"ollama", "lmstudio", "omlx", "turboquant"}


def test_list_manifests_returns_four_engines() -> None:
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
        ".",                  # matches every cmdline
        ".*",                 # explicit greedy
        ".*sshd",             # would target sshd
        "(ollama|sshd)",      # alternation
        "ollama|sshd",        # alternation no parens
        "[a-z]+",             # char class
        "ollama\\s",          # backslash escape
        "ollama$",            # anchor
        "^ollama",            # anchor start
        "ollama?",            # quantifier
        "0lookups",           # leading digit (we require leading letter)
        "",                   # empty
        " ollama",            # leading space
        "ollama ",            # trailing space
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
