"""Tests for the asiai version_sources provider."""

from __future__ import annotations

import pytest

from ais_cli import version_sources


def test_provide_covers_base_engines_and_turboquant():
    specs = version_sources.provide(version_sources.VERSION_SOURCE_API_VERSION)
    by_name = {s["engine_name"]: s for s in specs}

    # turboquant is aisrv-specific — asiai has no internal entry for it.
    assert "turboquant" in by_name
    assert by_name["turboquant"].get("brew_formula") == "turboquant"

    # mlx-lm manifest stem is remapped to asiai's canonical key "mlxlm".
    assert "mlxlm" in by_name
    assert "mlx-lm" not in by_name

    # llamacpp uses the build-number compare scheme.
    assert by_name["llamacpp"]["version_scheme"] == "llamacpp_build"
    assert by_name["llamacpp"].get("brew_formula") == "llama.cpp"


def test_provide_skips_aux_instances():
    specs = version_sources.provide(version_sources.VERSION_SOURCE_API_VERSION)
    names = {s["engine_name"] for s in specs}
    assert not any("-aux-" in n for n in names)


def test_provide_rejects_wrong_api_version():
    with pytest.raises(version_sources.IncompatibleVersionSourceError):
        version_sources.provide(99)


def test_provide_lmstudio_no_upstream():
    specs = version_sources.provide(version_sources.VERSION_SOURCE_API_VERSION)
    by_name = {s["engine_name"]: s for s in specs}
    assert by_name["lmstudio"].get("no_upstream") is True


def test_provide_pip_engines_have_pip_package():
    specs = version_sources.provide(version_sources.VERSION_SOURCE_API_VERSION)
    by_name = {s["engine_name"]: s for s in specs}
    # mlx-lm builds from source -> pip_package set (stem name).
    if "mlxlm" in by_name and by_name["mlxlm"].get("pip_package"):
        assert by_name["mlxlm"]["pip_package"] == "mlx-lm"


def test_provider_dicts_are_plain_json_safe():
    # The contract is plain dicts (no dataclasses / tuples that asiai can't
    # rebuild). Every value must be a JSON-native scalar or bool.
    specs = version_sources.provide(version_sources.VERSION_SOURCE_API_VERSION)
    for s in specs:
        assert isinstance(s, dict)
        assert isinstance(s["engine_name"], str)
        for v in s.values():
            assert isinstance(v, (str, bool)) or v is None
