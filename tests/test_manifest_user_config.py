"""User-config discovery — XDG override of bundled manifests / presets.

These tests redirect the user-config root to a pytest ``tmp_path`` via
``ASIAI_USER_CONFIG_DIR`` so they exercise the discovery path without
touching the real ``~/.config/asiai-inference-server/`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ais_core.manifest import (
    list_manifests,
    list_presets,
    load_manifest,
)


def _make_manifest_toml(name: str, port: int, instance_suffix: str) -> str:
    """Minimal but valid manifest TOML targeting the llamacpp-aux family pattern."""
    return f"""\
name = "{name}"
display = "user-added {name}"

[binary]
candidates = ["/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"]
process_pattern = "llms/gguf/aux{instance_suffix}/active.gguf"
model_path = "~/llms/gguf/aux{instance_suffix}/active.gguf"
program_args = ["--mlock", "--n-gpu-layers", "999"]
builds_from_source = false

[plist]
name = "com.asiai.{name}"
throttle_interval = 10
timeout = 30

[network]
port = {port}
bind = "0.0.0.0"
health_endpoint = "/health"
health_timeout = 30

[firewall]
supported = true
anchor_name = "com.asiai.{name}"

[logs]
dir = "~/Library/Logs/asiai/{name}"
stdout = "{name}.log"
stderr = "{name}.err"

[wrapper]
needed = false

[environment]
vars = []
"""


def _make_preset_toml(name: str, engine: str, port: int) -> str:
    """Minimal valid preset TOML — a self-contained manifest for ``engine``."""
    return f"""\
name = "{engine}"
display = "user preset {name}"

[binary]
candidates = ["/opt/homebrew/bin/llama-server"]
process_pattern = "llms/gguf/aux1/active.gguf"
model_path = "~/llms/gguf/aux1/active.gguf"
program_args = ["--ctx-size", "65536"]
builds_from_source = false

[plist]
name = "com.asiai.{engine}"
throttle_interval = 10
timeout = 30

[network]
port = {port}
bind = "0.0.0.0"
health_endpoint = "/health"
health_timeout = 30

[firewall]
supported = true
anchor_name = "com.asiai.{engine}"

[logs]
dir = "~/Library/Logs/asiai/{engine}"
stdout = "{engine}.log"
stderr = "{engine}.err"

[wrapper]
needed = false

[environment]
vars = []
"""


@pytest.fixture
def user_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the user-config root to ``tmp_path`` and create the
    standard subtree."""
    monkeypatch.setenv("ASIAI_USER_CONFIG_DIR", str(tmp_path))
    (tmp_path / "engine_manifests" / "presets").mkdir(parents=True)
    return tmp_path


def test_list_manifests_includes_user_added_aux_instance(user_cfg: Path) -> None:
    """A user can add a fifth llamacpp-aux instance without touching the package."""
    (user_cfg / "engine_manifests" / "llamacpp-aux-5.toml").write_text(
        _make_manifest_toml("llamacpp-aux-5", 8094, "5")
    )
    names = list_manifests()
    assert "llamacpp-aux-5" in names
    # Bundled siblings are still discoverable in parallel
    assert "llamacpp-aux-1" in names
    assert "llamacpp-aux-4" in names


def test_load_user_added_manifest_returns_typed_dataclass(user_cfg: Path) -> None:
    (user_cfg / "engine_manifests" / "llamacpp-aux-5.toml").write_text(
        _make_manifest_toml("llamacpp-aux-5", 8094, "5")
    )
    m = load_manifest("llamacpp-aux-5")
    assert m.name == "llamacpp-aux-5"
    assert m.network.port == 8094
    assert m.binary.process_pattern == "llms/gguf/aux5/active.gguf"
    assert m.plist.name == "com.asiai.llamacpp-aux-5"


def test_user_manifest_overrides_bundled(user_cfg: Path, caplog) -> None:
    """A user manifest sharing the stem with a bundled one wins, with a warning."""
    # Override llamacpp-aux-4 (bundled) with a user version using a different port.
    (user_cfg / "engine_manifests" / "llamacpp-aux-4.toml").write_text(
        _make_manifest_toml("llamacpp-aux-4", 9999, "4")
    )
    import logging

    caplog.set_level(logging.WARNING, logger="ais_core.manifest")
    m = load_manifest("llamacpp-aux-4")
    assert m.network.port == 9999  # user wins over bundled (8093)
    assert any(
        "user manifest llamacpp-aux-4 overrides bundled" in rec.message for rec in caplog.records
    )


def test_list_presets_includes_user_added_preset(user_cfg: Path) -> None:
    (user_cfg / "engine_manifests" / "presets" / "my-custom-preset.toml").write_text(
        _make_preset_toml("my-custom-preset", "llamacpp-aux-1", 8090)
    )
    presets = list_presets()
    assert "my-custom-preset" in presets


def test_load_user_added_preset(user_cfg: Path) -> None:
    (user_cfg / "engine_manifests" / "presets" / "my-custom-preset.toml").write_text(
        _make_preset_toml("my-custom-preset", "llamacpp-aux-1", 8090)
    )
    m = load_manifest("llamacpp-aux-1", preset="my-custom-preset")
    assert m.name == "llamacpp-aux-1"
    assert m.display == "user preset my-custom-preset"


def test_user_preset_overrides_bundled(user_cfg: Path, caplog) -> None:
    """A user preset sharing the stem with a bundled one wins, with a warning."""
    # Override the bundled aux-1 preset with a user version using a different port.
    (user_cfg / "engine_manifests" / "presets" / "qwen3-4b-instruct-hermes-aux-1.toml").write_text(
        _make_preset_toml("qwen3-4b-instruct-hermes-aux-1", "llamacpp-aux-1", 9090)
    )
    import logging

    caplog.set_level(logging.WARNING, logger="ais_core.manifest")
    m = load_manifest("llamacpp-aux-1", preset="qwen3-4b-instruct-hermes-aux-1")
    assert m.network.port == 9090  # user wins
    assert any(
        "user preset qwen3-4b-instruct-hermes-aux-1 overrides bundled" in rec.message
        for rec in caplog.records
    )


def test_user_dir_absent_falls_back_to_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the user-config dir does not exist, discovery degrades to bundled-only
    (no crash). CI runners don't have ~/.config/asiai-inference-server/ — this
    must keep working."""
    nonexistent = tmp_path / "definitely-not-there"
    monkeypatch.setenv("ASIAI_USER_CONFIG_DIR", str(nonexistent))

    names = list_manifests()
    # Bundled engines still discoverable
    assert "llamacpp" in names
    assert "llamacpp-aux-1" in names

    presets = list_presets()
    # Bundled presets still discoverable
    assert "qwen3-4b-instruct-hermes-aux-1" in presets


def test_cli_discovers_user_added_engine(user_cfg: Path) -> None:
    """End-to-end: ``aisctl list`` surfaces a user-added llamacpp-aux instance.

    Crucially this validates that the family-pattern dispatch in
    ``ais_cli.commands.get_driver_factory()`` resolves the new manifest
    name to a callable driver factory at runtime — the user did not have
    to add any Python code.
    """
    (user_cfg / "engine_manifests" / "llamacpp-aux-7.toml").write_text(
        _make_manifest_toml("llamacpp-aux-7", 8096, "7")
    )

    from ais_cli.commands import get_driver_factory

    factory = get_driver_factory("llamacpp-aux-7")
    driver = factory()
    assert driver.manifest.name == "llamacpp-aux-7"
    assert driver.manifest.network.port == 8096


# ---------------------------------------------------------------------------
# Pre-v0.3 preset location: <config>/presets/ is dead, loudly (issue #7)
# ---------------------------------------------------------------------------


def test_legacy_dir_presets_are_ignored_with_warning(user_cfg: Path, caplog) -> None:
    """TOMLs in the pre-v0.3 root-level presets/ are not resolved anymore,
    and the operator is told exactly which files moved out of resolution
    and where to put them — a preset silently dropping out after an
    upgrade is the same failure class issue #6 exists for."""
    legacy = user_cfg / "presets"
    legacy.mkdir()
    (legacy / "old-preset.toml").write_text(_make_preset_toml("old-preset", "llamacpp-aux-1", 8090))
    import logging

    caplog.set_level(logging.WARNING, logger="ais_core.manifest")
    assert "old-preset" not in list_presets()
    with pytest.raises(FileNotFoundError):
        load_manifest("llamacpp-aux-1", preset="old-preset")
    assert any(
        "deprecated location" in rec.message and "old-preset.toml" in rec.getMessage()
        for rec in caplog.records
    )
