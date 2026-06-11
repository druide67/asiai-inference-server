"""Tests for the SMAppService bundle builder (ais_core.bundle).

Pure plist rendering is tested directly; the full build (clang + swiftc +
sips/iconutil) runs as a real end-to-end test when the toolchain is present
(macOS CI runners ship it), and is skipped otherwise.
"""

from __future__ import annotations

import plistlib
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from ais_core import bundle, install_state
from ais_core.bundle import BundleError, BundleSpec
from ais_core.manifest import load_manifest, manifest_source_path

_SPEC = BundleSpec(
    services=("llamacpp-aux-1", "llamacpp-aux-2"),
    user="enginerunner",
    launcher_path="/opt/asiai/bin/asiai-launch",
    version="1.2.3",
)


# --- pure plist rendering ------------------------------------------------------


class TestInfoPlist:
    def test_identity_and_background_keys(self) -> None:
        info = bundle.build_info_plist_dict(_SPEC)
        assert info["CFBundleIdentifier"] == "dev.asiai.engines"
        assert info["CFBundleExecutable"] == "Asiai"
        assert info["CFBundleDisplayName"] == "asiai Inference Engines"
        assert info["CFBundlePackageType"] == "APPL"
        assert info["LSBackgroundOnly"] is True
        assert info["LSUIElement"] is True
        assert info["CFBundleIconFile"] == "AppIcon"
        assert info["CFBundleShortVersionString"] == "1.2.3"


class TestEmbeddedPlist:
    def test_long_running_regime_and_identity(self) -> None:
        m = load_manifest("llamacpp-aux-1")
        d = bundle.build_embedded_plist_dict(_SPEC, m)
        # Label stays com.asiai.* — compatible with the whole existing
        # lifecycle tooling (status, sudoers scope, process supervision).
        assert d["Label"] == "com.asiai.llamacpp-aux-1"
        # The stub inside the bundle + the service name: nothing tuning-
        # specific may live in a sealed plist.
        assert d["ProgramArguments"] == [
            "/Applications/Asiai.app/Contents/MacOS/Asiai",
            "llamacpp-aux-1",
        ]
        assert d["UserName"] == "enginerunner"
        assert d["AssociatedBundleIdentifiers"] == ["dev.asiai.engines"]
        # Long-running regime: KeepAlive on crash, and NEVER StartInterval
        # (a periodic daemon with KeepAlive would respawn-loop).
        assert d["KeepAlive"] == {"Crashed": True, "SuccessfulExit": False}
        assert "StartInterval" not in d
        assert d["RunAtLoad"] is True

    def test_env_carries_only_static_home_path(self) -> None:
        m = load_manifest("llamacpp-aux-1")
        d = bundle.build_embedded_plist_dict(_SPEC, m)
        # Tuning env (TURBO_* etc.) is the active manifest's job, applied by
        # asiai-launch at exec time — the sealed plist gets HOME/PATH only.
        assert set(d["EnvironmentVariables"]) == {"HOME", "PATH"}


class TestServiceValidation:
    def test_unknown_service_rejected(self) -> None:
        with pytest.raises(BundleError, match="unknown engine"):
            bundle._load_service_manifest("never-heard-of-it")

    def test_wrapper_engine_rejected(self) -> None:
        with pytest.raises(BundleError, match="wrapper-based engine"):
            bundle._load_service_manifest("turboquant")

    def test_build_requires_services(self, tmp_path: Path) -> None:
        spec = BundleSpec(services=(), user="u", launcher_path="/bin/true")
        with pytest.raises(BundleError, match="no services"):
            bundle.build_bundle(spec, tmp_path)


# --- active manifests ----------------------------------------------------------


class TestActivate:
    def test_writes_safe_perms_and_parses(self, tmp_path: Path) -> None:
        result = bundle.write_active_manifest("llamacpp-aux-1", dest_dir=tmp_path)
        dest = Path(result["active_manifest"])
        assert dest.is_file()
        mode = dest.stat().st_mode
        assert not mode & (stat.S_IWGRP | stat.S_IWOTH)  # launcher tamper gate
        # Round-trip: the launcher must accept what activate published.
        from ais_core.launch import load_active_manifest

        m = load_active_manifest(dest)
        assert m.name == "llamacpp-aux-1"

    def test_defaults_to_recorded_preset(self, tmp_path: Path) -> None:
        preset = "qwen3-4b-instruct-hermes-aux-1"
        src = manifest_source_path("llamacpp-aux-1", preset)
        assert src is not None
        install_state.record_install("llamacpp-aux-1", preset=preset, manifest_path=src)

        result = bundle.write_active_manifest("llamacpp-aux-1", dest_dir=tmp_path)
        assert result["preset"] == preset
        published = Path(result["active_manifest"]).read_bytes()
        assert published == src.read_bytes()

    def test_without_record_uses_base_manifest(self, tmp_path: Path) -> None:
        result = bundle.write_active_manifest("llamacpp-aux-2", dest_dir=tmp_path)
        assert result["preset"] is None
        src = manifest_source_path("llamacpp-aux-2")
        assert src is not None
        assert Path(result["active_manifest"]).read_bytes() == src.read_bytes()

    def test_wrapper_engine_refused(self, tmp_path: Path) -> None:
        with pytest.raises(BundleError, match="wrapper-based engine"):
            bundle.write_active_manifest("turboquant", dest_dir=tmp_path)

    def test_default_dir_follows_launcher_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """activate and asiai-launch must agree on the active dir."""
        monkeypatch.setenv("ASIAI_LAUNCH_MANIFEST_DIR", str(tmp_path / "active"))
        result = bundle.write_active_manifest("llamacpp-aux-1")
        assert result["active_manifest"] == str(tmp_path / "active" / "llamacpp-aux-1.toml")


# --- CLI surface ----------------------------------------------------------------


def test_bundle_subcommands_parse() -> None:
    from ais_cli.__main__ import build_parser

    parser = build_parser()
    parser.parse_args(["bundle", "build", "--services", "llamacpp-aux-1", "--sign", "X"])
    parser.parse_args(["bundle", "activate", "llamacpp-aux-1", "--preset", "p"])
    parser.parse_args(["bundle", "register", "llamacpp-aux-1"])
    parser.parse_args(["bundle", "unregister"])
    parser.parse_args(["bundle", "status"])


# --- end-to-end build (real toolchain, present on macOS CI) ---------------------


_toolchain = all(shutil.which(t) for t in ("clang", "swiftc", "sips", "iconutil"))


@pytest.mark.skipif(not _toolchain, reason="clang/swiftc/sips/iconutil not available")
class TestBuildEndToEnd:
    def test_build_unsigned_bundle(self, tmp_path: Path) -> None:
        launcher = tmp_path / "asiai-launch"
        launcher.write_text("#!/bin/sh\n")
        spec = BundleSpec(
            services=("llamacpp-aux-1",),
            user="enginerunner",
            launcher_path=str(launcher),
            version="9.9.9",
        )
        result = bundle.build_bundle(spec, tmp_path / "dist")

        app = Path(result["app"])
        assert app.name == "Asiai.app"
        info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
        assert info["CFBundleIdentifier"] == "dev.asiai.engines"

        embedded = app / "Contents" / "Library" / "LaunchDaemons" / "com.asiai.llamacpp-aux-1.plist"
        d = plistlib.loads(embedded.read_bytes())
        assert d["ProgramArguments"][1] == "llamacpp-aux-1"

        stub = app / "Contents" / "MacOS" / "Asiai"
        register = app / "Contents" / "MacOS" / "AsiaiRegister"
        for binary in (stub, register):
            assert binary.is_file()
            assert binary.stat().st_mode & stat.S_IXUSR

        assert (app / "Contents" / "Resources" / "AppIcon.icns").stat().st_size > 10_000
        assert result["signed"] is False

        # The register helper must at least answer usage without SMAppService.
        proc = subprocess.run([str(register)], capture_output=True, text=True, check=False)
        assert proc.returncode == 2
        assert "llamacpp-aux-1" in proc.stdout
