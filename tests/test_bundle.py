"""Tests for the SMAppService bundle builder (ais_core.bundle).

Pure plist rendering is tested directly; the full build (clang + swiftc +
sips/iconutil) runs as a real end-to-end test when the toolchain is present
(macOS CI runners ship it), and is skipped otherwise.
"""

from __future__ import annotations

import argparse
import os
import plistlib
import pwd
import shutil
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ais_core import bundle, install_state
from ais_core.bundle import BundleError, BundleSpec
from ais_core.manifest import load_manifest, manifest_source_path

#: An account that actually exists on the test machine (build_bundle resolves
#: the daemon user via getpwnam and refuses uid 0 / unknown accounts).
_REAL_USER = pwd.getpwuid(os.getuid()).pw_name

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


class TestDaemonUserValidation:
    """I2 parity (story 2.6): embedded plists never carry root or a ghost account."""

    def test_unknown_user_refused(self, tmp_path: Path) -> None:
        spec = BundleSpec(
            services=("llamacpp-aux-1",), user="no-such-acct", launcher_path="/bin/true"
        )
        with pytest.raises(BundleError, match="unknown daemon user"):
            bundle.build_bundle(spec, tmp_path)

    def test_root_refused(self, tmp_path: Path) -> None:
        """Decided on pw_uid (getpwnam is case-insensitive on macOS), not the string."""
        spec = BundleSpec(services=("llamacpp-aux-1",), user="root", launcher_path="/bin/true")
        with pytest.raises(BundleError, match="never run as root"):
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
    parser.parse_args(["bundle", "register", "--allow-headless"])
    parser.parse_args(["bundle", "unregister"])
    parser.parse_args(["bundle", "status"])


# --- register guards (story 2.6: hard refusals, machine-enforced) ---------------


def _fake_app(tmp_path: Path) -> Path:
    """A minimal installed bundle: just the Register helper the wrapper locates."""
    macos = tmp_path / "Asiai.app" / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    (macos / "AsiaiRegister").write_text("#!/bin/sh\n")
    return tmp_path / "Asiai.app"


def _ctl_args(app: Path, action: str = "register", **kw) -> argparse.Namespace:
    return argparse.Namespace(
        app=str(app),
        action=action,
        service=kw.get("service"),
        allow_headless=kw.get("allow_headless", False),
    )


class TestRegisterGuards:
    def test_headless_register_refused(self, tmp_path, monkeypatch, capsys) -> None:
        from ais_cli import commands

        monkeypatch.setattr(commands, "_gui_session_active", lambda: False)
        rc = commands.cmd_bundle_ctl(_ctl_args(_fake_app(tmp_path)))
        assert rc == 2
        assert "no GUI session" in capsys.readouterr().err

    def test_allow_headless_bypasses_gui_gate(self, tmp_path, monkeypatch) -> None:
        from ais_cli import commands

        monkeypatch.setattr(commands, "_gui_session_active", lambda: False)
        monkeypatch.setattr(commands, "_register_blockers", lambda names: [])
        forwarded: dict = {}

        def fake_run(argv, check):
            forwarded["argv"] = argv
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(commands.subprocess, "run", fake_run)
        rc = commands.cmd_bundle_ctl(_ctl_args(_fake_app(tmp_path), allow_headless=True))
        assert rc == 0
        assert forwarded["argv"][1] == "register"

    def test_blockers_hard_refuse(self, tmp_path, monkeypatch, capsys) -> None:
        """AC story 2.6: legacy plist present -> REFUSAL (was a warning), and the
        Swift register helper is never invoked."""
        from ais_cli import commands

        monkeypatch.setattr(commands, "_gui_session_active", lambda: True)
        monkeypatch.setattr(
            commands,
            "_register_blockers",
            lambda names: ["aux-1: legacy plist /Library/LaunchDaemons/x.plist still installed"],
        )
        invoked: list = []
        monkeypatch.setattr(commands.subprocess, "run", lambda argv, check: invoked.append(argv))
        rc = commands.cmd_bundle_ctl(_ctl_args(_fake_app(tmp_path)))
        assert rc == 2
        assert not invoked  # hard refusal: nothing forwarded to SMAppService
        err = capsys.readouterr().err
        assert "register refused" in err
        assert "aisctl uninstall" in err


class TestRegisterBlockers:
    def _manifest(self, label: str):
        return SimpleNamespace(plist=SimpleNamespace(name=label))

    def test_legacy_plist_blocks(self, tmp_path, monkeypatch) -> None:
        from ais_cli import commands

        label = "com.asiai.zz-guard"
        (tmp_path / f"{label}.plist").write_text("x")
        monkeypatch.setattr(commands, "load_manifest", lambda n: self._manifest(label))
        monkeypatch.setattr(commands, "_daemon_loaded_from", lambda lb: None)
        blockers = commands._register_blockers(["zz-guard"], daemons_dir=tmp_path)
        assert blockers and "legacy plist" in blockers[0]

    def test_loaded_outside_bundle_blocks(self, tmp_path, monkeypatch) -> None:
        """Plist file already removed but the label never booted out -> still blocked."""
        from ais_cli import commands

        label = "com.asiai.zz-guard"
        monkeypatch.setattr(commands, "load_manifest", lambda n: self._manifest(label))
        monkeypatch.setattr(
            commands, "_daemon_loaded_from", lambda lb: f"/Library/LaunchDaemons/{lb}.plist"
        )
        blockers = commands._register_blockers(["zz-guard"], daemons_dir=tmp_path)
        assert blockers and "still loaded" in blockers[0]

    def test_unidentified_source_blocks(self, tmp_path, monkeypatch) -> None:
        """Fail closed: loaded but launchctl names no source."""
        from ais_cli import commands

        monkeypatch.setattr(
            commands, "load_manifest", lambda n: self._manifest("com.asiai.zz-guard")
        )
        monkeypatch.setattr(commands, "_daemon_loaded_from", lambda lb: "")
        blockers = commands._register_blockers(["zz-guard"], daemons_dir=tmp_path)
        assert blockers and "unidentified source" in blockers[0]

    def test_already_bundle_loaded_is_fine(self, tmp_path, monkeypatch) -> None:
        """Re-registering an already bundle-loaded label is idempotent, not blocked."""
        from ais_cli import commands

        label = "com.asiai.zz-guard"
        monkeypatch.setattr(commands, "load_manifest", lambda n: self._manifest(label))
        monkeypatch.setattr(
            commands,
            "_daemon_loaded_from",
            lambda lb: f"/Applications/Asiai.app/Contents/Library/LaunchDaemons/{lb}.plist",
        )
        assert commands._register_blockers(["zz-guard"], daemons_dir=tmp_path) == []

    def test_not_loaded_no_legacy_is_fine(self, tmp_path, monkeypatch) -> None:
        from ais_cli import commands

        monkeypatch.setattr(
            commands, "load_manifest", lambda n: self._manifest("com.asiai.zz-guard")
        )
        monkeypatch.setattr(commands, "_daemon_loaded_from", lambda lb: None)
        assert commands._register_blockers(["zz-guard"], daemons_dir=tmp_path) == []

    def test_smd_submitted_is_bundle_loaded_not_blocked(self, tmp_path, monkeypatch) -> None:
        """0.6.1 finding 2: launchctl reports a bundle-registered daemon as
        'path = (submitted by smd.<pid>)', with no filesystem path. That must
        read as bundle-loaded (idempotent re-register), not a foreign loader."""
        from ais_cli import commands

        label = "com.asiai.zz-guard"
        monkeypatch.setattr(commands, "load_manifest", lambda n: self._manifest(label))
        monkeypatch.setattr(commands, "_daemon_loaded_from", lambda lb: "(submitted by smd.347)")
        assert commands._register_blockers(["zz-guard"], daemons_dir=tmp_path) == []


class TestBundleSourceRecognition:
    """0.6.1 finding 2: _is_bundle_source accepts both forms launchctl uses."""

    def test_in_bundle_path(self) -> None:
        from ais_cli import commands

        assert commands._is_bundle_source(
            "/Applications/Asiai.app/Contents/Library/LaunchDaemons/com.asiai.x.plist"
        )

    def test_smd_attribution(self) -> None:
        from ais_cli import commands

        assert commands._is_bundle_source("(submitted by smd.42)")

    def test_legacy_path_is_not_bundle(self) -> None:
        from ais_cli import commands

        assert not commands._is_bundle_source("/Library/LaunchDaemons/com.asiai.x.plist")

    def test_none_and_empty_are_not_bundle(self) -> None:
        from ais_cli import commands

        assert not commands._is_bundle_source(None)
        assert not commands._is_bundle_source("")


class TestBundleServiceNames:
    """0.6.1 finding 1: whole-bundle register scopes to the bundle's own
    services, not every manifest on disk (a wrapper engine blocked it)."""

    def _make_bundle(self, tmp_path, services: list[str]) -> Path:
        daemons = tmp_path / "Asiai.app" / "Contents" / "Library" / "LaunchDaemons"
        daemons.mkdir(parents=True)
        for svc in services:
            plist_data = {
                "Label": f"com.asiai.{svc}",
                "ProgramArguments": [
                    "/Applications/Asiai.app/Contents/MacOS/Asiai",
                    svc,
                ],
            }
            with (daemons / f"com.asiai.{svc}.plist").open("wb") as fh:
                plistlib.dump(plist_data, fh)
        return tmp_path / "Asiai.app"

    def test_reads_embedded_service_names(self, tmp_path) -> None:
        from ais_cli import commands

        app = self._make_bundle(tmp_path, ["llamacpp", "ollama"])
        assert sorted(commands._bundle_service_names(app)) == ["llamacpp", "ollama"]

    def test_missing_daemons_dir_returns_empty(self, tmp_path) -> None:
        from ais_cli import commands

        assert commands._bundle_service_names(tmp_path / "Nope.app") == []


# --- end-to-end build (real toolchain, present on macOS CI) ---------------------


_toolchain = all(shutil.which(t) for t in ("clang", "swiftc", "sips", "iconutil"))


@pytest.mark.skipif(not _toolchain, reason="clang/swiftc/sips/iconutil not available")
class TestBuildEndToEnd:
    def test_build_unsigned_bundle(self, tmp_path: Path) -> None:
        launcher = tmp_path / "asiai-launch"
        launcher.write_text("#!/bin/sh\n")
        spec = BundleSpec(
            services=("llamacpp-aux-1",),
            user=_REAL_USER,  # build_bundle resolves via getpwnam (non-root, must exist)
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
