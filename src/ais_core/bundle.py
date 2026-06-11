"""SMAppService bundle builder — ``Asiai.app`` for the Background Items panel.

Why a bundle
------------
Raw ``/Library/LaunchDaemons`` plists show up in macOS Settings > Login Items
& Extensions as unidentified Unix executables (generic icon, binary name).
Registering the same daemons through an app bundle (``SMAppService.daemon``)
gives them ONE entry with a proper display name and a custom icon.

Architecture (sealed-bundle constraint)
---------------------------------------
A signed bundle is sealed: changing anything inside (including an embedded
plist) invalidates the signature. So nothing machine- or tuning-specific
lives in the bundle:

* Each embedded LaunchDaemon plist runs ``<bundle>/Contents/MacOS/<App> <service>``
  — a tiny C stub that exec()s ``asiai-launch <service>``.
* ``asiai-launch`` (:mod:`ais_core.launch`) resolves the service's ACTIVE
  manifest (a TOML outside the bundle, written by ``aisctl bundle activate``)
  and execs the engine binary with its parameters and environment.

Changing a model, a context size or an env var = rewriting the active
manifest. The bundle itself never changes, so it is signed once.

Layout produced::

    <App>.app/Contents/
    ├── Info.plist
    ├── MacOS/
    │   ├── <App>             # exec stub (C), launcher path baked at build
    │   └── <App>Register     # SMAppService register/unregister/status (Swift)
    ├── Resources/AppIcon.icns
    └── Library/LaunchDaemons/<label>.plist   (one per service)

Code signing is the deployer's step (a locally-trusted code-signing cert;
on macOS 26+ BTM silently rejects custom icons from ad-hoc-signed bundles).
``build_bundle`` signs when an identity is supplied and otherwise leaves the
bundle unsigned — fully functional, generic icon.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ais_core import install_state
from ais_core.manifest import (
    EngineManifest,
    list_manifests,
    load_manifest,
    manifest_source_path,
)

#: Sizes of an .iconset entry mapped to their pixel dimension.
_ICONSET_SIZES: tuple[tuple[str, int], ...] = (
    ("icon_16x16.png", 16),
    ("icon_16x16@2x.png", 32),
    ("icon_32x32.png", 32),
    ("icon_32x32@2x.png", 64),
    ("icon_128x128.png", 128),
    ("icon_128x128@2x.png", 256),
    ("icon_256x256.png", 256),
    ("icon_256x256@2x.png", 512),
    ("icon_512x512.png", 512),
    ("icon_512x512@2x.png", 1024),
)


class BundleError(RuntimeError):
    """Bundle build/activation failed for a non-recoverable reason."""


@dataclass(frozen=True)
class BundleSpec:
    """Everything that parameterizes a bundle build.

    Defaults target the asiai namespace (``dev.asiai.engines``). Forks and
    third-party deployments should override ``bundle_id`` with their own
    reverse-DNS namespace.
    """

    services: tuple[str, ...]
    user: str
    launcher_path: str
    bundle_id: str = "dev.asiai.engines"
    app_name: str = "Asiai"
    display_name: str = "asiai Inference Engines"
    version: str = "0.0.0"
    icon_png: Path | None = None  # defaults to the bundled asiai logo

    def app_dir(self, output_dir: Path) -> Path:
        return output_dir / f"{self.app_name}.app"


def _bundle_data_dir() -> Path:
    """Locate ``data/bundle`` in either install layout (wheel or editable)."""
    here = Path(__file__).resolve().parent  # src/ais_core

    bundled = here / "data" / "bundle"
    if bundled.is_dir():
        return bundled

    repo_root = here.parent.parent
    editable = repo_root / "data" / "bundle"
    if editable.is_dir():
        return editable

    raise BundleError(f"data/bundle not found under {bundled} or {editable}")


# ---------------------------------------------------------------------------
# plist rendering (pure functions)
# ---------------------------------------------------------------------------


def build_info_plist_dict(spec: BundleSpec) -> dict:
    return {
        "CFBundleIdentifier": spec.bundle_id,
        "CFBundleName": spec.app_name,
        "CFBundleDisplayName": spec.display_name,
        "CFBundleExecutable": spec.app_name,
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": spec.version,
        "CFBundleVersion": spec.version,
        "CFBundleIconFile": "AppIcon",
        "CFBundleIconName": "AppIcon",
        "LSBackgroundOnly": True,
        "LSUIElement": True,
        "LSMinimumSystemVersion": "13.0",
    }


def build_embedded_plist_dict(spec: BundleSpec, manifest: EngineManifest) -> dict:
    """Embedded LaunchDaemon plist for one engine service.

    Long-running regime only (engines): ``KeepAlive`` on crash, no
    ``StartInterval`` — a periodic daemon would loop forever with KeepAlive.
    Everything machine-specific beyond the daemon user stays OUT of this
    plist (sealed bundle): tuning args and env come from the active manifest
    via ``asiai-launch``.
    """
    stub = f"/Applications/{spec.app_name}.app/Contents/MacOS/{spec.app_name}"
    user_home = os.path.expanduser(f"~{spec.user}")
    path_value = ":".join(
        [
            f"{user_home}/.local/bin",
            "/opt/homebrew/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ]
    )
    return {
        "Label": manifest.plist.name,
        "Comment": (
            f"{manifest.display} — managed by asiai-inference-server (SMAppService bundle)"
        ),
        "ProgramArguments": [stub, manifest.name],
        "UserName": spec.user,
        "EnvironmentVariables": {"HOME": user_home, "PATH": path_value},
        "RunAtLoad": True,
        "KeepAlive": {"Crashed": True, "SuccessfulExit": False},
        "ThrottleInterval": manifest.plist.throttle_interval,
        "ProcessType": "Interactive",
        "AssociatedBundleIdentifiers": [spec.bundle_id],
        "StandardOutPath": manifest.logs.stdout_path,
        "StandardErrorPath": manifest.logs.stderr_path,
        "TimeOut": manifest.plist.timeout,
        "Nice": 0,
    }


def _load_service_manifest(name: str) -> EngineManifest:
    if name not in list_manifests():
        raise BundleError(f"unknown engine {name!r}; known: {', '.join(list_manifests())}")
    manifest = load_manifest(name)
    if manifest.wrapper.needed:
        raise BundleError(
            f"{name!r} is a wrapper-based engine; it is launched via the boot "
            "prelude and cannot be embedded in the SMAppService bundle"
        )
    return manifest


# ---------------------------------------------------------------------------
# build steps
# ---------------------------------------------------------------------------


def _run(argv: list[str], *, what: str) -> None:
    try:
        subprocess.run(argv, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise BundleError(f"{what}: {argv[0]} not found — install Xcode CLT") from exc
    except subprocess.CalledProcessError as exc:
        raise BundleError(f"{what} failed (exit {exc.returncode}): {exc.stderr.strip()}") from exc


def _render_icns(source_png: Path, dest_icns: Path, work_dir: Path) -> None:
    """PNG -> .iconset (sips) -> .icns (iconutil). CLI-only, no GUI needed."""
    iconset = work_dir / "AppIcon.iconset"
    iconset.mkdir(parents=True, exist_ok=True)
    for filename, size in _ICONSET_SIZES:
        _run(
            [
                "sips",
                "-z",
                str(size),
                str(size),
                str(source_png),
                "--out",
                str(iconset / filename),
            ],
            what=f"icon resize {size}px",
        )
    _run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(dest_icns)],
        what="iconutil icns assembly",
    )


def _compile_stub(template: Path, dest: Path, launcher_path: str) -> None:
    _run(
        [
            "clang",
            "-O2",
            f'-DASIAI_LAUNCHER_PATH="{launcher_path}"',
            "-o",
            str(dest),
            str(template),
        ],
        what="exec stub compilation",
    )


def _compile_register(
    template: Path, dest: Path, services_map: dict[str, str], work_dir: Path
) -> None:
    literal_entries = ", ".join(
        f'"{service}": "{plist_name}"' for service, plist_name in sorted(services_map.items())
    )
    source = template.read_text().replace("__SERVICES_MAP__", f"[{literal_entries}]")
    swift_src = work_dir / "register.swift"
    swift_src.write_text(source)
    _run(
        ["swiftc", "-O", "-o", str(dest), str(swift_src), "-framework", "ServiceManagement"],
        what="register helper compilation",
    )


def sign_bundle(app_dir: Path, identity: str) -> None:
    """xattr-strip then deep-sign the bundle with a code-signing identity."""
    _run(["xattr", "-cr", str(app_dir)], what="xattr strip")
    _run(
        ["codesign", "--force", "--deep", "--sign", identity, str(app_dir)],
        what="codesign",
    )


def build_bundle(
    spec: BundleSpec,
    output_dir: Path,
    *,
    sign_identity: str | None = None,
) -> dict:
    """Build ``<App>.app`` under ``output_dir``. Returns a summary dict."""
    if not spec.services:
        raise BundleError("no services given; pass at least one engine name")
    manifests = [_load_service_manifest(s) for s in spec.services]

    data_dir = _bundle_data_dir()
    icon_png = spec.icon_png if spec.icon_png is not None else data_dir / "AppIcon-1024.png"
    if not icon_png.is_file():
        raise BundleError(f"icon source {icon_png} not found")
    if not Path(spec.launcher_path).is_file():
        raise BundleError(
            f"launcher {spec.launcher_path!r} not found; install asiai-inference-server "
            "(provides asiai-launch) or pass --launcher"
        )

    app = spec.app_dir(output_dir)
    if app.exists():
        shutil.rmtree(app)
    contents = app / "Contents"
    macos_dir = contents / "MacOS"
    resources = contents / "Resources"
    daemons_dir = contents / "Library" / "LaunchDaemons"
    for d in (macos_dir, resources, daemons_dir):
        d.mkdir(parents=True)

    (contents / "Info.plist").write_bytes(
        plistlib.dumps(build_info_plist_dict(spec), fmt=plistlib.FMT_XML, sort_keys=False)
    )

    services_map: dict[str, str] = {}
    for manifest in manifests:
        plist_name = f"{manifest.plist.name}.plist"
        services_map[manifest.name] = plist_name
        (daemons_dir / plist_name).write_bytes(
            plistlib.dumps(
                build_embedded_plist_dict(spec, manifest),
                fmt=plistlib.FMT_XML,
                sort_keys=False,
            )
        )

    work_dir = output_dir / ".bundle-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        _compile_stub(
            data_dir / "templates" / "stub.c", macos_dir / spec.app_name, spec.launcher_path
        )
        _compile_register(
            data_dir / "templates" / "register.swift",
            macos_dir / f"{spec.app_name}Register",
            services_map,
            work_dir,
        )
        _render_icns(icon_png, resources / "AppIcon.icns", work_dir)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if sign_identity:
        sign_bundle(app, sign_identity)

    return {
        "app": str(app),
        "bundle_id": spec.bundle_id,
        "services": dict(sorted(services_map.items())),
        "user": spec.user,
        "launcher": spec.launcher_path,
        "signed": bool(sign_identity),
        "register": str(macos_dir / f"{spec.app_name}Register"),
    }


# ---------------------------------------------------------------------------
# active manifests (what asiai-launch reads at boot)
# ---------------------------------------------------------------------------


def write_active_manifest(
    engine: str,
    *,
    preset: str | None = None,
    dest_dir: Path | None = None,
) -> dict:
    """Publish the manifest ``asiai-launch`` will exec ``engine`` from.

    Without ``preset``, reuses the preset recorded at install time (the
    install-record safety net) and falls back to the base manifest when no
    record exists. The file is written with safe permissions (0644) so the
    launcher's tamper gate accepts it.
    """
    from ais_core.launch import _active_manifest_dir

    record = install_state.read_install(engine)
    if preset is None and record is not None:
        preset = record.preset

    manifest = load_manifest(engine, preset=preset)
    if manifest.wrapper.needed:
        raise BundleError(
            f"{engine!r} is a wrapper-based engine; it cannot be activated "
            "for the SMAppService bundle"
        )
    src = manifest_source_path(engine, preset)
    if src is None:
        raise BundleError(f"no manifest source found for {engine!r} (preset={preset!r})")

    base = dest_dir if dest_dir is not None else _active_manifest_dir()
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"{engine}.toml"
    tmp = dest.with_suffix(".toml.tmp")
    tmp.write_bytes(src.read_bytes())
    tmp.chmod(0o644)
    tmp.replace(dest)
    return {"engine": engine, "preset": preset, "active_manifest": str(dest)}
