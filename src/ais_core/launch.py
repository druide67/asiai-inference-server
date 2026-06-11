"""asiai-launch — secure generic launcher for the SMAppService bundle.

The ``Asiai.app`` bundle ships **one** generic launcher. Each embedded
LaunchDaemon's ``ProgramArguments`` is ``[asiai-launch, <service>]``. At boot,
``asiai-launch`` resolves the *active manifest* for ``<service>`` (written by
``aisctl`` to a config dir), then ``exec()``s the engine binary.

Security model (confused-deputy guard)
--------------------------------------
A signed/trusted bundle must never ``exec`` whatever an *editable file* tells it
to — otherwise we have shipped a trusted deputy that runs arbitrary code. Two
guards, both mandatory (review condition):

1. **The executable path comes from a HARDCODED allowlist** (:data:`ALLOWED_BINARY_NAMES`),
   never from the manifest. The manifest may only *choose among* known engine
   binaries and supply **parameters** (model, ctx, port, flags).
2. **The active manifest file must be on safe permissions** — owned by root or
   the daemon user, and neither group- nor world-writable — or the launcher
   refuses to run.

Zero-dependency (``tomllib`` is stdlib ≥ 3.11), mirroring the rest of
:mod:`ais_core`.
"""

from __future__ import annotations

import os
import stat
import sys
import tomllib
from pathlib import Path

from ais_core.manifest import EngineManifest, _from_dict

# Binaries the launcher is allowed to exec, by basename. The manifest's
# ``binary.candidates`` may point only at one of these names — a tampered
# manifest pointing at ``/tmp/evil`` (basename ``evil``) is rejected. Adding a
# new engine binary is a deliberate, code-reviewed change here, not a config
# edit. (Python-backed aisrv-internal services — monitor, aisctl-serve — are
# launched by aisrv itself, not preset-driven, and are handled outside this
# allowlist.)
ALLOWED_BINARY_NAMES = frozenset(
    {
        "llama-server",
        "llama-server-turboquant",
        "ollama",
        "lms",
        "mlx_lm.server",
    }
)


class LaunchError(RuntimeError):
    """The service cannot be launched (bad manifest, missing/forbidden binary)."""


class LaunchSecurityError(LaunchError):
    """A guard refused to launch: unsafe manifest perms, or non-allowlisted binary."""


def _active_manifest_dir() -> Path:
    """Directory of per-service *active* manifests written by ``aisctl``.

    Overridable via ``ASIAI_LAUNCH_MANIFEST_DIR`` (tests, sandboxed installs).
    """
    override = os.environ.get("ASIAI_LAUNCH_MANIFEST_DIR")
    if override:
        return Path(override).expanduser()
    return Path("~/.local/share/asiai-inference-server/active").expanduser()


def assert_safe_manifest_perms(path: Path) -> None:
    """Refuse a manifest a non-privileged actor could have tampered with.

    Owner must be root (uid 0) or the current effective uid (the daemon user);
    the file must be a regular file and neither group- nor world-writable.
    """
    try:
        st = path.stat()
    except OSError as exc:
        raise LaunchSecurityError(f"cannot stat manifest {path}: {exc}") from exc
    if not stat.S_ISREG(st.st_mode):
        raise LaunchSecurityError(f"manifest {path} is not a regular file")
    if st.st_uid not in (0, os.geteuid()):
        raise LaunchSecurityError(
            f"manifest {path} owned by uid {st.st_uid}; must be root or the daemon user"
        )
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise LaunchSecurityError(
            f"manifest {path} is group/world-writable (mode {oct(st.st_mode & 0o777)}); "
            "refuse to exec a trusted daemon from an editable manifest"
        )


def _resolve_allowed_binary(manifest: EngineManifest) -> str:
    """First existing binary candidate, but only if its basename is allowlisted."""
    binary = manifest.binary.resolve()
    if binary is None:
        raise LaunchError(
            f"no binary candidate exists for {manifest.name!r}: {list(manifest.binary.candidates)}"
        )
    name = os.path.basename(binary)
    if name not in ALLOWED_BINARY_NAMES:
        raise LaunchSecurityError(
            f"binary {binary!r} (name {name!r}) is not in the launcher allowlist "
            f"{sorted(ALLOWED_BINARY_NAMES)}; a manifest may only select a known "
            "engine binary, never an arbitrary path"
        )
    return binary


def build_argv(manifest: EngineManifest) -> list[str]:
    """Build the exec argv from the manifest PARAMETERS.

    Mirrors :func:`ais_core.plist.build_plist_dict`'s ProgramArguments order
    (program_args → --model → --chat-template-file → --mmproj → --host/--port),
    minus the wrapper. The binary path comes from the allowlist, not the manifest.
    """
    if manifest.wrapper.needed:
        # Wrapper engines (e.g. turboquant: sysctl iogpu.wired_limit_mb then exec)
        # are started via the `aisctl boot` GPU prelude, not this generic launcher.
        raise LaunchError(
            f"{manifest.name!r}: wrapper-based engines are launched via the boot "
            "prelude, not asiai-launch"
        )
    binary = _resolve_allowed_binary(manifest)
    spec = manifest.binary
    argv = [binary, *spec.program_args]
    if spec.model_path:
        argv += ["--model", os.path.expanduser(spec.model_path)]
    if spec.template_path:
        argv += ["--chat-template-file", os.path.expanduser(spec.template_path)]
    if spec.mmproj_path:
        argv += ["--mmproj", os.path.expanduser(spec.mmproj_path)]
    if manifest.network.bind:
        argv += ["--host", manifest.network.bind, "--port", str(manifest.network.port)]
    return argv


def build_env(manifest: EngineManifest) -> dict[str, str]:
    """Process environment for the engine: inherited env + manifest [environment] vars.

    The embedded SMAppService plist is sealed inside a signed bundle, so it can
    only carry static env (HOME/PATH). Per-machine tuning env lives in the
    active manifest and is applied here at exec time — mirroring what
    :func:`ais_core.plist.build_plist_dict` does on the legacy plist path.
    """
    env = dict(os.environ)
    for entry in manifest.env_vars:
        key, _, value = entry.partition("=")
        env[key] = value
    return env


def load_active_manifest(path: Path) -> EngineManifest:
    """Perms-check then parse a per-service active manifest into an EngineManifest."""
    assert_safe_manifest_perms(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return _from_dict(raw, source=str(path))


def resolve_launch(
    service: str, *, manifest_dir: Path | None = None
) -> tuple[list[str], dict[str, str]]:
    """Full pipeline: validate service name → load active manifest → (argv, env)."""
    if not service or "/" in service or service.startswith("."):
        raise LaunchSecurityError(f"invalid service name {service!r}")
    base = manifest_dir if manifest_dir is not None else _active_manifest_dir()
    manifest = load_active_manifest(base / f"{service}.toml")
    return build_argv(manifest), build_env(manifest)


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``asiai-launch <service>`` → exec the engine (replaces process)."""
    argv = sys.argv if argv is None else argv
    if len(argv) != 2:
        print("usage: asiai-launch <service>", file=sys.stderr)
        return 2
    try:
        cmd, env = resolve_launch(argv[1])
    except LaunchError as exc:
        print(f"asiai-launch: {exc}", file=sys.stderr)
        return 1
    os.execve(cmd[0], cmd, env)  # on success this never returns
    return 127  # pragma: no cover — only reached if execve itself fails


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
