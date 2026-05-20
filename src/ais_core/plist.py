"""LaunchDaemon plist generation.

Ports the heredoc-XML logic from ``lib-engine.sh:352-470`` to
``plistlib.dumps()`` (stdlib). Producing the dict is a pure function and is
tested in isolation; writing the file requires sudo and is wrapped in a
single thin function so the privileged surface stays small.

Why plistlib over a Jinja2 template
-----------------------------------
``plistlib.dumps()`` is stdlib, handles XML escaping (e.g. ampersands in env
values), enforces the right DOCTYPE, and refuses non-string keys by
construction. The Bash version had to escape arguments by hand and was prone
to breakage on values containing quotes or angle brackets.

Atomic write
------------
We render to a per-invocation 0700 staging directory under ``/tmp`` (see
:mod:`ais_core.io`) and ``mv`` the file under sudo to the final
``/Library/LaunchDaemons/`` path. The 0700 staging dir prevents a local
attacker from swapping the source for a symlink between the write and the
privileged move (TOCTOU). Atomic ``mv`` also avoids leaving a half-written
plist if the process is killed mid-write.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
from pathlib import Path

from ais_core.io import secure_staging_dir
from ais_core.manifest import EngineManifest, is_valid_plist_label

LAUNCH_DAEMONS_DIR = "/Library/LaunchDaemons"


class PlistError(RuntimeError):
    """Raised when plist generation or write fails for a non-recoverable reason."""


def plist_path(manifest: EngineManifest) -> str:
    """Absolute path of the LaunchDaemon plist for this engine."""
    return f"{LAUNCH_DAEMONS_DIR}/{manifest.plist.name}.plist"


def build_plist_dict(
    manifest: EngineManifest,
    *,
    user: str,
    binary_path: str,
) -> dict:
    """Return the plist as a Python dict ready for ``plistlib.dumps()``.

    Pure function, no I/O. ``user`` is the macOS user that will own the daemon
    (typically ``$USER``); ``binary_path`` is the resolved engine binary path
    (or the wrapper path if ``manifest.wrapper.needed``).

    Mirrors lib-engine.sh:352-470 with two functional differences:
      * The ``Comment`` field is preserved with the same wording.
      * Environment variables are emitted as a real dict, not as alternating
        ``<key>``/``<string>`` pairs concatenated by hand — which means
        plistlib will refuse duplicate keys (a silent footgun in the Bash
        version).
    """
    if not is_valid_plist_label(manifest.plist.name):
        raise PlistError(
            f"Refusing to render plist with label {manifest.plist.name!r}: "
            "labels must match com.asiai.<engine>"
        )

    program_path = manifest.wrapper.install_path if manifest.wrapper.needed else binary_path
    if program_path is None:
        raise PlistError(
            f"{manifest.name}: wrapper.needed but install_path is unset; "
            "manifest validation should have caught this"
        )

    program_args: list[str] = [program_path]
    if not manifest.wrapper.needed:
        # Order: manifest.binary.program_args (user-controlled flags) first,
        # then --model (single-model engines), then --host/--port (always last
        # so they can't be accidentally overridden by program_args). All
        # currently-supported engines use named flags only — no positional
        # subcommand. If we add an engine that takes a positional command
        # before its flags, this order needs revisiting.
        program_args.extend(manifest.binary.program_args)
        if manifest.binary.model_path:
            program_args.extend(["--model", os.path.expanduser(manifest.binary.model_path)])
        if manifest.binary.template_path:
            program_args.extend(
                ["--chat-template-file", os.path.expanduser(manifest.binary.template_path)]
            )
        if manifest.binary.mmproj_path:
            program_args.extend(["--mmproj", os.path.expanduser(manifest.binary.mmproj_path)])
        if manifest.network.bind:
            program_args.extend(
                [
                    "--host",
                    manifest.network.bind,
                    "--port",
                    str(manifest.network.port),
                ]
            )

    user_home = os.path.expanduser(f"~{user}")
    binary_dir = str(Path(binary_path).parent) if binary_path else "/usr/local/bin"
    path_value = ":".join([binary_dir, "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"])

    env: dict[str, str] = {"HOME": user_home, "PATH": path_value}
    for entry in manifest.env_vars:
        key, _, value = entry.partition("=")
        if key in env and key not in ("HOME", "PATH"):
            raise PlistError(f"{manifest.name}: duplicate env var {key!r} in manifest")
        env[key] = value

    return {
        "Label": manifest.plist.name,
        "Comment": (f"{manifest.display} LLM Server — managed by asiai-inference-server"),
        "ProgramArguments": program_args,
        "UserName": user,
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": {"Crashed": True, "SuccessfulExit": False},
        "ThrottleInterval": manifest.plist.throttle_interval,
        "StandardOutPath": manifest.logs.stdout_path,
        "StandardErrorPath": manifest.logs.stderr_path,
        "TimeOut": manifest.plist.timeout,
        "Nice": 0,
    }


def render_plist_xml(
    manifest: EngineManifest,
    *,
    user: str,
    binary_path: str,
) -> bytes:
    """Render a plist dict to its XML representation (UTF-8 bytes)."""
    dct = build_plist_dict(manifest, user=user, binary_path=binary_path)
    return plistlib.dumps(dct, fmt=plistlib.FMT_XML, sort_keys=False)


def write_plist(
    manifest: EngineManifest,
    *,
    user: str,
    binary_path: str,
    dry_run: bool = False,
) -> str:
    """Write the plist atomically under ``/Library/LaunchDaemons/``.

    Returns the destination path. In ``dry_run=True`` mode, the rendered XML
    is printed and no privileged operations run.

    The atomic sequence is: render in a private temp file, ``sudo /bin/mv`` to
    final destination, then ``sudo /usr/sbin/chown root:wheel`` and
    ``sudo /bin/chmod 644``. The temp file is created under ``/tmp`` so
    SIP-protected ``/Library/LaunchDaemons`` is never touched without sudo.
    """
    xml_bytes = render_plist_xml(manifest, user=user, binary_path=binary_path)
    dst = plist_path(manifest)

    if dry_run:
        print(f"--- plist {dst} ---")
        print(xml_bytes.decode("utf-8"))
        print("--- end plist ---")
        return dst

    with secure_staging_dir() as staging:
        tmp_path = staging / f"{manifest.plist.name}.plist"
        tmp_path.write_bytes(xml_bytes)
        tmp_path.chmod(0o644)
        try:
            subprocess.run(["sudo", "/bin/mv", str(tmp_path), dst], check=True)
            subprocess.run(["sudo", "/usr/sbin/chown", "root:wheel", dst], check=True)
            subprocess.run(["sudo", "/bin/chmod", "644", dst], check=True)
        except subprocess.CalledProcessError as e:
            raise PlistError(
                f"Failed to install plist at {dst}: {e.cmd} returned {e.returncode}"
            ) from e
    return dst


def remove_plist(manifest: EngineManifest, *, dry_run: bool = False) -> bool:
    """Delete the plist file under sudo. Returns True if removed, False if absent."""
    dst = plist_path(manifest)
    if not Path(dst).exists():
        return False

    if dry_run:
        print(f"[dry-run] sudo /bin/rm -f {dst}")
        return True

    subprocess.run(["sudo", "/bin/rm", "-f", dst], check=True)
    return True
