"""LaunchDaemon plist generation (pure) + path helper.

Ports the heredoc-XML logic from ``lib-engine.sh:352-470`` to ``plistlib.dumps()`` (stdlib).
Producing the dict is a pure function, tested in isolation.

After AEP-01 (privileged helper) the **daemon plist is generated AND written by the root
helper** (``asiai-priv install-daemon``, generate-don't-validate) — there is no longer a
raw-``sudo`` write path in this module. ``build_plist_dict`` / ``render_plist_xml`` are kept
as the pure, tested **reference shape** (dry-run preview) — close to, but NOT byte-for-byte,
the helper's output: the helper omits the cosmetic ``Comment`` key and an explicit ``Nice``
(0 is launchd's default) and puts Standard*Path under ``/Library/Logs/asiai/<label>.{out,err}``
rather than the user home. ``plist_path`` is the canonical ``/Library/LaunchDaemons/<label>.plist``
locator used across the package. The helper, not this module, owns ``ProgramArguments[0]``
resolution and the live plist.

Why plistlib over a Jinja2 template
-----------------------------------
``plistlib.dumps()`` is stdlib, handles XML escaping (e.g. ampersands in env values), enforces
the right DOCTYPE, and refuses non-string keys by construction.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

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

    # Resolve ~ against the DAEMON's user, not whoever ran aisctl: the plist
    # runs as ``user`` (UserName=), so ``~/llms/gguf/...`` must point at that
    # account's home. os.path.expanduser keys on the caller's $HOME/passwd.
    user_home = os.path.expanduser(f"~{user}")
    if user_home == f"~{user}":
        raise PlistError(f"cannot resolve home directory for user {user!r}")

    def _for_user(path: str) -> str:
        if path == "~":
            return user_home
        if path.startswith("~/"):
            return f"{user_home}/{path[2:]}"
        return path

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
            program_args.extend(["--model", _for_user(manifest.binary.model_path)])
        if manifest.binary.template_path:
            program_args.extend(["--chat-template-file", _for_user(manifest.binary.template_path)])
        if manifest.binary.mmproj_path:
            program_args.extend(["--mmproj", _for_user(manifest.binary.mmproj_path)])
        if manifest.binary.api_key_file:
            # Path only — the key value lives in the (operator-created, 600)
            # file and must never reach the plist, which is world-readable.
            program_args.extend(["--api-key-file", _for_user(manifest.binary.api_key_file)])
        if manifest.network.bind:
            program_args.extend(
                [
                    "--host",
                    manifest.network.bind,
                    "--port",
                    str(manifest.network.port),
                ]
            )

    binary_dir = str(Path(binary_path).parent) if binary_path else "/usr/local/bin"
    # ``~/.local/bin`` is where ``uv tool install`` lands binaries by
    # default — put it ahead of system paths so a daemon launched by
    # launchd finds the user-installed CLI before any older system
    # variant (homebrew, /usr/local).
    user_local_bin = f"{user_home}/.local/bin"
    path_value = ":".join(
        [binary_dir, user_local_bin, "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"]
    )

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
        "StandardOutPath": f"{_for_user(manifest.logs.dir.rstrip('/'))}/{manifest.logs.stdout}",
        "StandardErrorPath": f"{_for_user(manifest.logs.dir.rstrip('/'))}/{manifest.logs.stderr}",
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


# NOTE: the privileged write/remove (the old raw-``sudo`` ``mv``/``chown``/``chmod``/``rm``
# path) was removed in AEP-01. The root helper (``asiai-priv install-daemon`` /
# ``uninstall-daemon``) now generates + writes + removes the daemon plist; ``ais_core``
# invokes it via :mod:`ais_core.privhelper`. This module is pure generation + ``plist_path``.
