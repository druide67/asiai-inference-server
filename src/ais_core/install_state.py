"""Persisted install records: which preset (if any) an engine was installed with.

Why this exists
---------------
``aisctl install <engine> --preset <name>`` writes a plist generated from the
preset manifest, but nothing on the host records WHICH preset produced it. A
later ``uninstall`` + plain ``install`` silently regenerates the service from
the base manifest — different binary candidates, default context size, none of
the tuned args — and the engine comes back *healthy*. Monitoring sees a
running instance while the tuned configuration is gone.

The record deliberately outlives ``uninstall``: it means "the last tuned
configuration this host chose for that engine", not "currently installed".
That is what lets a plain ``install`` warn before degrading a previously
preset-based service, and what ``aisctl reinstall`` replays.

Storage: one JSON file per engine under the XDG *state* directory
(``~/.local/state/asiai-inference-server/installs/``) — this is tool
bookkeeping, not operator-edited configuration, so it lives apart from the
``~/.config`` tree where presets and manifests are written by humans. The
whole state tree can be relocated via ``ASIAI_USER_STATE_DIR`` (tests,
sandboxed installs), mirroring ``ASIAI_USER_CONFIG_DIR``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

# Same charset as the engine segment of com.asiai.<engine> plist labels.
_ENGINE_NAME_RE = re.compile(r"^[a-z0-9-]+$")


@dataclass(frozen=True)
class InstallRecord:
    engine: str
    # Preset the service was generated from; None for a base-manifest install.
    preset: str | None
    # SHA-256 of the manifest TOML at install time; lets reinstall flag drift.
    manifest_sha256: str
    # Firewall mode chosen at install time ("lan-only" | "none").
    firewall: str = "none"
    recorded_at: str = ""


def _state_dir() -> Path:
    """Resolve the XDG state directory for asiai-inference-server.

    Order of precedence mirrors :func:`ais_core.manifest._user_config_dir`:

    1. ``ASIAI_USER_STATE_DIR`` — full override of the state tree.
    2. ``XDG_STATE_HOME/asiai-inference-server`` if ``XDG_STATE_HOME`` is set.
    3. ``~/.local/state/asiai-inference-server`` (POSIX default).
    """
    override = os.environ.get("ASIAI_USER_STATE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_STATE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "asiai-inference-server"
    return Path("~/.local/state/asiai-inference-server").expanduser()


def _record_path(engine: str) -> Path:
    # Engine names follow the com.asiai.<engine> label charset; rejecting
    # anything else keeps path traversal out of the state tree.
    if not _ENGINE_NAME_RE.match(engine):
        raise ValueError(f"invalid engine name {engine!r}")
    return _state_dir() / "installs" / f"{engine}.json"


def manifest_digest(path: Path) -> str:
    """SHA-256 hex digest of a manifest TOML's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_install(
    engine: str,
    *,
    preset: str | None,
    manifest_path: Path,
    firewall: str = "none",
) -> InstallRecord:
    """Write (atomically) the record of what was just installed."""
    record = InstallRecord(
        engine=engine,
        preset=preset,
        manifest_sha256=manifest_digest(manifest_path),
        firewall=firewall,
        recorded_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )
    path = _record_path(engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(dataclasses.asdict(record), indent=2) + "\n")
    tmp.replace(path)
    return record


def read_install(engine: str) -> InstallRecord | None:
    """Read the record for ``engine``; None if absent or unreadable.

    A corrupt record degrades to None rather than raising: the record is a
    safety net, and a broken net must not block install/status themselves.
    """
    try:
        raw = json.loads(_record_path(engine).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    preset = raw.get("preset")
    return InstallRecord(
        engine=str(raw.get("engine", engine)),
        preset=str(preset) if preset is not None else None,
        manifest_sha256=str(raw.get("manifest_sha256", "")),
        firewall=str(raw.get("firewall", "none")),
        recorded_at=str(raw.get("recorded_at", "")),
    )


def clear_install(engine: str) -> bool:
    """Remove the record; True if one existed.

    NOT called by ``uninstall`` (see module docstring — the record must
    survive it). Exposed for operators and tests that want a clean slate.
    """
    try:
        _record_path(engine).unlink()
        return True
    except FileNotFoundError:
        return False
