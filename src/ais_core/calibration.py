"""Measured memory footprints per preset — the ground truth behind ``aisctl plan``.

Declared and computed memory figures (see :mod:`ais_core.plan`) are estimates;
this module records what the engine process ACTUALLY costs on this host, so
the estimator can prefer reality over paperwork.

Storage
-------
One JSON file per engine under the XDG state tree
(``~/.local/state/asiai-inference-server/calibration/<engine>.json``),
holding a ring buffer of at most :data:`RING_SIZE` samples per
``<preset>@<manifest_sha256>`` key. The sha in the key makes staleness
detection structural: edit the preset TOML (new ctx, new model symlink
target) and its samples simply stop matching — they are ignored, not
deleted, so reverting the edit revives them.

Sample sources (and why they are weighted differently)
------------------------------------------------------
* ``health`` — the engine process's resident footprint (``ps -o rss``)
  right after ``wait_for_health`` succeeded on start/install/reinstall.
  Reliable: the model is fully loaded (health gates on it) and the
  measurement is scoped to the one process. Weight 1.0.
* ``unload`` — the host-wide active+wired+compressed delta observed
  across an unload (the same accounting as
  :attr:`ais_core.memory.PurgeReport.freed_mb`). Noisy: other processes
  allocate and free during the window. Weight 0.5, and negative deltas
  are discarded at record time.

Reads return the weighted median of the fresh samples: robust to the odd
outlier sample without the complexity of trimming heuristics.

Everything here is best-effort: calibration is advisory bookkeeping, so a
failed measurement or an unwritable state dir must never fail the
lifecycle operation that triggered it (callers use :func:`record_sample`
which logs and swallows I/O errors).
"""

from __future__ import annotations

import json
import logging
import re
import socket
import subprocess
import time
from pathlib import Path

from ais_core.install_state import _state_dir
from ais_core.manifest import EngineManifest

logger = logging.getLogger(__name__)

RING_SIZE = 10

# Per-source weights for the weighted median (see module docstring).
SOURCE_WEIGHTS = {"health": 1.0, "unload": 0.5}

# Same charset as install_state records — keeps path traversal out of the
# state tree (the engine name becomes a filename).
_ENGINE_NAME_RE = re.compile(r"^[a-z0-9-]+$")


def _calibration_path(engine: str) -> Path:
    if not _ENGINE_NAME_RE.match(engine):
        raise ValueError(f"invalid engine name {engine!r}")
    return _state_dir() / "calibration" / f"{engine}.json"


def _host() -> str:
    return socket.gethostname()


def _key(preset: str, manifest_sha256: str) -> str:
    return f"{preset}@{manifest_sha256}"


def _read_rings(engine: str) -> dict[str, list[dict]]:
    """Load the per-key rings; corrupt or absent files degrade to empty."""
    try:
        raw = json.loads(_calibration_path(engine).read_text())
    except (OSError, ValueError):
        return {}
    samples = raw.get("samples") if isinstance(raw, dict) else None
    if not isinstance(samples, dict):
        return {}
    rings: dict[str, list[dict]] = {}
    for key, ring in samples.items():
        if isinstance(ring, list):
            rings[key] = [s for s in ring if isinstance(s, dict)]
    return rings


def _write_rings(engine: str, rings: dict[str, list[dict]]) -> None:
    path = _calibration_path(engine)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"samples": rings}, indent=2) + "\n")
    tmp.replace(path)


def record_sample(
    engine: str,
    *,
    preset: str,
    manifest_sha256: str,
    phys_footprint_mb: float,
    source: str,
) -> bool:
    """Append a measurement to the ring for ``preset x manifest_sha256``.

    Returns True if recorded. Best-effort by contract: any I/O problem is
    logged and swallowed (False) — calibration must never fail the
    lifecycle operation that produced the measurement. Non-positive
    values are discarded (an unload delta can go negative when other
    processes allocate during the window; a "freed -2 GB" sample would
    poison the median).
    """
    # Caller bugs (bad source, malformed engine name) raise loudly; only
    # environmental I/O failures are swallowed below.
    if source not in SOURCE_WEIGHTS:
        raise ValueError(f"unknown sample source {source!r}; known: {sorted(SOURCE_WEIGHTS)}")
    if not _ENGINE_NAME_RE.match(engine):
        raise ValueError(f"invalid engine name {engine!r}")
    if phys_footprint_mb <= 0:
        logger.debug(
            "calibration: discarding non-positive sample %.1f MB for %s (%s)",
            phys_footprint_mb,
            engine,
            source,
        )
        return False
    sample = {
        "ts": int(time.time()),
        "host": _host(),
        "phys_footprint_mb": round(float(phys_footprint_mb), 1),
        "source": source,
    }
    try:
        rings = _read_rings(engine)
        ring = rings.setdefault(_key(preset, manifest_sha256), [])
        ring.append(sample)
        del ring[:-RING_SIZE]
        _write_rings(engine, rings)
        return True
    except (OSError, ValueError) as e:
        logger.warning("calibration: could not record sample for %s: %s", engine, e)
        return False


def _weighted_median(samples: list[dict]) -> float | None:
    """Weighted median of ``phys_footprint_mb`` over the given samples."""
    pairs: list[tuple[float, float]] = []
    for s in samples:
        mb = s.get("phys_footprint_mb")
        if isinstance(mb, bool) or not isinstance(mb, (int, float)) or mb <= 0:
            continue
        weight = SOURCE_WEIGHTS.get(str(s.get("source")), 0.0)
        if weight > 0:
            pairs.append((float(mb), weight))
    if not pairs:
        return None
    pairs.sort(key=lambda p: p[0])
    half = sum(w for _, w in pairs) / 2.0
    acc = 0.0
    for mb, weight in pairs:
        acc += weight
        if acc >= half:
            return mb
    return pairs[-1][0]


def measured_footprint_mb(
    engine: str,
    *,
    preset: str,
    manifest_sha256: str,
    host: str | None = None,
) -> tuple[float, int] | None:
    """Weighted-median measured footprint for this exact preset revision.

    Returns ``(median_mb, sample_count)`` or None when no fresh sample
    exists. Staleness is structural, not temporal: samples keyed under a
    different manifest sha never match (the key encodes it), and samples
    taken on a different host are skipped — a 128 GB machine's footprint
    says nothing about this 64 GB one. Stale samples are ignored, never
    deleted.
    """
    rings = _read_rings(engine)  # malformed names/files degrade to empty
    want_host = host if host is not None else _host()
    fresh = [s for s in rings.get(_key(preset, manifest_sha256), []) if s.get("host") == want_host]
    median = _weighted_median(fresh)
    if median is None:
        return None
    return median, len(fresh)


# ---------------------------------------------------------------------------
# Measurement helpers (the hooks in ais_cli.commands call these)
# ---------------------------------------------------------------------------


def engine_rss_mb(manifest: EngineManifest) -> float | None:
    """Resident footprint (MB) of the running process for THIS manifest.

    Finds the process by the manifest's ``process_pattern`` and, when the
    matched command lines carry ``--port``, disambiguates by the manifest's
    own port (same doctrine as :func:`ais_core.lifecycle.process_alive`:
    aux presets share one binary). Sums ``ps -o rss`` over the matched
    PIDs (engines that fork workers count as one footprint). None when no
    process matches or ``ps`` output is unusable.
    """
    proc = subprocess.run(
        ["pgrep", "-fl", manifest.binary.process_pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return None
    port_token = f"--port {manifest.network.port}"
    pids: list[str] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, cmdline = line.partition(" ")
        if not pid.isdigit():
            continue
        if "--port " in cmdline and port_token not in cmdline:
            continue  # a sibling instance pinned to another port
        pids.append(pid)
    if not pids:
        return None
    ps = subprocess.run(
        ["ps", "-o", "rss=", "-p", ",".join(pids)],
        check=False,
        capture_output=True,
        text=True,
    )
    if ps.returncode != 0:
        return None
    total_kb = 0
    for line in ps.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            total_kb += int(line)
    if total_kb <= 0:
        return None
    return total_kb / 1024.0
