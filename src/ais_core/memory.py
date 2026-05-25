"""Memory introspection, unload + purge pipeline, operations lock, repair mode.

This is the *killer building block* of asiai-inference-server. Without it,
benchmarking and engine switching on Apple Silicon are unreliable because the
unified-memory compressor doesn't release VRAM after a kill — the documented
pain that motivated the project.

Pipeline
--------
1. **Unload** loaded models per engine, preferring the engine's native API
   (Ollama ``keep_alive=0``, LM Studio ``lms unload``). Engines without an API
   (oMLX, TurboQuant, raw llama.cpp) fall back to a full LaunchDaemon restart
   via :mod:`ais_core.lifecycle`.
2. **purge** ``sudo /usr/sbin/purge`` flushes file system caches; on Apple
   Silicon it also nudges the unified-memory compressor.
3. **Verify** by comparing ``vm_stat`` snapshots before and after, and report
   the actual delta in MB. We refuse to make a marketing claim about freeing
   VRAM that isn't backed by the measurement.

Operations lock
---------------
:class:`OperationsLock` uses ``fcntl.flock`` on
``~/.local/share/asiai/operations.lock``. ``aisctl engine purge`` and any
``asiai bench`` worker grab the same lock, so a purge cannot interrupt a
running benchmark unless the user passes ``--force``. This is the
recommendation from Copilot Q5 (option A); a Unix domain socket dance was
considered overkill at this stage.

Repair mode
-----------
``--force-repair`` (Gemini Q5) cleans up residue from a crashed daemon:
stale operations lock files (PID dead), orphan
``/Library/LaunchDaemons/com.asiai.*.plist`` not registered in any known
manifest, and stuck pgrep matches that ``stop`` couldn't kill.
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import re
import subprocess
import time
from pathlib import Path

from ais_core import lifecycle, plist
from ais_core.manifest import EngineManifest, list_manifests, load_manifest

OPERATIONS_LOCK_PATH = Path.home() / ".local" / "share" / "asiai" / "operations.lock"
LAUNCH_DAEMONS_DIR = Path("/Library/LaunchDaemons")
PAGE_SIZE_DEFAULT = 16384  # Apple Silicon page size; verified with `pagesize` if available.


class MemoryError_(RuntimeError):
    """Raised when a memory operation fails irrecoverably (named to avoid shadowing builtin)."""


# ---------------------------------------------------------------------------
# vm_stat parsing
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class VmStat:
    page_size_bytes: int
    pages_free: int
    pages_active: int
    pages_inactive: int
    pages_wired: int
    pages_compressed: int  # "Pages occupied by compressor"

    @property
    def total_active_bytes(self) -> int:
        """Bytes currently held by active+wired+compressed pages.

        This is the metric we want to see drop after an unload+purge: it
        captures both the resident process working set and the compressor
        ballast.
        """
        return (self.pages_active + self.pages_wired + self.pages_compressed) * self.page_size_bytes

    @property
    def free_bytes(self) -> int:
        return self.pages_free * self.page_size_bytes


_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
# [^:\n] (not just [^:]) so the lazy match can't slurp across newlines and end
# up reporting "page size of 16384 bytes\nPages free" as the key.
_LINE_RE = re.compile(r"^([^:\n]+?)\s*:\s+(\d+)\.?\s*$", re.MULTILINE)


def vm_stat_parse(text: str | None = None) -> VmStat:
    """Parse the output of ``vm_stat`` (or a provided string in tests).

    The format on macOS 14/15 looks like::

        Mach Virtual Memory Statistics: (page size of 16384 bytes)
        Pages free:                               1234567.
        Pages active:                              987654.
        ...

    We are deliberately permissive: keys we don't care about are ignored, and
    a missing key falls back to 0 rather than raising — different macOS
    versions add and remove counters.
    """
    if text is None:
        proc = subprocess.run(["/usr/bin/vm_stat"], check=True, capture_output=True, text=True)
        text = proc.stdout

    page_size = PAGE_SIZE_DEFAULT
    m = _PAGE_SIZE_RE.search(text)
    if m:
        page_size = int(m.group(1))

    counters: dict[str, int] = {}
    for match in _LINE_RE.finditer(text):
        key = match.group(1).strip().lower()
        counters[key] = int(match.group(2))

    return VmStat(
        page_size_bytes=page_size,
        pages_free=counters.get("pages free", 0),
        pages_active=counters.get("pages active", 0),
        pages_inactive=counters.get("pages inactive", 0),
        pages_wired=counters.get("pages wired down", 0),
        pages_compressed=counters.get("pages occupied by compressor", 0),
    )


def memory_pressure() -> str:
    """Best-effort pressure level: ``"normal"`` / ``"warn"`` / ``"critical"`` / ``"unknown"``.

    macOS exposes this via the ``memory_pressure`` binary, whose output format
    is text-only and changes between releases. We grep for the well-known
    keywords and report ``unknown`` rather than crashing.
    """
    try:
        proc = subprocess.run(
            ["/usr/bin/memory_pressure"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"

    out = (proc.stdout + proc.stderr).lower()
    if "critical" in out:
        return "critical"
    if "warning" in out or "warn" in out:
        return "warn"
    if "normal" in out:
        return "normal"
    return "unknown"


# ---------------------------------------------------------------------------
# purge
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PurgeReport:
    before: VmStat
    after: VmStat
    pressure_after: str
    elapsed_s: float

    @property
    def freed_mb(self) -> int:
        """How many MB the active+wired+compressed footprint dropped.

        Negative values are possible if other processes allocated during the
        purge window. We return the signed integer; callers decide whether
        to report it or treat negatives as 0.
        """
        delta_bytes = self.before.total_active_bytes - self.after.total_active_bytes
        return delta_bytes // (1024 * 1024)

    @property
    def free_mb_delta(self) -> int:
        delta = self.after.free_bytes - self.before.free_bytes
        return delta // (1024 * 1024)


def purge_memory(*, dry_run: bool = False) -> PurgeReport:
    """Run ``sudo /usr/sbin/purge`` and report the measured delta.

    The before/after snapshots bracket the privileged call so a slow purge
    on a busy system doesn't skew the elapsed time we report.
    """
    before = vm_stat_parse()
    t0 = time.monotonic()

    if dry_run:
        print("[dry-run] sudo /usr/sbin/purge")
    else:
        subprocess.run(["sudo", "/usr/sbin/purge"], check=True, timeout=30)

    after = vm_stat_parse()
    return PurgeReport(
        before=before,
        after=after,
        pressure_after=memory_pressure(),
        elapsed_s=time.monotonic() - t0,
    )


# ---------------------------------------------------------------------------
# unload pipeline (engine-agnostic shell; per-engine drivers live in ais_engines)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class UnloadStep:
    engine: str
    method: str  # "api" | "restart" | "skipped" | "error"
    success: bool
    detail: str = ""


def unload_via_restart(manifest: EngineManifest) -> UnloadStep:
    """Engine-agnostic fallback: restart the LaunchDaemon, full process cycle.

    Always works (because the OS guarantees the process exits), but slower
    than an API unload and re-warms the model on the next request.
    """
    try:
        lifecycle.restart(manifest)
        return UnloadStep(
            engine=manifest.name,
            method="restart",
            success=True,
            detail=f"restarted {manifest.plist.name}",
        )
    except subprocess.CalledProcessError as e:
        return UnloadStep(
            engine=manifest.name,
            method="restart",
            success=False,
            detail=f"launchctl returned {e.returncode}",
        )


# ---------------------------------------------------------------------------
# OperationsLock — fcntl coordination with `asiai bench`
# ---------------------------------------------------------------------------


class OperationsLock:
    """Filesystem flock used to serialize destructive engine operations.

    Usage::

        with OperationsLock() as lock:
            ...

    If the lock is held by another process, ``acquire`` raises
    :class:`MemoryError_`. Pass ``force=True`` to skip locking entirely.
    """

    def __init__(self, *, path: Path = OPERATIONS_LOCK_PATH, force: bool = False):
        self.path = path
        self.force = force
        self._fd: int | None = None

    def __enter__(self) -> OperationsLock:
        if self.force:
            return self
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 0o600 — the file holds only a PID for repair() to consult, but the
        # principle of least privilege says no other local user needs to read it.
        self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(self._fd)
            self._fd = None
            holder = self._read_holder()
            raise MemoryError_(
                f"operations lock held by another process (PID {holder or '?'}); "
                "rerun with --force to bypass, or wait for it to release"
            ) from e
        # Stash our PID so a future repair pass can detect a stale lock.
        os.ftruncate(self._fd, 0)
        os.write(self._fd, f"{os.getpid()}\n".encode())

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def _read_holder(self) -> int | None:
        try:
            return int(self.path.read_text().strip())
        except (OSError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Repair mode
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RepairReport:
    stale_lock_cleared: bool
    orphan_plists: list[str]


def repair(*, dry_run: bool = False) -> RepairReport:
    """Clean up residue from a crashed daemon or interrupted operation.

    What we do:
      * If ``operations.lock`` exists but the recorded PID is dead, delete it
        so the next operation can acquire cleanly.
      * Find ``com.asiai.*.plist`` files that don't correspond to any known
        manifest and report them (we don't auto-delete because that would
        clobber a partially-installed engine the user may want to recover).
    """
    stale_cleared = False
    if OPERATIONS_LOCK_PATH.exists():
        holder_text = OPERATIONS_LOCK_PATH.read_text().strip()
        try:
            holder = int(holder_text)
            os.kill(holder, 0)  # signal 0 = "is the process alive?"
        except (ValueError, ProcessLookupError):
            if dry_run:
                print(f"[dry-run] would remove stale lock {OPERATIONS_LOCK_PATH}")
            else:
                OPERATIONS_LOCK_PATH.unlink()
            stale_cleared = True
        except PermissionError:
            # PID exists, just owned by another user — leave it alone.
            pass

    known_plist_names = set()
    for name in list_manifests():
        m = load_manifest(name)
        known_plist_names.add(Path(plist.plist_path(m)).name)

    orphans: list[str] = []
    if LAUNCH_DAEMONS_DIR.is_dir():
        for path in LAUNCH_DAEMONS_DIR.glob("com.asiai.*.plist"):
            if path.name not in known_plist_names:
                orphans.append(str(path))

    return RepairReport(
        stale_lock_cleared=stale_cleared,
        orphan_plists=orphans,
    )
