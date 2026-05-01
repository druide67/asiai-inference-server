"""Engine lifecycle: install / uninstall / start / stop / restart / status.

Ports the orchestration layer of ``lib-engine.sh`` (``engine_stop_existing``,
``engine_start_service``, ``engine_stop_service``) to Python.

Two design points worth flagging
--------------------------------

**Health check uses exponential backoff, not a flat 2s loop.** Bash polled the
HTTP endpoint every 2s; we poll at 0.5s, 1s, 2s, 4s, 8s capped at 8s. This is
~3x faster on healthy starts (Ollama responds in <1s) without burning the API
on slow ones like TurboQuant cold-starting a 70B model. A bare TCP connect
isn't enough — ``port UP ≠ API ready`` (Copilot Q3) — so we hit the manifest's
declared ``health_endpoint`` and require a 2xx.

**Idempotent install.** Running ``install`` twice in a row must not double the
LaunchDaemon or break pf.conf with a duplicate ``anchor`` line. The plist is
overwritten in place; the anchor line is added only if absent;
``launchctl load -w`` returns success on already-loaded daemons.
"""

from __future__ import annotations

import enum
import subprocess
import time
import urllib.error
import urllib.request

from ais_core import firewall, plist
from ais_core.manifest import EngineManifest


class LifecycleError(RuntimeError):
    """Raised when a lifecycle operation fails irrecoverably."""


class EngineState(enum.StrEnum):
    NOT_INSTALLED = "not_installed"     # no plist file
    STOPPED = "stopped"                  # plist exists, daemon not loaded
    LOADED_NOT_RUNNING = "loaded"        # launchctl knows it, no PID
    UNHEALTHY = "unhealthy"              # PID alive but health endpoint silent
    RUNNING = "running"                  # PID alive AND health endpoint 2xx


# ---------------------------------------------------------------------------
# install / uninstall
# ---------------------------------------------------------------------------


def install(
    manifest: EngineManifest,
    *,
    user: str,
    binary_path: str | None = None,
    enable_firewall: bool = False,
    subnets: tuple[str, ...] = firewall.DEFAULT_SUBNETS,
    dry_run: bool = False,
) -> dict:
    """Install an engine end-to-end.

    Steps:
      1. Stop any pre-existing daemon/process for this engine (idempotent).
      2. Resolve the binary if not supplied (manifest's first existing candidate).
      3. Write the plist atomically.
      4. Optionally write the pf anchor and reload pf.
      5. Load and start the daemon, wait for the health endpoint.

    Returns a dict with the resolved binary, plist path, anchor path (or None),
    and whether the health endpoint answered within the manifest's timeout.
    """
    if binary_path is None:
        resolved = manifest.binary.resolve()
        if resolved is None:
            raise LifecycleError(
                f"{manifest.name}: no binary found in candidates "
                f"{list(manifest.binary.candidates)}; install it or pass --binary"
            )
        binary_path = resolved

    stop_existing(manifest, dry_run=dry_run)

    plist_path_str = plist.write_plist(
        manifest, user=user, binary_path=binary_path, dry_run=dry_run
    )

    anchor_path_str: str | None = None
    if enable_firewall:
        anchor_path_str = firewall.install_anchor(
            manifest, subnets=subnets, dry_run=dry_run
        )

    if not dry_run:
        start(manifest)

    health_ok = (
        not dry_run
        and wait_for_health(manifest, timeout=manifest.network.health_timeout)
    )

    return {
        "engine": manifest.name,
        "binary": binary_path,
        "plist": plist_path_str,
        "anchor": anchor_path_str,
        "health_ok": health_ok if not dry_run else None,
        "dry_run": dry_run,
    }


def uninstall(
    manifest: EngineManifest,
    *,
    keep_logs: bool = True,
    dry_run: bool = False,
) -> dict:
    """Tear down an engine: stop daemon, remove plist, remove pf anchor."""
    stop(manifest, dry_run=dry_run)
    plist_removed = plist.remove_plist(manifest, dry_run=dry_run)
    fw_changed = (
        firewall.remove_anchor(manifest, dry_run=dry_run)
        if manifest.firewall.supported
        else False
    )

    return {
        "engine": manifest.name,
        "plist_removed": plist_removed,
        "firewall_removed": fw_changed,
        "logs_kept": keep_logs,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# start / stop / restart
# ---------------------------------------------------------------------------


def start(manifest: EngineManifest) -> None:
    """Load the LaunchDaemon. ``-w`` overrides any prior 'disabled' state."""
    subprocess.run(
        ["sudo", "/bin/launchctl", "load", "-w", plist.plist_path(manifest)],
        check=True,
    )


def stop(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Stop the daemon, unload it, kill any straggler process."""
    if dry_run:
        print(f"[dry-run] would stop {manifest.plist.name}")
        return

    subprocess.run(
        ["sudo", "/bin/launchctl", "stop", manifest.plist.name],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["sudo", "/bin/launchctl", "unload", "-w", plist.plist_path(manifest)],
        check=False,
        capture_output=True,
    )
    if process_alive(manifest):
        subprocess.run(
            ["pkill", "-f", manifest.binary.process_pattern],
            check=False,
            capture_output=True,
        )
        time.sleep(2)


def restart(manifest: EngineManifest) -> None:
    """Stop then start. Health check is the caller's responsibility."""
    stop(manifest)
    start(manifest)


def stop_existing(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Best-effort cleanup before a fresh install.

    Mirrors ``engine_stop_existing()`` (lib-engine.sh:301-330): unload the
    plist if it's there and kill the process if it's there. Both arms are
    independent so a missing plist doesn't prevent a process kill.
    """
    if dry_run:
        print(f"[dry-run] would stop existing {manifest.name} services")
        return

    if subprocess.run(
        ["test", "-f", plist.plist_path(manifest)], check=False
    ).returncode == 0:
        subprocess.run(
            ["sudo", "/bin/launchctl", "unload", plist.plist_path(manifest)],
            check=False,
            capture_output=True,
        )
        time.sleep(2)

    if process_alive(manifest):
        subprocess.run(
            ["pkill", "-f", manifest.binary.process_pattern],
            check=False,
            capture_output=True,
        )
        time.sleep(2)


# ---------------------------------------------------------------------------
# health / status
# ---------------------------------------------------------------------------


def wait_for_health(
    manifest: EngineManifest,
    *,
    timeout: int | None = None,
    initial_delay: float = 0.5,
    cap_delay: float = 8.0,
) -> bool:
    """Poll the manifest's health endpoint with exponential backoff.

    Returns True as soon as a 2xx response comes back. Returns False if
    ``timeout`` seconds elapse without success.

    The poll interval doubles each round (0.5, 1, 2, 4, 8, 8, 8, …) and is
    capped at ``cap_delay``. The total elapsed time is checked before each
    sleep so we never overshoot the deadline.
    """
    timeout = timeout if timeout is not None else manifest.network.health_timeout
    deadline = time.monotonic() + timeout
    delay = initial_delay

    while time.monotonic() < deadline:
        if probe_health(manifest):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(delay, remaining))
        delay = min(delay * 2, cap_delay)
    return False


def probe_health(manifest: EngineManifest, *, request_timeout: float = 2.0) -> bool:
    """Single non-throwing health probe. True iff the endpoint returns 2xx."""
    url = manifest.network.health_url
    try:
        with urllib.request.urlopen(url, timeout=request_timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def process_alive(manifest: EngineManifest) -> bool:
    """True iff a process matching the manifest's pattern is running."""
    proc = subprocess.run(
        ["pgrep", "-f", manifest.binary.process_pattern],
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def is_loaded(manifest: EngineManifest) -> bool:
    """True iff launchctl knows about this label (loaded, even if not running)."""
    proc = subprocess.run(
        ["/bin/launchctl", "list", manifest.plist.name],
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def current_state(manifest: EngineManifest) -> EngineState:
    """Best-effort state machine answer.

    The order matters: we ask the cheap questions first and only fall through
    to the network probe if the daemon claims to be running.
    """
    from pathlib import Path

    if not Path(plist.plist_path(manifest)).exists():
        return EngineState.NOT_INSTALLED
    if not is_loaded(manifest):
        return EngineState.STOPPED
    if not process_alive(manifest):
        return EngineState.LOADED_NOT_RUNNING
    if not probe_health(manifest):
        return EngineState.UNHEALTHY
    return EngineState.RUNNING
