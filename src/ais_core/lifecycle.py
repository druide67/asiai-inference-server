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
from pathlib import Path

from ais_core import firewall, plist
from ais_core.manifest import EngineManifest


class LifecycleError(RuntimeError):
    """Raised when a lifecycle operation fails irrecoverably."""


class EngineState(enum.StrEnum):
    NOT_INSTALLED = "not_installed"  # no plist file
    STOPPED = "stopped"  # plist exists, daemon not loaded
    DISABLED = "disabled"  # plist exists, durably off (survives reboot)
    LOADED_NOT_RUNNING = "loaded"  # launchctl knows it, no PID
    UNHEALTHY = "unhealthy"  # PID alive but health endpoint silent
    RUNNING = "running"  # PID alive AND health endpoint 2xx
    DEGRADED = "degraded"  # health 2xx BUT generation fails (GPU backend zombie)


class GenVerdict(enum.StrEnum):
    """Outcome of a 1-token generation probe (``gen_probe``).

    Only ZOMBIE is alarming. BUSY (request timed out — slots saturated by a
    long prefill) and UNSUPPORTED (engine has no OpenAI-compatible chat
    endpoint, or requires fields we don't send) are normal conditions and
    must never trigger a restart.
    """

    OK = "ok"
    ZOMBIE = "zombie"  # HTTP answer carrying "Compute error" — Metal backend dead
    BUSY = "busy"  # request timeout: instance alive but slots occupied
    DOWN = "down"  # connection refused / unreachable
    UNSUPPORTED = "unsupported"  # 4xx without compute error (no compat endpoint)
    ERROR = "error"  # anything else (5xx without the known marker, bad JSON…)


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
      3. Ensure the user-space log directory exists.
      4. Write the plist atomically.
      5. Optionally write the pf anchor and reload pf.
      6. Load and start the daemon, wait for the health endpoint.

    Returns a dict with the resolved binary, plist path, anchor path (or None),
    and whether the health endpoint answered within the manifest's timeout.

    Note on dry-run vs binary resolution: in dry-run we skip the binary
    existence check (the user may want to preview the install on a host
    where the engine isn't installed yet). A placeholder path is recorded
    in the dict so the caller still sees the resolved candidate slot.
    """
    # Check every precondition before touching the system: install_anchor
    # would reject an unsupported firewall anyway, but only after the
    # existing daemon was stopped and the plist rewritten.
    if enable_firewall and not manifest.firewall.supported:
        raise firewall.FirewallError(
            f"{manifest.name}: firewall.supported=false — refusing to install anchor"
        )

    if binary_path is None:
        resolved = manifest.binary.resolve()
        if resolved is None:
            if dry_run:
                # Dry-run shouldn't block on missing binary — show the first
                # candidate as a placeholder so the user sees what *would* be used.
                binary_path = (
                    manifest.binary.candidates[0]
                    if manifest.binary.candidates
                    else "(no candidates)"
                )
            else:
                raise LifecycleError(
                    f"{manifest.name}: no binary found in candidates "
                    f"{list(manifest.binary.candidates)}; install it or pass --binary"
                )
        else:
            binary_path = resolved

    stop_existing(manifest, dry_run=dry_run)

    # Ensure user-space log directory exists before launchctl tries to open
    # StandardOut/ErrorPath. Using ~/Library/Logs/asiai/<engine>/ avoids the
    # need for sudo (vs the BSD legacy /var/log/<engine>/).
    if not dry_run:
        Path(manifest.logs.expanded_dir).mkdir(parents=True, exist_ok=True)

    plist_path_str = plist.write_plist(
        manifest, user=user, binary_path=binary_path, dry_run=dry_run
    )

    anchor_path_str: str | None = None
    if enable_firewall:
        anchor_path_str = firewall.install_anchor(manifest, subnets=subnets, dry_run=dry_run)

    if not dry_run:
        start(manifest)

    health_ok = not dry_run and wait_for_health(manifest, timeout=manifest.network.health_timeout)

    return {
        "engine": manifest.name,
        "binary": binary_path,
        "plist": plist_path_str,
        "anchor": anchor_path_str,
        "health_ok": health_ok if not dry_run else None,
        "dry_run": dry_run,
    }


def uninstall(manifest: EngineManifest, *, dry_run: bool = False) -> dict:
    """Tear down an engine: stop daemon, remove plist, remove pf anchor.

    Log files in ``manifest.logs.dir`` are left untouched. If a future user
    needs log purging on uninstall, surface it as a separate command rather
    than a flag on ``uninstall`` (irreversible side effects don't belong on
    a removal verb the operator may run by reflex).
    """
    stop(manifest, dry_run=dry_run)
    plist_removed = plist.remove_plist(manifest, dry_run=dry_run)
    fw_changed = (
        firewall.remove_anchor(manifest, dry_run=dry_run) if manifest.firewall.supported else False
    )

    return {
        "engine": manifest.name,
        "plist_removed": plist_removed,
        "firewall_removed": fw_changed,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# start / stop / restart
# ---------------------------------------------------------------------------


def start(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Load the LaunchDaemon. ``-w`` overrides any prior 'disabled' state."""
    if dry_run:
        print(f"[dry-run] would start {manifest.plist.name}")
        return
    try:
        subprocess.run(
            ["sudo", "/bin/launchctl", "load", "-w", plist.plist_path(manifest)],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise LifecycleError(f"{manifest.name}: launchctl load failed: {e}") from e


def stop(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Stop the daemon, unload it, kill any straggler process.

    The unload deliberately has no ``-w``: that flag writes the durable
    disabled override, which is ``disable()``'s job. ``stop`` is the
    temporary state — the engine rejoins the boot sequence via RunAtLoad.
    """
    if dry_run:
        print(f"[dry-run] would stop {manifest.plist.name}")
        return

    subprocess.run(
        ["sudo", "/bin/launchctl", "stop", manifest.plist.name],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["sudo", "/bin/launchctl", "unload", plist.plist_path(manifest)],
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


def restart(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Stop then start. Health check is the caller's responsibility."""
    if dry_run:
        print(f"[dry-run] would restart {manifest.plist.name}")
        return
    stop(manifest)
    start(manifest)


def disable(manifest: EngineManifest, *, dry_run: bool = False) -> dict:
    """Durable cold standby: stop now AND stay off across reboots.

    Writes the launchd disabled override for the label, then stops the
    daemon. The plist (and therefore the tuned preset configuration) is
    left untouched — this is the middle state between ``stop`` (back at
    next boot via RunAtLoad) and ``uninstall`` (gone entirely).

    The override is written *before* stopping so a KeepAlive daemon can't
    respawn in the gap. ``aisctl start`` re-enables (``load -w`` clears
    the override) — starting a disabled engine is an explicit operator
    action, not an accident.
    """
    if dry_run:
        print(f"[dry-run] would disable {manifest.plist.name}")
        return {"engine": manifest.name, "disabled": True, "dry_run": True}

    try:
        subprocess.run(
            ["sudo", "/bin/launchctl", "disable", f"system/{manifest.plist.name}"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise LifecycleError(f"{manifest.name}: launchctl disable failed: {e}") from e
    stop(manifest)
    return {"engine": manifest.name, "disabled": True, "dry_run": False}


def enable(manifest: EngineManifest, *, start_now: bool = False, dry_run: bool = False) -> dict:
    """Clear the launchd disabled override; optionally start right away.

    Without ``start_now`` the engine simply rejoins the boot sequence at
    the next reboot — that's the cold-standby contract: enabling is not
    the same decision as paying the model-load memory spike now.
    """
    if dry_run:
        return {
            "engine": manifest.name,
            "enabled": True,
            "started": start_now,
            "dry_run": True,
        }

    try:
        subprocess.run(
            ["sudo", "/bin/launchctl", "enable", f"system/{manifest.plist.name}"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise LifecycleError(f"{manifest.name}: launchctl enable failed: {e}") from e
    if start_now:
        start(manifest)
    return {"engine": manifest.name, "enabled": True, "started": start_now, "dry_run": False}


def is_disabled(manifest: EngineManifest) -> bool:
    """True iff the label carries a launchd disabled override.

    Reads ``launchctl print-disabled system`` (no sudo needed). Output
    lines look like ``"com.asiai.llamacpp-aux-4" => disabled`` (older
    launchd prints ``=> true``). Any failure degrades to False — the
    state machine then reports STOPPED, which is the safe default.
    """
    proc = subprocess.run(
        ["/bin/launchctl", "print-disabled", "system"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False
    needle = f'"{manifest.plist.name}"'
    for line in proc.stdout.splitlines():
        if needle in line:
            return "=> disabled" in line or "=> true" in line
    return False


def stop_existing(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Best-effort cleanup before a fresh install.

    Mirrors ``engine_stop_existing()`` (lib-engine.sh:301-330): unload the
    plist if it's there and kill the process if it's there. Both arms are
    independent so a missing plist doesn't prevent a process kill.
    """
    if dry_run:
        print(f"[dry-run] would stop existing {manifest.name} services")
        return

    if subprocess.run(["test", "-f", plist.plist_path(manifest)], check=False).returncode == 0:
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

    def _healthy() -> bool:
        if not probe_health(manifest):
            return False
        if not manifest.network.gen_check:
            return True
        # Opt-in ([network] gen_check = true): certify SERVING health, not
        # just HTTP health — a fresh start should answer a 1-token request.
        # The probe timeout is clamped to the remaining deadline budget so
        # a hung generation cannot overshoot the caller's timeout.
        remaining = deadline - time.monotonic()
        probe_timeout = min(10.0, max(0.5, remaining))
        return gen_probe(manifest, request_timeout=probe_timeout) is GenVerdict.OK

    while time.monotonic() < deadline:
        if _healthy():
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


_GEN_PROBE_PAYLOAD = (
    b'{"messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0}'
)
_COMPUTE_ERROR_MARKER = b"Compute error"


def gen_probe(manifest: EngineManifest, *, request_timeout: float = 45.0) -> GenVerdict:
    """1-token generation probe — the only health signal that survives a GPU OOM.

    On Apple Silicon, a Metal out-of-memory during compute leaves llama-server
    in a permanent backend-error state: the process stays up, ``/health``
    keeps answering 2xx, but every generation returns HTTP 500 "Compute
    error" until the process is restarted. This probe detects that zombie
    state by actually generating one token.

    Verdict mapping is deliberately conservative: only an HTTP *answer*
    carrying the "Compute error" marker is a ZOMBIE. A timeout means the
    instance is busy (e.g. a long prefill holds every slot) — restarting it
    would kill legitimate work.
    """
    req = urllib.request.Request(
        manifest.network.gen_url,
        data=_GEN_PROBE_PAYLOAD,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
            body = resp.read(65536)
            # Marker first: a 2xx body carrying both "choices" and the
            # compute-error marker (e.g. an error field next to an empty
            # choices array) means a broken backend, not a healthy one.
            if _COMPUTE_ERROR_MARKER in body:
                return GenVerdict.ZOMBIE
            if b'"choices"' in body:
                return GenVerdict.OK
            return GenVerdict.ERROR
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read(65536)
        except OSError:
            body = b""
        if _COMPUTE_ERROR_MARKER in body:
            return GenVerdict.ZOMBIE
        if 400 <= exc.code < 500:
            return GenVerdict.UNSUPPORTED
        return GenVerdict.ERROR
    except TimeoutError:
        return GenVerdict.BUSY
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        # urllib wraps socket timeouts in URLError(reason=TimeoutError).
        if isinstance(getattr(exc, "reason", None), TimeoutError):
            return GenVerdict.BUSY
        return GenVerdict.DOWN


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


def probe_state(
    manifest: EngineManifest, *, deep: bool = False
) -> tuple[EngineState, GenVerdict | None]:
    """Best-effort state machine answer, plus the generation verdict behind it.

    Order rationale: ``launchctl list`` for system daemons (in
    ``/Library/LaunchDaemons/``) requires sudo from a non-root caller — without
    it, ``is_loaded`` returns False even when the daemon is up and serving.
    To avoid that false negative, we trust the network probe first: if the
    health endpoint answers 2xx, the daemon is RUNNING regardless of what
    ``launchctl list`` says. Only when the daemon is silent do we drill down
    via ``process_alive``, ``is_loaded`` and ``is_disabled`` to distinguish
    UNHEALTHY / LOADED_NOT_RUNNING / DISABLED / STOPPED.

    With ``deep=True``, a RUNNING engine is additionally generation-probed —
    exactly once, and the verdict is returned alongside the state so callers
    that display it (``aisctl status --deep``) don't have to probe again.
    A confirmed "Compute error" answer (Metal backend zombie — /health lies
    after a GPU OOM) downgrades the state to DEGRADED. BUSY/UNSUPPORTED/ERROR
    keep RUNNING — only the unambiguous zombie marker may raise the alarm.
    The verdict is ``None`` when ``deep`` is False or the engine isn't RUNNING.
    """
    if not Path(plist.plist_path(manifest)).exists():
        return EngineState.NOT_INSTALLED, None

    if probe_health(manifest):
        if deep:
            verdict = gen_probe(manifest)
            if verdict is GenVerdict.ZOMBIE:
                return EngineState.DEGRADED, verdict
            return EngineState.RUNNING, verdict
        return EngineState.RUNNING, None

    if process_alive(manifest):
        return EngineState.UNHEALTHY, None

    if is_loaded(manifest):
        return EngineState.LOADED_NOT_RUNNING, None

    if is_disabled(manifest):
        return EngineState.DISABLED, None

    return EngineState.STOPPED, None


def current_state(manifest: EngineManifest, *, deep: bool = False) -> EngineState:
    """State-only convenience over :func:`probe_state` (same single probe)."""
    return probe_state(manifest, deep=deep)[0]
