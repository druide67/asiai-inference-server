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
LaunchDaemon or break pf.conf with a duplicate ``anchor`` line. ``stop_existing``
boots out + unlinks any prior daemon first, so the helper's ``install-daemon``
bootstrap starts from a clean slate; the pf anchor line is added only if absent.

**Privileged ops route through the helper (AEP-01).** Daemon lifecycle no longer
shells out to raw ``sudo launchctl``/``sudo mv``: it calls the root-owned
``asiai-priv`` helper via :mod:`ais_core.privhelper` (the sudoers fragment grants
NOPASSWD on that one binary only). The helper uses the modern launchctl model
(bootstrap/bootout/kickstart/kill/enable/disable). Firewall (pf) stays raw
``sudo``, password-gated (see :mod:`ais_core.firewall`).
"""

from __future__ import annotations

import enum
import logging
import os
import plistlib
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ais_core import firewall, plist, privhelper
from ais_core.manifest import EngineManifest

logger = logging.getLogger(__name__)


class LifecycleError(RuntimeError):
    """Raised when a lifecycle operation fails irrecoverably."""


class EngineState(enum.StrEnum):
    NOT_INSTALLED = "not_installed"  # binary absent, nothing provisioned
    AVAILABLE = "available"  # binary present, service not provisioned (no plist)
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
      1. Resolve the binary if not supplied (manifest's first existing candidate).
      2. Boot out + unlink any pre-existing daemon/process (``stop_existing``, idempotent).
      3. ``asiai-priv install-daemon``: the helper generates the plist, writes it root:wheel,
         pre-creates the root-owned ``/Library/Logs/asiai`` leaves, and bootstraps it
         (RunAtLoad starts it) — so there is no separate plist write, log mkdir, or start here.
      4. Optionally write the pf anchor and reload pf (firewall, password-gated).
      5. Wait for the health endpoint.

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
    # Same fail-fast for the sudo password: if the pf write WILL be needed (anchor
    # missing or stale) and sudo cannot prompt (no TTY, no ticket), refuse now —
    # not after the daemon was stopped and the plist rewritten (2026-07-01 retex).
    if (
        enable_firewall
        and not dry_run
        and not firewall.anchor_up_to_date(manifest, subnets=subnets)
    ):
        firewall.preflight_sudo(f"{manifest.name}: installing the pf anchor")

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

    # The privileged helper GENERATES the plist (generate-don't-validate), writes it
    # root:wheel into /Library/LaunchDaemons, pre-creates the root-owned log leaves under
    # /Library/Logs/asiai, and bootstraps (RunAtLoad starts it). We therefore no longer
    # write the plist or mkdir a user log dir here, and no longer call start() afterwards
    # (bootstrap already started it). Standard*Path live at /Library/Logs/asiai/<label>.{out,err}.
    privhelper.run(
        "install-daemon",
        *_install_args(manifest, user=user, binary_path=binary_path),
        dry_run=dry_run,
    )

    anchor_path_str: str | None = None
    if enable_firewall:
        # Firewall stays raw-sudo (password-gated post-cutover): pf.conf editing is opt-in,
        # install-time, operator-present — deliberately NOT in the NOPASSWD helper surface.
        anchor_path_str = firewall.install_anchor(manifest, subnets=subnets, dry_run=dry_run)

    health_ok = not dry_run and wait_for_health(manifest, timeout=manifest.network.health_timeout)

    return {
        "engine": manifest.name,
        "binary": binary_path,
        "plist": plist.plist_path(manifest),
        "anchor": anchor_path_str,
        "health_ok": health_ok if not dry_run else None,
        "dry_run": dry_run,
    }


def _resolve_user_file(raw: str) -> str:
    """Resolve symlinks in an absolute model/template/mmproj path before the helper call.

    Audit finding #1: the helper refuses a symlink final component on these paths
    (anti swap-TOCTOU inside a user-writable home — a deliberate hardening we keep),
    which broke the ``active.gguf``/``active.jinja`` switch convention. Resolving the
    symlink client-side pins the plist to the real target of the moment; switching
    models = re-point the symlink + reinstall.

    Tilde paths pass through untouched: they expand against the DAEMON account's home
    inside the helper — realpath'ing here would wrongly expand against the INVOKER's.
    A non-existing path also passes through: the helper owns the existence refusal
    and its message.
    """
    if raw.startswith("~"):
        return raw
    return os.path.realpath(raw)


def _install_args(manifest: EngineManifest, *, user: str, binary_path: str) -> list[str]:
    """Translate a manifest into ``asiai-priv install-daemon`` flags.

    The helper is generate-don't-validate: we pass bounded fields, never a plist.

    **Every value is joined to its flag as ``--flag=value`` (not ``--flag value``).** This is
    load-bearing: the helper's argparse uses ``action="append"`` for ``--program-arg``/``--env``,
    which reads a *separate* token beginning with ``-`` as a new option and errors out — and
    every llama-server program-arg is dash-prefixed (``--flash-attn``, ``--mlock``,
    ``--n-gpu-layers``, ``--jinja``), as is the ``--host`` we inject. ``--flag=value`` parses the
    value verbatim regardless of a leading dash.

    ``--host``/``--port`` are emitted together **only when the manifest binds** — parity with the
    old ``plist.build_plist_dict`` (ollama has ``bind=''`` and sets its port via the
    ``OLLAMA_HOST`` env var; ``ollama serve --port N`` is a fatal unknown-flag).

    Absolute model/template/mmproj paths are realpath'd HERE (audit #1): the helper refuses a
    symlink final component (anti swap-TOCTOU — its target home is user-writable), which broke
    the ``active.gguf`` switch mechanism. Resolving client-side keeps both: the plist is pinned
    to the real target of the moment (re-switch = re-point the symlink + reinstall) and the
    helper's hardening stays intact. Tilde paths are still passed raw — the helper expands them
    against the DAEMON account's home (not ours) and confines them there.
    ``binary_path`` is the manifest's resolved candidate (the STABLE
    ``/opt/homebrew/bin/<engine>`` symlink for brew engines — survives ``brew upgrade``); the
    helper re-validates it (I5: realpath under an allowlisted prefix; the symlink is accepted).
    """
    label = manifest.plist.name
    program = manifest.wrapper.install_path if manifest.wrapper.needed else binary_path
    args = [f"--label={label}", f"--binary={program}", f"--user={user}"]
    if not manifest.wrapper.needed:
        for pa in manifest.binary.program_args:
            args.append(f"--program-arg={pa}")
        if manifest.binary.model_path:
            args.append(f"--model-path={_resolve_user_file(manifest.binary.model_path)}")
        if manifest.binary.template_path:
            args.append(f"--template-path={_resolve_user_file(manifest.binary.template_path)}")
        if manifest.binary.mmproj_path:
            args.append(f"--mmproj-path={_resolve_user_file(manifest.binary.mmproj_path)}")
        if manifest.network.bind:
            # --host/--port only when bound (parity): ollama (bind='') must NOT get --port.
            args.append("--program-arg=--host")
            args.append(f"--program-arg={manifest.network.bind}")
            args.append(f"--port={manifest.network.port}")
    for entry in manifest.env_vars:
        args.append(f"--env={entry}")
    args.append(f"--throttle-interval={manifest.plist.throttle_interval}")
    args.append(f"--timeout={manifest.plist.timeout}")
    return args


def uninstall(
    manifest: EngineManifest, *, keep_firewall: bool = False, dry_run: bool = False
) -> dict:
    """Tear down an engine: boot out the daemon, remove plist, remove pf anchor.

    ``keep_firewall=True`` leaves the anchor and pf.conf untouched — for a reinstall
    that will need the identical anchor right back (skipping the password-gated pf
    path entirely).

    The pf preflight runs BEFORE the daemon is touched: ``remove_anchor`` is raw
    password-gated sudo, and failing it mid-sequence leaves the engine half-removed
    (daemon gone, anchor behind — 2026-07-01 retex on a helper-only host without a TTY).

    Log files under /Library/Logs/asiai are left untouched. If a future user needs log
    purging on uninstall, surface it as a separate command rather than a flag on
    ``uninstall`` (irreversible side effects don't belong on a removal verb the operator
    may run by reflex).
    """
    fw_removal_needed = (
        manifest.firewall.supported and not keep_firewall and firewall.anchor_present(manifest)
    )
    if fw_removal_needed and not dry_run:
        firewall.preflight_sudo(f"{manifest.name}: removing the pf anchor")

    existed = Path(plist.plist_path(manifest)).exists()
    # The helper boots out the daemon AND unlinks the plist in one idempotent action (a
    # missing label/plist still exits 0). Replaces the old stop()+remove_plist() pair.
    privhelper.run("uninstall-daemon", "--label", manifest.plist.name, dry_run=dry_run)
    fw_changed = firewall.remove_anchor(manifest, dry_run=dry_run) if fw_removal_needed else False

    return {
        "engine": manifest.name,
        "plist_removed": existed,
        "firewall_removed": fw_changed,
        "firewall_kept": keep_firewall,
        "dry_run": dry_run,
    }


# ---------------------------------------------------------------------------
# start / stop / restart
# ---------------------------------------------------------------------------


def start(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Start an installed engine (modern launchctl model, via the helper).

    Clears any persistent disabled override first (``enable-daemon`` — parity with the legacy
    ``load -w`` that cleared disable), then ``start-daemon`` (``kickstart -k``). The plist must
    already exist + be bootstrapped (i.e. ``install`` ran); ``start`` does not (re)generate it.
    """
    label = manifest.plist.name
    privhelper.run("enable-daemon", "--label", label, dry_run=dry_run)
    privhelper.run("start-daemon", "--label", label, dry_run=dry_run)


def stop(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Stop the engine via ``stop-daemon`` (``launchctl kill SIGTERM``).

    The daemon stays bootstrapped (state becomes LOADED_NOT_RUNNING). With
    ``KeepAlive{SuccessfulExit:false}`` a clean (exit 0) shutdown stays down, and SIGTERM is
    not a crash signal so ``KeepAlive{Crashed:true}`` does not respawn it either. Best-effort
    (``check=False``): ``kill`` on an already-stopped service returns non-zero — that's a no-op,
    not a failure. **No ``pkill``**: a SIGKILL on a still-bootstrapped KeepAlive daemon would
    read as a crash and be RESPAWNED by launchd (the legacy ``pkill`` only worked because
    ``unload`` had already deregistered the job).
    """
    label = manifest.plist.name
    privhelper.run("stop-daemon", "--label", label, check=False, dry_run=dry_run)


def restart(manifest: EngineManifest, *, dry_run: bool = False) -> None:
    """Restart the engine. ``start-daemon`` is ``kickstart -k`` — it kills the current instance
    and starts a fresh one atomically — so a restart is just ``start`` (enable + kickstart).
    Health check is the caller's responsibility."""
    start(manifest, dry_run=dry_run)


def disable(manifest: EngineManifest, *, dry_run: bool = False) -> dict:
    """Durable cold standby: stay off across reboots AND stop now.

    ``disable-daemon`` writes the launchd disabled override (survives reboot) *before* the
    SIGTERM, so a KeepAlive daemon can't respawn in the gap. The plist (and the tuned preset)
    is left untouched — the middle state between ``stop`` (back at next boot via RunAtLoad) and
    ``uninstall`` (gone). ``aisctl start`` re-enables (``enable-daemon`` clears the override),
    an explicit operator action.
    """
    label = manifest.plist.name
    privhelper.run("disable-daemon", "--label", label, dry_run=dry_run)
    privhelper.run("stop-daemon", "--label", label, check=False, dry_run=dry_run)
    return {"engine": manifest.name, "disabled": True, "dry_run": dry_run}


def enable(manifest: EngineManifest, *, start_now: bool = False, dry_run: bool = False) -> dict:
    """Clear the launchd disabled override; optionally start right away.

    Without ``start_now`` the engine simply rejoins the boot sequence at the next reboot —
    that's the cold-standby contract: enabling is not the same decision as paying the
    model-load memory spike now.
    """
    label = manifest.plist.name
    privhelper.run("enable-daemon", "--label", label, dry_run=dry_run)
    if start_now:
        privhelper.run("start-daemon", "--label", label, dry_run=dry_run)
    return {"engine": manifest.name, "enabled": True, "started": start_now, "dry_run": dry_run}


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

    Boots out + unlinks any prior daemon under this label (idempotent ``uninstall-daemon``,
    ``check=False``) so the upcoming ``install-daemon`` bootstrap won't trip on an
    already-loaded label. Because the label is then **deregistered**, ``pkill``'ing a straggler
    is safe here — unlike in ``stop``, there is no KeepAlive job left to respawn it.
    """
    if dry_run:
        print(f"[dry-run] would stop existing {manifest.name} services")
        return

    privhelper.run("uninstall-daemon", "--label", manifest.plist.name, check=False)
    time.sleep(1)

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


# Key files already warned about this process — the perms warning fires once
# per file, not on every probe (monitors probe frequently).
_warned_key_files: set[str] = set()


def _auth_headers(manifest: EngineManifest, *, url: str) -> dict[str, str]:
    """Bearer header for engines that gate every endpoint behind an API key.

    Some engines (MTPLX) require an API key for any non-localhost bind and
    apply the auth middleware to ALL routes — including ``/health`` — so an
    unauthenticated probe would read 401 and misreport a serving engine as
    UNHEALTHY forever. When the manifest declares ``[network].api_key_file``,
    read the key at probe time and attach it.

    The key is only ever attached to loopback requests: probes run on the
    host itself, so a non-loopback target URL (strict-bind health probe, or
    a hostile manifest) gets no header — the key never leaves the host.

    Fail-soft: an absent/unreadable/empty key file yields no header — the
    probe then honestly reports whatever the engine answers (401 → not
    healthy), instead of masking a misconfiguration. The key is held only
    for the duration of the request and never logged.
    """
    key_file = manifest.network.api_key_file
    if not key_file:
        return {}
    if urllib.parse.urlsplit(url).hostname != "127.0.0.1":
        return {}
    path = Path(os.path.expanduser(key_file))
    try:
        if path.stat().st_mode & 0o077 and str(path) not in _warned_key_files:
            _warned_key_files.add(str(path))
            logger.warning(
                "api key file %s is readable by group/others — chmod 600 recommended", path
            )
        key = path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not key:
        return {}
    return {"Authorization": f"Bearer {key}"}


def probe_health(manifest: EngineManifest, *, request_timeout: float = 2.0) -> bool:
    """Single non-throwing health probe. True iff the endpoint returns 2xx."""
    url = manifest.network.health_url
    req = urllib.request.Request(url, headers=_auth_headers(manifest, url=url))
    try:
        with urllib.request.urlopen(req, timeout=request_timeout) as resp:
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
    url = manifest.network.gen_url
    req = urllib.request.Request(
        url,
        data=_GEN_PROBE_PAYLOAD,
        headers={"Content-Type": "application/json", **_auth_headers(manifest, url=url)},
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
    """True iff a process belonging to THIS manifest is running.

    Manifests can share a binary — every aux preset runs the same
    llama-server fork — so a bare pattern hit may belong to ANOTHER
    manifest's instance (the bug that pinned a stopped engine to
    UNHEALTHY for weeks: health silent + a *neighbour's* process alive).
    When the matched command lines carry ``--port``, the manifest's own
    port disambiguates; matches without a port token (ollama-style
    launchers, wrapper scripts) keep the historical pattern-only answer.
    """
    proc = subprocess.run(
        ["pgrep", "-fl", manifest.binary.process_pattern],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0:
        return False
    lines = [ln for ln in proc.stdout.decode(errors="replace").splitlines() if ln.strip()]
    if not lines:
        return False
    port_token = f"--port {manifest.network.port}"
    if any(port_token in ln for ln in lines):
        return True
    # False iff every match is pinned to some OTHER manifest's port;
    # port-less matches keep the historical pattern-only answer.
    return not all("--port " in ln for ln in lines)


def is_loaded(manifest: EngineManifest) -> bool:
    """True iff launchctl knows about this label (loaded, even if not running)."""
    proc = subprocess.run(
        ["/bin/launchctl", "list", manifest.plist.name],
        check=False,
        capture_output=True,
    )
    return proc.returncode == 0


def bundle_provisioned(manifest: EngineManifest) -> bool:
    """True iff an installed app bundle embeds a LaunchDaemon for this label.

    A bundle-managed service (SMAppService, story 2.6) keeps NO plist under
    ``/Library/LaunchDaemons`` and its label is UNLOADED while dormant —
    both provisioning traces the legacy checks rely on. The embedded plist
    inside the installed ``.app`` is the remaining evidence. The app path
    follows the build default; forks that renamed the app at build time
    point ``ASIAI_BUNDLE_APP`` at theirs.
    """
    app = Path(os.environ.get("ASIAI_BUNDLE_APP", "/Applications/Asiai.app"))
    embedded = app / "Contents" / "Library" / "LaunchDaemons" / f"{manifest.plist.name}.plist"
    return embedded.is_file()


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
    # Network FIRST: a serving engine is RUNNING no matter what the launchd
    # paperwork says. Bundle-managed daemons (SMAppService) are invisible to
    # a user-domain ``launchctl list``, desktop apps and hand-launched
    # servers have no plist at all — the old plist-first gate reported all
    # of them NOT_INSTALLED while they were answering requests.
    if probe_health(manifest):
        if deep:
            verdict = gen_probe(manifest)
            if verdict is GenVerdict.ZOMBIE:
                return EngineState.DEGRADED, verdict
            return EngineState.RUNNING, verdict
        return EngineState.RUNNING, None

    # Nothing provisioned for this label: no legacy plist, launchd has never
    # heard of it, and no installed app bundle embeds it (a DORMANT
    # bundle-managed label is unloaded — only its embedded plist remains as
    # provisioning evidence). Split by what's on disk: the software being
    # present without a provisioned service (ollama installed via brew but
    # never ``aisctl install``ed) is AVAILABLE, not "not installed" — an
    # operator reading the fleet must not be told to install something that
    # is already there.
    if (
        not Path(plist.plist_path(manifest)).exists()
        and not is_loaded(manifest)
        and not bundle_provisioned(manifest)
    ):
        if manifest.binary.resolve() is not None:
            return EngineState.AVAILABLE, None
        return EngineState.NOT_INSTALLED, None

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


def _model_display_name(path: str) -> str | None:
    """Filename an operator recognizes behind a model path.

    Preset-managed engines load a stable symlink (active.gguf); the
    operator needs the real filename behind it. Local resolve is
    legitimate: this code runs on the engine's own host.
    """
    try:
        if os.path.islink(path):
            path = os.path.realpath(path)
    except OSError:
        pass
    return os.path.basename(path) or None


def _active_manifest_model(manifest: EngineManifest) -> str | None:
    """Model the ACTIVE manifest (what ``asiai-launch`` would exec) declares.

    Bundle-managed services have no legacy plist to read argv from; their
    tuning lives in the active manifest published by ``aisctl bundle
    activate``. Best-effort: no active manifest, or one without a
    ``model_path`` (wrapper engines never get activated), yields None.
    """
    import tomllib

    from ais_core.launch import _active_manifest_dir

    active = _active_manifest_dir() / f"{manifest.name}.toml"
    try:
        with open(active, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    binary = data.get("binary")
    raw = binary.get("model_path") if isinstance(binary, dict) else None
    if not isinstance(raw, str) or not raw:
        return None
    return _model_display_name(os.path.expanduser(raw))


def installed_model(manifest: EngineManifest) -> str | None:
    """Basename of the model this provisioned engine would serve, or None.

    Answers "what would this engine serve if started" for a provisioned but
    non-running engine — the manifest itself carries no model (that is the
    preset's job). Sources, in order: the ``--model`` argument of the
    INSTALLED legacy plist, then the active manifest for bundle-managed
    services (no legacy plist on disk). Best-effort: unreadable or
    model-less sources (wrapper engines) simply yield None.
    """
    try:
        with open(plist.plist_path(manifest), "rb") as f:
            data = plistlib.load(f)
    except (OSError, plistlib.InvalidFileException):
        return _active_manifest_model(manifest)
    args = data.get("ProgramArguments")
    if not isinstance(args, list):
        return None
    for i, arg in enumerate(args):
        if arg == "--model" and i + 1 < len(args):
            return _model_display_name(str(args[i + 1]))
    return None
