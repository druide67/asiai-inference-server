"""Command handlers for the ``aisctl`` CLI.

Each handler is a thin orchestrator over :mod:`ais_core` and
:mod:`ais_engines`: parse args, call the right module, format the result.
Handlers return an integer exit code. They print human-friendly text by
default and a JSON envelope when ``--json`` is set.

Why JSON output is a first-class flag
-------------------------------------
Agents (Claude, MCP clients) and shell pipelines parse JSON. The default
human format prints the same data with a couple of color hints; a CI script
or a monitoring agent should always pass ``--json`` and never depend on the
human format staying stable across versions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ais_core import bootstrap, calibration, firewall, install_state, lifecycle, memory, sudoers
from ais_core.manifest import (
    EngineManifest,
    ManifestError,
    list_manifests,
    list_presets,
    load_manifest,
    manifest_source_path,
    preset_search_dirs,
    preset_summary,
)
from ais_core.upgrade import upgrade_argv
from ais_engines.llamacpp import LlamaCppDriver
from ais_engines.llamacpp_aux import LlamaCppAuxDriver
from ais_engines.lmstudio import LMStudioDriver
from ais_engines.mlx_lm import MlxLmDriver
from ais_engines.mtplx import MtplxDriver
from ais_engines.ollama import OllamaDriver
from ais_engines.omlx import OmlxDriver
from ais_engines.rapidmlx import RapidMlxDriver
from ais_engines.turboquant import TurboquantDriver
from ais_engines.vmlx import VmlxDriver

logger = logging.getLogger(__name__)

# Static driver registrations — engines that exist as a single instance,
# one driver class each. Lookups for these names short-circuit ahead of
# the family-pattern dispatch below.
_STATIC_DRIVER_FACTORIES: dict[str, Callable[[], Any]] = {
    "ollama": OllamaDriver.from_manifest,
    "lmstudio": LMStudioDriver.from_manifest,
    "omlx": OmlxDriver.from_manifest,
    "turboquant": TurboquantDriver.from_manifest,
    "llamacpp": LlamaCppDriver.from_manifest,
    "vmlx": VmlxDriver.from_manifest,
    "mlx-lm": MlxLmDriver.from_manifest,
    "rapidmlx": RapidMlxDriver.from_manifest,
    "mtplx": MtplxDriver.from_manifest,
}

# Family-pattern registrations — names matching the regex resolve to the
# associated driver class, instantiated with the matched manifest name.
# This lets a new instance (e.g. ``llamacpp-aux-5``) be added by dropping
# a TOML on disk, with no Python code change. New families (e.g.
# ``mlx-lm-aux-N``) follow the same pattern: regex + class.
_FAMILY_PATTERNS: list[tuple[re.Pattern[str], type]] = [
    (re.compile(r"^llamacpp-aux-\d+$"), LlamaCppAuxDriver),
]


def get_driver_factory(name: str) -> Callable[[], Any]:
    """Return a zero-arg factory that builds the driver for ``name``.

    Looks first in the mutable :data:`DRIVER_FACTORIES` snapshot (the
    static registry plus any family instance discovered at module load,
    plus anything tests may have injected via ``patch.dict``). Falls back
    to the family-pattern dispatch for engine names whose manifest landed
    on disk after import time (user added a new ``llamacpp-aux-5.toml``
    in their XDG dir, hot deployment, etc.). Raises ``KeyError`` if
    neither path matches — the CLI handler converts that into a user-
    facing ``SystemExit``.
    """
    snapshot = DRIVER_FACTORIES.get(name)
    if snapshot is not None:
        return snapshot
    for pattern, driver_cls in _FAMILY_PATTERNS:
        if pattern.match(name):
            return lambda manifest=None, n=name, cls=driver_cls: (
                cls.from_manifest(manifest) if manifest is not None else cls.from_manifest_name(n)
            )
    raise KeyError(name)


def _known_engine_names() -> set[str]:
    """Names the CLI knows how to dispatch to, regardless of which TOMLs
    happen to be on disk right now.

    This is the union of (a) static registry names and (b) any manifest
    name on disk that matches a family pattern. It's the set used to
    answer "is this a known engine" at the CLI boundary.
    """
    names = set(_STATIC_DRIVER_FACTORIES)
    for n in list_manifests():
        for pattern, _ in _FAMILY_PATTERNS:
            if pattern.match(n):
                names.add(n)
                break
    return names


def _build_driver_factories() -> dict[str, Callable[[], Any]]:
    """Snapshot the dispatch table: static engines + discovered family instances.

    Family instances are discovered by scanning ``list_manifests()`` and
    matching against ``_FAMILY_PATTERNS``. Manifests added on disk after
    this module is imported are NOT in the returned dict — runtime
    dispatch for those goes through :func:`get_driver_factory` directly,
    which re-evaluates the family patterns on each call.

    The dict is mutable so existing test helpers (``unittest.mock.patch.dict``,
    insertion of fakes) keep working. New code should prefer
    :func:`get_driver_factory` to benefit from on-disk discovery.
    """
    factories: dict[str, Callable[[], Any]] = dict(_STATIC_DRIVER_FACTORIES)
    for n in list_manifests():
        if n in factories:
            continue
        for pattern, driver_cls in _FAMILY_PATTERNS:
            if pattern.match(n):
                # Same contract as the static factories: an optional manifest
                # positional. Binding it to a keyword-only-by-convention slot
                # used to swallow the manifest into the *name* parameter and
                # crash from_manifest_name with a FileNotFoundError.
                factories[n] = lambda manifest=None, name=n, cls=driver_cls: (
                    cls.from_manifest(manifest)
                    if manifest is not None
                    else cls.from_manifest_name(name)
                )
                break
    return factories


DRIVER_FACTORIES: dict[str, Callable[[], Any]] = _build_driver_factories()


def _emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        # allow_nan=False: NaN/Infinity are not valid JSON tokens; a producer
        # bug must crash HERE, never emit unparseable output to a consumer.
        print(json.dumps(_jsonify(payload), indent=2, default=str, allow_nan=False))
    else:
        print(payload)


def _jsonify(obj: Any) -> Any:
    """Recursively convert dataclasses/sets/Path-like to JSON-serializable forms."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonify(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonify(v) for v in obj]
    return obj


def _resolve_manifest(name: str, preset: str | None = None) -> EngineManifest:
    if name not in list_manifests():
        raise SystemExit(f"unknown engine {name!r}; known: {', '.join(list_manifests())}")
    if preset is not None and preset not in list_presets():
        lines = [f"unknown preset {preset!r}; searched (in precedence order):"]
        for d in preset_search_dirs():
            if d.is_dir():
                found = sorted(p.stem for p in d.glob("*.toml"))
                lines.append(f"  {d}: {', '.join(found) or '(empty)'}")
            else:
                lines.append(f"  {d}: (absent)")
        raise SystemExit("\n".join(lines))
    return load_manifest(name, preset=preset)


def _resolve_installed_manifest(name: str) -> EngineManifest:
    """Resolve an engine as INSTALLED (preset recorded at install time).

    Lifecycle verbs must target the ports/paths the service was actually
    generated from — the baseline may differ (a preset can move the port).
    ``install`` keeps taking its preset explicitly; everything that acts
    on an existing service goes through here.
    """
    if name not in list_manifests():
        raise SystemExit(f"unknown engine {name!r}; known: {', '.join(list_manifests())}")
    return install_state.load_installed_manifest(name)


def _driver_for(manifest: EngineManifest):
    try:
        factory = get_driver_factory(manifest.name)
    except KeyError as e:
        raise SystemExit(f"no driver registered for engine {manifest.name!r}") from e
    # Every factory (static registry and family dispatch) accepts an
    # optional manifest positional and uses it when given. The TypeError
    # fallback keeps zero-arg fakes injected by tests working.
    try:
        return factory(manifest)
    except TypeError:
        return factory()


# ---------------------------------------------------------------------------
# list-engines
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace) -> int:
    names = list_manifests()
    if args.json:
        _emit({"engines": names}, as_json=True)
        return 0
    print("Known engines:")
    for n in names:
        m = load_manifest(n)
        print(f"  {n:<12} port={m.network.port} plist={m.plist.name}")
    return 0


def cmd_list_presets(args: argparse.Namespace) -> int:
    """List bundled tuned-manifest presets (full TOMLs under presets/)."""
    summaries = [preset_summary(n) for n in list_presets()]
    if args.json:
        _emit({"presets": summaries}, as_json=True)
        return 0
    if not summaries:
        print("No presets bundled.")
        return 0
    print("Bundled presets (use with: aisctl install <engine> --preset <name>):")
    for s in summaries:
        print(f"  {s['preset']}")
        print(f"    engine: {s['engine']}")
        print(f"    display: {s['display']}")
    return 0


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    """``aisctl plan <preset>`` — estimate the preset's memory cost (read-only).

    No OperationsLock: nothing is mutated, and an operator planning the
    next install must be able to run it while a bench or install holds
    the lock. ``--json`` emits the frozen wire contract (the same payload
    ``GET /internal/v1/plan`` serves).
    """
    from ais_core import plan as plan_mod

    try:
        cost = plan_mod.plan_for_preset(args.preset, engine=args.engine)
    except ManifestError as e:
        # e.g. --engine pinned to an engine the preset does not target.
        raise SystemExit(str(e)) from None
    except FileNotFoundError:
        lines = [f"unknown preset {args.preset!r}; searched (in precedence order):"]
        for d in preset_search_dirs():
            if d.is_dir():
                found = sorted(p.stem for p in d.glob("*.toml"))
                lines.append(f"  {d}: {', '.join(found) or '(empty)'}")
            else:
                lines.append(f"  {d}: (absent)")
        raise SystemExit("\n".join(lines)) from None

    if args.json:
        _emit(cost.payload(), as_json=True)
        return 0

    engine = args.engine or preset_summary(args.preset)["engine"]
    print(f"Preset: {cost.preset} (engine: {engine})")
    print("Components:")
    for name, comp in cost.components.items():
        mb = f"{comp.mb:>10.1f} MB" if comp.mb is not None else f"{'?':>10} MB"
        detail = f" — {comp.detail}" if comp.detail else ""
        print(f"  {name:<15} {mb}  [{comp.source}]{detail}")
    if cost.confidence == "unknown":
        print("Estimated total: unknown — at least one component has no usable source")
        print("(declare it in the preset's [memory] section, or let calibration measure it)")
    else:
        print(
            f"Estimated total: {cost.total_mb_low:.0f}-{cost.total_mb_high:.0f} MB "
            f"(confidence: {cost.confidence})"
        )
    print("Advisory only — the estimate never blocks an install.")
    return 0


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    deep = getattr(args, "deep", False)
    targets = [args.engine] if args.engine else list_manifests()
    rows = []
    for name in targets:
        m = _resolve_installed_manifest(name)
        # probe_state probes generation at most once per engine (deep mode)
        # and returns the verdict alongside the state. ZOMBIE downgrades the
        # state to DEGRADED; BUSY/UNSUPPORTED/ERROR are surfaced as-is so the
        # operator sees why certification was inconclusive — they are normal
        # conditions, not alarms.
        state, verdict = lifecycle.probe_state(m, deep=deep)
        record = install_state.read_install(name)
        row = {
            "engine": name,
            "state": state.value,
            "port": m.network.port,
            "plist": m.plist.name,
            # Recorded at install time — what the service was generated
            # from, not a live introspection of the running process.
            "preset": record.preset if record else None,
        }
        if deep:
            row["gen"] = verdict.value if verdict is not None else None
        rows.append(row)

    if args.json:
        _emit({"engines": rows}, as_json=True)
        return 0
    gen_header = " gen" if deep else ""
    print(f"{'engine':<12} {'state':<22} {'port':<6} plist preset{gen_header}")
    for r in rows:
        preset = r["preset"] or "-"
        gen = f" {r['gen'] or '-'}" if deep else ""
        print(f"{r['engine']:<12} {r['state']:<22} {r['port']:<6} {r['plist']} {preset}{gen}")
    return 0


# ---------------------------------------------------------------------------
# calibration hooks (advisory, best-effort — never fail the lifecycle verb)
# ---------------------------------------------------------------------------


def _record_health_calibration(m: EngineManifest) -> None:
    """Record the engine's measured resident footprint after a healthy start.

    This is the RELIABLE calibration source (full weights loaded — health
    gated on it; measurement scoped to the one process). Only applies to
    preset-based installs: the plan estimator answers per-preset questions,
    and the install record carries the preset name + manifest sha that key
    the calibration ring. Strictly best-effort: calibration is advisory
    bookkeeping and must never fail the operation that produced it.
    """
    try:
        record = install_state.read_install(m.name)
        if record is None or record.preset is None or not record.manifest_sha256:
            return
        rss_mb = calibration.engine_rss_mb(m)
        if rss_mb is None:
            return
        calibration.record_sample(
            m.name,
            preset=record.preset,
            manifest_sha256=record.manifest_sha256,
            phys_footprint_mb=rss_mb,
            source="health",
        )
    except Exception as e:  # advisory hook — swallow everything
        logger.debug("calibration (health) skipped for %s: %s", m.name, e)


def _record_unload_calibration(
    m: EngineManifest,
    before: memory.VmStat | None,
    after: memory.VmStat | None,
) -> None:
    """Record the host-wide footprint drop across a successful unload.

    Both snapshots are captured by the caller — the ``after`` one INSIDE the
    OperationsLock, right after the unload, so no concurrent locked operation
    (purge, another unload) can allocate/free between the unload and the
    measurement. Still noisy by nature (unlocked processes run freely), so
    :mod:`ais_core.calibration` weights these samples at 0.5 and negative
    deltas are discarded. Reuses :attr:`ais_core.memory.PurgeReport.freed_mb`
    for the active+wired+compressed accounting.
    """
    if before is None or after is None:
        return
    try:
        record = install_state.read_install(m.name)
        if record is None or record.preset is None or not record.manifest_sha256:
            return
        report = memory.PurgeReport(before=before, after=after, pressure_after="", elapsed_s=0.0)
        calibration.record_sample(
            m.name,
            preset=record.preset,
            manifest_sha256=record.manifest_sha256,
            phys_footprint_mb=float(report.freed_mb),
            source="unload",
        )
    except Exception as e:  # advisory hook — swallow everything
        logger.debug("calibration (unload) skipped for %s: %s", m.name, e)


def _vm_snapshot_or_none() -> memory.VmStat | None:
    """Non-throwing vm_stat snapshot for the unload calibration bracket."""
    try:
        return memory.vm_stat_parse()
    except Exception as e:  # advisory hook — swallow everything
        logger.debug("calibration: vm_stat snapshot failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# install / uninstall / start / stop / restart
# ---------------------------------------------------------------------------


def _resolve_install_user(requested: str | None) -> str:
    """Resolve the account an engine daemon runs as, failing CLOSED on root (D4, story 2.2).

    ``--user`` wins, then ``$SUDO_USER`` (the human behind ``sudo``), then ``$USER``; NEVER a
    ``"root"`` fallback. An engine daemon must run under a non-root account (I2); the root helper
    would refuse uid 0 anyway, but we fail closed in the CLI with a friendly message rather than
    hand the helper a ``"root"`` it has to reject. Shared by ``cmd_install`` and ``cmd_reinstall``
    so both install entry points behave identically.
    """
    user = requested or os.environ.get("SUDO_USER") or os.environ.get("USER")
    if not user or user == "root":
        raise SystemExit(
            "refusing to install as root: pass --user <name> or run as a regular user "
            "(engine daemons must run under a non-root account)."
        )
    return user


def cmd_install(args: argparse.Namespace) -> int:
    preset = getattr(args, "preset", None)
    record = install_state.read_install(args.engine)
    if preset is None and record is not None and record.preset is not None and not args.force:
        # The dangerous path issue #6 exists for: a plain install over a
        # preset-based one regenerates the service from the base manifest
        # and comes back *healthy* — the degradation is silent.
        raise SystemExit(
            f"{args.engine} was last installed with preset {record.preset!r}; a plain "
            f"install would silently regenerate it from the base manifest.\n"
            f"Use 'aisctl reinstall {args.engine}' to replay the preset, pass "
            f"'--preset {record.preset}' explicitly, or --force to install the base manifest."
        )
    m = _resolve_manifest(args.engine, preset=preset)
    user = _resolve_install_user(args.user)  # D4: fail closed on root (no "root" fallback)
    enable_fw = args.firewall == "lan-only"

    with memory.OperationsLock(force=args.force):
        result = lifecycle.install(
            m,
            user=user,
            binary_path=args.binary,
            enable_firewall=enable_fw,
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        src = manifest_source_path(args.engine, preset)
        if src is not None:
            install_state.record_install(
                args.engine, preset=preset, manifest_path=src, firewall=args.firewall
            )
    result["preset"] = preset
    if not args.dry_run and result.get("health_ok"):
        _record_health_calibration(m)
    _emit(result, as_json=args.json)
    return 0 if (args.dry_run or result.get("health_ok")) else 2


def cmd_reinstall(args: argparse.Namespace) -> int:
    """``aisctl reinstall <engine>`` — uninstall + install replaying the record.

    Exists because the manual sequence (``uninstall`` then plain ``install``)
    silently degrades a preset-based install to the base manifest while coming
    back healthy. ``reinstall`` replays what was actually installed: same
    preset, same firewall mode (unless overridden on the command line).
    """
    record = install_state.read_install(args.engine)
    if record is None:
        raise SystemExit(
            f"no install record for {args.engine!r} — nothing to replay.\n"
            f"Records are written by 'aisctl install'; run "
            f"'aisctl install {args.engine} [--preset <name>]' once to create one."
        )
    m = _resolve_manifest(args.engine, preset=record.preset)
    user = _resolve_install_user(args.user)  # D4: fail closed on root (was: ... or "root")
    firewall_mode = args.firewall or record.firewall
    enable_fw = firewall_mode == "lan-only"

    src = manifest_source_path(args.engine, record.preset)
    manifest_changed = (
        src is not None
        and record.manifest_sha256 != ""
        and install_state.manifest_digest(src) != record.manifest_sha256
    )

    # When the reinstall will regenerate the IDENTICAL anchor (same port/subnets —
    # the nominal flag-change reinstall), keep it across the uninstall: the whole
    # password-gated pf path is skipped (install_anchor detects up-to-date and
    # no-ops). Without this, a reinstall on a helper-only host without a TTY died
    # mid-sequence with the daemon already removed (2026-07-01 retex).
    fw_unchanged = enable_fw and firewall.anchor_up_to_date(m)

    with memory.OperationsLock(force=args.force):
        lifecycle.uninstall(m, keep_firewall=fw_unchanged, dry_run=args.dry_run)
        result = lifecycle.install(
            m,
            user=user,
            binary_path=args.binary,
            enable_firewall=enable_fw,
            dry_run=args.dry_run,
        )
    result["preset"] = record.preset
    # The reinstall picks up the manifest file as it is NOW; flag drift so
    # the operator knows the regenerated service may differ from the original.
    result["manifest_changed_since_install"] = manifest_changed
    if not args.dry_run and src is not None:
        install_state.record_install(
            args.engine, preset=record.preset, manifest_path=src, firewall=firewall_mode
        )
    if not args.dry_run and result.get("health_ok"):
        _record_health_calibration(m)
    _emit(result, as_json=args.json)
    return 0 if (args.dry_run or result.get("health_ok")) else 2


def cmd_uninstall(args: argparse.Namespace) -> int:
    m = _resolve_installed_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        result = lifecycle.uninstall(m, dry_run=args.dry_run)
    _emit(result, as_json=args.json)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    m = _resolve_installed_manifest(args.engine)
    dry_run = getattr(args, "dry_run", False)
    lifecycle.start(m, dry_run=dry_run)
    if dry_run:
        _emit({"engine": m.name, "started": False, "dry_run": True}, as_json=args.json)
        return 0
    healthy = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
    if healthy:
        _record_health_calibration(m)
    payload = {"engine": m.name, "started": True, "healthy": healthy}
    _emit(payload, as_json=args.json)
    return 0 if healthy else 2


def cmd_stop(args: argparse.Namespace) -> int:
    m = _resolve_installed_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        lifecycle.stop(m, dry_run=args.dry_run)
    _emit({"engine": m.name, "stopped": True, "dry_run": args.dry_run}, as_json=args.json)
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    m = _resolve_installed_manifest(args.engine)
    dry_run = getattr(args, "dry_run", False)
    with memory.OperationsLock(force=args.force):
        lifecycle.restart(m, dry_run=dry_run)
    if dry_run:
        _emit({"engine": m.name, "restarted": False, "dry_run": True}, as_json=args.json)
        return 0
    healthy = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
    _emit(
        {"engine": m.name, "restarted": True, "healthy": healthy},
        as_json=args.json,
    )
    return 0 if healthy else 2


def cmd_disable(args: argparse.Namespace) -> int:
    """``aisctl disable <engine>`` — durable cold standby (survives reboot).

    Fills the gap between ``stop`` (RunAtLoad brings the engine back at
    next boot) and ``uninstall`` (loses the install): the engine stays
    installed with its tuned configuration but is excluded from the boot
    sequence — and from its model-load memory spike.
    """
    m = _resolve_installed_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        result = lifecycle.disable(m, dry_run=args.dry_run)
    _emit(result, as_json=args.json)
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    m = _resolve_installed_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        result = lifecycle.enable(m, start_now=args.start, dry_run=args.dry_run)
    if args.start and not args.dry_run:
        healthy = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
        if healthy:
            _record_health_calibration(m)
        result["healthy"] = healthy
        _emit(result, as_json=args.json)
        return 0 if healthy else 2
    _emit(result, as_json=args.json)
    return 0


def cmd_upgrade(args: argparse.Namespace) -> int:
    """``aisctl upgrade <engine>`` — brew-upgrade a whitelisted engine.

    The formula whitelist lives in :mod:`ais_core.upgrade` (shared with the
    loopback command server). After a successful upgrade the running daemon
    still executes the *old* binary; pass ``--restart`` to reconcile it
    (otherwise we print a hint and leave the process running so an in-flight
    request isn't interrupted without the operator asking).
    """
    m = _resolve_installed_manifest(args.engine)
    try:
        argv = upgrade_argv(m.name)
    except ValueError as e:
        _emit({"engine": m.name, "ok": False, "error": str(e)}, as_json=args.json)
        return 2

    if args.dry_run:
        _emit(
            {"engine": m.name, "dry_run": True, "argv": argv, "would_restart": args.restart},
            as_json=args.json,
        )
        return 0

    started = time.monotonic()
    with memory.OperationsLock(force=args.force):
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    ok = proc.returncode == 0
    result: dict[str, Any] = {
        "engine": m.name,
        "ok": ok,
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-4096:],
        "stderr": proc.stderr[-4096:],
        "duration_ms": int((time.monotonic() - started) * 1000),
    }

    restarted = False
    if ok and args.restart:
        with memory.OperationsLock(force=args.force):
            lifecycle.restart(m)
        restarted = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
        result["restarted"] = restarted
    elif ok:
        result["hint"] = f"running process still on the old build; run 'aisctl restart {m.name}'"

    _emit(result, as_json=args.json)
    if not ok:
        return 2
    if args.restart and not restarted:
        return 2
    return 0


# ---------------------------------------------------------------------------
# unload / purge / repair
# ---------------------------------------------------------------------------


def cmd_unload(args: argparse.Namespace) -> int:
    m = _resolve_installed_manifest(args.engine)
    driver = _driver_for(m)
    before = _vm_snapshot_or_none()
    with memory.OperationsLock(force=args.force):
        outcome = driver.unload(args.model)
        # Snapshot INSIDE the lock, right after the unload: a concurrent
        # locked operation must not be able to skew the measured delta.
        after = _vm_snapshot_or_none() if outcome.success else None
    if outcome.success:
        # The footprint drop across the unload is a (noisy) calibration
        # sample for the plan estimator — recorded at half weight.
        _record_unload_calibration(m, before, after)
    _emit(outcome, as_json=args.json)
    return 0 if outcome.success else 2


def cmd_load(args: argparse.Namespace) -> int:
    """Warm-load a model on a running engine so the first inference is hot.

    The implementation is engine-specific:

    - **Ollama** sends a ``POST /api/generate`` with an empty prompt + a
      ``keep_alive`` window so the model is paged into VRAM and stays
      there for the configured duration.
    - **LM Studio** uses the same trick over the OpenAI-compatible
      ``/v1/chat/completions`` endpoint with ``max_tokens=1``.
    - Other engines load their model at daemon start (llama.cpp's
      ``--warmup``, mlx-lm's eager init); for those we return a
      ``noop`` outcome rather than failing, so the fleet caller can
      issue a uniform ``load`` everywhere.
    """
    import urllib.error
    import urllib.request

    m = _resolve_installed_manifest(args.engine)
    # Driver is intentionally not used here: this command targets the
    # engine's HTTP surface directly (different transport per engine).
    # We still resolve the manifest to enforce that the engine is known.
    _driver_for(m)
    model = args.model
    keep_alive = getattr(args, "keep_alive", "5m") or "5m"

    name = m.name
    base = f"http://{m.network.bind}:{m.network.port}"
    started = time.monotonic()

    def _outcome(method: str, success: bool, detail: str = "") -> dict:
        return {
            "engine": name,
            "model": model,
            "method": method,
            "success": success,
            "detail": detail,
            "duration_ms": int((time.monotonic() - started) * 1000),
        }

    if name == "ollama":
        payload = json.dumps(
            {"model": model, "prompt": "", "keep_alive": keep_alive, "stream": False}
        ).encode()
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                resp.read(1024)
            _emit(_outcome("api", True, f"keep_alive={keep_alive}"), as_json=args.json)
            return 0
        except (urllib.error.URLError, OSError) as e:
            _emit(_outcome("api", False, f"ollama error: {e}"), as_json=args.json)
            return 2

    if name == "lmstudio":
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1,
                "temperature": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                resp.read(1024)
            _emit(_outcome("api", True, "1-token warm-up"), as_json=args.json)
            return 0
        except (urllib.error.URLError, OSError) as e:
            _emit(_outcome("api", False, f"lmstudio error: {e}"), as_json=args.json)
            return 2

    # Engines that load their model at daemon start: report noop success.
    _emit(
        _outcome("noop", True, f"{name} loads its model at daemon start"),
        as_json=args.json,
    )
    return 0


def cmd_purge(args: argparse.Namespace) -> int:
    with memory.OperationsLock(force=args.force):
        report = memory.purge_memory(dry_run=args.dry_run)
    _emit(report, as_json=args.json)
    return 0


def cmd_repair(args: argparse.Namespace) -> int:
    """No lock here: repair runs *because* a previous lock might be stale."""
    report = memory.repair(dry_run=args.dry_run)
    _emit(report, as_json=args.json)
    return 0


# ---------------------------------------------------------------------------
# bootstrap (install sudoers fragment)
# ---------------------------------------------------------------------------


def cmd_bootstrap(args: argparse.Namespace) -> int:
    if args.verify:
        return _bootstrap_verify()
    if args.rollback:
        return _bootstrap_rollback(dry_run=args.dry_run)
    if args.install:
        return _bootstrap_full(dedicated_user=args.dedicated_user, dry_run=args.dry_run)
    if args.install_sudoers:
        return _bootstrap_sudoers_only(dry_run=args.dry_run)
    print(
        "bootstrap: nothing to do.\n"
        "  --install          full one-time setup: install the privileged helper "
        "(/Library/PrivilegedHelperTools/asiai-priv, root:wheel 0755) THEN the helper-only\n"
        "                     sudoers fragment. I0-checked, strict order, idempotent.\n"
        "  --install-sudoers  install only the sudoers fragment (granular/legacy).\n"
        "  --rollback         revert: restore the pre-bootstrap sudoers, then remove the helper.\n"
        "  --verify           recompute the helper SHA-256 and compare to its sidecar.\n"
        "Add --dry-run to preview without touching the system."
    )
    return 0


def _bootstrap_full(*, dedicated_user: bool, dry_run: bool) -> int:
    """One-time idempotent bootstrap in the strict order: I0 chain check -> install helper
    (root:wheel 0755, invariant #3) -> sign helper (NFR11 sidecar) -> log dir + audit log
    (helper runtime, findings #3b/#2) -> [opt create the dedicated _aisrv account, NFR12] ->
    install the helper-only sudoers fragment (visudo-c'd).
    """
    if dry_run:
        bootstrap.install_helper(dry_run=True)
        bootstrap.write_helper_signature(dry_run=True)
        bootstrap.ensure_log_dir(dry_run=True)
        bootstrap.ensure_audit_log(dry_run=True)
        if dedicated_user:
            bootstrap.create_dedicated_user(dry_run=True)
        sudoers.backup_existing_sudoers(dry_run=True)
        sudoers.install_sudoers(dry_run=True)
        return 0
    try:
        # I0 first: refuse if ANY root-write target's chain is unlocked, before any write.
        bootstrap.assert_fleet_chain_locked()
        helper_path = bootstrap.install_helper()
        sidecar_path = bootstrap.write_helper_signature()  # NFR11, after the copy
        # Helper runtime: the log dir the helper refuses to create itself (#3b), then the
        # audit log 0640 root:admin so refusals are operator-readable without sudo (#2).
        log_dir_path = bootstrap.ensure_log_dir()
        audit_log_path = bootstrap.ensure_audit_log()
        user_info = bootstrap.create_dedicated_user() if dedicated_user else None  # NFR12, opt-in
        # FR8: record the pre-bootstrap sudoers state ONCE, BEFORE the helper-only fragment
        # overwrites it, so `--rollback` can restore exactly what was there (the rollback net).
        backup_path = sudoers.backup_existing_sudoers()
        sudoers_path = sudoers.install_sudoers()
    except (bootstrap.BootstrapError, sudoers.SudoersError) as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    print(f"installed helper:    {helper_path}")
    print(f"wrote signature:     {sidecar_path}")
    print(f"log dir:             {log_dir_path} (root:wheel 0755)")
    print(f"audit log:           {audit_log_path} (root:admin 0640 — operator-readable)")
    if user_info is not None:
        state = "created" if user_info["created"] else "already present"
        print(f"dedicated user:      {user_info['user']} (uid {user_info.get('uid')}, {state})")
    if backup_path is not None:
        print(f"backed up sudoers:   {backup_path} (for --rollback)")
    print(f"installed sudoers:   {sudoers_path}")
    print("bootstrap complete. Engine lifecycle now routes through the helper (NOPASSWD).")
    return 0


def _bootstrap_verify() -> int:
    """``aisctl bootstrap --verify`` — recompute the helper's SHA-256 and compare to its sidecar."""
    try:
        ok = bootstrap.verify_helper()
    except bootstrap.BootstrapError as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    if ok:
        print(f"helper integrity OK: {sudoers.PRIVILEGED_HELPER_PATH} matches its SHA-256 sidecar")
        return 0
    print(
        f"helper integrity MISMATCH: {sudoers.PRIVILEGED_HELPER_PATH} does not match "
        f"{bootstrap.HELPER_SHA256_PATH} — the helper was modified or the sidecar is stale",
        file=sys.stderr,
    )
    return 1


def _bootstrap_sudoers_only(*, dry_run: bool) -> int:
    content = sudoers.generate_sudoers_content()
    if dry_run:
        # mirror the live order: backup-before-install
        sudoers.backup_existing_sudoers(dry_run=True)
        print(content)
        return 0
    try:
        sudoers.validate_content(content)
    except sudoers.SudoersError as e:
        print(f"sudoers validation FAILED: {e}", file=sys.stderr)
        return 2
    try:
        # FR8: record the prior state ONCE before overwriting, so --rollback stays possible even
        # via the granular path (no-op if already recorded by a prior install).
        sudoers.backup_existing_sudoers()
        path = sudoers.install_sudoers(content)
    except sudoers.SudoersError as e:
        # US-004: non-TTY guard surfaces clear instructions; print them
        # cleanly rather than as an uncaught traceback.
        print(f"\n{e}", file=sys.stderr)
        return 2
    print(f"installed {path}")
    return 0


def _bootstrap_rollback(*, dry_run: bool) -> int:
    """``aisctl bootstrap --rollback`` (FR8) — revert the helper model, never locking out sudo.

    Order matters: restore the sudoers FIRST (raw sudo / the pre-bootstrap state is back, and the
    restore itself is anti-lockout — visudo-validated, never publishes broken content), THEN remove
    the helper + its signature sidecar. The dedicated ``_aisrv`` account is intentionally NOT
    removed (out of scope, harmless, reversible by the operator via ``sysadminctl -deleteUser``).
    """
    if dry_run:
        sudoers.restore_sudoers(dry_run=True)
        bootstrap.remove_helper(dry_run=True)
        return 0
    try:
        desc = sudoers.restore_sudoers()  # sudoers first — re-establish a working sudo
        removed = bootstrap.remove_helper()  # then the helper, after sudo is healthy
    except (bootstrap.BootstrapError, sudoers.SudoersError) as e:
        print(f"\n{e}", file=sys.stderr)
        return 2
    print(f"sudoers:             {desc}")
    print(f"removed helper:      {removed[0]}")
    print(f"removed signature:   {removed[1]}")
    print(
        "rollback complete. Raw sudo is restored; re-run `aisctl bootstrap --install` to "
        "re-apply the helper model."
    )
    return 0


# ---------------------------------------------------------------------------
# bundle (SMAppService — Background Items panel identity)
# ---------------------------------------------------------------------------


def cmd_bundle_build(args: argparse.Namespace) -> int:
    """``aisctl bundle build`` — produce <App>.app from the engine manifests."""
    from ais_core import __version__ as core_version
    from ais_core import bundle

    launcher = args.launcher or shutil.which("asiai-launch")
    if not launcher:
        print(
            "asiai-launch not found on PATH; install asiai-inference-server or pass --launcher",
            file=sys.stderr,
        )
        return 2
    # Never fall back to root: engines run as a regular account (I2). With no
    # resolvable invoker the caller must say who — same doctrine as the helper.
    user = args.user or os.environ.get("USER")
    if not user:
        print(
            "cannot determine the daemon user ($USER unset); pass --user explicitly "
            "(bundle daemons never run as root)",
            file=sys.stderr,
        )
        return 2
    spec = bundle.BundleSpec(
        services=tuple(s.strip() for s in args.services.split(",") if s.strip()),
        user=user,
        launcher_path=launcher,
        bundle_id=args.bundle_id,
        app_name=args.app_name,
        display_name=args.display_name,
        version=core_version,
    )
    try:
        result = bundle.build_bundle(spec, Path(args.output).expanduser(), sign_identity=args.sign)
    except bundle.BundleError as e:
        print(f"bundle build failed: {e}", file=sys.stderr)
        return 2
    if not args.sign:
        result["hint"] = (
            "unsigned bundle: functional, but macOS 26+ shows the generic icon "
            "in Background Items unless signed with a locally-trusted "
            "code-signing identity (--sign)"
        )
    _emit(result, as_json=args.json)
    return 0


def cmd_bundle_activate(args: argparse.Namespace) -> int:
    """``aisctl bundle activate <engine>`` — publish the active manifest.

    This is what ``asiai-launch`` reads at daemon start. Defaults to the
    preset recorded at install time, so a bundle-launched engine keeps the
    exact tuning of its last install.
    """
    from ais_core import bundle

    try:
        result = bundle.write_active_manifest(args.engine, preset=getattr(args, "preset", None))
    except (bundle.BundleError, FileNotFoundError) as e:
        print(f"activate failed: {e}", file=sys.stderr)
        return 2
    _emit(result, as_json=args.json)
    return 0


def _daemon_loaded_from(label: str) -> str | None:
    """Source plist path launchd loaded ``label`` from, or None if not loaded.

    ``launchctl print system/<label>`` exits non-zero when the service is not
    in the system domain; when loaded, its output carries a ``path = ...``
    line naming the plist it was bootstrapped from (a bundle-registered
    daemon shows a path inside ``<App>.app/Contents/Library/LaunchDaemons``).
    Returns ``""`` when loaded but the source line is missing (treat as
    unidentified — callers must fail closed).
    """
    proc = subprocess.run(
        ["launchctl", "print", f"system/{label}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("path = "):
            return stripped[len("path = ") :]
    return ""


def _is_bundle_source(loaded_from: str | None) -> bool:
    """True iff ``loaded_from`` (from :func:`_daemon_loaded_from`) is an
    SMAppService bundle load, in either form launchctl reports it:

    * a plist path INSIDE an app bundle (``.../Contents/Library/LaunchDaemons/``);
    * the smd attribution launchctl prints for a registered daemon,
      ``(submitted by smd.<pid>)`` — no filesystem path at all.

    Recognising the smd form is what makes a re-register idempotent: without
    it a label already loaded from the bundle looks like a foreign loader and
    the register guard falsely blocks (deploy-M5 finding, 0.6.1).
    """
    if not loaded_from:
        return False
    return "/Contents/Library/LaunchDaemons/" in loaded_from or loaded_from.startswith(
        "(submitted by smd"
    )


def _bundle_service_names(app: Path) -> list[str]:
    """Engine service names embedded in an installed bundle.

    Each embedded LaunchDaemon runs ``<stub> <service>`` — the service name is
    ``ProgramArguments[1]``. This is the authoritative "what this bundle
    manages" list; using it (instead of every manifest on disk) keeps the
    register guard from tripping on wrapper/legacy engines that are not, and
    cannot be, in the bundle (deploy-M5 finding: mlx-lm, a wrapper engine,
    blocked a whole-bundle register).
    """
    daemons = app / "Contents" / "Library" / "LaunchDaemons"
    services: list[str] = []
    if not daemons.is_dir():
        return services
    for pl in sorted(daemons.glob("*.plist")):
        try:
            with pl.open("rb") as fh:
                data = plistlib.load(fh)
        except (OSError, ValueError):
            continue
        argv = data.get("ProgramArguments")
        if isinstance(argv, list) and len(argv) >= 2 and isinstance(argv[1], str):
            services.append(argv[1])
    return services


def _gui_session_active() -> bool:
    """True when running inside a GUI (Aqua) session.

    SMAppService registration needs one: the one-time approval toggle lives in
    System Settings, and a headless register parks the daemon in
    ``requiresApproval`` forever — a silent fleet outage (ADR-002).
    """
    proc = subprocess.run(["launchctl", "managername"], capture_output=True, text=True, check=False)
    return proc.returncode == 0 and proc.stdout.strip() == "Aqua"


def _register_blockers(
    names: list[str], *, daemons_dir: Path = Path("/Library/LaunchDaemons")
) -> list[str]:
    """Double-load guard: what forbids registering ``names`` with SMAppService.

    A label driven by two loaders (legacy plist AND the bundle) double-loads
    under launchd. Registration is blocked while, for any selected service,
    the legacy plist still exists on disk OR the label is still loaded from
    anywhere outside an app bundle (plist gone but never booted out, or an
    unidentifiable source — fail closed). Already loaded FROM the bundle is
    fine: re-registering is idempotent.
    """
    blockers: list[str] = []
    for name in names:
        try:
            m = load_manifest(name)
        except Exception:
            continue
        label = m.plist.name
        legacy = daemons_dir / f"{label}.plist"
        loaded_from = _daemon_loaded_from(label)
        if legacy.exists():
            blockers.append(f"{name}: legacy plist {legacy} still installed")
        elif loaded_from is not None and not _is_bundle_source(loaded_from):
            source = loaded_from or "an unidentified source"
            blockers.append(f"{name}: label {label} still loaded from {source}")
    return blockers


def cmd_bundle_ctl(args: argparse.Namespace) -> int:
    """register / unregister / status — thin wrapper over <App>Register.

    The Swift helper inside the bundle owns the SMAppService calls; this
    wrapper locates it and forwards the action. ``register`` is guarded
    (story 2.6 acceptance criteria, both machine-enforced):

    * HARD refusal while a legacy ``/Library/LaunchDaemons`` plist for a
      selected service still exists or is still loaded — the same label
      driven by two loaders double-loads under launchd. Remedy first:
      ``aisctl uninstall <engine>``.
    * HARD refusal outside a GUI session (unless ``--allow-headless``): the
      approval toggle is GUI-only, so a headless register silently parks the
      daemon in ``requiresApproval``.
    """
    app = Path(args.app).expanduser()
    register_bin = app / "Contents" / "MacOS" / f"{app.stem}Register"
    if not register_bin.is_file():
        print(f"register helper not found: {register_bin}", file=sys.stderr)
        return 2

    if args.action == "register":
        if not _gui_session_active() and not getattr(args, "allow_headless", False):
            print(
                "register refused: no GUI session (launchctl managername != Aqua). "
                "SMAppService needs the one-time approval toggle in System Settings; "
                "registering headless would leave the daemons in requiresApproval, "
                "silently not running. Register from the Mac's console (or Screen "
                "Sharing) — or pass --allow-headless if you will approve via GUI later.",
                file=sys.stderr,
            )
            return 2
        # Scope the guard to what the bundle actually manages. A whole-bundle
        # register ("all") must not be blocked by an unrelated wrapper/legacy
        # engine that isn't — and can't be — in the bundle (deploy-M5 finding).
        if args.service and args.service != "all":
            targets = [args.service]
        else:
            targets = _bundle_service_names(app) or list_manifests()
        blockers = _register_blockers(targets)
        if blockers:
            print(
                "register refused — the same label driven by two loaders double-loads "
                "under launchd (story 2.6):",
                file=sys.stderr,
            )
            for b in blockers:
                print(f"  - {b}", file=sys.stderr)
            print("run 'aisctl uninstall <engine>' first, then register.", file=sys.stderr)
            return 2

    argv = [str(register_bin), args.action]
    if args.service:
        argv.append(args.service)
    proc = subprocess.run(argv, check=False)
    return proc.returncode
