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
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from typing import Any

from ais_core import bootstrap, install_state, lifecycle, memory, sudoers
from ais_core.manifest import (
    EngineManifest,
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
from ais_engines.ollama import OllamaDriver
from ais_engines.omlx import OmlxDriver
from ais_engines.rapidmlx import RapidMlxDriver
from ais_engines.turboquant import TurboquantDriver
from ais_engines.vmlx import VmlxDriver

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
        print(json.dumps(_jsonify(payload), indent=2, default=str))
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
# status
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    deep = getattr(args, "deep", False)
    targets = [args.engine] if args.engine else list_manifests()
    rows = []
    for name in targets:
        m = _resolve_manifest(name)
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
# install / uninstall / start / stop / restart
# ---------------------------------------------------------------------------


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
    # D4 (story 2.2): never fall back to "root". An engine daemon must run under a non-root
    # account (I2); the helper would refuse uid 0 anyway, but we fail closed in the CLI rather
    # than hand the helper a "root" it has to reject.
    user = args.user or os.environ.get("SUDO_USER") or os.environ.get("USER")
    if not user or user == "root":
        raise SystemExit(
            "refusing to install as root: pass --user <name> or run as a regular user "
            "(engine daemons must run under a non-root account)."
        )
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
    user = args.user or os.environ.get("USER") or "root"
    firewall_mode = args.firewall or record.firewall
    enable_fw = firewall_mode == "lan-only"

    src = manifest_source_path(args.engine, record.preset)
    manifest_changed = (
        src is not None
        and record.manifest_sha256 != ""
        and install_state.manifest_digest(src) != record.manifest_sha256
    )

    with memory.OperationsLock(force=args.force):
        lifecycle.uninstall(m, dry_run=args.dry_run)
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
    _emit(result, as_json=args.json)
    return 0 if (args.dry_run or result.get("health_ok")) else 2


def cmd_uninstall(args: argparse.Namespace) -> int:
    m = _resolve_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        result = lifecycle.uninstall(m, dry_run=args.dry_run)
    _emit(result, as_json=args.json)
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    m = _resolve_manifest(args.engine)
    dry_run = getattr(args, "dry_run", False)
    lifecycle.start(m, dry_run=dry_run)
    if dry_run:
        _emit({"engine": m.name, "started": False, "dry_run": True}, as_json=args.json)
        return 0
    healthy = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
    payload = {"engine": m.name, "started": True, "healthy": healthy}
    _emit(payload, as_json=args.json)
    return 0 if healthy else 2


def cmd_stop(args: argparse.Namespace) -> int:
    m = _resolve_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        lifecycle.stop(m, dry_run=args.dry_run)
    _emit({"engine": m.name, "stopped": True, "dry_run": args.dry_run}, as_json=args.json)
    return 0


def cmd_restart(args: argparse.Namespace) -> int:
    m = _resolve_manifest(args.engine)
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
    m = _resolve_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        result = lifecycle.disable(m, dry_run=args.dry_run)
    _emit(result, as_json=args.json)
    return 0


def cmd_enable(args: argparse.Namespace) -> int:
    m = _resolve_manifest(args.engine)
    with memory.OperationsLock(force=args.force):
        result = lifecycle.enable(m, start_now=args.start, dry_run=args.dry_run)
    if args.start and not args.dry_run:
        healthy = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
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
    m = _resolve_manifest(args.engine)
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
    m = _resolve_manifest(args.engine)
    driver = _driver_for(m)
    with memory.OperationsLock(force=args.force):
        outcome = driver.unload(args.model)
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

    m = _resolve_manifest(args.engine)
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
    (root:wheel 0755, invariant #3) -> sign helper (NFR11 sidecar) -> [opt create the dedicated
    _aisrv account, NFR12] -> install the helper-only sudoers fragment (visudo-c'd).
    """
    if dry_run:
        bootstrap.install_helper(dry_run=True)
        bootstrap.write_helper_signature(dry_run=True)
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
