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

from ais_core import lifecycle, memory, sudoers
from ais_core.manifest import (
    EngineManifest,
    list_manifests,
    list_presets,
    load_manifest,
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
            return lambda n=name, cls=driver_cls: cls.from_manifest_name(n)
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
                factories[n] = lambda name=n, cls=driver_cls: cls.from_manifest_name(name)
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
        raise SystemExit(
            f"unknown preset {preset!r}; available: {', '.join(list_presets()) or '(none)'}"
        )
    return load_manifest(name, preset=preset)


def _driver_for(manifest: EngineManifest):
    try:
        factory = get_driver_factory(manifest.name)
    except KeyError as e:
        raise SystemExit(f"no driver registered for engine {manifest.name!r}") from e
    # The static-registry factories accept an optional manifest argument;
    # the family-pattern factories ignore it (they re-load by name).
    # Calling with the manifest passes through cleanly in both shapes:
    # the static ones use it, the family ones drop it silently. This
    # keeps the call site uniform and lets tests inject a fake factory
    # that asserts on the manifest if they want to.
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
    targets = [args.engine] if args.engine else list_manifests()
    rows = []
    for name in targets:
        m = _resolve_manifest(name)
        state = lifecycle.current_state(m)
        rows.append(
            {
                "engine": name,
                "state": state.value,
                "port": m.network.port,
                "plist": m.plist.name,
            }
        )

    if args.json:
        _emit({"engines": rows}, as_json=True)
        return 0
    print(f"{'engine':<12} {'state':<22} {'port':<6} plist")
    for r in rows:
        print(f"{r['engine']:<12} {r['state']:<22} {r['port']:<6} {r['plist']}")
    return 0


# ---------------------------------------------------------------------------
# install / uninstall / start / stop / restart
# ---------------------------------------------------------------------------


def cmd_install(args: argparse.Namespace) -> int:
    m = _resolve_manifest(args.engine, preset=getattr(args, "preset", None))
    user = args.user or os.environ.get("USER") or "root"
    enable_fw = args.firewall == "lan-only"

    with memory.OperationsLock(force=args.force):
        result = lifecycle.install(
            m,
            user=user,
            binary_path=args.binary,
            enable_firewall=enable_fw,
            dry_run=args.dry_run,
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
    lifecycle.start(m)
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
    with memory.OperationsLock(force=args.force):
        lifecycle.restart(m)
    healthy = lifecycle.wait_for_health(m, timeout=m.network.health_timeout)
    _emit(
        {"engine": m.name, "restarted": True, "healthy": healthy},
        as_json=args.json,
    )
    return 0 if healthy else 2


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
    if not args.install_sudoers:
        print(
            "bootstrap: nothing to do without --install-sudoers.\n"
            "Run with --install-sudoers to write /etc/sudoers.d/asiai-inference\n"
            "(validated via visudo -cf before any sudo move into place)."
        )
        return 0

    content = sudoers.generate_sudoers_content()
    if args.dry_run:
        print(content)
        return 0

    try:
        sudoers.validate_content(content)
    except sudoers.SudoersError as e:
        print(f"sudoers validation FAILED: {e}", file=sys.stderr)
        return 2

    try:
        path = sudoers.install_sudoers(content)
    except sudoers.SudoersError as e:
        # US-004: non-TTY guard surfaces clear instructions; print them
        # cleanly rather than as an uncaught traceback.
        print(f"\n{e}", file=sys.stderr)
        return 2
    print(f"installed {path}")
    return 0
