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
import sys
from typing import Any

from ais_core import lifecycle, memory, sudoers
from ais_core.manifest import (
    EngineManifest,
    list_manifests,
    list_presets,
    load_manifest,
    preset_summary,
)
from ais_engines.llamacpp import LlamaCppDriver
from ais_engines.llamacpp_aux import LlamaCppAuxDriver
from ais_engines.lmstudio import LMStudioDriver
from ais_engines.ollama import OllamaDriver
from ais_engines.omlx import OmlxDriver
from ais_engines.turboquant import TurboquantDriver

DRIVER_FACTORIES = {
    "ollama": OllamaDriver.from_manifest,
    "lmstudio": LMStudioDriver.from_manifest,
    "omlx": OmlxDriver.from_manifest,
    "turboquant": TurboquantDriver.from_manifest,
    "llamacpp": LlamaCppDriver.from_manifest,
    "llamacpp-aux": LlamaCppAuxDriver.from_manifest,
}


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
    factory = DRIVER_FACTORIES.get(manifest.name)
    if factory is None:
        raise SystemExit(f"no driver registered for engine {manifest.name!r}")
    return factory(manifest)


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
