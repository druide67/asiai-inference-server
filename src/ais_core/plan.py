"""Preset memory-cost estimator — the "cost" half of the UMA plan contract.

``asiai`` (the fleet companion) owns the VERDICT (does this preset fit next
to what already runs on the node); this module owns the COST (how many MB
the preset needs). The wire contract between the two is frozen::

    {"preset": "<name>",
     "cost": {"total_mb_low": <float>, "total_mb_high": <float>,
              "confidence": "measured|declared|computed|unknown",
              "components": {<detail>}}}

Everything is advisory: the estimate never blocks an install.

Estimation ladder (best source wins, per component)
---------------------------------------------------
1. **measured** — calibration samples for this exact preset revision on
   this host (:mod:`ais_core.calibration`). A fresh measurement covers the
   WHOLE process footprint (weights + KV + runtime overhead: llama.cpp
   allocates its KV up front and health gates on the loaded model), so it
   replaces the component decomposition entirely rather than feeding the
   ``weights`` slot — plugging a full-footprint number into ``weights`` and
   then adding KV and overhead again would double-count. Band ±10%.
2. **declared** — the preset's ``[memory]`` figures, written by a human
   for the exact quant/architecture. Band ±20%.
3. **computed** — derived here: weights from the model file's on-disk
   size (``st_size`` of the realpath'd GGUF + mmproj), overhead from a
   per-engine-family default constant. Band ±35%.

We deliberately NEVER parse the GGUF to compute KV per token: hybrid-
attention models (e.g. Qwen3.6-27B — only 16 of 64 layers carry KV) make
the naive layersxheadsxhead_dim formula wrong by ~4x. KV comes only from
a declared ``kv_bytes_per_token`` multiplied by the ``--ctx-size`` parsed
out of the preset's ``program_args`` (the TOTAL context, shared between
slots — never multiplied by ``--parallel``).

Fail-closed: any required component that cannot be sourced makes the
whole estimate ``unknown`` with zeroed bounds. An honest "I don't know"
beats an optimistic invented number that greenlights an install into a
swap storm.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass

from ais_core import calibration, install_state
from ais_core.manifest import (
    EngineManifest,
    list_presets,
    load_manifest,
    manifest_source_path,
    preset_summary,
)

# Confidence levels, best to worst. The estimate's global confidence is the
# WORST of its used components.
CONFIDENCE_ORDER = ("measured", "declared", "computed", "unknown")

# Uncertainty band per source (fraction of the point estimate).
BANDS = {"measured": 0.10, "declared": 0.20, "computed": 0.35}

# Default runtime overhead (MB) per engine family, used when the preset does
# not declare ``[memory].overhead_mb``. Deliberately coarse constants — the
# ±35% computed band owns the imprecision:
# * llama.cpp on Metal: compute buffers + graph + host copies typically land
#   between ~500 MB (small dense models) and ~1.5 GB (large ctx / MoE);
#   1024 MB is the midpoint.
# * everything else (Ollama supervisor, MLX runtimes, LM Studio): 512 MB as
#   a conservative floor for the bare runtime.
LLAMACPP_OVERHEAD_MB = 1024.0
GENERIC_OVERHEAD_MB = 512.0


@dataclass(frozen=True)
class CostComponent:
    mb: float | None
    source: str  # "measured" | "declared" | "computed" | "unknown"
    detail: str = ""


@dataclass(frozen=True)
class PresetCost:
    preset: str
    total_mb_low: float
    total_mb_high: float
    confidence: str
    components: dict[str, CostComponent]

    def payload(self) -> dict:
        """The frozen wire-contract dict (used verbatim by CLI --json and serve)."""
        return {
            "preset": self.preset,
            "cost": {
                "total_mb_low": self.total_mb_low,
                "total_mb_high": self.total_mb_high,
                "confidence": self.confidence,
                "components": {
                    name: dataclasses.asdict(comp) for name, comp in self.components.items()
                },
            },
        }


def _parse_ctx_size(program_args: tuple[str, ...]) -> int | None:
    """Total context size from llama.cpp-style args (``--ctx-size`` / ``-c``).

    This is the TOTAL KV budget, shared between ``--parallel`` slots —
    callers must NOT multiply by the parallel count. Returns None when the
    flag is absent or its value is not a positive integer (fail-closed:
    the KV component then reports unknown rather than guessing a default).
    """
    args = list(program_args)
    for i, tok in enumerate(args):
        value: str | None = None
        if tok in ("--ctx-size", "-c"):
            if i + 1 < len(args):
                value = args[i + 1]
        elif tok.startswith("--ctx-size="):
            value = tok.split("=", 1)[1]
        if value is None:
            continue
        try:
            ctx = int(value)
        except ValueError:
            return None
        return ctx if ctx > 0 else None
    return None


def _file_size_mb(path: str) -> float | None:
    """On-disk size (MB) of a manifest file path, symlinks resolved.

    Tilde paths expand against the CURRENT user — this estimator runs on
    the engine's own host, where the operator convention keeps the model
    symlink in the invoking account's home. Missing file → None.
    """
    real = os.path.realpath(os.path.expanduser(path))
    try:
        size = os.stat(real).st_size
    except OSError:
        return None
    return size / (1024.0 * 1024.0)


def _weights_component(manifest: EngineManifest) -> CostComponent:
    """Weights footprint: declared ``[memory].weights_mb``, else disk size."""
    declared = manifest.memory.weights_mb
    if declared is not None:
        return CostComponent(mb=declared, source="declared", detail="[memory].weights_mb")
    if manifest.binary.model_path:
        model_mb = _file_size_mb(manifest.binary.model_path)
        if model_mb is None:
            return CostComponent(
                mb=None,
                source="unknown",
                detail=f"model file not found: {manifest.binary.model_path}",
            )
        detail = "st_size of model file"
        if manifest.binary.mmproj_path:
            mmproj_mb = _file_size_mb(manifest.binary.mmproj_path)
            if mmproj_mb is None:
                return CostComponent(
                    mb=None,
                    source="unknown",
                    detail=f"mmproj file not found: {manifest.binary.mmproj_path}",
                )
            model_mb += mmproj_mb
            detail = "st_size of model + mmproj files"
        return CostComponent(mb=round(model_mb, 1), source="computed", detail=detail)
    return CostComponent(
        mb=None,
        source="unknown",
        detail="no [memory].weights_mb declared and no model_path to size",
    )


def _kv_component(manifest: EngineManifest) -> CostComponent:
    """KV cache: declared bytes/token x the ctx parsed from program_args."""
    bpt = manifest.memory.kv_bytes_per_token
    if bpt is None:
        return CostComponent(
            mb=None,
            source="unknown",
            detail="no [memory].kv_bytes_per_token declared (GGUF parsing is deliberately "
            "not attempted: wrong on hybrid-attention models)",
        )
    ctx = _parse_ctx_size(manifest.binary.program_args)
    if ctx is None:
        return CostComponent(
            mb=None,
            source="unknown",
            detail="kv_bytes_per_token declared but no --ctx-size/-c in program_args",
        )
    kv_mb = bpt * ctx / (1024.0 * 1024.0)
    return CostComponent(
        mb=round(kv_mb, 1),
        source="declared",
        detail=f"{bpt:g} B/token x ctx {ctx} (total, shared across slots)",
    )


def _overhead_component(manifest: EngineManifest) -> CostComponent:
    """Engine runtime overhead: declared override, else family default."""
    declared = manifest.memory.overhead_mb
    if declared is not None:
        return CostComponent(mb=declared, source="declared", detail="[memory].overhead_mb")
    if manifest.name.startswith("llamacpp"):
        return CostComponent(
            mb=LLAMACPP_OVERHEAD_MB,
            source="computed",
            detail="llama.cpp family default (Metal buffers ~500-1500 MB)",
        )
    return CostComponent(
        mb=GENERIC_OVERHEAD_MB,
        source="computed",
        detail="generic engine runtime default",
    )


def _worst_confidence(sources: list[str]) -> str:
    return max(sources, key=CONFIDENCE_ORDER.index)


def estimate_preset_cost(
    manifest: EngineManifest,
    *,
    preset: str,
    manifest_sha256: str | None = None,
    host: str | None = None,
) -> PresetCost:
    """Estimate the resident memory cost of running ``preset``.

    ``manifest_sha256`` is the digest of the preset TOML the manifest was
    loaded from; it keys the calibration lookup (pass None to skip the
    measured path, e.g. for ad-hoc manifests not backed by a file).
    ``host`` overrides the calibration host filter (tests).
    """
    peak = manifest.memory.peak_extra_mb

    # 1. Measured: the whole-process footprint for this exact preset
    #    revision on this host beats any decomposition.
    if manifest_sha256:
        measured = calibration.measured_footprint_mb(
            manifest.name, preset=preset, manifest_sha256=manifest_sha256, host=host
        )
        if measured is not None:
            median_mb, n = measured
            band = BANDS["measured"]
            components: dict[str, CostComponent] = {
                "measured_total": CostComponent(
                    mb=round(median_mb, 1),
                    source="measured",
                    detail=f"weighted median of {n} calibration sample(s) on this host "
                    "(covers weights + KV + overhead)",
                )
            }
            high = median_mb * (1 + band)
            if peak is not None:
                components["peak_extra"] = CostComponent(
                    mb=peak,
                    source="declared",
                    detail="[memory].peak_extra_mb — load/warmup transient, added to the "
                    "high bound only",
                )
                high += peak
            return PresetCost(
                preset=preset,
                total_mb_low=round(median_mb * (1 - band), 1),
                total_mb_high=round(high, 1),
                confidence="measured",
                components=components,
            )

    # 2./3. Decompose: weights + KV + overhead, each from its best source.
    components = {
        "weights": _weights_component(manifest),
        "kv_cache": _kv_component(manifest),
        "overhead": _overhead_component(manifest),
    }
    if peak is not None:
        components["peak_extra"] = CostComponent(
            mb=peak,
            source="declared",
            detail="[memory].peak_extra_mb — load/warmup transient, added to the high bound only",
        )

    summed = [components[name] for name in ("weights", "kv_cache", "overhead")]
    confidence = _worst_confidence([c.source for c in summed])
    if confidence == "unknown":
        # Fail-closed: one unsourceable component poisons the total. Zeroed
        # bounds, never a partial sum an operator could mistake for the cost.
        return PresetCost(
            preset=preset,
            total_mb_low=0.0,
            total_mb_high=0.0,
            confidence="unknown",
            components=components,
        )

    low = 0.0
    high = 0.0
    for comp in summed:
        assert comp.mb is not None  # confidence != unknown guarantees it
        band = BANDS[comp.source]
        low += comp.mb * (1 - band)
        high += comp.mb * (1 + band)
    if peak is not None:
        # A declared transient allowance, not a steady-state term: widens the
        # high bound only (the low bound describes settled occupancy).
        high += peak

    return PresetCost(
        preset=preset,
        total_mb_low=round(low, 1),
        total_mb_high=round(high, 1),
        confidence=confidence,
        components=components,
    )


def plan_for_preset(preset: str, *, engine: str | None = None) -> PresetCost:
    """Resolve ``preset`` to its manifest and estimate its cost.

    The engine is read from the preset itself unless the caller pins it
    (``--engine``); either way :func:`load_manifest` re-validates that the
    preset's ``name`` matches. Raises FileNotFoundError for an unknown
    preset — callers map it to their own 404/exit-code surface.
    """
    if preset not in list_presets():
        raise FileNotFoundError(f"unknown preset {preset!r}")
    engine_name = engine or preset_summary(preset)["engine"]
    manifest = load_manifest(engine_name, preset=preset)
    source = manifest_source_path(engine_name, preset)
    sha = install_state.manifest_digest(source) if source is not None else None
    return estimate_preset_cost(manifest, preset=preset, manifest_sha256=sha)
