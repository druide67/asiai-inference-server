"""TurboQuant driver — restart-only.

TurboQuant is a llama.cpp fork with custom KV-cache compression. It exposes
the upstream ``/v1/*`` OpenAI-compatible endpoints but no model-unload API
(llama-server doesn't have one upstream either). VRAM is reclaimed only by
process exit.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver


class TurboquantDriver(RestartOnlyDriver):
    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> TurboquantDriver:
        m = manifest if manifest is not None else load_manifest("turboquant")
        # No asiai engine wrapper for turboquant yet; the read-side path goes
        # through the generic OpenAI-compatible probe in lifecycle.probe_health.
        return cls(m, asiai_engine=None)
