"""llama.cpp driver — restart-only.

llama-server (upstream llama.cpp binary) does not expose a model-unload
endpoint as of late 2026: each instance is bound to a single model file
at launch via ``--model <path>``. Reclaiming VRAM requires bouncing the
LaunchDaemon — which is what :class:`RestartOnlyDriver` does by default.

Read-side metadata (``/health``, ``/props``, ``/v1/models``, ``/metrics``)
is exposed by asiai's :class:`asiai.engines.llamacpp.LlamaCppEngine`,
wired in through :func:`make_asiai_engine_proxy` so ``aisctl status`` can
introspect the running model and context length.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class LlamaCppDriver(RestartOnlyDriver):
    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> LlamaCppDriver:
        m = manifest if manifest is not None else load_manifest("llamacpp")
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.llamacpp", class_name="LlamaCppEngine"
        )
        return cls(m, engine)
