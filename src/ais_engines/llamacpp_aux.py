"""llama.cpp auxiliary driver — restart-only.

Companion to :class:`ais_engines.llamacpp.LlamaCppDriver`: same binary,
different model, different port. Discriminated from the main instance by
the ``process_pattern`` in the manifest (path of the bound GGUF), so
``pkill``/``pgrep`` operations don't cross-contaminate.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class LlamaCppAuxDriver(RestartOnlyDriver):
    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> LlamaCppAuxDriver:
        m = manifest if manifest is not None else load_manifest("llamacpp-aux")
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.llamacpp", class_name="LlamaCppEngine"
        )
        return cls(m, engine)
