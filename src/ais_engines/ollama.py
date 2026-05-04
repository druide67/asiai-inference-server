"""Ollama driver — uses the asiai engine's native API unload (keep_alive=0)."""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import EngineDriver, make_asiai_engine_proxy


class OllamaDriver(EngineDriver):
    """No customization needed: asiai's OllamaEngine.unload_model() does the right thing.

    POST /api/generate with ``keep_alive=0`` causes Ollama to evict the model
    from its keep-alive pool immediately. Verified in
    ``asiai/engines/ollama.py:107-119``.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> OllamaDriver:
        m = manifest if manifest is not None else load_manifest("ollama")
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.ollama", class_name="OllamaEngine"
        )
        return cls(m, engine)
