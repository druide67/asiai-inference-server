"""Ollama driver — uses the asiai engine's native API unload (keep_alive=0)."""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import EngineDriver


class OllamaDriver(EngineDriver):
    """No customization needed: asiai's OllamaEngine.unload_model() does the right thing.

    POST /api/generate with ``keep_alive=0`` causes Ollama to evict the model
    from its keep-alive pool immediately. Verified in
    ``asiai/engines/ollama.py:107-119``.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> OllamaDriver:
        m = manifest if manifest is not None else load_manifest("ollama")
        engine = _build_asiai_engine(m)
        return cls(m, engine)


def _build_asiai_engine(manifest: EngineManifest) -> object | None:
    """Construct the asiai engine wrapper if asiai is importable.

    We import lazily so a user who installs asiai-inference-server in an
    environment without asiai (rare but possible during dev) still gets
    restart-fallback unload, just no API path.
    """
    try:
        from asiai.engines.ollama import OllamaEngine  # type: ignore[import-not-found]
    except ImportError:
        return None
    base_url = f"http://127.0.0.1:{manifest.network.port}"
    try:
        return OllamaEngine(base_url=base_url)
    except TypeError:
        # asiai's constructor may evolve; fall back to the no-arg form.
        try:
            return OllamaEngine()
        except Exception:
            return None
