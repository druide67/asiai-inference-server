"""LM Studio driver — native unload via the ``lms unload`` CLI."""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import EngineDriver


class LMStudioDriver(EngineDriver):
    """asiai's ``LMStudioEngine.unload_model()`` shells out to ``lms unload --yes``.

    See ``asiai/engines/lmstudio.py:121-139``. The CLI is the only published
    way to unload an LM Studio model — there's no HTTP endpoint as of v0.4.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> LMStudioDriver:
        m = manifest if manifest is not None else load_manifest("lmstudio")
        engine = _build_asiai_engine(m)
        return cls(m, engine)


def _build_asiai_engine(manifest: EngineManifest) -> object | None:
    try:
        from asiai.engines.lmstudio import LMStudioEngine  # type: ignore[import-not-found]
    except ImportError:
        return None
    base_url = f"http://127.0.0.1:{manifest.network.port}"
    try:
        return LMStudioEngine(base_url=base_url)
    except TypeError:
        try:
            return LMStudioEngine()
        except Exception:
            return None
