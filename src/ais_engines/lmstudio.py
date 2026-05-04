"""LM Studio driver — native unload via the ``lms unload`` CLI."""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import EngineDriver, make_asiai_engine_proxy


class LMStudioDriver(EngineDriver):
    """asiai's ``LMStudioEngine.unload_model()`` shells out to ``lms unload --yes``.

    See ``asiai/engines/lmstudio.py:121-139``. The CLI is the only published
    way to unload an LM Studio model — there's no HTTP endpoint as of v0.4.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> LMStudioDriver:
        m = manifest if manifest is not None else load_manifest("lmstudio")
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.lmstudio", class_name="LMStudioEngine"
        )
        return cls(m, engine)
