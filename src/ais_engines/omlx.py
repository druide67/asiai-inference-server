"""oMLX driver — restart-only, no native unload API."""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class OmlxDriver(RestartOnlyDriver):
    """oMLX (port 8800) does not expose a model-unload endpoint as of late 2026.

    The asiai engine wrapper exists (``asiai/engines/omlx.py``) for read-side
    introspection, but ``unload_model`` returns False. The only way to free
    its VRAM is to bounce the LaunchDaemon — which is what the parent class
    does by default.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> OmlxDriver:
        m = manifest if manifest is not None else load_manifest("omlx")
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.omlx", class_name="OmlxEngine"
        )
        return cls(m, engine)
