"""vMLX driver — restart-only, no native unload API.

vMLX (https://vmlx.net) is a MLX-native OpenAI-compatible inference server
with first-class Mamba/SSM hybrid architecture support. As of late 2026
it does not expose a model-unload endpoint, so freeing VRAM requires
bouncing the LaunchDaemon (parent class default).
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class VmlxDriver(RestartOnlyDriver):
    """Driver for vMLX (port 8003, offset from upstream default 8000 to
    avoid collisions with mlx-lm and vllm-mlx), restart-only."""

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> VmlxDriver:
        m = manifest if manifest is not None else load_manifest("vmlx")
        engine = make_asiai_engine_proxy(m, module="asiai.engines.vmlx", class_name="VmlxEngine")
        return cls(m, engine)
