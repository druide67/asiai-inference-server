"""mlx-lm driver — restart-only Apple-maintained MLX inference.

mlx-lm exposes ``python -m mlx_lm.server`` as an OpenAI-compatible API.
The server has no unload endpoint — restarting the daemon is the only
way to swap models. Apple-maintained, but the server's own documentation
warns ``not recommended for production``: it implements basic security
checks only. Operators are expected to front it with a reverse proxy
(bearer auth, rate-limit, audit) when exposing beyond LAN.

TODO(security): wire up a reverse-proxy auth profile as a follow-up
user story. The MVP driver assumes LAN-only deployment behind a pf
anchor; bearer-token auth is not implemented here.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class MlxLmDriver(RestartOnlyDriver):
    """Driver for mlx-lm (port 8000 by default), restart-only.

    The asiai engine wrapper at ``asiai/engines/mlxlm.py`` provides
    read-side introspection (models list, health). It has no
    ``unload_model`` because mlx-lm exposes no such endpoint — the only
    way to free VRAM is bouncing the LaunchDaemon, which the parent
    class handles.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> MlxLmDriver:
        m = manifest if manifest is not None else load_manifest("mlx-lm")
        engine = make_asiai_engine_proxy(m, module="asiai.engines.mlxlm", class_name="MlxLmEngine")
        return cls(m, engine)
