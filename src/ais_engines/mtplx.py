"""MTPLX driver — restart-only, native MTP speculative decoding on Apple Silicon.

MTPLX (github.com/youssofal/MTPLX) is an MLX-based OpenAI-compatible
inference server built around native multi-token-prediction speculative
decoding: the model's own MTP head drafts tokens verified by the target
in the same pass (no separate draft model). Installed via the brew tap
``youssofal/mtplx/mtplx``; the daemon launches through the stable
``/opt/homebrew/bin/mtplx`` wrapper, which resolves the keg-versioned
virtualenv of the moment and execs ``python -m mtplx.server.openai``
in-place (see the manifest header for the full rationale).

One instance serves one model loaded at launch time; there is no unload
endpoint (``mtplx stop --port N`` stops the whole daemon, which under
launchd supervision is exactly what :func:`ais_core.lifecycle.restart`
does). Bouncing the LaunchDaemon is the only way to free the VRAM —
the ``RestartOnlyDriver`` contract.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class MtplxDriver(RestartOnlyDriver):
    """Driver for MTPLX (port 8005 baseline), restart-only.

    The asiai read-side adapter (``asiai.engines.mtplx``) does not exist
    yet — ``make_asiai_engine_proxy`` degrades to None gracefully, which
    keeps the driver in restart-only mode (its only mode anyway) and
    becomes forward-compatible the day asiai ships the adapter.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> MtplxDriver:
        m = manifest if manifest is not None else load_manifest("mtplx")
        engine = make_asiai_engine_proxy(m, module="asiai.engines.mtplx", class_name="MtplxEngine")
        return cls(m, engine)
