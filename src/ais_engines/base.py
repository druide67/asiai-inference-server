"""EngineDriver — wraps an asiai engine + a manifest with write operations.

Composition over inheritance: a driver does NOT subclass
``asiai.engines.base.InferenceEngine``; it *holds* one (when asiai is
installed and the engine is reachable) and adds write-side operations on
top.

The fallback strategy
---------------------
Every driver implements ``unload_with_fallback(model)`` which:

1. Calls the asiai engine's ``unload_model(model)`` if the engine wraps a
   native API (Ollama ``keep_alive=0``, LM Studio ``lms unload``).
2. If that returns False or no native unload exists, restarts the
   LaunchDaemon — the only way to reclaim VRAM from oMLX, TurboQuant, and
   raw llama.cpp on Apple Silicon.

The result is reported with ``method`` so the CLI/MCP can show the user
exactly what happened: ``"api"`` (clean), ``"restart"`` (full bounce),
or ``"error"``.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging

from ais_core import lifecycle
from ais_core.manifest import EngineManifest

logger = logging.getLogger(__name__)


def make_asiai_engine_proxy(
    manifest: EngineManifest,
    *,
    module: str,
    class_name: str,
) -> object | None:
    """Build the asiai engine wrapper for a driver, or None if unavailable.

    Replaces the four near-identical ``_build_asiai_engine`` helpers that used
    to live in each driver module. None is returned (rather than raising) so
    that a missing or broken asiai install drops the driver to restart-only
    mode instead of breaking the whole CLI.

    Failure paths are *logged*, not silenced (Q4 audit fix): if asiai is
    present but the engine class can't be instantiated, the operator sees
    why their driver fell back to restart-only.
    """
    try:
        engine_module = importlib.import_module(module)
    except ImportError:
        # asiai not installed at all — silent is fine, this is the documented
        # "dev environment without asiai" path.
        return None
    try:
        cls = getattr(engine_module, class_name)
    except AttributeError:
        logger.warning(
            "asiai module %s imported but class %s not found — driver will "
            "use restart-only fallback for unload",
            module,
            class_name,
        )
        return None

    base_url = f"http://127.0.0.1:{manifest.network.port}"
    try:
        return cls(base_url=base_url)
    except TypeError:
        # asiai's constructor signature may have evolved — try the no-arg form.
        try:
            return cls()
        except Exception as exc:
            logger.warning(
                "asiai %s.%s instantiation failed: %s — driver will use "
                "restart-only fallback for unload",
                module,
                class_name,
                exc,
            )
            return None
    except Exception as exc:
        logger.warning(
            "asiai %s.%s base_url constructor failed: %s — falling back to no-arg",
            module,
            class_name,
            exc,
        )
        try:
            return cls()
        except Exception as exc2:
            logger.warning(
                "asiai %s.%s no-arg constructor also failed: %s — driver "
                "will use restart-only fallback for unload",
                module,
                class_name,
                exc2,
            )
            return None


@dataclasses.dataclass(frozen=True)
class UnloadOutcome:
    engine: str
    model: str | None
    method: str  # "api" | "restart" | "error"
    success: bool
    detail: str = ""


class EngineDriver:
    """Default driver: API unload if the wrapped engine offers one, else restart.

    Subclasses override ``_try_native_unload`` to add engine-specific logic
    or to short-circuit the API path entirely (oMLX/TurboQuant don't have
    one).
    """

    def __init__(self, manifest: EngineManifest, asiai_engine: object | None = None):
        self.manifest = manifest
        self.asiai_engine = asiai_engine

    @property
    def name(self) -> str:
        return self.manifest.name

    # -- public API ----------------------------------------------------------

    def unload(self, model: str | None) -> UnloadOutcome:
        """Try native unload first, fall back to a full daemon restart.

        Pass ``model=None`` to force a daemon restart (effectively unloads
        everything). Useful for pre-bench cleanup when the caller doesn't
        track which models are loaded.
        """
        if model is not None:
            try:
                ok = self._try_native_unload(model)
            except Exception as e:
                return UnloadOutcome(
                    engine=self.name,
                    model=model,
                    method="error",
                    success=False,
                    detail=f"native unload raised: {e}",
                )
            if ok:
                return UnloadOutcome(
                    engine=self.name,
                    model=model,
                    method="api",
                    success=True,
                    detail=f"unloaded {model} via native API",
                )

        # Fallback: full daemon bounce.
        try:
            lifecycle.restart(self.manifest)
        except Exception as e:
            return UnloadOutcome(
                engine=self.name,
                model=model,
                method="error",
                success=False,
                detail=f"restart failed: {e}",
            )
        return UnloadOutcome(
            engine=self.name,
            model=model,
            method="restart",
            success=True,
            detail=f"restarted {self.manifest.plist.name}",
        )

    def list_loaded_models(self) -> list[str]:
        """Best-effort introspection via the wrapped asiai engine."""
        if self.asiai_engine is None or not hasattr(self.asiai_engine, "list_running"):
            return []
        try:
            return [m.name for m in self.asiai_engine.list_running()]
        except Exception as exc:
            logger.debug("list_running failed for %s: %s", self.name, exc)
            return []

    # -- override hooks ------------------------------------------------------

    def _try_native_unload(self, model: str) -> bool:
        """Return True iff the model was unloaded via a native API.

        Default implementation delegates to the wrapped asiai engine's
        ``unload_model``. Subclasses for engines without a native API
        (oMLX, TurboQuant, raw llama.cpp) override this to return False
        unconditionally — the lifecycle restart will run instead.
        """
        if self.asiai_engine is None:
            return False
        unload = getattr(self.asiai_engine, "unload_model", None)
        if unload is None:
            return False
        return bool(unload(model))


class RestartOnlyDriver(EngineDriver):
    """Driver for engines without any native unload API.

    Concretely: oMLX, TurboQuant, plain llama.cpp, mlx-lm, vllm-mlx, Exo.
    A full daemon restart is the only way to reclaim VRAM from these.
    """

    def _try_native_unload(self, model: str) -> bool:
        return False
