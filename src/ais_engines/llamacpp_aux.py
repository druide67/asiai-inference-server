"""llama.cpp auxiliary driver — restart-only family.

A single ``LlamaCppAuxDriver`` class powers the whole ``llamacpp-aux-*``
family of engine instances. Each instance is a regular ``llama-server``
process bound to its own port and model, discriminated from the main
``llamacpp`` (8080) and from sibling auxiliary instances by the model
path declared in the manifest (``llms/gguf/auxN/active.gguf``), so
``pgrep``/``pkill`` operations never cross-contaminate.

How instances are declared
--------------------------
Adding a new auxiliary slot is a config-only change:

1. Drop a manifest TOML in ``data/engine_manifests/llamacpp-aux-N.toml``
   (or, for a user-local instance never to be committed to OSS, in
   ``$XDG_CONFIG_HOME/asiai-inference-server/engine_manifests/``).
2. Pick a free port and a distinct ``llms/gguf/auxN/`` segment in
   ``process_pattern`` and ``model_path``.

No Python code change is required: the CLI discovers the manifest by
name and dispatches to this driver via the family pattern registered in
``ais_cli.commands``.

The bundled package ships ``llamacpp-aux-1`` through ``llamacpp-aux-5``
as a complete example stack on dedicated ports 8090-8094, paired with
example presets for representative model sizes (sub-1B title generation
through a vision-enabled 7B, plus a dedicated long-context slot). The
example presets target an external
orchestrator use case (see ``data/engine_manifests/presets/README.md``)
and exist as starting points to copy or override, not as a fixed product
configuration.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class LlamaCppAuxDriver(RestartOnlyDriver):
    """One driver class, N instances at runtime — parameterised by manifest name."""

    @classmethod
    def from_manifest_name(cls, name: str) -> LlamaCppAuxDriver:
        m = load_manifest(name)
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.llamacpp", class_name="LlamaCppEngine"
        )
        return cls(m, engine)

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> LlamaCppAuxDriver:
        """Adapter for the standard ``from_manifest`` factory contract.

        Most callers go through ``from_manifest_name(name)`` because the
        ``llamacpp-aux-*`` family has multiple instances and the call site
        knows which one it wants. ``from_manifest`` is kept for symmetry
        with the single-instance drivers (Ollama, LM Studio, ...) and
        requires the manifest to be passed explicitly: there is no default
        instance to fall back to.
        """
        if manifest is None:
            raise ValueError(
                "LlamaCppAuxDriver.from_manifest requires an explicit manifest "
                "(the aux family has multiple instances — use from_manifest_name(name) "
                "or pass a loaded EngineManifest)."
            )
        engine = make_asiai_engine_proxy(
            manifest, module="asiai.engines.llamacpp", class_name="LlamaCppEngine"
        )
        return cls(manifest, engine)
