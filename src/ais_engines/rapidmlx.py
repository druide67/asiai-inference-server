"""Rapid-MLX driver — restart-only Apple-Silicon-optimized MLX inference.

Rapid-MLX (github.com/raullenchai/Rapid-MLX) is a third-party MLX-based
OpenAI-compatible inference server. Key claim validated 2026-05-23 night
on M5 Max + Qwen3.6-27B-UD-MLX-4bit: ``prefix_cache_reuse`` works for
hybrid-attention models (Qwen3.6 DeltaNet + full_attention) via "snapshots
RNN" technique — TTFT prefix-test 17x faster than cold on the bench.

This is currently the **only** engine that handles cross-USER prefix cache
reuse on Qwen3.6 hybrid arch (llama.cpp ``--cache-reuse`` is silently
disabled via ``llama_memory_can_shift=false``, mlx-lm/oMLX/LM-Studio-MLX
all broken on hybrid prefix cache per upstream issues mlx-lm #980, oMLX
#825).

Like mlx-lm, the binary is a Python entry-point; install via
``brew install raullenchai/rapid-mlx/rapid-mlx`` (preferred) or
``pip install rapid-mlx``. Server has no native unload — restart is the
only way to swap models.
"""

from __future__ import annotations

from ais_core.manifest import EngineManifest, load_manifest
from ais_engines.base import RestartOnlyDriver, make_asiai_engine_proxy


class RapidMlxDriver(RestartOnlyDriver):
    """Driver for Rapid-MLX (port 8004 by default), restart-only.

    The asiai engine wrapper at ``asiai/engines/rapidmlx.py`` provides
    read-side introspection (models list, health). Like mlx-lm, no
    ``unload_model`` since Rapid-MLX exposes no such endpoint — bouncing
    the LaunchDaemon is the only way to free VRAM, handled by the parent
    class.
    """

    @classmethod
    def from_manifest(cls, manifest: EngineManifest | None = None) -> RapidMlxDriver:
        m = manifest if manifest is not None else load_manifest("rapidmlx")
        engine = make_asiai_engine_proxy(
            m, module="asiai.engines.rapidmlx", class_name="RapidMlxEngine"
        )
        return cls(m, engine)
