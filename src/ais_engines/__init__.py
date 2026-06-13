"""asiai-inference-server engine drivers.

Each driver wraps an inference engine (Ollama, LM Studio, oMLX, TurboQuant, ...)
and adds unload-with-fallback (native unload API where the engine has one,
full daemon restart otherwise) plus best-effort loaded-model introspection,
on top of asiai's read-only engine adapters. Lifecycle operations (install,
start, stop, restart) are manifest-driven and live in ``ais_core.lifecycle``,
not in the drivers.
"""
