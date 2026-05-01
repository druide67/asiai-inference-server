"""asiai-inference-server engine drivers.

Each driver wraps an inference engine (Ollama, LM Studio, oMLX, TurboQuant, ...)
and adds write-side operations on top of asiai's read-only engine adapters:
install, start, stop, restart, unload-with-fallback.
"""
