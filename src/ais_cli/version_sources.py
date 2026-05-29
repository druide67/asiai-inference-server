"""asiai ``version_sources`` provider.

Exposed through the ``asiai.version_sources`` entry-point group. When asiai
runs ``asiai versions`` (or the doctor recap, or the web page) it discovers
this callable and merges the returned table over its internal fallback —
this is how the engines aisrv manages (notably ``turboquant``) and the
authoritative brew-formula mapping reach asiai without either package
importing the other.

Contract: ``provide(api_version: int) -> list[dict]``. Returns plain dicts
keyed by ``EngineVersionSpec`` field names (asiai rebuilds its own dataclass,
ignoring unknown keys). The version is validated against the asiai-offered
``api_version``, mirroring ``as_subcmd.validate_asiai_compat``.
"""

from __future__ import annotations

import re

# Major version of the asiai version-source contract this provider implements.
VERSION_SOURCE_API_VERSION = 1

# llamacpp-aux-N instances share the llama.cpp binary — they carry no
# distinct version, so we don't emit a row per instance (it would just
# duplicate the llamacpp row). The base ``llamacpp`` entry covers them.
_AUX_RE = re.compile(r"-aux-\d+$")

# Manifest stem -> asiai canonical engine key (engine_map in asiai.cli).
# Only stems that differ from asiai's key need an entry.
_CANONICAL = {
    "mlx-lm": "mlxlm",
}

# Upstream GitHub repos for the OSS engines aisrv knows about. Used only as
# a changelog link / optional upstream source; brew stays authoritative for
# brew-backed engines.
_GITHUB = {
    "ollama": "ollama/ollama",
    "llamacpp": "ggml-org/llama.cpp",
    "mlx-lm": "ml-explore/mlx-lm",
}


class IncompatibleVersionSourceError(RuntimeError):
    """Raised when asiai's version-source API version disagrees with ours."""


def _validate(api_version: int) -> None:
    if api_version != VERSION_SOURCE_API_VERSION:
        raise IncompatibleVersionSourceError(
            f"asiai-inference-server version_sources expects "
            f"api_version={VERSION_SOURCE_API_VERSION}, asiai offered {api_version}."
        )


def _canonical(stem: str) -> str:
    return _CANONICAL.get(stem, stem)


def provide(api_version: int) -> list[dict]:
    """Return the engine→version-source mapping aisrv contributes to asiai."""
    _validate(api_version)

    # Imported lazily so a failure inside aisrv internals degrades to "no
    # provider" on the asiai side rather than an import-time crash.
    from ais_core.manifest import list_manifests, load_manifest
    from ais_core.upgrade import UPGRADE_FORMULAS

    out: list[dict] = []
    for stem in list_manifests():
        if _AUX_RE.search(stem):
            continue  # derived instance of a base engine

        try:
            m = load_manifest(stem)
            display = m.display
            builds_from_source = bool(getattr(m.binary, "builds_from_source", False))
        except Exception:
            display = stem
            builds_from_source = False

        entry: dict = {
            "engine_name": _canonical(stem),
            "display": display,
            "version_scheme": "llamacpp_build" if stem.startswith("llamacpp") else "semver",
        }

        formula = UPGRADE_FORMULAS.get(stem)
        if formula:
            entry["brew_formula"] = formula
        if builds_from_source:
            # pip-installed engine; the manifest stem is the pip distribution
            # name (mlx-lm, omlx, vmlx). brew (if also present) wins at
            # resolution time — listing both is harmless.
            entry["pip_package"] = stem
        if stem in _GITHUB:
            entry["github_repo"] = _GITHUB[stem]
        if stem == "lmstudio":
            entry["no_upstream"] = True

        out.append(entry)

    return out
