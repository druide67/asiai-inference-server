"""Homebrew upgrade whitelist — single source of truth.

Both the loopback command server (``ais_cli.serve``) and the native
``aisctl upgrade`` subcommand (``ais_cli.commands``) must agree on exactly
which formulas an upgrade may target. Keeping the whitelist in one place
(here, in ``ais_core``) prevents the two call sites from drifting — a
drift would be a real security bug, since the whitelist is what stops an
attacker who bypasses the engine-name regex from passing ``coreutils`` or
a tap with arbitrary post-install hooks.
"""

from __future__ import annotations

import shutil

# Engine name -> Homebrew formula. Anything not here is rejected before any
# subprocess is spawned.
UPGRADE_FORMULAS: dict[str, str] = {
    "ollama": "ollama",
    "llamacpp": "llama.cpp",
    "lmstudio": "lm-studio",
    "rapidmlx": "rapid-mlx",
    "turboquant": "turboquant",
}


def upgrade_argv(engine: str) -> list[str]:
    """Return the ``brew upgrade <formula>`` argv for *engine*.

    Raises ValueError if the engine is not in the whitelist.
    """
    formula = UPGRADE_FORMULAS.get(engine)
    if not formula:
        raise ValueError(
            f"upgrade is not whitelisted for engine '{engine}' "
            f"(allowed: {', '.join(sorted(UPGRADE_FORMULAS))})"
        )
    brew = shutil.which("brew") or "/opt/homebrew/bin/brew"
    return [brew, "upgrade", formula]
