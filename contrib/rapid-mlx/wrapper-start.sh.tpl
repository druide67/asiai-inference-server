#!/usr/bin/env bash
# =============================================================================
# Rapid-MLX headless server startup script
# Installed at /usr/local/bin/rapid-mlx-server-start from this template.
# =============================================================================
#
# Rapid-MLX (github.com/raullenchai/Rapid-MLX) is a Python entry-point
# distributed via pip / brew. This wrapper:
#   1. Ensures ``~/.cache/huggingface/hub`` exists (MLX engines crash on
#      ``scan_cache_dir()`` when it does not).
#   2. Discovers the ``rapid-mlx`` binary (env override > brew > pip).
#   3. Execs ``rapid-mlx serve <model> --host H --port P``.
#
# Override via environment variables (set in the LaunchDaemon plist):
#   RAPIDMLX_BIN    path to the rapid-mlx binary (skips auto-discovery)
#   RAPIDMLX_MODEL  model alias or HF repo id passed positionally
#   RAPIDMLX_PORT   listening port (default 8004)
#   RAPIDMLX_HOST   bind address (default 0.0.0.0)
#
# Recommended install:  ``brew install raullenchai/rapid-mlx/rapid-mlx``
# Alternative:          ``pip install rapid-mlx`` (then export RAPIDMLX_BIN
#                       to the pip-installed entry point if not in PATH)
# =============================================================================

set -euo pipefail

# HuggingFace cache directory must exist before MLX engines start.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
mkdir -p "${HF_HUB_CACHE}"

# Resolve the rapid-mlx binary: explicit env override wins, otherwise probe
# a short list of conventional install layouts.
BIN="${RAPIDMLX_BIN:-}"
if [[ -z "${BIN}" ]]; then
    for candidate in \
        "/opt/homebrew/bin/rapid-mlx" \
        "/usr/local/bin/rapid-mlx" \
        "${HOME}/.local/bin/rapid-mlx" \
        "${HOME}/.local/share/uv/tools/rapid-mlx/bin/rapid-mlx"
    do
        if [[ -x "${candidate}" ]]; then
            BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${BIN}" || ! -x "${BIN}" ]]; then
    echo "ERROR: rapid-mlx binary not found." >&2
    echo "Hint: install via 'brew install raullenchai/rapid-mlx/rapid-mlx' or set RAPIDMLX_BIN." >&2
    exit 1
fi

MODEL="${RAPIDMLX_MODEL:-qwen3.6-27b-ud}"
PORT="${RAPIDMLX_PORT:-8004}"
HOST="${RAPIDMLX_HOST:-0.0.0.0}"

exec "${BIN}" serve "${MODEL}" \
    --host "${HOST}" \
    --port "${PORT}"
