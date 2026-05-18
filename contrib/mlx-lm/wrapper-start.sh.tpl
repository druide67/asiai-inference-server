#!/usr/bin/env bash
# =============================================================================
# mlx-lm headless server startup script
# Installed at /usr/local/bin/mlx-lm-server-start from this template.
# =============================================================================
#
# mlx-lm has no native CLI binary — the OpenAI-compatible server is a
# Python module (``python -m mlx_lm.server``). This wrapper:
#   1. Ensures ``~/.cache/huggingface/hub`` exists (mlx-lm crashes on
#      ``scan_cache_dir()`` when it does not).
#   2. Discovers the venv python (env override > known paths).
#   3. Execs ``python -m mlx_lm.server`` with the configured args.
#
# Override via environment variables (set in the LaunchDaemon plist):
#   MLXLM_PYTHON  path to the venv python (skips auto-discovery)
#   MLXLM_MODEL   model path or HF repo id passed to ``--model``
#   MLXLM_PORT    listening port (default 8000)
#   MLXLM_HOST    bind address (default 0.0.0.0)
#
# Recommended install:  ``uv tool install mlx-lm``
# =============================================================================

set -euo pipefail

# HuggingFace cache directory must exist before mlx-lm starts.
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
mkdir -p "${HF_HUB_CACHE}"

# Resolve the venv python: explicit env override wins, otherwise probe a
# short list of conventional install layouts.
PY="${MLXLM_PYTHON:-}"
if [[ -z "${PY}" ]]; then
    for candidate in \
        "${HOME}/.local/share/uv/tools/mlx-lm/bin/python" \
        "${HOME}/.local/pipx/venvs/mlx-lm/bin/python" \
        "${HOME}/.local/share/pipx/venvs/mlx-lm/bin/python" \
        "${HOME}/.venvs/mlx-lm/bin/python" \
        "/opt/homebrew/bin/python3" \
        "/usr/local/bin/python3"
    do
        if [[ -x "${candidate}" ]]; then
            PY="${candidate}"
            break
        fi
    done
fi

if [[ -z "${PY}" || ! -x "${PY}" ]]; then
    echo "ERROR: no mlx-lm python interpreter found." >&2
    echo "Hint: install via 'uv tool install mlx-lm' or set MLXLM_PYTHON." >&2
    exit 1
fi

# Sanity-check: the chosen interpreter must import mlx_lm.
if ! "${PY}" -c "import mlx_lm" >/dev/null 2>&1; then
    echo "ERROR: interpreter ${PY} cannot import mlx_lm." >&2
    echo "Hint: ensure mlx-lm is installed in that environment." >&2
    exit 1
fi

MODEL_PATH="${MLXLM_MODEL:-${HOME}/llms/mlx/active}"
# Expand a leading ~ to $HOME since launchd does not pre-expand env vars
# and quoted bash parameters skip tilde expansion. mlx-lm receives the
# argument literally and would fail to find the model directory.
MODEL_PATH="${MODEL_PATH/#\~/${HOME}}"
PORT="${MLXLM_PORT:-8000}"
HOST="${MLXLM_HOST:-0.0.0.0}"

exec "${PY}" -m mlx_lm.server \
    --model "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}"
