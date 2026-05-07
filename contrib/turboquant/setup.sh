#!/bin/bash
# =============================================================================
# setup.sh — Installation TurboQuant (llama.cpp fork) en serveur headless
# Utilise lib-engine.sh pour la logique commune.
#
# TurboQuant necessite :
#   1. Compilation from source (cmake + Metal)
#   2. sysctl iogpu.wired_limit_mb pour lever la limite GPU
#   3. Wrapper script (args complexes : KV cache turbo, flash attention, etc.)
#
# Options supplementaires :
#   --model URL         Telecharger un GGUF depuis HuggingFace
#   --model-from-ollama NAME  Symlink depuis un blob Ollama existant
#   --config FILE       Profil modele (dans configs/, default: llama-70b-turbo3.conf)
#   --rebuild           Forcer la recompilation meme si le binaire existe
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib-engine.sh"
engine_load_conf "${SCRIPT_DIR}/engine.conf"

# --- Variables TurboQuant-specifiques ---

TURBOQUANT_REPO="https://github.com/TheTom/llama-cpp-turboquant.git"
TURBOQUANT_BRANCH="feature/turboquant-kv-cache"
TURBOQUANT_SRC_DIR="/usr/local/src/llama-cpp-turboquant"
TURBOQUANT_MODELS_DIR="/usr/local/share/turboquant/models"
TURBOQUANT_BIN="/usr/local/bin/llama-server-turboquant"

MODEL_URL=""
MODEL_FROM_OLLAMA=""
MODEL_CONFIG="llama-70b-turbo3.conf"
FORCE_REBUILD=false

# --- Parsing supplementaire ---

_remaining_args=()
for arg in "$@"; do
    if [[ "${_next_is:-}" == "model" ]]; then
        MODEL_URL="$arg"; _next_is=""; continue
    fi
    if [[ "${_next_is:-}" == "ollama" ]]; then
        MODEL_FROM_OLLAMA="$arg"; _next_is=""; continue
    fi
    if [[ "${_next_is:-}" == "config" ]]; then
        MODEL_CONFIG="$arg"; _next_is=""; continue
    fi
    case "$arg" in
        --model)            _next_is="model" ;;
        --model-from-ollama) _next_is="ollama" ;;
        --config)           _next_is="config" ;;
        --rebuild)          FORCE_REBUILD=true ;;
        *)                  _remaining_args+=("$arg") ;;
    esac
done

# --- Hooks TurboQuant-specifiques ---

engine_pre_setup() {
    log_step "Pre-requis TurboQuant"

    # 1. Verifier cmake
    if ! command -v cmake &>/dev/null; then
        if command -v /opt/homebrew/bin/brew &>/dev/null; then
            log_info "Installation de cmake via Homebrew..."
            [[ "$DRY_RUN" == "false" ]] && /opt/homebrew/bin/brew install cmake
        else
            log_error "cmake requis. Installez via: brew install cmake"
            exit 1
        fi
    fi
    check_pass "cmake disponible"

    # 2. Cloner/mettre a jour le repo source
    if [[ -d "$TURBOQUANT_SRC_DIR/.git" ]]; then
        log_info "Mise a jour du repo TurboQuant..."
        if [[ "$DRY_RUN" == "false" ]]; then
            cd "$TURBOQUANT_SRC_DIR"
            git fetch origin
            git checkout "$TURBOQUANT_BRANCH"
            git pull --ff-only origin "$TURBOQUANT_BRANCH"
        fi
        check_pass "Repo mis a jour: $TURBOQUANT_SRC_DIR"
    else
        log_info "Clonage du repo TurboQuant..."
        if [[ "$DRY_RUN" == "false" ]]; then
            sudo mkdir -p "$(dirname "$TURBOQUANT_SRC_DIR")"
            sudo git clone "$TURBOQUANT_REPO" "$TURBOQUANT_SRC_DIR"
            cd "$TURBOQUANT_SRC_DIR"
            git checkout "$TURBOQUANT_BRANCH"
        fi
        check_pass "Repo clone: $TURBOQUANT_SRC_DIR"
    fi

    # 3. Compiler (si binaire absent ou --rebuild)
    if [[ ! -x "$TURBOQUANT_BIN" ]] || [[ "$FORCE_REBUILD" == "true" ]]; then
        log_step "Compilation TurboQuant (Metal + Release)"
        if [[ "$DRY_RUN" == "false" ]]; then
            cd "$TURBOQUANT_SRC_DIR"
            cmake -B build \
                -DGGML_METAL=ON \
                -DGGML_METAL_EMBED_LIBRARY=ON \
                -DCMAKE_BUILD_TYPE=Release
            cmake --build build -j"$(sysctl -n hw.ncpu)"

            # Installer sous un nom unique
            sudo cp build/bin/llama-server "$TURBOQUANT_BIN"
            sudo chmod 755 "$TURBOQUANT_BIN"
        fi
        check_pass "Binaire compile: $TURBOQUANT_BIN"
    else
        check_pass "Binaire deja present: $TURBOQUANT_BIN"
    fi

    # 4. Repertoire modeles
    if [[ "$DRY_RUN" == "false" ]]; then
        sudo mkdir -p "$TURBOQUANT_MODELS_DIR"
        sudo chown "${ENGINE_USER}:staff" "$TURBOQUANT_MODELS_DIR"
    fi
    check_pass "Repertoire modeles: $TURBOQUANT_MODELS_DIR"

    # 5. Telecharger un modele si demande
    if [[ -n "$MODEL_URL" ]]; then
        local model_name
        model_name=$(basename "$MODEL_URL")
        local model_path="${TURBOQUANT_MODELS_DIR}/${model_name}"
        if [[ -f "$model_path" ]]; then
            log_info "Modele deja present: $model_path"
        else
            log_step "Telechargement du modele..."
            if [[ "$DRY_RUN" == "false" ]]; then
                curl -L --progress-bar -o "$model_path" "$MODEL_URL"
                sudo chown "${ENGINE_USER}:staff" "$model_path"
            fi
            check_pass "Modele telecharge: $model_path"
        fi
    fi

    # 6. Symlink depuis Ollama si demande
    if [[ -n "$MODEL_FROM_OLLAMA" ]]; then
        log_step "Resolution du blob Ollama: $MODEL_FROM_OLLAMA"
        if [[ "$DRY_RUN" == "false" ]]; then
            local ollama_manifest
            local model_tag="${MODEL_FROM_OLLAMA##*:}"
            local model_base="${MODEL_FROM_OLLAMA%%:*}"
            ollama_manifest="$HOME/.ollama/models/manifests/registry.ollama.ai/library/${model_base}/${model_tag}"

            if [[ -f "$ollama_manifest" ]]; then
                local blob_sha
                blob_sha=$(python3 -c "
import json
with open('$ollama_manifest') as f:
    m = json.load(f)
for layer in m['layers']:
    if layer['mediaType'].endswith('.model'):
        print(layer['digest'].replace(':', '-'))
        break
")
                local blob_path="$HOME/.ollama/models/blobs/${blob_sha}"
                if [[ -f "$blob_path" ]]; then
                    local link_name="${model_base}-${model_tag}.gguf"
                    ln -sf "$blob_path" "${TURBOQUANT_MODELS_DIR}/${link_name}"
                    check_pass "Symlink Ollama: ${TURBOQUANT_MODELS_DIR}/${link_name}"
                else
                    log_warn "Blob Ollama introuvable: $blob_path"
                fi
            else
                log_warn "Manifest Ollama introuvable: $ollama_manifest"
            fi
        fi
    fi

    # 7. Appliquer sysctl GPU memory (persistant via wrapper, applique ici pour le setup)
    log_info "Configuration GPU memory limit..."
    if [[ "$DRY_RUN" == "false" ]]; then
        sudo sysctl iogpu.wired_limit_mb=61440 2>/dev/null || log_warn "sysctl iogpu.wired_limit_mb non supporte sur ce systeme"
    fi
}

# Arguments supplementaires pour le plist : pas utilises car tout est dans le wrapper
engine_extra_args() {
    echo ""
}

# Infos supplementaires dans le resume
engine_print_extra() {
    local config_path="${SCRIPT_DIR}/configs/${MODEL_CONFIG}"
    echo "  # Config active: ${MODEL_CONFIG}"
    if [[ -f "$config_path" ]]; then
        echo "  # $(grep '^MODEL_PATH=' "$config_path" 2>/dev/null || echo "MODEL_PATH non defini")"
        echo "  # $(grep '^CACHE_TYPE_V=' "$config_path" 2>/dev/null || echo "CACHE_TYPE_V non defini")"
    fi
    echo ""
    echo "  # Modeles: $TURBOQUANT_MODELS_DIR"
    echo "  # Sources: $TURBOQUANT_SRC_DIR"
    echo "  # Binaire: $TURBOQUANT_BIN"
    echo ""
    echo "  # Problemes connus:"
    echo "  #   - Qwen 3.5 MoE: rope.dimension_sections incompatible (attend 4, recoit 3)"
    echo "  #   - Template llama3 buggue dans ce fork, utiliser chatml"
    echo ""
    echo "  # Changer de modele:"
    echo "  sudo $WRAPPER_INSTALL_PATH  # (modifier le .conf dans configs/)"
    echo ""
}

engine_setup "${_remaining_args[@]}"
