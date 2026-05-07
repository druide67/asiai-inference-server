#!/bin/bash
# =============================================================================
# uninstall.sh — Desinstallation TurboQuant
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib-engine.sh"
engine_load_conf "${SCRIPT_DIR}/engine.conf"

engine_uninstall "$@"

# Nettoyage supplementaire TurboQuant
echo ""
echo "Nettoyage supplementaire (manuel) :"
echo "  sudo rm -f /usr/local/bin/llama-server-turboquant"
echo "  sudo rm -rf /usr/local/src/llama-cpp-turboquant"
echo "  sudo rm -rf /usr/local/share/turboquant"
echo "  # Restaurer la limite GPU si necessaire :"
echo "  sudo sysctl iogpu.wired_limit_mb=0"
