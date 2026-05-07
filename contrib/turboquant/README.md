# TurboQuant — llama.cpp fork avec KV cache turbo

Fork de llama.cpp par [TheTom](https://github.com/TheTom/llama-cpp-turboquant) qui compresse le KV cache via quantization asymetrique.

## Principe

Le KV cache standard consomme beaucoup de VRAM. TurboQuant applique une quantization differenciee :
- **Keys** : `q8_0` (haute precision, les keys sont sensibles)
- **Values** : `turbo3` (compression 5x, les values sont plus tolerantes)

Resultat : modeles 70B a context 32k qui tiennent dans 64 Go de RAM unifiee.

## Installation

```bash
# Depuis le MacBook (via admin CLI)
./admin/oc engine setup turboquant --firewall --yes

# Avec telechargement d'un modele
./admin/oc engine setup turboquant --firewall --yes \
  --model https://huggingface.co/bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf

# Avec symlink depuis Ollama
./admin/oc engine setup turboquant --firewall --yes \
  --model-from-ollama qwen3.5:35b-a3b
```

## Profils modele

Les configs sont dans `configs/` :

| Fichier | Modele | VRAM | Status |
|---------|--------|------|--------|
| `llama-70b-turbo3.conf` | Llama 3.1 70B Q4_K_M | ~42 Go | OK |
| `qwen35-turbo3.conf` | Qwen 3.5 35B-A3B | ~30 Go | NON FONCTIONNEL (rope bug) |

Pour changer de profil : editer `TURBOQUANT_CONFIG` dans le plist ou passer `--config <file>` au setup.

## Gestion

```bash
./admin/oc engine status                   # Etat de tous les engines
./admin/oc engine health turboquant        # Health check HTTP
./admin/oc engine start turboquant         # Demarrer
./admin/oc engine stop turboquant          # Arreter
./admin/oc engine logs turboquant          # Logs stdout
./admin/oc engine logs turboquant --err    # Logs stderr
```

## Problemes connus

- **Qwen 3.5 MoE** : erreur `rope.dimension_sections` (attend 4, recoit 3). Le fork ne gere pas les architectures MoE.
- **Template llama3** : produit du texte aleatoire. Utiliser `chatml` a la place.
- **PATH Homebrew SSH** : `/opt/homebrew/bin` n'est pas dans le PATH SSH par defaut. Le setup utilise le chemin complet.

## Pre-requis systeme

- macOS avec Apple Silicon (Metal obligatoire)
- cmake (`brew install cmake`)
- `sysctl iogpu.wired_limit_mb` pour les modeles > 50 Go VRAM
- Suffisamment de RAM unifiee (64 Go recommande pour le 70B)
