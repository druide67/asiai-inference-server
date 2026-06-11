# TurboQuant — llama.cpp fork with turbo KV cache

Fork of llama.cpp by [TheTom](https://github.com/TheTom/llama-cpp-turboquant)
that compresses the KV cache with asymmetric quantization.

## How it works

The standard KV cache is a major VRAM consumer at long context.
TurboQuant applies differentiated quantization:

- **Keys**: `q8_0` (high precision — keys are sensitive)
- **Values**: `turbo3` (5x compression — values tolerate it better),
  or `turbo2` for a more conservative trade-off

Result: 70B models at 32K context fit in 64 GB of unified memory.

## Installation with aisctl

TurboQuant is **built from source** (the fork is not distributed via
brew). Build it once, then let `aisctl` manage the daemon:

```bash
# 1. Build the fork (cmake + Metal required)
git clone -b feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git
cmake -B build -DGGML_METAL=ON && cmake --build build -j
cp build/bin/llama-server /usr/local/bin/llama-server-turboquant

# 2. Provision the LaunchDaemon (wrapper script + plist + optional pf anchor)
aisctl install turboquant
```

The bundled `turboquant` manifest expects the binary at
`/usr/local/bin/llama-server-turboquant` (a distinct name, so it never
conflicts with a brew-installed llama.cpp). The install renders the
multi-step wrapper from `wrapper-start.sh.tpl` — it raises
`sysctl iogpu.wired_limit_mb` before starting the server, which models
above ~50 GB of VRAM need.

## Management

```bash
aisctl status turboquant          # state (+ --deep for a generation probe)
aisctl start turboquant
aisctl stop turboquant
aisctl restart turboquant
aisctl uninstall turboquant
```

Logs land in `~/Library/Logs/asiai/turboquant/`.

## Model profiles

Example profiles live in `configs/`:

| File | Model | VRAM | Status |
|------|-------|------|--------|
| `llama-70b-turbo3.conf` | Llama 3.1 70B Q4_K_M | ~42 GB | OK |
| `qwen35-turbo3.conf` | Qwen 3.5 35B-A3B | ~30 GB | NOT WORKING (rope bug) |

## Known issues

- **Qwen 3.5 MoE**: `rope.dimension_sections` error (expects 4, gets 3).
  The fork does not handle MoE architectures on this branch.
- **llama3 chat template**: produces garbled output. Use `chatml` instead.
- **Homebrew PATH over SSH**: `/opt/homebrew/bin` is not in the default
  SSH PATH; use full paths in remote commands.

## Legacy scripts

`setup.sh`, `uninstall.sh` and `engine.conf` in this directory are the
original bash tooling this engine was ported from. They depend on a
`lib-engine.sh` library that is **not** part of this repository and are
kept for reference only — use the `aisctl` flow above.

## System prerequisites

- macOS on Apple Silicon (Metal required)
- cmake (`brew install cmake`)
- `sysctl iogpu.wired_limit_mb` for models above ~50 GB of VRAM
  (handled by the wrapper)
- Enough unified memory (64 GB recommended for the 70B)
