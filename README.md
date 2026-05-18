# asiai-inference-server

Fleet manager for local LLM inference engines on Apple Silicon.

> **Status: v0.0.1 pre-alpha — skeleton only.** Not yet functional. See the
> roadmap below.

`asiai-inference-server` is the **control plane** companion to
[`asiai`](https://asiai.dev) (the observability/benchmark CLI). Where `asiai`
*observes* what's running on your Mac, this project *manages* it: install,
start, stop, unload, and orchestrate inference engines (llama.cpp,
Ollama, LM Studio, oMLX, TurboQuant, mlx-lm, vMLX, …) across one or
several Apple Silicon machines.

It also fixes the long-standing macOS pain point: **engine memory that never
gets freed** because of the unified-memory compressor. Killing a process
doesn't release the VRAM. This tool combines per-engine unload APIs, full
LaunchDaemon restart, and `sudo purge` to reclaim memory deterministically —
and reports the actual delta measured, not a marketing promise.

## Why

After a year of running multi-engine LLM inference on Apple Silicon
(MacBook M1 Max, Mac Mini M4 Pro, MacBook M5 Max), the operational gap
became obvious:

- **Install/uninstall** an engine should not require chasing brews, plists
  and firewall rules across READMEs.
- **Switching profiles** ("coding agent on Qwen-Coder 32B" → "70B chat on
  TurboQuant") should be a single command, not five.
- **Memory unload** should actually free the VRAM, not let the macOS
  compressor sit on it.
- **A fleet of Macs** should be a single dashboard, not three SSH sessions.
- **AI agents** (Claude Code, Cursor, Windsurf) should be able to manage
  the fleet via MCP, not just observe it.

`asiai-inference-server` ships these one at a time, building on the
`asiai` observability stack.

## Roadmap

| Version | Scope | Status |
|---------|-------|--------|
| v0.0 | Repo skeleton + packaging | in progress |
| v0.1 | Install/uninstall/start/stop + **unload + purge memory** | next |
| v0.2 | Profile switching (TOML profiles, apply/rollback) | planned |
| v0.3 | Fleet manager (multi-Mac inventory, SSH dispatch) | planned |
| v0.4 | Web cockpit + optional HTTP agent | planned |
| v1.0 | MCP write tools + PyPI/Homebrew release | planned |

## Architecture (high level)

- **CLI**: `aisctl <command>` (standalone) or `asiai engine <command>`
  (auto-injected sub-CLI when `asiai-inference-server` is installed
  alongside `asiai`).
- **Python stdlib only** for the core (cohérent avec asiai). Optional
  extras: `mcp` (for v1.0 write tools).
- **macOS Apple Silicon only**. We rely on `launchctl`, `vm_stat`,
  `sudo purge`, `pfctl`, `iogpu.wired_limit_mb`.
- **SSH-first** for fleet operations. Optional HTTP agent in v0.4 for
  agent-to-agent orchestration.
- **TOML** for human-edited files (engine manifests, profiles, fleet
  inventory). JSON for runtime state.

The full design rationale (architecture diagram, sequencing, file map,
risk mitigations) lives in the validated plan at
`~/.claude/plans/iterative-wiggling-crystal.md`.

## License

Apache-2.0 © 2026 Jean-Marc Nahlovsky
