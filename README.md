# asiai-inference-server

Fleet manager for local LLM inference engines on Apple Silicon.

> **Status: v0.3.0 — released on
> [PyPI](https://pypi.org/project/asiai-inference-server/).** Lifecycle,
> memory reclaim, presets, upgrades, fleet push and health probing are
> functional. See the roadmap below for what each version shipped.

`asiai-inference-server` is the **control plane** companion to
[`asiai`](https://asiai.dev) (the observability/benchmark CLI). Where `asiai`
*observes* what's running on your Mac, this project *manages* it: install,
start, stop, unload, and orchestrate inference engines (llama.cpp,
Ollama, LM Studio, oMLX, TurboQuant, mlx-lm, vMLX, Rapid-MLX, MTPLX, …)
across one or several Apple Silicon machines.

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
| v0.1 | Install/uninstall/start/stop/restart + **unload + purge memory**, pf firewall anchor, sudoers bootstrap | shipped |
| v0.2 | `aisctl upgrade` (brew, whitelisted), `asiai versions` provider, fleet Phase 2 (loopback `serve` + orchestrator `fleet` push) | shipped |
| v0.3 | `status --deep` (generation probe, catches GPU-OOM zombies), install records + `reinstall`, single preset location, `disable`/`enable` (cold standby) | shipped |
| v0.4 | **`asiai-priv` privileged helper** — a root-owned, content-validating helper replaces the wildcard sudoers: generate-don't-validate plists, a closed action allowlist, forced non-root run-as, and an append-only audit log; NOPASSWD is reduced to the helper alone | shipped |
| v0.5 | SMAppService app bundle + `asiai-launch` (named entry with custom icon in Background Items) | in review |
| later | Profile switching (apply/rollback), fleet inventory, web cockpit, MCP write tools | planned |

## Architecture (high level)

- **CLI**: `aisctl <command>` (standalone) or `asiai engine <command>`
  (auto-injected sub-CLI when `asiai-inference-server` is installed
  alongside `asiai`).
- **Python stdlib only** for the core (consistent with `asiai`). The
  package depends on `asiai>=1.15.0` (`auth.loopback` + the shared
  `fleet.command_spec`). Optional extras: `mcp` (for future write tools).
- **macOS Apple Silicon only**. We rely on `launchctl`, `vm_stat`,
  `sudo purge`, `pfctl`, `iogpu.wired_limit_mb`.
- **HTTP agent** for fleet operations (loopback `serve` + `fleet push`
  orchestrator, shipped v0.2).
- **TOML** for human-edited files (engine manifests, profiles, fleet
  inventory). JSON for runtime state.
- **Engine API keys stay out of plists and `ps`**: manifests declare
  `[binary].api_key_file` (path to an operator-created, mode-600 key
  file the engine reads at startup — e.g. `llama-server
  --api-key-file`) and `[network].api_key_file` (same file, used by the
  health/generation probes to authenticate). Only paths appear in
  manifests, generated plists, and process arguments — never key values.

## License

Apache-2.0 © 2026 Jean-Marc Nahlovsky
