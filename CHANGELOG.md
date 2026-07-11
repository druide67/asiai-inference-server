# Changelog

All notable changes to asiai-inference-server (the `aisctl` CLI and the
`ais_core` library) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **MTPLX engine driver** — full lifecycle support for
  [MTPLX](https://github.com/youssofal/MTPLX) (MLX-based OpenAI-compatible
  server with native multi-token-prediction speculative decoding, brew tap
  `youssofal/mtplx`): baseline manifest (loopback, port 8005), restart-only
  driver, upgrade whitelist + `asiai versions` provider entries. The daemon
  launches through the stable brew wrapper (`mtplx quickstart …`), which
  execs into the server inside the keg-versioned virtualenv — surviving
  `brew upgrade` without a reinstall. The SSD SessionBank cache is pinned
  `off` explicitly in every bundled manifest (upstream defaults it to
  on/100GB — youssofal/MTPLX#140).
- Bundled preset `qwen3.6-27b-mtplx-hermes-agent`: Qwen3.6-27B
  Optimized-Speed on the main-inference slot (8080, LAN bind), native MTP
  depth 3, turbo profile, thinking disabled at launch (`--reasoning off`),
  Qwen3.6 sampling recommendations, `[memory]` figures for the plan
  estimator.
- Optional `[network].api_key_file` manifest field: the health and
  generation probes attach `Authorization: Bearer <key>` read from the
  file at probe time — required for engines whose auth middleware covers
  `/health` itself on non-localhost binds (MTPLX). Only the path lives in
  the manifest, never the key.
- Optional `[memory].ctx_tokens` field: declared context budget for
  engines whose CLI takes no context-size flag (window is model-native).
  Used as the KV multiplier only when `program_args` carries no
  `--ctx-size`/`-c`; an explicit CLI flag always wins, and a
  present-but-malformed flag stays fail-closed `unknown`.

## [0.11.0](https://github.com/druide67/asiai-inference-server/compare/v0.10.0...v0.11.0) — 2026-07-09

### Added

- **Preset memory-cost estimator** (#32): `aisctl plan <preset> [--engine] [--json]`
  (and `asiai engine plan`) estimates the resident memory a preset needs —
  weights + KV cache + runtime overhead, each from its best source
  (measured ±10 % / declared ±20 % / computed ±35 %, worst source wins).
  Fail-closed: an unsourceable component makes the whole estimate `unknown`
  with zeroed bounds. GGUF metadata is deliberately never parsed (hybrid-
  attention models make the naive KV formula wrong by ~4×).
- Optional `[memory]` manifest section (`weights_mb`, `kv_bytes_per_token`,
  `overhead_mb`, `peak_extra_mb`) — strictly validated (positive finite
  numbers, unknown keys rejected). Bundled tuned presets now carry the
  figures previously recorded as comments.
- Calibration rings (`~/.local/state/asiai-inference-server/calibration/`):
  real footprint samples recorded after healthy start/install/reinstall/
  enable (full weight) and from the unload memory delta (half weight),
  keyed by preset × manifest digest, host-filtered, 10-sample ring.
  A fresh measurement beats any decomposition (±10 %).
- `GET /internal/v1/plan?preset=` on `aisctl serve` (loopback, Bearer):
  returns the frozen cost contract consumed by asiai ≥ 1.25's
  `/api/v1/plan` verdict route.

## [0.10.0](https://github.com/druide67/asiai-inference-server/compare/v0.9.0...v0.10.0) — 2026-07-07

Preset funnel — the companion release for asiai 1.22.0's install preset
picker, closing the silent-baseline trap end to end.

### Added

- **`GET /internal/v1/presets`** on `aisctl serve` (same loopback Bearer
  gate as the other read endpoints): the bundled tuned-manifest presets
  as `{preset, engine, display}` summaries, so the fleet cockpit's
  Install modal can offer them.
- **`install` accepts `args.preset`** through the loopback command
  funnel — shape-checked before any subprocess (leading-alphanumeric,
  ≤ 64 chars), then validated against the bundled preset registry by
  `aisctl install` itself. A preset smuggled onto any other verb never
  reaches the argv.
- **`engines-state` carries each engine's install-time preset**, so
  dashboards can label what a service was generated from.

## [0.9.0](https://github.com/druide67/asiai-inference-server/compare/v0.8.2...v0.9.0) — 2026-07-06

### Added

- **Cold-standby verbs through the loopback dispatcher**: `enable` and
  `disable` are now routable via `POST /internal/v1/command`, with the
  same engine-argv shape as start/stop. Pairs with asiai ≥ 1.18's shared
  command spec (60 s / 120 s budgets, REVERSIBLE classification) and the
  cockpit's Standby/Enable menu.

### Changed

- Depends on `asiai >= 1.18.0` (single-sourced command spec).

## [0.8.2](https://github.com/druide67/asiai-inference-server/compare/v0.8.1...v0.8.2) — 2026-07-06

### Fixed

- **Bundle-aware provisioned state**: engines whose LaunchDaemon lives in
  the SMAppService bundle (not `/Library/LaunchDaemons/`) are recognized
  as provisioned, so dormant bundle services report truthful
  STOPPED/DISABLED instead of falling back to AVAILABLE.
- `installed_model` falls back to the ACTIVE manifest
  (`ASIAI_LAUNCH_MANIFEST_DIR`) when the plist embeds none, so dormant
  services still display the model they would serve.

## [0.8.1](https://github.com/druide67/asiai-inference-server/compare/v0.8.0...v0.8.1) — 2026-07-06

### Fixed

- `installed_model` resolves preset symlinks, so a model published
  through a preset's `active.gguf` link shows its real name.

## [0.8.0](https://github.com/druide67/asiai-inference-server/compare/v0.7.0...v0.8.0) — 2026-07-06

Truthful lifecycle states.

### Changed

- **Port-aware liveness**: an engine is RUNNING only if ITS port
  answers — a neighbor's server can no longer make a stopped engine
  look alive.
- **New AVAILABLE state**: software present but service not
  provisioned, distinct from NOT_INSTALLED.
- **Network-first probing** with the process table as fallback, and
  `installed_model` read from the installed plist so non-running
  engines are labeled by what they WOULD serve.

## [0.7.0](https://github.com/druide67/asiai-inference-server/compare/v0.6.0...v0.7.0) — 2026-07-05

### Added

- **`GET /internal/v1/engines-state`** on `aisctl serve` (loopback
  Bearer): the rich lifecycle state of every manifest, cached a few
  seconds, so the co-located `asiai web` snapshot can show
  stopped/disabled/unhealthy engines instead of the poor
  reachable/unreachable split HTTP detection alone offers.

### Fixed

- Three first-real-deploy bundle findings (0.6.1): `bundle status`
  parsing, register scope, and smd job recognition.

## [0.6.0](https://github.com/druide67/asiai-inference-server/compare/v0.5.1...v0.6.0) — 2026-07-04

SMAppService bundle: the fleet's daemons get ONE named, iconed entry in the
macOS Background Items panel instead of a collection of unidentified Unix
executables. The packaged helper changes (no-double-load guard) — re-run
`aisctl bootstrap --install` per node after upgrading.

### Added

- **`asiai-launch`** — secure generic launcher the bundle's embedded
  LaunchDaemons exec. Confused-deputy guards: the executable comes from a
  hardcoded basename allowlist (never from the manifest), the per-service
  active manifest must be owned by root or the daemon user and not
  group/world-writable, and dynamic-linker env vars (`DYLD_*`/`LD_*`) are
  refused outright (I3 parity with the helper).
- **`aisctl bundle build`** — produces a signable `<App>.app` embedding one
  LaunchDaemon per engine service (sealed-bundle constraint: nothing
  machine- or tuning-specific inside; tuning lives in the active manifest).
- **`aisctl bundle activate`** — publishes the active manifest a service
  execs from (defaults to the preset recorded at install time).
- **`aisctl bundle register/unregister/status`** — thin wrapper over the
  bundle's SMAppService Swift helper.

### Security

- **No-double-load, machine-enforced on both sides** (story 2.6 acceptance
  criteria): `bundle register` hard-refuses while a legacy
  `/Library/LaunchDaemons` plist for a selected service still exists or its
  label is still loaded outside an app bundle; the privileged helper's
  `install-daemon`/`install-reserved-service` refuse a label already loaded
  from an app bundle (fail closed on unidentified sources). Helper diff
  security-gated line-by-line.
- **`bundle register` refuses without a GUI session** (`--allow-headless`
  opts out): the SMAppService approval toggle is GUI-only; a headless
  register parks daemons in `requiresApproval`, silently not running.
- **Bundle daemon user validated**: resolved via `getpwnam`, uid 0 or
  unknown accounts refused — no `$USER`-or-root fallback.

## [0.5.1](https://github.com/druide67/asiai-inference-server/compare/v0.5.0...v0.5.1) — 2026-07-03

Bootstrap idempotence fix — pin `>=0.5.1` for cutovers that rely on the
re-run-to-converge recovery path.

### Fixed

- **Bootstrap:** re-running `aisctl bootstrap --install` over an existing
  audit log no longer aborts with `chain component not root:wheel`. The
  audit log is deliberately `root:admin` (operator-readable refusals) and
  the I0 chain check now targets its parent directory — which the
  bootstrap itself pins `root:wheel`, non-group-writable — instead of the
  leaf. The invariant check itself is unchanged. Caught by a cutover
  rollback→reinstall rehearsal; covered by a regression test that
  exercises the real chain check.
- Refreshed the stale committed `uv.lock` (still resolved `asiai==1.8.0`;
  reproducible dev/CI now matches `pyproject`'s `asiai>=1.15.0`).

## [0.5.0](https://github.com/druide67/asiai-inference-server/compare/v0.4.0...v0.5.0) — 2026-07-02

Fleet groundwork + hardening on top of the privileged-helper model. This is
the release whose packaged helper matches the security-reviewed sources —
pin `>=0.5.0` when bootstrapping a new machine.

### Added

- The loopback server (`aisctl serve`) and the fleet client (`aisctl fleet
  push`) now consume the shared `asiai.fleet.command_spec` (single source of
  truth for the write-command whitelist and timeouts). The nesting invariant
  `client > edge > loopback` holds by construction; the wrapped `upgrade`
  passes an inner tool deadline so the operator sees the tool's real error,
  not a timeout kill. Requires `asiai>=1.15.0`.

### Fixed

- **Firewall:** `aisctl reinstall`/`uninstall` no longer tears the daemon
  down before discovering it cannot touch pf — a sudo/TTY preflight runs
  first, and anchor installation is idempotent (no-op when already current).
- **Bootstrap:** the helper's log directory is created root:wheel `0755` and
  the audit log is pre-created `0640 root:admin`, so refusal entries are
  operator-readable without sudo.
- **Helper:** a missing log directory now yields a clear refusal pointing at
  `aisctl bootstrap` instead of an opaque internal error; the audit log is
  created group-readable.
- **Lifecycle:** user-supplied model/template paths are resolved
  (`realpath`) before reaching the helper, so a symlinked model file (e.g.
  an `active.gguf` pointer) installs again.

## [0.4.0](https://github.com/druide67/asiai-inference-server/compare/v0.3.2...v0.4.0) — 2026-06-24

Privileged-helper release (retro-added entry).

### Added

- Root-owned privileged helper (`asiai-priv`) replacing wildcard sudoers
  rules: closed action allowlist, plist content validation at load,
  refuse-by-default, append-only audit log, `aisctl bootstrap
  --install/--verify/--rollback` with a recorded pre-bootstrap sudoers
  backup for lossless rollback.
- Reserved-service installs (`asiai web` / `aisctl serve`) through the
  helper with content-validated specs.

## [0.3.2](https://github.com/druide67/asiai-inference-server/compare/v0.3.1...v0.3.2) — 2026-06-13

OSS hygiene. No functional change.

### Changed

- Test suite uses a neutral `testuser` daemon-user placeholder instead of a
  personal login.

## [0.3.1](https://github.com/druide67/asiai-inference-server/compare/v0.3.0...v0.3.1) — 2026-06-13

Audit follow-up (2026-06-11). Correctness and hardening; no breaking changes.

### Fixed

- **Firewall (critical):** `block-port` anchors were a no-op. The generated
  `pf.conf` referenced the anchor but was missing the `load anchor … from …`
  directive, so it was never actually loaded. Added the load directive and an
  IPv6 block rule; a failing `pfctl -f` now raises instead of being swallowed.
- **Lifecycle:** `stop` no longer disables the daemon (it had wrongly passed
  `launchctl … -w`), so a stopped engine still comes back on reboot as
  documented — `disable` remains the durable cold-standby path. `start` and
  `restart` now honour `--dry-run`.
- **Manifest / plist:** `~` resolves against the daemon user rather than the
  invoking user, and the health URL maps a `0.0.0.0`/empty bind to
  `127.0.0.1`.
- Hardened error paths and preconditions across `ais_core` (manifest parsing,
  memory repair/purge, install-record validation) and surfaced previously
  silenced failures in the driver proxy.
- `aisctl fleet` with no subcommand prints a usage error instead of crashing;
  closed `asiai engine` parity gaps.

### Changed

- The test suite is now hermetic — it never touches the developer's real
  config or state — and several weak assertions were tightened.

### Docs

- README, roadmap, presets and engine docs realigned with v0.3.0.

## [0.3.0](https://github.com/druide67/asiai-inference-server/compare/v0.2.0...v0.3.0) — 2026-06-11

### Added

- **`aisctl status --deep`:** a generation-based health probe that catches a
  process which is up but serving a stale/zombie model.
- **`aisctl disable` / `enable`:** durable cold standby that survives reboot
  (`launchctl … -w`).
- **Install records + `aisctl reinstall`:** the preset and options used at
  install time are persisted, so a reinstall can no longer silently degrade
  the configuration.

### Changed

- User presets now resolve from a single location, `engine_manifests/presets/`.

### Fixed

- Deterministic synchronization for the cross-process lock test (a CI flake).

## [0.2.0](https://github.com/druide67/asiai-inference-server/compare/v0.1.0...v0.2.0) — 2026-05-30

### Added

- **`aisctl upgrade <engine>`:** native engine upgrade through a shared Homebrew
  whitelist, run behind the OperationsLock, with `--restart` and `--dry-run`.
- **`asiai.version_sources` provider:** exposes engine and tooling versions to
  `asiai versions`.

## [0.1.0](https://github.com/druide67/asiai-inference-server/releases/tag/v0.1.0) — 2026-05-28

First public release.

### Added

- The `aisctl` lifecycle CLI and the `ais_core` building blocks
  (install/uninstall, start/stop/restart, unload, purge, an fcntl-based
  operations lock and `RegistrationContext`) for managing local inference
  engines on Apple Silicon.
- Engines: Ollama, LM Studio, llama.cpp (llama-server), mlx-lm, vMLX and
  Rapid-MLX, plus the config-driven `llamacpp-aux` family and a TOML preset
  mechanism for tuned Hermes Agent stacks.
- `aisctl serve` loopback command server and Fleet Phase 2 (`fleet push`,
  `install-service`); sudoers bootstrap; pf firewall anchors.
- `asiai.engines` entry points so `asiai` discovers the engines managed here.
- CI matrix Python 3.11/3.12/3.13 × macOS 14/15, with OIDC trusted publishing
  to PyPI.

### Security

- Pre-release audit fixes: H1 (TOCTOU on `/tmp` → a `0700` staging dir),
  H2 (process-pattern validation), M2 (lock mode `0o600`), and removal of the
  `--keep-logs` escape hatch.
